# server/server.py
import os
import asyncio
import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.websockets import WebSocketState

FILES_ROOT = os.getenv("FILES_ROOT", "./files")

# 1) Create the app FIRST
app = FastAPI(title="ScoutIQO Analyzer")

# 2) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# 3) Mount static artifacts (/files/jobs/<id>/...)
app.mount("/files", StaticFiles(directory=FILES_ROOT), name="files")

# Global storage for live analysis
live_sessions: Dict[str, Dict[str, Any]] = {}
active_websockets: Dict[str, List[WebSocket]] = {}

# 4) Include routers AFTER app exists (imports may reference `app`)
#    These modules must each define `router = APIRouter(...)`
from server.monitor import router as monitor_router       # GET /monitor/{job_id} (SSE)
from server.progress import router as progress_router     # POST /progress/{job_id}
from server.train_routes import router as train_router    # Training routes
from server.live_track_demo import router as demo_router  # POST /demo/track (local proof)
from server.simple_demo import router as simple_demo_router  # POST /demo/simple (simple proof)
# Optional: your segmented flow
# from server.routes_ingest import router as ingest_router # /upload, /analyze, /files/...

app.include_router(monitor_router)
app.include_router(progress_router)
app.include_router(train_router)
app.include_router(demo_router)
app.include_router(simple_demo_router)
# app.include_router(ingest_router)

# Basic endpoints
@app.get("/")
def root():
    return {"ok": True, "message": "ScoutIQO Analyzer Server"}

@app.get("/health")
def health():
    return {"ok": True}

# (Optional) Friendly GET to avoid 405 confusion
from fastapi.responses import JSONResponse
@app.get("/ai/analyze")
def analyze_get():
    return JSONResponse({"detail":"Use POST /analyze with JSON body: {\"job_id\":\"...\",\"segment_urls\":[]}"})

# ==================== LIVE VIDEO ANALYSIS ====================

class LiveVideoAnalyzer:
    """Real-time video analysis with actual tracking data"""
    
    def __init__(self, video_path: str, session_id: str):
        self.video_path = video_path
        self.session_id = session_id
        self.cap = None
        self.detector = None
        self.tracker = None
        self.ball_tracker = None
        self.team_assigner = None
        self.is_running = False
        self.current_frame = 0
        self.fps = 30.0
        self.total_frames = 0

        # Tracking data
        self.players = []
        self.ball = None
        self.events = []
        self.metrics = {}
        
        # Training integration
        self.training_enabled = True
        
    async def initialize(self):
        """Initialize video capture and AI models"""
        try:
            # Import AI components
            from detector import Detector
            from tracker_players import PlayerTracker
            from ball_tracker import BallTracker
            from team_assign import TeamAssigner
            
            # Import Phase 1 & 2 components
            from calibration import HomographyEstimator
            from metrics.physical import PhysicalMetricsManager
            
            # Import adaptive filtering components
            from calibration.pose_filter import CameraPoseFilter
            from tracking.world_tracker import WorldTrackerManager
            from tracking.ball_physics import BallPhysicsTracker
            from models.confidence_heads import (
                PlayerConfidenceEstimator, 
                BallConfidenceEstimator,
                CameraPoseConfidenceEstimator
            )
            
            # Import Phoenix adaptive monocular system
            from analyzers.phoenix_runner import PhoenixRunner
            
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize AI models
            self.detector = Detector({"weights_bucket": "local", "weights_path": "yolov8n.pt"})
            self.tracker = PlayerTracker({"max_age": 30, "min_hits": 3})
            self.ball_tracker = BallTracker({"min_conf": 0.10, "class_id": 32})
            self.team_assigner = TeamAssigner()
            
            # Initialize Phase 1: Camera pose and world mapping
            self.homography_estimator = HomographyEstimator()
            self.current_homography = None
            
            # Initialize Phase 2: Physical metrics
            self.physical_metrics = PhysicalMetricsManager()
            
            # Initialize adaptive filtering components
            self.pose_filter = CameraPoseFilter()
            self.world_tracker = WorldTrackerManager()
            self.ball_physics_tracker = BallPhysicsTracker()
            
            # Initialize confidence estimators
            self.player_confidence_estimator = PlayerConfidenceEstimator()
            self.ball_confidence_estimator = BallConfidenceEstimator()
            self.camera_pose_confidence_estimator = CameraPoseConfidenceEstimator()
            
            # Initialize Phoenix adaptive monocular system
            self.phoenix_runner = PhoenixRunner(device="cpu")
            
            # Set camera intrinsics (will be estimated or provided)
            K_default = np.array([
                [1000, 0, 640],
                [0, 1000, 360],
                [0, 0, 1]
            ], dtype=np.float32)
            self.phoenix_runner.set_camera_intrinsics(K_default)
            
            return True
        except Exception as e:
            print(f"Failed to initialize analyzer: {e}")
            return False
    
    async def process_frame(self):
        """Process a single frame and return tracking data with world coordinates"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.current_frame += 1
        timestamp = self.current_frame / self.fps
        
        try:
            # Run detection first
            detections = self.detector.infer(frame)
            
            # PHOENIX ADAPTIVE MONOCULAR MODE
            # Use sliding window bundle optimization for true adaptive tracking
            phoenix_detections = {
                "players": [],
                "ball": None
            }
            
            # Convert detections to Phoenix format
            player_dets = {"xyxy": [], "conf": [], "cls": []}
            ball_dets = {"xyxy": [], "conf": [], "cls": []}
            
            for i, (x1, y1, x2, y2) in enumerate(detections.get("xyxy", [])):
                conf = detections.get("conf", [0.0])[i]
                cls = detections.get("cls", [-1])[i]
                
                if cls == 0:  # person
                    player_dets["xyxy"].append([x1, y1, x2, y2])
                    player_dets["conf"].append(conf)
                    player_dets["cls"].append(cls)
                    
                    # Add to Phoenix detections
                    phoenix_detections["players"].append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf
                    })
                elif cls == 32:  # sports ball
                    ball_dets["xyxy"].append([x1, y1, x2, y2])
                    ball_dets["conf"].append(conf)
                    ball_dets["cls"].append(cls)
                    
                    # Add to Phoenix detections
                    phoenix_detections["ball"] = {
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf
                    }
            
            # Process with Phoenix system
            phoenix_result = self.phoenix_runner.process_frame(frame, phoenix_detections)
            
            # Submit window data for online training if available
            if hasattr(self, 'session_id') and phoenix_result.get("phoenix_optimized", False):
                await self._submit_training_window(phoenix_detections, frame)
            
            # Use Phoenix results if available, otherwise fall back to adaptive filtering
            if phoenix_result.get("phoenix_optimized", False):
                return phoenix_result
            
            # FALLBACK: ADAPTIVE CAMERA POSE FILTERING
            # Update pose filter every frame for smooth tracking
            self.current_homography = self.pose_filter.update(frame, dt=1.0/self.fps)
            
            # Get pose confidence
            pose_confidence = self.pose_filter.get_confidence()
            
            # Track players (legacy tracker for pixel coordinates)
            self.players = self.tracker.update(player_dets) or []
            
            # Track ball (legacy tracker for pixel coordinates)
            self.ball = self.ball_tracker.update(ball_dets)
            
            # Team assignment
            if self.players:
                self.team_assigner.observe(
                    frame,
                    [{"id": p["id"], "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"]} for p in self.players]
                )
            
            # ADAPTIVE WORLD-SPACE TRACKING
            # Update world tracker with detections
            world_tracks = self.world_tracker.update(
                self.players, 
                self.current_homography, 
                timestamp, 
                dt=1.0/self.fps
            )
            
            # Update ball physics tracker
            ball_world_state = self.ball_physics_tracker.update(
                self.ball,
                self.current_homography,
                timestamp,
                dt=1.0/self.fps
            )
            
            # Convert to enhanced format with confidence scores
            players_with_world_coords = []
            for track in world_tracks:
                # Get confidence scores for this player
                player_bbox = None
                for p in self.players:
                    if p["id"] == track["global_id"]:
                        player_bbox = (p["x1"], p["y1"], p["x2"], p["y2"])
                        break
                
                confidence_scores = {}
                if player_bbox is not None:
                    try:
                        # Extract player crop for confidence estimation
                        x1, y1, x2, y2 = [int(coord) for coord in player_bbox]
                        player_crop = frame[y1:y2, x1:x2]
                        
                        if player_crop.size > 0:
                            confidence_scores = self.player_confidence_estimator.estimate_confidence(
                                player_crop, player_bbox, {
                                    "speed": track["speed_mps"],
                                    "acceleration": 0.0,  # Could calculate from history
                                    "direction_change": 0.0,
                                    "prediction_error": 0.0
                                }
                            )
                    except Exception as e:
                        print(f"Player confidence estimation failed: {e}")
                        confidence_scores = {"combined": 0.5}
                
                players_with_world_coords.append({
                    "id": track["global_id"],
                    "position_px": track.get("position_px", [0, 0]),
                    "position_world": track["position"],
                    "velocity_world": track["velocity"],
                    "speed_mps": track["speed_mps"],
                    "speed_kmh": track["speed_kmh"],
                    "team": self.team_assigner.get_team(track["global_id"]),
                    "bbox": track.get("bbox", [0, 0, 0, 0]),
                    "confidence_scores": confidence_scores,
                    "visibility_score": track["visibility_score"]
                })
            
            # Enhanced ball data with physics
            ball_data = None
            if ball_world_state:
                ball_confidence_scores = {}
                if self.ball:
                    try:
                        # Extract ball crop for confidence estimation
                        x1, y1, x2, y2 = [int(coord) for coord in [self.ball["x1"], self.ball["y1"], self.ball["x2"], self.ball["y2"]]]
                        ball_crop = frame[y1:y2, x1:x2]
                        
                        if ball_crop.size > 0:
                            ball_confidence_scores = self.ball_confidence_estimator.estimate_confidence(
                                ball_crop, (x1, y1, x2, y2)
                            )
                    except Exception as e:
                        print(f"Ball confidence estimation failed: {e}")
                        ball_confidence_scores = {"combined": 0.5}
                
                ball_data = {
                    "position_px": [self.ball["x1"], self.ball["y1"], self.ball["x2"], self.ball["y2"]] if self.ball else None,
                    "position_world": ball_world_state["position"],
                    "velocity_world": ball_world_state["velocity"],
                    "speed_mps": ball_world_state["speed_mps"],
                    "speed_kmh": ball_world_state["speed_kmh"],
                    "height": ball_world_state["height"],
                    "in_flight": ball_world_state["in_flight"],
                    "confidence": self.ball.get("conf", 0.0) if self.ball else 0.0,
                    "confidence_scores": ball_confidence_scores,
                    "visibility_score": ball_world_state["visibility_score"]
                }
            
            # Calculate enhanced metrics
            self.metrics = self._calculate_enhanced_metrics()
            
            return {
                "frame": self.current_frame,
                "timestamp": timestamp,
                "players": players_with_world_coords,
                "ball": ball_data,
                "homography_available": self.current_homography is not None,
                "pose_confidence": pose_confidence,
                "adaptive_tracking": True,
                "metrics": self.metrics
            }
            
        except Exception as e:
            print(f"Error processing frame {self.current_frame}: {e}")
            return None
    
    def _calculate_enhanced_metrics(self):
        """Calculate enhanced real-time metrics with physical data"""
        base_metrics = {
            "players_count": len(self.players),
            "ball_detected": self.ball is not None,
            "frame_rate": self.fps,
            "progress": (self.current_frame / self.total_frames) * 100 if self.total_frames > 0 else 0,
            "homography_available": self.current_homography is not None
        }
        
        # Add physical metrics if available
        if hasattr(self, 'physical_metrics'):
            all_physical = self.physical_metrics.get_all_metrics()
            
            # Team assignments for aggregation
            team_assignments = {}
            for p in self.players:
                team_assignments[p["id"]] = self.team_assigner.get_team(p["id"])
            
            team_metrics = self.physical_metrics.get_team_metrics(team_assignments)
            
            base_metrics.update({
                "physical_metrics": all_physical,
                "team_metrics": team_metrics,
                "total_distance_km": sum(m["distance_km"] for m in all_physical.values()),
                "max_speed_kmh": max((m["max_speed_kmh"] for m in all_physical.values()), default=0.0),
                "total_sprints": sum(m["sprint_bursts"] for m in all_physical.values())
            })
        
        return base_metrics
    
    async def _submit_training_window(self, detections, frame):
        """Submit window data for online training"""
        try:
            # Prepare training data
            training_data = {
                "init": {
                    "rvec": np.zeros((25, 3)),  # Window size
                    "tvec": np.zeros((25, 3)),
                    "players_xy": np.random.uniform(10, 95, (25, 22, 2)),  # 22 players
                    "ball_xyz": np.zeros((25, 3)),
                    "ball_v": np.zeros((25, 3))
                },
                "meas": {
                    "u_players": np.zeros((25, 22, 2)),  # Player measurements
                    "u_ball": [None] * 25,  # Ball measurements
                    "h_ball": [None] * 25   # Ball height measurements
                },
                "feats": {
                    "players": np.random.randn(25 * 22, 64),  # Player features
                    "ball_xy": np.random.randn(25, 64),       # Ball features
                    "ball_h": np.random.randn(25, 64),        # Ball height features
                    "field": np.random.randn(25, 64)          # Field features
                },
                "pitch_pts_world": self.phoenix_runner.optimizer.pitch_keypoints_world,
                "pitch_pts_img": [None] * 25,  # Would be populated from field detection
                "dt": 1.0 / self.fps
            }
            
            # Submit to training system
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:8080/train/window/{self.session_id}",
                    json=training_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"Submitted training window: {result}")
            
        except Exception as e:
            print(f"Failed to submit training window: {e}")
    
    async def start_analysis(self):
        """Start the live analysis loop"""
        self.is_running = True
        
        while self.is_running and self.cap and self.cap.isOpened():
            tracking_data = await self.process_frame()
            
            if tracking_data:
                # Broadcast to all connected WebSockets
                await self._broadcast_data(tracking_data)
            
            # Control frame rate
            await asyncio.sleep(1.0 / self.fps)
        
        await self.cleanup()
    
    async def _broadcast_data(self, data: dict):
        """Broadcast tracking data to all connected WebSockets"""
        if self.session_id in active_websockets:
            disconnected = []
            for websocket in active_websockets[self.session_id]:
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(data)
                    else:
                        disconnected.append(websocket)
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected WebSockets
            for ws in disconnected:
                active_websockets[self.session_id].remove(ws)
    
    async def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        self.is_running = False

# Live analysis endpoints
@app.post("/live/upload")
async def upload_video_for_live_analysis(file: UploadFile = File(...), bg: BackgroundTasks = BackgroundTasks()):
    """Upload video for live analysis"""
    session_id = f"live_{int(time.time())}"
    
    # Save uploaded video
    video_path = Path(FILES_ROOT) / "live" / f"{session_id}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(video_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Initialize analyzer
    analyzer = LiveVideoAnalyzer(str(video_path), session_id)
    success = await analyzer.initialize()
    
    if not success:
        return JSONResponse({"error": "Failed to initialize analyzer"}, status_code=500)
    
    # Store session
    live_sessions[session_id] = {
        "analyzer": analyzer,
        "video_path": str(video_path),
        "upload_time": time.time(),
        "status": "ready"
    }
    
    return JSONResponse({
        "session_id": session_id,
        "status": "ready",
        "message": "Video uploaded successfully. Start analysis with /live/start/{session_id}"
    })

@app.post("/live/start/{session_id}")
async def start_live_analysis(session_id: str, bg: BackgroundTasks = BackgroundTasks()):
    """Start live analysis of uploaded video"""
    if session_id not in live_sessions:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    session = live_sessions[session_id]
    analyzer = session["analyzer"]
    
    if session["status"] == "running":
        return JSONResponse({"error": "Analysis already running"}, status_code=400)
    
    # Start analysis in background
    bg.add_task(analyzer.start_analysis)
    session["status"] = "running"
    
    return JSONResponse({
        "session_id": session_id,
        "status": "running",
        "message": "Live analysis started. Connect to WebSocket at /live/stream/{session_id}"
    })

@app.websocket("/live/stream/{session_id}")
async def live_data_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for live tracking data"""
    await websocket.accept()
    
    if session_id not in live_sessions:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    # Add WebSocket to active connections
    if session_id not in active_websockets:
        active_websockets[session_id] = []
    active_websockets[session_id].append(websocket)
    
    try:
        # Keep connection alive and send periodic updates
        while True:
            if session_id in live_sessions:
                session = live_sessions[session_id]
                if session["status"] == "running":
                    # Send session info
                    await websocket.send_json({
                        "type": "session_info",
                        "session_id": session_id,
                        "status": session["status"],
                        "timestamp": time.time()
                    })
                else:
                    await websocket.send_json({
                        "type": "status",
                        "status": session["status"],
                        "message": "Analysis not running"
                    })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Session not found"
                })
                break
            
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        pass
    finally:
        # Remove WebSocket from active connections
        if session_id in active_websockets:
            if websocket in active_websockets[session_id]:
                active_websockets[session_id].remove(websocket)

    @app.get("/live/status/{session_id}")
    async def get_live_status(session_id: str):
        """Get status of live analysis session"""
        if session_id not in live_sessions:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        
        session = live_sessions[session_id]
        analyzer = session["analyzer"]
        
        # Get enhanced status with adaptive metrics
        status_data = {
            "session_id": session_id,
            "status": session["status"],
            "current_frame": analyzer.current_frame,
            "total_frames": analyzer.total_frames,
            "fps": analyzer.fps,
            "progress": (analyzer.current_frame / analyzer.total_frames) * 100 if analyzer.total_frames > 0 else 0,
            "connected_clients": len(active_websockets.get(session_id, [])),
            "adaptive_tracking": True,
            "homography_available": analyzer.current_homography is not None,
            "pose_confidence": analyzer.pose_filter.get_confidence() if hasattr(analyzer, 'pose_filter') else 0.0
        }
        
        # Add physical metrics if available
        if hasattr(analyzer, 'physical_metrics'):
            all_physical = analyzer.physical_metrics.get_all_metrics()
            status_data.update({
                "total_distance_km": sum(m["distance_km"] for m in all_physical.values()),
                "max_speed_kmh": max((m["max_speed_kmh"] for m in all_physical.values()), default=0.0),
                "total_sprints": sum(m["sprint_bursts"] for m in all_physical.values())
            })
        
        return JSONResponse(status_data)

@app.post("/live/stop/{session_id}")
async def stop_live_analysis(session_id: str):
    """Stop live analysis"""
    if session_id not in live_sessions:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    session = live_sessions[session_id]
    analyzer = session["analyzer"]
    
    analyzer.is_running = False
    session["status"] = "stopped"
    
    return JSONResponse({
        "session_id": session_id,
        "status": "stopped",
        "message": "Live analysis stopped"
    })