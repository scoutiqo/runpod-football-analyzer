#!/usr/bin/env python3
"""
Simple video analysis - processes your actual video with minimal dependencies
"""
import os
import cv2
import numpy as np
import json
from pathlib import Path

def find_video_file():
    """Find a video file to analyze"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Look in current directory
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            return file
    
    # Look in files directory
    files_dir = Path('files')
    if files_dir.exists():
        for file in files_dir.iterdir():
            if any(file.name.lower().endswith(ext) for ext in video_extensions):
                return str(file)
    
    return None

def analyze_video_simple(video_path):
    """Simple video analysis using OpenCV and basic computer vision"""
    print(f"Analyzing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize YOLO for object detection
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Use nano model for speed
        print("YOLO model loaded successfully")
        yolo_available = True
    except Exception as e:
        print(f"YOLO not available: {e}")
        yolo_available = False
    
    # Analysis results
    results = {
        "video_info": {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration_seconds": total_frames / fps
        },
        "analysis": {
            "frames_processed": 0,
            "players_detected": [],
            "ball_detections": [],
            "events": [],
            "tracking_data": []
        }
    }
    
    frame_count = 0
    player_tracks = {}
    ball_tracks = []
    
    print("Starting frame-by-frame analysis...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # Process every 5th frame for speed
        if frame_count % 5 != 0:
            continue
        
        print(f"Processing frame {frame_count}/{total_frames} ({timestamp:.1f}s)")
        
        # Object detection with YOLO
        if yolo_available:
            try:
                detections = model.predict(frame, conf=0.3, verbose=False)[0]
                
                # Process detections
                frame_players = []
                frame_ball = None
                
                for box in detections.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Class 0 = person, Class 32 = sports ball
                    if cls == 0 and conf > 0.5:  # Person
                        player_id = len(frame_players)
                        player_data = {
                            "id": player_id,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(conf),
                            "center_x": float((x1 + x2) / 2),
                            "center_y": float((y1 + y2) / 2),
                            "timestamp": timestamp
                        }
                        frame_players.append(player_data)
                        
                        # Track player movement
                        if player_id not in player_tracks:
                            player_tracks[player_id] = []
                        player_tracks[player_id].append({
                            "timestamp": timestamp,
                            "x": player_data["center_x"],
                            "y": player_data["center_y"]
                        })
                    
                    elif cls == 32 and conf > 0.3:  # Sports ball
                        frame_ball = {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(conf),
                            "center_x": float((x1 + x2) / 2),
                            "center_y": float((y1 + y2) / 2),
                            "timestamp": timestamp
                        }
                        ball_tracks.append(frame_ball)
                
                # Store frame results
                results["analysis"]["players_detected"].extend(frame_players)
                if frame_ball:
                    results["analysis"]["ball_detections"].append(frame_ball)
                
            except Exception as e:
                print(f"Detection error on frame {frame_count}: {e}")
        
        # Simple motion analysis (even without YOLO)
        if not yolo_available:
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple motion detection using frame difference
            if hasattr(analyze_video_simple, 'prev_gray'):
                diff = cv2.absdiff(gray, analyze_video_simple.prev_gray)
                motion_pixels = np.sum(diff > 30)
                
                if motion_pixels > 1000:  # Significant motion
                    results["analysis"]["events"].append({
                        "timestamp": timestamp,
                        "type": "motion_detected",
                        "intensity": int(motion_pixels)
                    })
            
            analyze_video_simple.prev_gray = gray
        
        # Limit processing for demo
        if frame_count > 300:  # Process first 300 frames
            break
    
    cap.release()
    
    # Finalize results
    results["analysis"]["frames_processed"] = frame_count
    results["analysis"]["unique_players"] = len(player_tracks)
    results["analysis"]["ball_detections_count"] = len(ball_tracks)
    results["analysis"]["player_tracks"] = player_tracks
    results["analysis"]["ball_tracks"] = ball_tracks
    
    print(f"Analysis complete!")
    print(f"Frames processed: {frame_count}")
    print(f"Players detected: {len(player_tracks)}")
    print(f"Ball detections: {len(ball_tracks)}")
    print(f"Events detected: {len(results['analysis']['events'])}")
    
    return results

def main():
    print("=" * 60)
    print("SIMPLE VIDEO ANALYSIS")
    print("=" * 60)
    print("This will analyze your ACTUAL video file")
    print("=" * 60)
    
    # Find video file
    video_file = find_video_file()
    if not video_file:
        print("ERROR: No video file found!")
        print("Please place a video file (.mp4, .avi, .mov, .mkv) in:")
        print("1. Current directory, or")
        print("2. files/ directory")
        return
    
    print(f"Found video: {video_file}")
    
    # Check file size
    file_size = os.path.getsize(video_file) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.1f} MB")
    
    # Run analysis
    print("\nStarting REAL video analysis...")
    results = analyze_video_simple(video_file)
    
    if results:
        # Save results
        output_file = f"real_analysis_{Path(video_file).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("This is REAL data from your video:")
        print(f"- Frames processed: {results['analysis']['frames_processed']}")
        print(f"- Players detected: {results['analysis']['unique_players']}")
        print(f"- Ball detections: {results['analysis']['ball_detections_count']}")
        print(f"- Events detected: {len(results['analysis']['events'])}")
        print("\nThis is NOT simulation - it's your actual video data!")
    else:
        print("\n" + "=" * 60)
        print("ANALYSIS FAILED")
        print("=" * 60)

if __name__ == "__main__":
    main()
