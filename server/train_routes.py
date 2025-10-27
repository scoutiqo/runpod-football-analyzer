# server/train_routes.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import asyncio
import json
import time
import logging
from typing import Dict, Optional
import torch
import numpy as np

from trainer.online_trainer import OnlineTrainer

logger = logging.getLogger(__name__)

router = APIRouter()

# Global storage for training sessions
TRAIN_QUEUES = {}     # job_id -> asyncio.Queue
TRAINERS = {}         # job_id -> OnlineTrainer
TRAINING_LOOPS = {}   # job_id -> asyncio.Task
WINDOW_BUFFERS = {}   # job_id -> list of window data

async def _publish(job_id: str, msg: dict):
    """Publish message to training stream"""
    q = TRAIN_QUEUES.get(job_id)
    if q:
        try:
            await q.put(json.dumps(msg))
        except asyncio.QueueFull:
            logger.warning(f"Training queue full for job {job_id}")

async def get_next_window(job_id: str) -> Optional[dict]:
    """Get next window data for training"""
    buffer = WINDOW_BUFFERS.get(job_id, [])
    if buffer:
        return buffer.pop(0)
    return None

async def consume_window_solution(job_id: str, solution: dict):
    """Consume optimized window solution"""
    # This would integrate with your existing analyzer to update tracks
    logger.info(f"Consumed window solution for job {job_id}")

@router.get("/train/stream/{job_id}")
async def train_stream(job_id: str):
    """SSE stream for training metrics"""
    q = asyncio.Queue(maxsize=256)
    TRAIN_QUEUES[job_id] = q
    
    async def gen():
        # Send initial connection message
        yield b"retry: 2000\n\n"
        yield b"data: " + json.dumps({"type": "connected", "job_id": job_id}).encode() + b"\n\n"
        
        try:
            while True:
                try:
                    # Wait for message with timeout
                    m = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield b"data: " + m.encode() + b"\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield b"data: " + json.dumps({"type": "keepalive", "timestamp": time.time()}).encode() + b"\n\n"
        except asyncio.CancelledError:
            logger.info(f"Training stream cancelled for job {job_id}")
        finally:
            TRAIN_QUEUES.pop(job_id, None)
            logger.info(f"Training stream closed for job {job_id}")
    
    return StreamingResponse(gen(), media_type="text/event-stream")

@router.post("/train/start/{job_id}")
async def train_start(job_id: str, background: BackgroundTasks):
    """Start online training for a job"""
    if job_id in TRAINERS:
        raise HTTPException(400, "Training already active for this job")
    
    # Initialize trainer with default camera intrinsics
    K = np.array([
        [1500, 0, 960],
        [0, 1500, 540],
        [0, 0, 1]
    ], dtype=np.float32)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainer = OnlineTrainer(K_np=K, device=device, ckpt_dir=f"./files/models/{job_id}")
    
    # Try to load existing checkpoint
    trainer.load(tag="latest")
    
    TRAINERS[job_id] = trainer
    WINDOW_BUFFERS[job_id] = []
    
    async def training_loop():
        """Main training loop"""
        step = 0
        last_checkpoint = 0
        
        try:
            while not trainer._stop:
                # Get next window data
                data = await get_next_window(job_id)
                if data is None:
                    await _publish(job_id, {"type": "train_idle", "step": step})
                    await asyncio.sleep(0.2)
                    continue
                
                # Training callback
                def cb(msg):
                    asyncio.create_task(_publish(job_id, msg))
                
                # Train on window
                try:
                    out = trainer.train_on_window(**data, cb=cb)
                    
                    # Consume solution
                    await consume_window_solution(job_id, out)
                    
                    # Periodic checkpoint
                    if step - last_checkpoint >= 25:  # Every 25 steps
                        checkpoint_path = trainer.save(tag=f"{job_id}_step_{step}")
                        await _publish(job_id, {
                            "type": "checkpoint",
                            "path": checkpoint_path,
                            "step": step
                        })
                        last_checkpoint = step
                    
                    step += 1
                    
                except Exception as e:
                    logger.error(f"Training step failed: {e}")
                    await _publish(job_id, {
                        "type": "train_error",
                        "error": str(e),
                        "step": step
                    })
                    await asyncio.sleep(1.0)  # Wait before retrying
        
        except Exception as e:
            logger.error(f"Training loop failed: {e}")
            await _publish(job_id, {
                "type": "train_fatal_error",
                "error": str(e)
            })
        finally:
            # Cleanup
            TRAINERS.pop(job_id, None)
            WINDOW_BUFFERS.pop(job_id, None)
            TRAINING_LOOPS.pop(job_id, None)
            await _publish(job_id, {"type": "train_stopped"})
    
    # Start training loop
    loop_task = asyncio.create_task(training_loop())
    TRAINING_LOOPS[job_id] = loop_task
    
    await _publish(job_id, {
        "type": "train_started",
        "job_id": job_id,
        "device": device
    })
    
    return {"ok": True, "job_id": job_id, "device": device}

@router.post("/train/stop/{job_id}")
async def train_stop(job_id: str):
    """Stop online training for a job"""
    trainer = TRAINERS.get(job_id)
    if trainer:
        trainer.stop()
    
    # Cancel training loop
    loop_task = TRAINING_LOOPS.get(job_id)
    if loop_task:
        loop_task.cancel()
    
    await _publish(job_id, {"type": "train_stopping"})
    
    return {"ok": True, "job_id": job_id}

@router.get("/train/status/{job_id}")
async def train_status(job_id: str):
    """Get training status for a job"""
    trainer = TRAINERS.get(job_id)
    if not trainer:
        return {"status": "not_training", "job_id": job_id}
    
    metrics = trainer.get_metrics_summary()
    metrics["job_id"] = job_id
    return metrics

@router.post("/train/window/{job_id}")
async def submit_window(job_id: str, window_data: dict):
    """Submit window data for training"""
    if job_id not in WINDOW_BUFFERS:
        raise HTTPException(404, "Training not started for this job")
    
    # Convert numpy arrays to lists for JSON serialization
    processed_data = {}
    for key, value in window_data.items():
        if isinstance(value, np.ndarray):
            processed_data[key] = value.tolist()
        elif isinstance(value, dict):
            processed_data[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    processed_data[key][subkey] = subvalue.tolist()
                else:
                    processed_data[key][subkey] = subvalue
        else:
            processed_data[key] = value
    
    WINDOW_BUFFERS[job_id].append(processed_data)
    
    # Keep buffer size reasonable
    if len(WINDOW_BUFFERS[job_id]) > 10:
        WINDOW_BUFFERS[job_id] = WINDOW_BUFFERS[job_id][-10:]
    
    return {"ok": True, "buffer_size": len(WINDOW_BUFFERS[job_id])}

@router.get("/train/ui/{job_id}")
def train_ui(job_id: str):
    """Live training UI with curves"""
    return HTMLResponse(f"""
<!doctype html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Online Training – {job_id}</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
        }}
        .status {{
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-weight: bold;
        }}
        .status.training {{ background: #d4edda; color: #155724; }}
        .status.idle {{ background: #fff3cd; color: #856404; }}
        .status.error {{ background: #f8d7da; color: #721c24; }}
        .row {{
            display: flex;
            gap: 24px;
            margin-bottom: 20px;
        }}
        .log-panel {{
            width: 420px;
            height: 400px;
            overflow: auto;
            background: #111;
            color: #eee;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }}
        .chart-panel {{
            flex: 1;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .controls {{
            margin-bottom: 20px;
        }}
        button {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
        }}
        button:hover {{ background: #0056b3; }}
        button:disabled {{ background: #6c757d; cursor: not-allowed; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Online Training Dashboard</h1>
        <div id="status" class="status idle">Connecting...</div>
        
        <div class="controls">
            <button id="startBtn" onclick="startTraining()">Start Training</button>
            <button id="stopBtn" onclick="stopTraining()" disabled>Stop Training</button>
            <button onclick="clearLog()">Clear Log</button>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value" id="stepCount">0</div>
                <div class="metric-label">Training Steps</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="lossValue">0.000</div>
                <div class="metric-label">Loss</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="rewardValue">0.000</div>
                <div class="metric-label">Total Reward</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="lrValue">0.0001</div>
                <div class="metric-label">Learning Rate</div>
            </div>
        </div>
        
        <div class="row">
            <div class="log-panel" id="log"></div>
            <div class="chart-panel">
                <canvas id="chart" width="600" height="400"></canvas>
            </div>
        </div>
    </div>

    <script>
        const jobId = '{job_id}';
        const log = document.getElementById('log');
        const status = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        const ctx = document.getElementById('chart').getContext('2d');
        const data = {{
            labels: [],
            loss: [],
            reward: [],
            reward_proj: [],
            reward_field: [],
            reward_phys: []
        }};
        
        let es = null;
        
        function logMessage(msg) {{
            const timestamp = new Date().toLocaleTimeString();
            log.textContent += `[${{timestamp}}] ${{msg}}\\n`;
            log.scrollTop = log.scrollHeight;
        }}
        
        function updateStatus(type, message) {{
            status.className = `status ${{type}}`;
            status.textContent = message;
        }}
        
        function updateMetrics(step, loss, reward, lr) {{
            document.getElementById('stepCount').textContent = step;
            document.getElementById('lossValue').textContent = loss.toFixed(4);
            document.getElementById('rewardValue').textContent = reward.toFixed(4);
            document.getElementById('lrValue').textContent = lr.toExponential(2);
        }}
        
        function connectStream() {{
            if (es) es.close();
            
            es = new EventSource(`/train/stream/${{jobId}}`);
            
            es.onopen = () => {{
                logMessage('Connected to training stream');
                updateStatus('training', 'Connected');
            }};
            
            es.onmessage = (e) => {{
                try {{
                    const msg = JSON.parse(e.data);
                    handleMessage(msg);
                }} catch (err) {{
                    logMessage(`Parse error: ${{err.message}}`);
                }}
            }};
            
            es.onerror = () => {{
                logMessage('Stream error - reconnecting...');
                updateStatus('error', 'Connection Error');
                setTimeout(connectStream, 2000);
            }};
        }}
        
        function handleMessage(msg) {{
            switch (msg.type) {{
                case 'connected':
                    logMessage(`Connected to job ${{msg.job_id}}`);
                    break;
                    
                case 'train_started':
                    logMessage(`Training started on ${{msg.device}}`);
                    updateStatus('training', 'Training Active');
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    break;
                    
                case 'train_tick':
                    logMessage(`Step ${{msg.step}}: Loss=${{msg.loss.toFixed(4)}}, Reward=${{msg.total_reward.toFixed(4)}}`);
                    updateMetrics(msg.step, msg.loss, msg.total_reward, msg.lr);
                    
                    // Update chart
                    data.labels.push(msg.step);
                    data.loss.push(msg.loss);
                    data.reward.push(msg.total_reward);
                    if (msg.rewards) {{
                        data.reward_proj.push(msg.rewards.reward_proj || 0);
                        data.reward_field.push(msg.rewards.reward_field || 0);
                        data.reward_phys.push(msg.rewards.reward_phys || 0);
                    }}
                    
                    // Keep only recent data
                    const maxPoints = 100;
                    if (data.labels.length > maxPoints) {{
                        data.labels = data.labels.slice(-maxPoints);
                        data.loss = data.loss.slice(-maxPoints);
                        data.reward = data.reward.slice(-maxPoints);
                        data.reward_proj = data.reward_proj.slice(-maxPoints);
                        data.reward_field = data.reward_field.slice(-maxPoints);
                        data.reward_phys = data.reward_phys.slice(-maxPoints);
                    }}
                    
                    drawChart();
                    break;
                    
                case 'train_idle':
                    updateStatus('idle', 'Waiting for data...');
                    break;
                    
                case 'checkpoint':
                    logMessage(`Checkpoint saved: ${{msg.path}}`);
                    break;
                    
                case 'train_error':
                    logMessage(`Training error: ${{msg.error}}`);
                    updateStatus('error', 'Training Error');
                    break;
                    
                case 'train_stopped':
                    logMessage('Training stopped');
                    updateStatus('idle', 'Training Stopped');
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    break;
                    
                default:
                    logMessage(`Unknown message: ${{msg.type}}`);
            }}
        }}
        
        function drawChart() {{
            if (data.labels.length < 2) return;
            
            ctx.clearRect(0, 0, 600, 400);
            
            // Draw axes
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 1;
            ctx.strokeRect(50, 20, 520, 350);
            
            // Draw grid
            ctx.strokeStyle = '#f0f0f0';
            for (let i = 1; i < 5; i++) {{
                const y = 20 + (350 / 5) * i;
                ctx.beginPath();
                ctx.moveTo(50, y);
                ctx.lineTo(570, y);
                ctx.stroke();
            }}
            
            // Plot loss (red)
            if (data.loss.length > 1) {{
                ctx.strokeStyle = '#dc3545';
                ctx.lineWidth = 2;
                plotLine(data.labels, data.loss, 'Loss');
            }}
            
            // Plot reward (blue)
            if (data.reward.length > 1) {{
                ctx.strokeStyle = '#007bff';
                ctx.lineWidth = 2;
                plotLine(data.labels, data.reward, 'Reward');
            }}
            
            // Plot reward components (lighter colors)
            if (data.reward_proj.length > 1) {{
                ctx.strokeStyle = '#6f42c1';
                ctx.lineWidth = 1;
                plotLine(data.labels, data.reward_proj, 'Proj Reward');
            }}
        }}
        
        function plotLine(xs, ys, label) {{
            if (xs.length < 2) return;
            
            const maxX = Math.max(...xs);
            const minX = Math.min(...xs);
            const maxY = Math.max(...ys);
            const minY = Math.min(...ys);
            
            const fx = (x) => 50 + 520 * (x - minX) / (maxX - minX + 1e-6);
            const fy = (y) => 370 - 350 * (y - minY) / ((maxY - minY) || 1);
            
            ctx.beginPath();
            for (let i = 0; i < xs.length; i++) {{
                const x = fx(xs[i]);
                const y = fy(ys[i]);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();
            
            // Label
            ctx.fillStyle = ctx.strokeStyle;
            ctx.font = '12px sans-serif';
            ctx.fillText(label, 500, 30 + Math.random() * 20);
        }}
        
        async function startTraining() {{
            try {{
                const response = await fetch(`/train/start/${{jobId}}`, {{method: 'POST'}});
                const result = await response.json();
                if (result.ok) {{
                    logMessage('Training start requested');
                }} else {{
                    logMessage(`Failed to start training: ${{result.error}}`);
                }}
            }} catch (err) {{
                logMessage(`Error starting training: ${{err.message}}`);
            }}
        }}
        
        async function stopTraining() {{
            try {{
                const response = await fetch(`/train/stop/${{jobId}}`, {{method: 'POST'}});
                const result = await response.json();
                if (result.ok) {{
                    logMessage('Training stop requested');
                }}
            }} catch (err) {{
                logMessage(`Error stopping training: ${{err.message}}`);
            }}
        }}
        
        function clearLog() {{
            log.textContent = '';
        }}
        
        // Initialize
        connectStream();
        
        // Check training status on load
        fetch(`/train/status/${{jobId}}`)
            .then(response => response.json())
            .then(status => {{
                if (status.status === 'training') {{
                    updateStatus('training', 'Training Active');
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                }}
            }})
            .catch(err => logMessage(`Status check failed: ${{err.message}}`));
    </script>
</body>
</html>
""")

@router.get("/train/checkpoints/{job_id}")
async def list_checkpoints(job_id: str):
    """List available checkpoints for a job"""
    from pathlib import Path
    ckpt_dir = Path(f"./files/models/{job_id}")
    if not ckpt_dir.exists():
        return {"checkpoints": []}
    
    checkpoints = []
    for ckpt_file in ckpt_dir.glob("online_*.pt"):
        stat = ckpt_file.stat()
        checkpoints.append({
            "name": ckpt_file.name,
            "path": str(ckpt_file),
            "size": stat.st_size,
            "modified": stat.st_mtime
        })
    
    return {"checkpoints": sorted(checkpoints, key=lambda x: x["modified"], reverse=True)}

