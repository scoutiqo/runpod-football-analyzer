# ScoutIQO Analyzer - Quick Start Guide

## Overview

ScoutIQO Analyzer is a production-lean MVP that analyzes full-match football videos, extracts tracking + events, and outputs player insights. The system processes video segments through YOLO tracking, event detection, value models, and generates comprehensive analytics.

## Architecture

```
Upload → Segment → Analyze → Progress → Artifacts
   ↓        ↓         ↓         ↓         ↓
FastAPI  FFmpeg   RunPod    WebSocket   JSON
```

### Key Components

- **FastAPI Server** (`server/server.py`): Handles upload, segmentation, and monitoring
- **RunPod Handler** (`handler.py`): Processes video segments with AI pipeline
- **Analyzers**: Tracking, events, value models
- **Metrics**: Player aggregation and insights
- **Render**: Heatmap generation
- **Merge**: Segment merging and final output

## PowerShell Runbook

### 1. Environment Setup

```powershell
# Clone repository
git clone <repository-url>
cd runpod-football-analyzer-2

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Fix NumPy compatibility issues and start server
.\fix_and_start.ps1
```

### Alternative: Manual Fix

If the PowerShell script doesn't work, fix dependencies manually:

```powershell
# Uninstall problematic packages
pip uninstall -y numpy scipy ultralytics filterpy opencv-python

# Install compatible versions
pip install numpy==1.24.3
pip install scipy==1.11.4
pip install opencv-python==4.8.1.78
pip install ultralytics==8.0.196
pip install filterpy==1.4.5

# Install other required packages
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6
pip install pydantic==2.5.0
pip install requests==2.31.0
pip install yt-dlp==2023.11.16
pip install matplotlib==3.8.2
pip install pandas==2.1.3

# Start the simplified server
python server_simple.py
```

### 2. Environment Variables

Create `.env` file:

```env
# Server Configuration
PUBLIC_BASE_URL=http://127.0.0.1:8080
CALLBACK_SECRET=your-secret-key

# RunPod Configuration (optional)
RUNPOD_ENDPOINT=https://api.runpod.ai/v2/<endpoint-id>/run
RUNPOD_API_KEY=your-runpod-api-key

# Supabase Configuration (optional)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_BUCKET=scoutiqo
```

### 3. Start the Server

```powershell
# Start FastAPI server
python -m uvicorn server.server:app --host 0.0.0.0 --port 8080 --reload
```

### 4. Test the System

#### Option A: Easy Runner (Web UI)

1. Open browser to `http://127.0.0.1:8080/easy`
2. Upload a video file (.mp4)
3. Configure settings:
   - Segment seconds: 20
   - Limit segments: 3
   - Simulate: true (for quick test)
4. Click "Run analysis"
5. Monitor progress at the redirected URL

#### Option B: API Testing

```powershell
# Test health endpoint
Invoke-RestMethod -Uri "http://127.0.0.1:8080/health"

# Upload video (replace with your video file)
$file = "test_video.mp4"
$uploadResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8080/upload" -Method Post -Form @{
    file = Get-Item $file
    segment_seconds = 20
    fast = 1
    simulate = $true
}

Write-Host "Job ID: $($uploadResponse.job_id)"
Write-Host "Segments: $($uploadResponse.segment_urls.Count)"

# Start analysis
$analyzeRequest = @{
    job_id = $uploadResponse.job_id
    segment_urls = $uploadResponse.segment_urls
    simulate = $true
} | ConvertTo-Json

$analyzeResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8080/analyze" -Method Post -Body $analyzeRequest -ContentType "application/json"

Write-Host "Analysis started: $($analyzeResponse.status)"

# Monitor progress
$jobId = $uploadResponse.job_id
$monitorUrl = "http://127.0.0.1:8080/monitor/$jobId"
Write-Host "Monitor at: $monitorUrl"
Start-Process $monitorUrl
```

### 5. Check Results

```powershell
# Get tracks.json
$tracksResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8080/files/jobs/$jobId/tracks.json"
Write-Host "Tracks JSON: $($tracksResponse | ConvertTo-Json -Depth 3)"

# Get player insights (replace 1 with actual player ID)
$playerResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8080/files/jobs/$jobId/players/tid_1.json"
Write-Host "Player Insights: $($playerResponse | ConvertTo-Json -Depth 3)"
```

## API Endpoints

### Core Endpoints

- `POST /upload` - Upload and segment video
- `POST /analyze` - Start analysis of segments
- `GET /monitor/{job_id}` - Real-time monitoring page
- `GET /files/jobs/{job_id}/tracks.json` - Get tracks JSON
- `GET /files/jobs/{job_id}/players/tid_{player_id}.json` - Get player insights

### Monitoring Endpoints

- `GET /status/{job_id}` - Job status
- `GET /progress/{job_id}/dump` - Full progress dump
- `POST /progress/{job_id}` - Progress callback (RunPod → Server)

## Data Contracts

### Tracks JSON Structure

```json
{
  "job_id": "UUID",
  "video": {
    "duration_s": 0,
    "width": 0,
    "height": 0,
    "fps": 0
  },
  "calibration": {
    "homography": null,
    "units": "px"
  },
  "players": [
    {
      "tid": 12,
      "team": "home|away|unknown",
      "jersey": null,
      "role_hint": null,
      "primary_position": null,
      "metrics": {
        "touches_total": 0,
        "passes": {"attempts": 0, "completed": 0, "progressive": 0},
        "carries": {"count": 0, "prog_dist_px": 0},
        "dribbles": {"att": 0, "won": 0},
        "shots": {"att": 0, "on_target": 0, "goals": 0, "xg_sum": 0},
        "pressures": {"total": 0, "successful": 0},
        "duels": {"air_won": 0, "air_att": 0, "grd_won": 0, "grd_att": 0},
        "distance_px": 0,
        "sprints": 0,
        "max_speed_pxps": 0
      },
      "heatmap": "signed://url/heatmap_tid12.png",
      "events_idx": [0,5,9]
    }
  ],
  "events": [
    {
      "id": 0,
      "t": 123.45,
      "phase": "build|defend|press|transition",
      "type": "pass|carry|shot|tackle|interception|reception|dribble|cross|foul|recovery|clearance|press|counterpress|setpiece|goal|save",
      "team": "home|away",
      "actor_tid": 12,
      "to_tid": 21,
      "loc": {"x": 0.53, "y": 0.27},
      "orient_deg": 135,
      "pressure": {"count_n": 2, "nearest_m": 2.4},
      "value": {"xT": 0.037, "xA": 0.01, "EPV": 0.021, "VAEP": 0.015},
      "outcome": "complete|incomplete|won|lost|shot_on|shot_off|goal|blocked|foul_won|foul_committed",
      "nextN": {"shot": false, "regain_s": 7.2, "turnover": false}
    }
  ],
  "auto_tune": {"params": {...}, "notes": "AI loop decisions"},
  "artifacts": {
    "overlays": ["signed://url/overlay_seg_000.mp4", "..."],
    "logs": ["signed://url/log.txt"]
  }
}
```

### Player Insights JSON Structure

```json
{
  "job_id": "UUID",
  "tid": 12,
  "identity": {"name": null, "squad_number": null, "foot": null},
  "minutes": {"on": 0, "off": null, "played": 0},
  "formations": [{"ts": 0, "shape": "4-3-3", "role": "RB"}],
  "opponent_matchups": [{"phase": "build", "vs_tid": 5}],
  "game_states": [{"minute": 17, "state": "level"}],
  "summary": {
    "touches": 0, "passes_att": 0, "passes_cmp": 0, "xA": 0,
    "carries": 0, "prog_carry_px": 0, "dribbles_won": 0,
    "shots": 0, "goals": 0, "xG": 0, "psxg_on_target": 0,
    "pressures": 0, "press_success": 0,
    "duels_air_won%": 0.0, "duels_grd_won%": 0.0,
    "distance_px": 0, "sprints": 0, "max_speed_pxps": 0
  },
  "value_models": {"xT_sum": 0, "EPV_sum": 0, "VAEP_sum": 0, "packing_for": 0, "packing_against": 0},
  "role_specific_notes": ["overlapped 7x", "line-breaking passes: 6"]
}
```

## Testing

### Run Unit Tests

```powershell
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_events.py -v
python -m pytest tests/test_value_models.py -v
python -m pytest tests/test_segment_merging.py -v
python -m pytest tests/test_packing_count.py -v
```

### Test Coverage

```powershell
# Install coverage
pip install coverage

# Run tests with coverage
coverage run -m pytest tests/
coverage report
coverage html
```

## Troubleshooting

### Common Issues

1. **Server won't start**: Check if port 8080 is available
2. **Upload fails**: Ensure video file is valid MP4
3. **Analysis fails**: Check RunPod configuration and API keys
4. **No results**: Verify job completed successfully in monitor

### Logs

```powershell
# Check server logs
Get-Content server.log -Tail 50

# Check job progress
Invoke-RestMethod -Uri "http://127.0.0.1:8080/progress/{job_id}/dump"
```

### Performance Tips

1. Use `simulate=true` for quick testing
2. Limit segments for faster processing
3. Use fast segmentation (copy-only) when possible
4. Monitor memory usage during processing

## Production Deployment

### RunPod Setup

1. Create RunPod endpoint
2. Set environment variables
3. Deploy handler code
4. Configure callback URLs

### Supabase Setup

1. Create Supabase project
2. Set up storage bucket
3. Configure service role key
4. Update environment variables

### Monitoring

- Use `/monitor/{job_id}` for real-time progress
- Check `/status/{job_id}` for job status
- Monitor server logs for errors
- Set up alerts for failed jobs

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review server logs
3. Test with simulation mode first
4. Verify all environment variables are set correctly

