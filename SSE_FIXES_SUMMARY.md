# SSE Streaming Fixes - Summary

## Issues Fixed

### 1. **Callback Not Publishing to SSE** ✅
- **Problem**: `/progress` endpoint was using WebSocket `publish()` instead of SSE bus
- **Fix**: Updated to use `BUS.publish()` from `sse_bus.py`
- **Location**: `server/server.py` line 359

### 2. **PUBLIC_BASE_URL Validation** ✅
- **Problem**: RunPod can't reach `http://127.0.0.1:8080` callbacks
- **Fix**: Added validation to catch localhost URLs before launching RunPod
- **Location**: `server/server.py` lines 316-319

### 3. **Demo Endpoint for Local Testing** ✅
- **Problem**: No way to test locally without RunPod
- **Fix**: Mounted `server/live_track_demo.py` router at `/demo/track`
- **Location**: `server/server.py` lines 41-46

### 4. **Proper SSE Monitor Endpoint** ✅
- **Problem**: `/monitor` was HTML with polling, not real SSE
- **Fix**: Replaced with proper `StreamingResponse` using SSE bus
- **Location**: `server/server.py` lines 385-418

### 5. **Friendly Error for GET /ai/analyze** ✅
- **Problem**: 405 Method Not Allowed was confusing
- **Fix**: Added GET endpoint explaining POST requirement
- **Location**: `server/server.py` lines 294-309

## How to Test

### Local Testing (No RunPod needed)

1. **Start the server**:
   ```bash
   uvicorn server.server:app --host 0.0.0.0 --port 8080
   ```

2. **Test with demo endpoint**:
   ```bash
   # Upload video for local processing
   curl -F "file=@test_video.mp4" http://127.0.0.1:8080/demo/track
   # Returns: {"job_id": "abc12345"}
   
   # Monitor with SSE
   curl -N http://127.0.0.1:8080/monitor/abc12345
   # Should show: started → segment_done → done
   ```

3. **Run automated test**:
   ```bash
   python test_sse_flow.py
   ```

### RunPod Testing (Live processing)

1. **Set up public URL** (ngrok/Cloudflare tunnel):
   ```bash
   # Example with ngrok
   ngrok http 8080
   # Use the https URL as PUBLIC_BASE_URL
   ```

2. **Set environment variables**:
   ```bash
   export PUBLIC_BASE_URL="https://your-tunnel.ngrok.app"
   export CALLBACK_SECRET="your-secret-key"
   export RUNPOD_ENDPOINT="https://api.runpod.ai/v2/your-endpoint/run"
   export RUNPOD_API_KEY="your-api-key"
   ```

3. **Test full flow**:
   ```bash
   # Upload and segment
   curl -F "file=@match.mp4" -F "simulate=false" http://127.0.0.1:8080/upload
   
   # Start analysis
   curl -X POST http://127.0.0.1:8080/analyze \
        -H "Content-Type: application/json" \
        -d '{"job_id":"up_12345678","segment_urls":["https://..."]}'
   
   # Monitor progress
   curl -N http://127.0.0.1:8080/monitor/up_12345678
   ```

## Expected Behavior

### Local Demo Flow
1. Upload video → `/demo/track` returns `job_id`
2. SSE monitor shows: `started` → `segment_done` → `done`
3. Output files created: `files/jobs/{job_id}/overlay.mp4`, `tracks.json`

### RunPod Flow
1. Upload video → `/upload` segments and uploads to Supabase
2. Start analysis → `/analyze` launches RunPod job
3. RunPod worker processes segments and POSTs to `/progress/{job_id}`
4. Server logs show: `PROGRESS {job_id} {...}`
5. SSE monitor shows: `started` → multiple `segment_done` → `done`

## Key Files Modified

- `server/server.py` - Main server with all fixes
- `test_sse_flow.py` - Automated test script
- `sse_bus.py` - SSE bus implementation (unchanged)
- `server/live_track_demo.py` - Demo endpoint (unchanged)

## Troubleshooting

### "No PROGRESS lines in server logs"
- Check `PUBLIC_BASE_URL` is public (not localhost)
- Verify `CALLBACK_SECRET` matches between server and worker
- Ensure RunPod worker can reach your server

### "SSE stream is silent"
- Use `/demo/track` for local testing first
- Check server logs for "PROGRESS" messages
- Verify `/monitor/{job_id}` endpoint is working

### "Method Not Allowed on /ai/analyze"
- This is expected - use POST method
- GET `/ai/analyze` now shows helpful usage info

