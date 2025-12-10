# --- Supabase (server-side) ---
export SUPABASE_URL='https://dirsscpuzqrjftawgotz.supabase.co'
export SUPABASE_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRpcnNzY3B1enFyamZ0YXdnb3R6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjI1MjAwOCwiZXhwIjoyMDY3ODI4MDA4fQ.wfnOamYV8yNdr1BLZllwP_2jom4VCMTYKSUct3pBak4'
# also export under common aliases just in case code expects them
export SUPABASE_SERVICE_ROLE_KEY="$SUPABASE_KEY"
export SUPABASE_ANON_KEY=''   # leave empty for the worker

# --- Tables (adjust if you renamed) ---
export JOBS_TABLE='jobs'

# --- Optional runtime knobs ---
export ANALYZE_CMD_TEMPLATE='python3 pipeline.py --input {input} --export_json --out_json {out_json}'
export POLLER_SLEEP_SECONDS='3'
