import os
import sys
from supabase import create_client, Client

# CONFIG
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def main():
    print("🔄 RESETTING STUCK JOBS...")
    
    # 1. Find jobs that are 'running' (stuck) or 'error'
    response = supabase.table("ml_jobs").select("*").in_("status", ["running", "error"]).execute()
    jobs = response.data
    
    if not jobs:
        print("   ✅ No stuck jobs found. The queue is clean.")
        return

    print(f"   ⚠️ Found {len(jobs)} stuck/failed jobs.")
    
    # 2. Reset them to 'queued'
    for job in jobs:
        print(f"      - Resetting {job['id']} ({job.get('video_name', 'Unknown')})...")
        supabase.table("ml_jobs").update({
            "status": "queued",
            "error_message": None,
            "worker_id": None,
            "started_at": None
        }).eq("id", job["id"]).execute()
        
    print(f"\n✅ All {len(jobs)} jobs have been pushed back to the Queue.")
    print("   👉 Restart 'worker.py' now to pick them up.")

if __name__ == "__main__":
    main()
