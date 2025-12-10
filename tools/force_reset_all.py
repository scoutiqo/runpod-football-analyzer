import os
import sys
from supabase import create_client, Client

# CONFIG
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def main():
    print("🔄 FORCE RESETTING ALL JOBS...")
    
    # Select ONLY ID and Status to avoid column errors
    try:
        response = supabase.table("ml_jobs").select("id, status").execute()
        jobs = response.data
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return
    
    if not jobs:
        print("   ✅ No jobs found.")
        return

    print(f"   ⚠️ Found {len(jobs)} total jobs. Resetting to 'queued'...")
    
    for job in jobs:
        # Reset EVERYTHING so it looks like a new upload
        supabase.table("ml_jobs").update({
            "status": "queued",
            "worker_id": None,
            "started_at": None,
            "finished_at": None,
            "error_message": None,
            "tracks_url": None,
            "events_url": None
        }).eq("id", job["id"]).execute()
        
    print(f"\n✅ All {len(jobs)} jobs have been Reset.")
    print("   👉 The Worker will now re-download and re-analyze everything.")

if __name__ == "__main__":
    main()
