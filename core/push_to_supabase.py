import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    from server.supabase_client import _get_client
except ImportError:
    sys.exit(1)

EVENTS_FILE = "runs/json/final_events_viewer.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", help="Job ID", default="manual_run") 
    parser.add_argument("--match_id", required=True, help="The Supabase Match ID (Must be valid UUID)") 
    args = parser.parse_args()

    sb = _get_client()
    if not sb:
        print("❌ No Supabase Client available. Check env vars.")
        return

    if not os.path.exists(EVENTS_FILE):
        print("⚠️ No events file found.")
        return

    data = json.loads(Path(EVENTS_FILE).read_text())
    if not data:
        print("⚠️ Events file is empty.")
        return

    print(f"🔄 Preparing to push {len(data)} events to table 'match_events'...")

    db_rows = []
    current_time = datetime.now(timezone.utc).isoformat()
    
    for evt in data:
        # MINIMALIST SCHEMA (Final Attempt: Using 'ts')
        row = {
            "type": evt.get("label"),
            "t": evt.get("frame"), # <--- FINAL GUESS: Using 't'
            "created_at": current_time,
            "match_id": args.match_id
        }
        db_rows.append(row)

    # Insert Attempt
    try:
        res = sb.table("match_events").insert(db_rows).execute()
        
        inserted_count = len(res.data) if res.data else 0

        print(f"✅ Success! Uploaded {inserted_count} events to match {args.match_id}.")

    except Exception as e:
        print(f"❌ FINAL UPLOAD FAILED: {e}")
        print("   -> This means the column name 'ts' is still wrong. We have eliminated all standard names.")

if __name__ == '__main__':
    main()
