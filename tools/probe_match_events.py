import sys
import os
import uuid
from pathlib import Path

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from server.supabase_client import _get_client
except ImportError:
    print("❌ Could not import supabase_client.")
    sys.exit(1)

def main():
    sb = _get_client()
    if not sb: return

    print("🔍 Probing 'match_events' table columns...")
    
    # List of candidate names for the "Event Label" column
    candidates_label = ["event_type", "type", "label", "name", "event", "category"]
    
    # List of candidate names for the "Frame" column
    candidates_frame = ["frame", "frame_idx", "frame_number", "timestamp", "time"]

    success_schema = {}

    # Test 1: Try to find the Label column
    for col in candidates_label:
        print(f"   Trying column: '{col}' ...")
        try:
            # Attempt insert with just this column (and maybe match_id if strict)
            # We use a random UUID for match_id just in case it's required UUID type
            payload = {col: "PROBE_TEST"}
            sb.table("match_events").insert(payload).execute()
            print(f"   ✅ SUCCESS! Found label column: '{col}'")
            success_schema["label"] = col
            break
        except Exception as e:
            err = str(e)
            if "Could not find the" in err:
                continue # Wrong column name
            elif "null value in column" in err:
                # This means the column exists but we missed a required field!
                print(f"   ✅ FOUND IT! '{col}' exists (but insert failed due to missing required fields).")
                success_schema["label"] = col
                break
            else:
                print(f"   ❓ Unknown error on '{col}': {err}")

    # Test 2: Try to find the Frame column (if label found)
    if "label" in success_schema:
        lbl_col = success_schema["label"]
        for col in candidates_frame:
            print(f"   Trying frame column: '{col}' ...")
            try:
                payload = {lbl_col: "PROBE_TEST", col: 100}
                sb.table("match_events").insert(payload).execute()
                print(f"   ✅ SUCCESS! Found frame column: '{col}'")
                success_schema["frame"] = col
                break
            except Exception as e:
                err = str(e)
                if "Could not find the" in err:
                    continue
                elif "null value" in err:
                     print(f"   ✅ FOUND IT! '{col}' exists.")
                     success_schema["frame"] = col
                     break

    print("\n🔎 PROBE RESULTS:")
    print(success_schema)

if __name__ == "__main__":
    main()
