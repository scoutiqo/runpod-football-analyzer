import sys
import os
from pathlib import Path
import json

# Add root to path to import supabase_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from server.supabase_client import _get_client
except ImportError:
    print("❌ Could not import supabase_client.")
    sys.exit(1)

def main():
    sb = _get_client()
    if not sb:
        print("❌ Connection failed.")
        return

    print("🔍 Inspecting 'match_events' table...")
    
    # Try to insert a dummy row to trigger a specific error that lists columns
    # OR try to select 1 row to see keys
    try:
        res = sb.table("match_events").select("*").limit(1).execute()
        if res.data:
            print("✅ Found existing data. Columns:")
            print(list(res.data[0].keys()))
        else:
            print("⚠️ Table is empty. Attempting detailed insert to probe schema...")
            # We'll fail on purpose to see if it hints valid columns, 
            # but usually select is safer.
            print("Cannot deduce columns from empty table via simple select.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

