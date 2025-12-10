import sys
import os
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
    if not sb:
        print("❌ Connection failed.")
        return

    print("🔍 Querying System Catalog for 'match_events' columns...")
    
    try:
        # Query the standard PostgreSQL information_schema
        # Note: Supabase-py client might restrict raw SQL, but rpc is standard.
        # If we can't run SQL, we might be stuck.
        # Let's try a different trick: Use the PostgREST introspection.
        
        # This usually returns the definition if we access the root of the table options
        # But simpler: let's try to insert a row with a key we KNOW doesn't exist to get the error,
        # usually the error lists valid columns? No, Supabase usually just says "column X not found".
        
        # Let's try to assume a standard schema based on the "Auditor" output earlier.
        # But for now, let's try to fetch columns via a Remote Procedure Call (RPC) if you have one, 
        # or just try to insert a row with minimal columns and see if it works.
        
        # EXPERIMENTAL: Try to insert ONLY 'event_type' and 'match_id'
        print("Attempting minimal insert...")
        res = sb.table("match_events").insert({
            "event_type": "TEST_PROBE",
            # We assume match_id might be nullable or we need a real UUID. 
            # If this fails, we know 'match_id' is required.
        }).execute()
        
        print("✅ Minimal insert succeeded!")
        print(res.data)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # The error message is our clue.

if __name__ == "__main__":
    main()
