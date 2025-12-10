import json
import sys
import os
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    from server.supabase_client import _get_client
except ImportError:
    sys.exit(1)

def main():
    sb = _get_client()
    if not sb:
        print("❌ Supabase connection failed.")
        return

    print("🔍 Querying live 'match_events' schema...")
    
    # Execute a query against the information schema
    # We are looking for columns in the 'public' schema belonging to 'match_events'
    try:
        query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'match_events' AND table_schema = 'public' 
        ORDER BY ordinal_position;
        """
        
        # Note: Supabase's Python client doesn't always support raw queries easily, 
        # so we rely on the client's internal query mechanism or standard view access.
        # Since standard query to information_schema is often restricted, 
        # we will use the fetch API and hope for success.

        # For reliability, we use the client to select data, which might return an error message
        # listing the columns if the original query failed (as we saw earlier). 
        # Since we cannot run raw SQL here, we must rely on the query builder.
        
        # Let's try to select all and output the column names from the result data structure.
        
        res = sb.table("match_events").select("*").limit(0).execute()
        
        # This relies on the internal response structure where column names are often stored.
        # This will vary based on the specific version of the Supabase client.
        
        if res.data is not None and len(res.data) == 0:
            # If the table is empty, we print the keys from the result metadata
            print("⚠️ Table is empty. Checking result keys (may not show all columns)...")
            
            # Use an alternative method if direct select fails to yield keys
            # Final fallback: we print the full response structure to find the keys.
            print("Full response structure might contain column names.")
            print(json.dumps(res, indent=2, default=str))


    except Exception as e:
        print(f"❌ Failed to inspect schema: {e}")
        print("   Likely cause: Insufficient permissions (RLS/Key issue) or unsupported query.")


if __name__ == "__main__":
    main()
