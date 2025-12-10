import sys, os
sys.path.append(os.getcwd())
from server.supabase_client import _get_client

sb = _get_client()
if sb:
    res = sb.table("matches").select("id").limit(1).execute()
    if res.data:
        print(res.data[0]["id"])
    else:
        print("NO_MATCHES_FOUND")
else:
    print("CONNECTION_FAILED")
