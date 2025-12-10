#!/usr/bin/env python
import os
import json
import sys
import requests

def main():
    url = os.getenv("LABEL_STUDIO_URL", "http://127.0.0.1:8081").rstrip("/")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    project_id = os.getenv("LABEL_STUDIO_PROJECT_ID", "1")

    if not api_key:
        print("ERROR: LABEL_STUDIO_API_KEY not set")
        sys.exit(1)

    export_url = f"{url}/api/projects/{project_id}/export"
    params = {
        "exportType": "JSON",
        "download_all_tasks": "true",
    }
    headers = {
        "Authorization": f"Token {api_key}"
    }

    print(f"Requesting export from {export_url} ...")
    resp = requests.get(export_url, params=params, headers=headers, timeout=120)
    try:
        resp.raise_for_status()
    except Exception as e:
        print("ERROR calling Label Studio export:", e)
        print("Status code:", resp.status_code)
        print("Body:", resp.text[:500])
        sys.exit(1)

    try:
        data = resp.json()
    except Exception as e:
        print("ERROR decoding JSON:", e)
        print("Raw body:", resp.text[:500])
        sys.exit(1)

    os.makedirs("runs/json", exist_ok=True)
    out_path = "runs/json/ls_export_project1.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} with {len(data)} tasks")

if __name__ == "__main__":
    main()
