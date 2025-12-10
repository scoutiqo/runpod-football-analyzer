import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from supabase import create_client

# CONFIG
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BACKUP_BUCKET = "ml-artifacts" # Using your existing bucket
PROJECT_DIR = Path("/workspace/runpod-football-analyzer")

def zip_project(output_filename):
    print(f"📦 Zipping project to {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_DIR):
            # Exclude heavy trash
            if 'tmp_jobs' in root or '__pycache__' in root or '.git' in root or 'runs/videos' in root:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, PROJECT_DIR)
                zipf.write(file_path, arcname)
                
    return output_filename

def main():
    print("🚨 STARTING EMERGENCY BACKUP...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"scoutiqo_backup_{timestamp}.zip"
    zip_path = PROJECT_DIR / zip_name
    
    # 1. Create Zip
    zip_project(zip_path)
    
    # 2. Upload
    print(f"☁️ Uploading {zip_name} to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    try:
        with open(zip_path, "rb") as f:
            supabase.storage.from_(BACKUP_BUCKET).upload(f"backups/{zip_name}", f.read())
        print(f"✅ BACKUP SECURE! Saved to {BACKUP_BUCKET}/backups/{zip_name}")
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
    finally:
        if zip_path.exists(): os.remove(zip_path)

if __name__ == "__main__":
    main()
