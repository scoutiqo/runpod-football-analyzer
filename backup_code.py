import os
import zipfile
from datetime import datetime
from data_manager import StorageManager

def create_code_archive(zip_name="code_backup.zip"):
    """Creates a ZIP file containing only .py files and configuration."""
    print("📦 Creating local code archive...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the current directory
        for root, dirs, files in os.walk('.'):
            # Exclude massive data directories
            dirs[:] = [d for d in dirs if d not in ['runs', 'tmp_jobs', 'datasets', '__pycache__', '.venv', '.git']]
            
            for file in files:
                # Include only code and configuration files
                if file.endswith(('.py', '.json', '.sh', '.yaml', '.txt', '.csv', '.h5', '.pt', '.pkl')):
                    full_path = os.path.join(root, file)
                    # Use relative path inside the zip
                    zipf.write(full_path, full_path)
    
    print(f"✅ Code archive saved locally: {zip_name}")
    return zip_name

def backup_to_backblaze():
    manager = StorageManager()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"code_and_config_{timestamp}.zip"
    cloud_key = f"code_backups/{archive_name}"
    
    local_path = create_code_archive(archive_name)
    
    # Upload the code bundle
    manager.upload_file(local_path, cloud_key)
    
    # Upload the new LSTM brain if it exists (it was the last thing to run)
    lstm_path = "models/event_lstm_master.h5"
    if os.path.exists(lstm_path):
        manager.upload_file(lstm_path, f"models_latest/event_lstm_master.h5")
        print(f"✅ Saved NEW LSTM Brain to models_latest/")
    
    # Cleanup local zip
    os.remove(local_path)
    print(f"🧹 Cleaned up local archive.")

if __name__ == "__main__":
    backup_to_backblaze()
