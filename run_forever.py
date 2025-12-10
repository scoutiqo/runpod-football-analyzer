import time
import os
import subprocess
import shutil
from data_manager import StorageManager

# --- CONFIGURATION ---
DOWNLOAD_DIR = "./temp_downloads"
OUTPUT_DIR = "./temp_outputs"
RAW_UPLOAD_PREFIX = "uploads/"
PROCESSED_PREFIX = "processed/"

# Ensure clean directories
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the Connection to Backblaze
manager = StorageManager()

def compress_video_ffmpeg(input_path, output_path):
    """
    Compresses video to 720p/1080p for web viewing.
    Reduces 1GB file to ~100MB.
    """
    print(f"⏳ Compressing {input_path}...")
    start_time = time.time()
    
    command = [
        "ffmpeg", "-y",                 # Overwrite output
        "-i", input_path,               # Input file
        "-vcodec", "libx264",           # H.264 Codec (Web standard)
        "-crf", "28",                   # Quality (Lower=Better, Higher=Smaller). 28 is good balance.
        "-preset", "faster",            # Encoding speed
        output_path
    ]
    
    # Run FFmpeg silently (hide the wall of text)
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if result.returncode == 0:
        duration = time.time() - start_time
        print(f"✅ Compression finished in {duration:.1f}s")
        return True
    else:
        print("❌ Compression Failed.")
        return False

def run_ai_analysis(video_path, match_id):
    """
    Calls your existing AI logic.
    We will hook this into your 'worker.py' or 'pipeline.py' next.
    """
    print(f"🧠 Starting AI Analysis on: {match_id}")
    
    # TODO: HERE IS WHERE WE IMPORT YOUR AI WORKER
    # from worker import analyze_match
    # results = analyze_match(video_path)
    
    # For now, simulate processing so we can test the infrastructure loop
    time.sleep(5) 
    
    # Simulate a JSON result
    dummy_json_path = os.path.join(OUTPUT_DIR, f"{match_id}_stats.json")
    with open(dummy_json_path, "w") as f:
        f.write('{"match_id": "' + match_id + '", "events": 150}')
        
    return dummy_json_path

def main_loop():
    print("🚀 ScoutIQO Autonomous Worker Started")
    print(f"👀 Watching Backblaze Bucket: {manager.bucket}/{RAW_UPLOAD_PREFIX}")

    while True:
        try:
            # 1. SCAN: Look for new files in the cloud
            new_files = manager.list_new_videos()
            
            if not new_files:
                print("💤 No new matches. Sleeping 30s...")
                time.sleep(30)
                continue

            # 2. SELECT: Take the first video
            cloud_file_key = new_files[0]
            filename = os.path.basename(cloud_file_key)
            match_id = filename.split('.')[0] # Assuming filename is 'match_id.mp4'
            
            local_raw_path = os.path.join(DOWNLOAD_DIR, filename)
            local_compressed_path = os.path.join(OUTPUT_DIR, f"compressed_{filename}")

            print(f"\n🎬 Processing Job: {filename}")

            # 3. DOWNLOAD: Pull from Backblaze
            if manager.download_file(cloud_file_key, local_raw_path):
                
                # 4. COMPRESS: Shrink for Frontend
                if compress_video_ffmpeg(local_raw_path, local_compressed_path):
                    
                    # 5. UPLOAD COMPRESSED: Send to 'web-ready' folder
                    manager.upload_file(local_compressed_path, f"web-ready/{filename}")
                    
                    # 6. RUN AI: Analyze the raw high-quality video
                    json_result = run_ai_analysis(local_raw_path, match_id)
                    manager.upload_file(json_result, f"stats/{match_id}.json")

                    # 7. CLEANUP: Move original file to 'archive' so we don't process it again
                    manager.move_to_processed(cloud_file_key)
                    
                    print("✨ Job Done!")
                
                # 8. WIPE DISK: Remove local files to save GPU space
                if os.path.exists(local_raw_path): os.remove(local_raw_path)
                if os.path.exists(local_compressed_path): os.remove(local_compressed_path)
                
            else:
                print("❌ Failed to download.")
                time.sleep(10)

        except Exception as e:
            print(f"❌ Critical Loop Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("🛑 Worker Stopped.")

