import boto3
import os
from botocore.exceptions import ClientError

# --- CONFIGURATION ---
KEY_ID = "00315134647906c0000000001"
APP_KEY = "K003PeN0006kV8qRc3TGjO4c33jnesw"
BUCKET_NAME = "scoutiqo-backup-main"
ENDPOINT_URL = "https://s3.eu-central-003.backblazeb2.com"

class StorageManager:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=APP_KEY
        )
        self.bucket = BUCKET_NAME
        print(f"✅ Connected to Storage Vault: {self.bucket}")

    def list_new_videos(self):
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix="uploads/")
            video_files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if key.endswith(('.mp4', '.avi', '.mov', '.mkv')) and obj['Size'] > 0:
                        video_files.append(key)
            return video_files
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return []

    def download_file(self, cloud_path, local_path):
        try:
            print(f"⬇️ Downloading: {cloud_path}...")
            self.s3.download_file(self.bucket, cloud_path, local_path)
            return True
        except Exception as e:
            print(f"❌ Download Failed: {e}")
            return False

    def upload_file(self, local_path, cloud_path):
        try:
            print(f"⬆️ Uploading: {cloud_path}...")
            self.s3.upload_file(local_path, self.bucket, cloud_path)
            return True
        except Exception as e:
            print(f"❌ Upload Failed: {e}")
            return False

    def move_to_processed(self, file_key):
        new_key = file_key.replace("uploads/", "processed/")
        try:
            # Standard S3 Copy
            self.s3.copy_object(
                Bucket=self.bucket, 
                CopySource={'Bucket': self.bucket, 'Key': file_key}, 
                Key=new_key
            )
            self.s3.delete_object(Bucket=self.bucket, Key=file_key)
            print(f"✅ Moved {file_key} -> {new_key}")
        except Exception as e:
            print(f"❌ Failed to move file: {e}")
