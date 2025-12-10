import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import os
import glob
from pathlib import Path
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# CONFIG
DATA_FILE = "datasets/master_bank/clean_master_dataset.csv"
VIDEO_DIR = "tmp_jobs" 
MODEL_OUT = "models/distilled_vision_v1.pth"
SEQUENCE_LENGTH = 5
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20

class VideoEventDataset(Dataset):
    def __init__(self, csv_path, video_dir, transform=None):
        self.video_dir = Path(video_dir)
        self.transform = transform
        
        # 1. Load & Clean Data
        print(f"   Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"   Raw Rows: {len(df)}")
        
        # Ensure columns exist
        if 'video_id' not in df.columns and 'job_id' in df.columns:
            df['video_id'] = df['job_id']
            
        # Drop rows with critical missing info
        df = df.dropna(subset=['frame', 'video_id', 'label'])
        
        # Convert frame to numeric and drop errors (NaNs)
        df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
        df = df.dropna(subset=['frame'])
        df['frame'] = df['frame'].astype(int)
        
        # Filter: Keep only rows where we actually HAVE the video file
        # This prevents "black image" training
        print("   Verifying video files availability...")
        valid_indices = []
        existing_videos = {p.name for p in self.video_dir.glob("*.mp4")}
        
        for idx, row in df.iterrows():
            vid_name = f"{row['video_id']}.mp4"
            if vid_name in existing_videos:
                valid_indices.append(idx)
                
        self.df = df.loc[valid_indices].copy().reset_index(drop=True)
        print(f"   ✅ Valid Training Rows: {len(self.df)}")

        if not self.df.empty:
            self.label_enc = LabelEncoder()
            self.df['encoded_label'] = self.label_enc.fit_transform(self.df['label'])
            self.num_classes = len(self.label_enc.classes_)
            print(f"   Classes: {self.label_enc.classes_}")
        else:
            self.num_classes = 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            vid_id = row['video_id']
            frame_idx = int(row['frame'])
            label = row['encoded_label']
            
            vid_path = self.video_dir / f"{vid_id}.mp4"
            
            cap = cv2.VideoCapture(str(vid_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1)) 
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                img = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img = self.transform(frame)
                else:
                    img = frame

            return img, label
        except Exception as e:
            print(f"Error: {e}")
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), 0

def main():
    print("⚗️ STARTING DISTILLATION (Training Cheap Vision Brain)...")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file not found: {DATA_FILE}")
        return

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VideoEventDataset(DATA_FILE, VIDEO_DIR, transform=transform)
    
    if len(dataset) == 0:
        print("❌ Dataset is empty after cleaning. Check your videos/CSVs.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Load Pre-trained ResNet
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, dataset.num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"   🚀 Training on {len(dataset)} samples for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"   Epoch {epoch+1} | Loss: {running_loss/len(loader):.4f} | Acc: {correct/total:.2f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"✅ DISTILLED BRAIN SAVED: {MODEL_OUT}")

if __name__ == "__main__":
    main()
