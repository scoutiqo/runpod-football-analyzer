import json
import os
import cv2
import shutil
from pathlib import Path
from collections import Counter

# CONFIG
INPUT_FILE = "runs/json/final_events_viewer.json"
VIDEO_DIR = "tmp_jobs"
OUTPUT_DIR = "runs/viz/smartness_proofs"

# TIERED ONTOLOGY (Hierarchy of Intelligence)
TIER_1_BASIC = ["pass", "duel", "carry", "ball_carry", "ball_loss", "unknown", "none"]
TIER_2_INTERMEDIATE = ["cross", "shot", "goal", "save", "corner_taken", "foul_committed", "interception", "clearance", "throw_in"]
TIER_3_PRO = ["cutback", "keeper_sweep", "sliding_tackle", "through_ball", "shot_volley", "shot_header", "penalty_awarded", "offside", "pressing_action", "defensive_error"]

def get_latest_video():
    # Find the video file corresponding to the last run
    # This is a heuristic; assumes only one video in tmp or we pick one
    videos = sorted(Path(VIDEO_DIR).glob("*.mp4"), key=os.path.getmtime, reverse=True)
    if videos: return str(videos[0])
    return None

def main():
    print("🧠 GENERATING INTELLIGENCE REPORT...")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ No events file found. Run the pipeline first.")
        return

    events = json.loads(Path(INPUT_FILE).read_text())
    if not events:
        print("   ⚠️ AI found 0 events. It might be asleep.")
        return

    # 1. Analyze Vocabulary
    counts = Counter([e['detail'] for e in events])
    
    tier_1_count = sum(counts[k] for k in counts if k in TIER_1_BASIC)
    tier_2_count = sum(counts[k] for k in counts if k in TIER_2_INTERMEDIATE)
    tier_3_count = sum(counts[k] for k in counts if k in TIER_3_PRO)
    
    total = len(events)
    smartness_score = ((tier_2_count * 5) + (tier_3_count * 20)) / total if total > 0 else 0
    
    print(f"\n📊 REPORT CARD:")
    print(f"   Total Events: {total}")
    print(f"   -----------------------------")
    print(f"   👶 Basic (Pass/Duel):    {tier_1_count} ({round(tier_1_count/total*100)}%)")
    print(f"   🎓 Interm. (Shot/Cross): {tier_2_count} ({round(tier_2_count/total*100)}%)")
    print(f"   🧠 PRO (Cutback/Slide):  {tier_3_count} ({round(tier_3_count/total*100)}%)")
    print(f"   -----------------------------")
    print(f"   IQ SCORE: {round(smartness_score, 1)} / 10.0")
    
    if tier_3_count == 0 and tier_2_count == 0:
        print("\n   📉 DIAGNOSIS: The AI is playing it safe. It only sees the basics.")
    else:
        print("\n   📈 DIAGNOSIS: The AI is 'Thinking'. It sees complex tactical events.")

    # 2. Visual Proof (Extract the Smartest Frames)
    print(f"\n📸 Extracting Proof of Intelligence...")
    video_path = get_latest_video()
    if not video_path:
        print("   ⚠️ Video file not found. Cannot extract frames.")
        return

    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    cap = cv2.VideoCapture(video_path)
    
    # Prioritize Tier 3, then Tier 2
    interesting_events = [e for e in events if e['detail'] in TIER_3_PRO]
    if len(interesting_events) < 5:
        interesting_events.extend([e for e in events if e['detail'] in TIER_2_INTERMEDIATE])
        
    # Extract max 10 proofs
    saved_count = 0
    for evt in interesting_events[:10]:
        frame_idx = evt['frame']
        label = evt['detail'].upper()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # Draw Label
        cv2.putText(frame, f"AI SEES: {label}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        out_path = f"{OUTPUT_DIR}/proof_{frame_idx}_{label}.jpg"
        cv2.imwrite(out_path, frame)
        saved_count += 1
        print(f"   ✅ Saved: {out_path}")

    cap.release()
    print(f"\n📁 Proofs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
