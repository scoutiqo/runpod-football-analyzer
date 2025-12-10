#!/usr/bin/env python
import argparse
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

def run_step(script_cmd, description):
    print(f"\n🚀 STEP: {description}")
    print(f"   Exec: python {script_cmd}")
    ret = os.system(f"python {script_cmd}")
    if ret != 0:
        print(f"❌ FAILED: {description}")
        sys.exit(ret)
    print(f"✅ DONE: {description}")

def main():
    parser = argparse.ArgumentParser(description="ScoutIQO Event Pipeline")
    
    parser.add_argument("--video", help="Path to source video (Required for stabilization)", default=None)
    parser.add_argument("--tracks", help="Path to main tracks file", required=True)
    parser.add_argument("--match_id", required=True, help="Supabase Match ID")
    parser.add_argument("--pitch_mask", help="Polygon coords (Normalized)", default=None)
    
    args = parser.parse_args()

    print(f"⚽ STARTING SCOUTIQO PIPELINE on: {args.tracks}")
    
    # -------------------------------------------------------
    # 1. DATA PREP & CLEANING
    # -------------------------------------------------------

    # Mine Events (Pass/Duel Candidates)
    run_step(f"core/events_from_tracks_pipeline.py --input \"{args.tracks}\"", "Mine Weak Labels")

    # Stabilization (Optical Flow)
    if args.video:
        run_step(f"core/stabilize_tracks.py --video \"{args.video}\"", "Stabilize Camera Movement")
    
    # CLEANING LOGIC
    if args.pitch_mask:
        # A) PRO MODE (Manual Mask)
        # 1. Remove everything outside the drawn pitch
        run_step(f"core/clean_tracks_manual.py --mask \"{args.pitch_mask}\" --video \"{args.video}\"", "Manual Pitch Masking")
        
        # 2. Smooth tracks (Anti-Flicker) - Critical for speed calculation
        run_step("core/clean_tracks.py", "Anti-Flicker Smoothing")

        # 3. Physics (Speed & Distance) - Uses the mask to map pixels to meters
        run_step(f"core/speed_and_distance.py --mask \"{args.pitch_mask}\"", "Calculate Physical Metrics")
        
        # 4. Tactics (Heatmaps & Possession) - Uses the meter coordinates
        run_step("core/tactical_metrics.py", "Generate Heatmaps & Possession")

    else:
        # B) AUTO MODE (Fallback)
        # Uses Density clustering to find the pitch
        run_step(f"core/clean_tracks_pose.py --input \"{args.tracks}\" --output \"{args.tracks}\"", "AI Geometry Filter")

    # -------------------------------------------------------
    # 2. INTELLIGENCE (Machine Learning)
    # -------------------------------------------------------

    # Build Features for LSTM
    run_step(f"training/build_event_dataset.py --input_tracks \"{args.tracks}\" --input_labels runs/json/silver_labels.json", "Build Dataset")

    # Inference (Using MASTER BRAIN)
    run_step("core/predict_events_lstm.py", "Deep Learning Inference")

    # -------------------------------------------------------
    # 3. EXPORT & UPLOAD
    # -------------------------------------------------------

    # Consolidate Events
    run_step("core/export_events_for_viewer.py", "Export Clean Events")

    # Finalize Tracks (Format for Frontend)
    run_step("core/finalize_export.py", "Finalize Tracks for Frontend")

    # Upload to Supabase
    if os.getenv("SUPABASE_URL"):
        run_step(f"core/push_to_supabase.py --match_id {args.match_id}", "Upload Events to Supabase")
    
    # Harvest Data for Future Training
    MASTER_DB = "datasets/master_bank"
    if os.path.exists("runs/json/event_dataset.csv"):
        shutil.copy("runs/json/event_dataset.csv", f"{MASTER_DB}/{args.match_id}.csv")
        print("💾 Harvested Training Data.")

    print("\n🏆 PIPELINE COMPLETE.")

if __name__ == "__main__":
    main()
