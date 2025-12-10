import subprocess
import time
import os

def run_daemon(cmd, log_file):
    print(f"🚀 Starting Daemon: {cmd}")
    with open(log_file, "w") as f:
        subprocess.Popen(cmd.split(), stdout=f, stderr=f)

def main():
    print("🌍 SCOUTIQO AUTONOMOUS SYSTEM STARTING...")
    
    # 1. Start the Day Worker (Serves Users)
    run_daemon("python worker.py", "logs_worker.txt")
    
    # 2. Start the Night Researcher (Evolves Physics)
    # We run this nicely so it doesn't hog CPU from the worker
    # Uses 'nice -n 10' to lower priority
    run_daemon("nice -n 10 python server/auto_tune.py", "logs_brain.txt")
    
    # 3. Start the Teacher Loop (Asks GPT-4o)
    # We run this every hour via a simple loop here
    print("   ✅ Daemons started. Entering Supervisor Loop.")
    
    while True:
        time.sleep(3600) # Wait 1 hour
        print("   🎓 Hourly Audit: Asking Teacher (GPT-4o)...")
        os.system("python core/active_learning_loop.py")
        
        print("   🧠 Hourly Upgrade: Retraining Master Brain...")
        os.system("python training/train_master_brain.py")

if __name__ == "__main__":
    main()
