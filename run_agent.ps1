param([string]$Tracks="C:\path\to\tracks.json",[string]$RepoRoot=".",[string]$PatchOut="proposed.patch")
python ai_agent.py --tracks "$Tracks" --repo_root "$RepoRoot" --out_patch "$PatchOut"
Write-Host "Agent finished. Patch at $PatchOut"
