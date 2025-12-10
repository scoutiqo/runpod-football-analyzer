set -e
pkill -f python || true
pip cache purge || true
rm -rf ~/.cache/pip ~/.cache/torch /tmp/* || true
pip uninstall -y opencv-python opencv-contrib-python || true
pip install --no-cache-dir -r requirements.cpu-min.txt
python - << 'PY'
import torch, cv2
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
print("OpenCV headless OK:", hasattr(cv2, "getBuildInformation"))
PY
