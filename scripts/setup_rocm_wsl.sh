#!/bin/bash
# ROCm 7.2 for WSL2 + a second venv with AMD's PyTorch wheels, so the PPO update can run on the RX 9070 XT.
# Run inside WSL in a terminal tab (sudo asks for the password; the user types it):
#   wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/setup_rocm_wsl.sh
# Prerequisite on Windows: "AMD Software: Adrenalin Edition for WSL2" (26.1.1 or newer), then a reboot.
# Source: rocm.docs.amd.com, Radeon/Ryzen docs-7.2, WSL install and PyTorch pages. This distro is Ubuntu 26.04,
# outside AMD's 22.04/24.04 matrix: the noble (24.04) installer is used; the WSL use case needs no kernel module.
set -euo pipefail
cd /tmp
ROCMINFO=/opt/rocm/bin/rocminfo
if [ "$(id -u)" = "0" ]; then
  # system half, run as root:  wsl.exe -d Ubuntu -u root -- bash <this script>
  if [ ! -x "$ROCMINFO" ]; then
    apt update
    [ -f amdgpu-install_7.2.70200-1_all.deb ] || wget -q https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
    apt install -y ./amdgpu-install_7.2.70200-1_all.deb
    amdgpu-install -y --usecase=wsl,rocm --no-dkms
  fi
  $ROCMINFO | grep -E "Name:|gfx" | head -8
  echo "system half done; now run the same script as the normal user for the venv"
  exit 0
fi
[ -x "$ROCMINFO" ] || { echo "ROCm not installed yet: run this script as root first (wsl.exe -d Ubuntu -u root -- bash $0)"; exit 1; }
$ROCMINFO | grep -E "Name:|gfx" | head -8
VENV=$HOME/dotasim-rocm-venv
if [ ! -x "$VENV/bin/python" ]; then
  ~/.local/bin/uv venv --python 3.12 "$VENV"
fi
source "$VENV/bin/activate"
B=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2
for w in torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl \
         torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
         triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl; do
  f=$(echo "$w" | sed 's/%2B/+/'); [ -f "$f" ] || wget -q "$B/$w" -O "$f"
done
~/.local/bin/uv pip install --python "$VENV/bin/python" torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl \
  torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl triton-3.5.1+rocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl \
  numpy protobuf psutil tensorboard pytest
# WSL uses the host's HSA runtime (librocdxg path); the wheel's copy must go
rm -f "$VENV"/lib/python3.12/site-packages/torch/lib/libhsa-runtime64.so*
python -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python - <<'PY'
import torch, time
if torch.cuda.is_available():
    x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); t = time.time()
    for _ in range(20): y = x @ x
    torch.cuda.synchronize(); print("20 matmuls 4096^2:", round(time.time() - t, 2), "s")
PY
echo "to use: scripts/wsl.sh with PATH=$VENV/bin first, or run python -m dotasim.train.selfplay --device cuda from this venv"
