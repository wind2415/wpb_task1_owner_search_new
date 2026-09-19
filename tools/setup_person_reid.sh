#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PKG_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
THIRD_PARTY_DIR="$PKG_DIR/third_party"
REID_DIR="$THIRD_PARTY_DIR/deep-person-reid"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REID_ZIP_URL="https://codeload.github.com/KaiyangZhou/deep-person-reid/zip/refs/heads/master"
REID_GIT_URL="https://github.com/KaiyangZhou/deep-person-reid.git"
MODEL_DIR="$PKG_DIR/models/reid"
DEFAULT_WEIGHT="$MODEL_DIR/osnet_x0_25_msmt17.pth"
DEFAULT_WEIGHT_ID="1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e"

mkdir -p "$THIRD_PARTY_DIR" "$MODEL_DIR"

download_repo_zip() {
  "$PYTHON_BIN" - "$REID_ZIP_URL" "$THIRD_PARTY_DIR" "$REID_DIR" <<'PY'
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

url, base_dir, target_dir = sys.argv[1:4]
fd, zip_path = tempfile.mkstemp(prefix="deep-person-reid-", suffix=".zip", dir="/tmp")
os.close(fd)
request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/zip"})
try:
    print("Downloading", url)
    with urllib.request.urlopen(request, timeout=90) as response, open(zip_path, "wb") as output:
        shutil.copyfileobj(response, output)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(base_dir)
    extracted = os.path.join(base_dir, "deep-person-reid-master")
    if os.path.exists(target_dir):
        os.rmdir(target_dir)
    os.rename(extracted, target_dir)
finally:
    if os.path.exists(zip_path):
        os.unlink(zip_path)
PY
}

if [ ! -f "$REID_DIR/setup.py" ]; then
  if command -v git >/dev/null 2>&1 && git help -a 2>/dev/null | grep -q 'remote-https'; then
    git clone --depth 1 "$REID_GIT_URL" "$REID_DIR"
  else
    download_repo_zip
  fi
else
  echo "deep-person-reid already exists: $REID_DIR"
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "python pip is missing. Install python3-pip first, then rerun this script." >&2
  exit 1
fi

if [ "${INSTALL_TORCH:-1}" = "1" ]; then
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    "$PYTHON_BIN" -m pip install --user torch torchvision --index-url "$TORCH_INDEX_URL"
  else
    "$PYTHON_BIN" -m pip install --user torch torchvision
  fi
fi

"$PYTHON_BIN" -m pip install --user -r "$REID_DIR/requirements.txt"
"$PYTHON_BIN" -m pip install --user -e "$REID_DIR"

if [ "${DOWNLOAD_REID_WEIGHT:-0}" = "1" ] && [ ! -s "$DEFAULT_WEIGHT" ]; then
  "$PYTHON_BIN" -m pip install --user gdown
  "$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=$DEFAULT_WEIGHT_ID" -O "$DEFAULT_WEIGHT"
  echo "Downloaded Re-ID weight: $DEFAULT_WEIGHT"
fi

cat <<EOF2

Person Re-ID setup complete.

Source: $REID_DIR
Optional weight: $DEFAULT_WEIGHT

If you downloaded the optional weight, launch with:
  roslaunch wpb_task1_owner_search person_reid_owner_test.launch reid_model_path:=$DEFAULT_WEIGHT

EOF2
