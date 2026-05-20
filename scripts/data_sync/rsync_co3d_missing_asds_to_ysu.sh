#!/usr/bin/env bash
set -euo pipefail

# Sync missing CO3D categories from ASDS -> YSU without overwriting existing files.
#
# Example:
#   bash scripts/data_sync/rsync_co3d_missing_asds_to_ysu.sh \
#     --asds-user asds --asds-host dgx \
#     --asds-root /home/asds/project_Hayk_Minasyan/data/co3d_processed_10cat8seq_fixed \
#     --ysu-root /mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed

ASDS_USER=""
ASDS_HOST=""
ASDS_ROOT=""
YSU_ROOT="/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asds-user) ASDS_USER="$2"; shift 2 ;;
    --asds-host) ASDS_HOST="$2"; shift 2 ;;
    --asds-root) ASDS_ROOT="$2"; shift 2 ;;
    --ysu-root) YSU_ROOT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${ASDS_USER}" || -z "${ASDS_HOST}" || -z "${ASDS_ROOT}" ]]; then
  echo "Missing required args. Need --asds-user --asds-host --asds-root" >&2
  exit 1
fi

missing_categories=(
  bottle couch cup donut frisbee hairdryer handbag hotdog hydrant kite
  laptop microwave motorcycle parkingmeter pizza sandwich skateboard stopsign
  teddybear toaster toybus toyplane toytrain tv wineglass
)

mkdir -p "${YSU_ROOT}"

for cat in "${missing_categories[@]}"; do
  src="${ASDS_USER}@${ASDS_HOST}:${ASDS_ROOT}/${cat}/"
  dst="${YSU_ROOT}/${cat}/"
  echo "==> Sync ${cat}"
  # --ignore-existing prevents overwriting files already on YSU.
  rsync -avz --ignore-existing "${src}" "${dst}"
done

echo "Done. Synced missing categories into ${YSU_ROOT}"
