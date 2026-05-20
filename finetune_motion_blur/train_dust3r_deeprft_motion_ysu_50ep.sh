#!/usr/bin/env bash
# Thin wrapper to keep a stable filename for transfer/submission.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_dust3r_motion_spatial_ysu_50ep.sh" "$@"
