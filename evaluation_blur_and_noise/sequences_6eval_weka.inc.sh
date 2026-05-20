# YSU weka: /mnt/weka/.../data/co3d_processed often has only this fixed 6 (no donut/couch).
# Use: export SEQUENCES_FILE=.../sequences_6eval_weka.inc.sh
# (source this file; do not execute directly)
# shellcheck disable=SC2034
declare -A SEQUENCES=(
  ["bottle"]="34_1397_4376"
  ["cup"]="12_100_593"
  ["hydrant"]="106_12648_23157"
  ["teddybear"]="101_11758_21048"
  ["toybus"]="111_13154_25988"
  ["toytrain"]="104_12352_22039"
)
