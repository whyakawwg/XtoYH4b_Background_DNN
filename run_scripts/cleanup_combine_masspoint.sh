#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Usage: bash cleanup_combine_masspoint.sh <mass-point-directory> <final-root-filename>" >&2
    exit 2
fi

if [[ ! -d "$1" ]]; then
    echo "[ERROR] Cleanup directory does not exist: $1" >&2
    exit 1
fi

workdir=$(realpath "$1")
final_file="$2"

if [[ "$(basename "$workdir")" != MX-*_MY-* ]]; then
    echo "[ERROR] Refusing cleanup outside an MX-<MX>_MY-<MY> directory: $workdir" >&2
    exit 1
fi
parent_path=$(dirname "$workdir")
parent_dir=$(basename "$parent_path")
grandparent_dir=$(basename "$(dirname "$parent_path")")
if [[ ! "$parent_dir" =~ ^CombineInput_(Run2_)?[34]b$ && \
      ! ( "$parent_dir" =~ ^SplitIndex[0-4]$ && "$grandparent_dir" =~ ^CombineInput_(Run2_)?3b$ ) && \
      ! ( "$parent_dir" == "NC3bAverage5Splits" && "$grandparent_dir" =~ ^CombineInput_(Run2_)?4b$ ) ]]; then
    echo "[ERROR] Refusing cleanup outside a CombineInput_<region> directory: $workdir" >&2
    exit 1
fi
if [[ "$final_file" == */* || "$final_file" != *.root ]]; then
    echo "[ERROR] Final output must be a ROOT filename without a directory: $final_file" >&2
    exit 1
fi
if [[ ! -f "$workdir/$final_file" ]]; then
    echo "[ERROR] Final output is missing; no cleanup performed: $workdir/$final_file" >&2
    exit 1
fi

find "$workdir" -maxdepth 1 -type f -name '*.root' ! -name "$final_file" -print -delete
echo "[INFO] Cleanup complete; retained $workdir/$final_file"
