#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash combineCards_eras.sh "YEAR [YEAR ...]" "SPLIT_INDEX [SPLIT_INDEX ...]" MX MY [OUTPUT_DIR]

Example:
  bash combineCards_eras.sh "2024 2025" "0 1" 600 150

The year and split-index lists are positional pairs. The example combines
2024/SplitIndex0 with 2025/SplitIndex1. One combined card is produced for
each 3b GoF combination (1, 2, and 3).
EOF
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
    usage >&2
    exit 2
fi

read -r -a years <<< "$1"
read -r -a split_indices <<< "$2"
mx=$3
my=$4

if [[ ${#years[@]} -lt 2 ]]; then
    echo "ERROR: At least two years are required." >&2
    exit 2
fi

if [[ ${#years[@]} -ne ${#split_indices[@]} ]]; then
    echo "ERROR: Each year must have exactly one corresponding split index." >&2
    exit 2
fi

if [[ ! $mx =~ ^[0-9]+$ || ! $my =~ ^[0-9]+$ ]]; then
    echo "ERROR: MX and MY must be non-negative integers." >&2
    exit 2
fi

for index in "${split_indices[@]}"; do
    if [[ ! $index =~ ^[0-4]$ ]]; then
        echo "ERROR: Invalid split index '$index'; expected an integer from 0 to 4." >&2
        exit 2
    fi
done

script_dir="/afs/desy.de/user/w/wanghaoy/private/work/CombineTools/XYHto4b/GoF"
base_dir="${script_dir}/results_3b"

combined_year="combined"
split_tag="SplitIndex"
for i in "${!years[@]}"; do
    combined_year+="_${years[$i]}"
    if [[ $i -gt 0 ]]; then
        split_tag+="_"
    fi
    split_tag+="${split_indices[$i]}"
done

run_tag="${combined_year}_${split_tag}_MX${mx}_MY${my}"
output_dir=${5:-"${base_dir}/${run_tag}/datacards"}
mkdir -p "$output_dir"

for combination in 1 2 3; do
    combine_args=()

    for i in "${!years[@]}"; do
        year=${years[$i]}
        split_index=${split_indices[$i]}
        input_run_tag="${year}_SplitIndex${split_index}_MX${mx}_MY${my}"
        card_name="XYH_3b_${combination}_13p6TeV_${year}_XtoYHto4B_Par-MX-${mx}-MY-${my}_SplitIndex${split_index}.txt"
        card_path="${base_dir}/${input_run_tag}/datacards/${card_name}"

        if [[ ! -f $card_path ]]; then
            echo "ERROR: Missing input card: $card_path" >&2
            exit 1
        fi

        combine_args+=("era${year}=${card_path}")
    done

    output_name="XYH_3b_${combination}_13p6TeV_${combined_year}_XtoYHto4B_Par-MX-${mx}-MY-${my}_${split_tag}.txt"
    output_card="${output_dir}/${output_name}"
    temporary_card=$(mktemp "${output_card}.tmp.XXXXXX")

    if ! combineCards.py "${combine_args[@]}" > "$temporary_card"; then
        rm -f "$temporary_card"
        echo "ERROR: combineCards.py failed for combination ${combination}." >&2
        exit 1
    fi

    mv -f "$temporary_card" "$output_card"
    echo "Combined card written to $output_card"
done

echo "Created combined 3b GoF cards in $output_dir"

# combineCards_eras.sh "2024 2025" "0 1" 600 150