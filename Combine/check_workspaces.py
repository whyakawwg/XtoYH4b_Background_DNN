import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--YEAR", default="2024", type=str, help="Which era?")
parser.add_argument("--SUFFIX", default="", type=str, help="Suffix for workspace and limit directories")
args = parser.parse_args()

path_dir = Path(f"/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/CombineResults/bkg/{args.YEAR}")
input_dir = path_dir / f"datacards{args.SUFFIX}"
output_dir = path_dir / f"workspace{args.SUFFIX}"

templates = [1]
missing_files = []
expected_count = 0

for temp in templates:
    # Find all matching txt files in the input directory
    for txt_file in input_dir.glob(f"XYH_4b_{temp}_*.txt"):
        expected_count += 1
        
        # txt_file.stem gets the filename without the '.txt' extension
        expected_workspace_name = f"workspace_{txt_file.stem}.root"
        expected_workspace_path = output_dir / expected_workspace_name
        
        # Check if the file exists
        if not expected_workspace_path.exists():
            missing_files.append(expected_workspace_name)

print(f"--- Verification Summary ---")
if expected_count == 0:
    print("No input datacards found. Please check your input directory path.")
elif not missing_files:
    print(f"Success! All {expected_count} workspaces were successfully created.")
else:
    print(f"Warning: {len(missing_files)} out of {expected_count} workspaces are MISSING!")
    print("Missing files:")
    for missing in missing_files:
        print(f"  - {missing}")