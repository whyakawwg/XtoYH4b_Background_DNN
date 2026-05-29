#!/bin/bash
year=$1
suffix=$2

path_dir="/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/CombineResults/bkg"
path_dir="${path_dir}/${year}"


input_dir="${path_dir}/datacards${suffix}"
output_dir="${path_dir}/workspace${suffix}"

if [ -d "$output_dir" ]; then
	echo "Directory already exists: $output_dir"
else
	mkdir -p "$output_dir"
fi

#templates=(1 2 3 4 5 6 7 8 9 10 11 12)
#templates=(5 6 11 12)
templates=(1)

for temp in "${templates[@]}"; do
    submission_file="$output_dir/condor_submit_text2workspace_${temp}.sh"
    : > "$submission_file"  # clear old file for this template

    for file in "$input_dir"/XYH_4b_${temp}_*.txt; do
        [[ -f "$file" ]] || continue

        file_name=$(basename "$file")
        output_workspace="workspace_${file_name/.txt/.root}"
        output_file="$output_dir/execute_${file_name/.txt/}.csh"
        output_name="${file_name/.txt/}"

        # Create execution script for Condor job
        cat << EOF > "$output_file"
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd \$CMSSW_BASE/src
eval \`scramv1 runtime -sh\`

cd $output_dir
text2workspace.py $file -o $output_dir/$output_workspace
EOF

        chmod +x "$output_file"
        echo "Created execution script: $output_file"

        # Create Condor submission file
        output_file_sub="$output_dir/submit_${output_name}.sh"

        cat << EOF > "$output_file_sub"
universe = vanilla
executable = $output_file
getenv = TRUE
log = $output_dir/job_${output_name}.log
output = $output_dir/job_${output_name}.out
error = $output_dir/job_${output_name}.err
notification = never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
queue
+MaxRuntime = 100000
EOF

        echo "Created condor submission script: $output_file_sub"
        echo "condor_submit $output_file_sub" >> "$submission_file"

    done

    chmod +x "$submission_file"
    echo "All jobs for template $temp prepared. Submit with: bash $submission_file"
done
