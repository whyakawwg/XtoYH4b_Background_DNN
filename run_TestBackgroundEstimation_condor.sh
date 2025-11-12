#!/bin/bash

test_region="4btest" # Change the test region: "4btest", "3btest", "3bHiggsMW"

CMSSW_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/"

Test_script_dir="/afs/desy.de/user/w/wanghaoy/private/work/XtoYH4b_Background_DNN/Test_BackgroundEstimation_DATA_DATA_UParTAK4.py"

input_dir="/data/dust/user/wanghaoy/XtoYH4b/without_jetmass"
output_dir="${input_dir}/TestBackgroundEstimation_condor"
mkdir -p "$output_dir"

output_job_dir="${output_dir}/${test_region}/job3"
mkdir -p "$output_job_dir"

cp "$Test_script_dir" "$input_dir"

# Use only the selected job here
unset jobs
declare -A jobs

jobs["DNN_${test_region}_DATA_DATA_UParTAK4"]="python3 Test_BackgroundEstimation_DATA_DATA_UParTAK4.py --YEAR 2024 --isScaling 1 --isBalanceClass 0 --TestRegion ${test_region} "


master_submit="$output_job_dir/condor_submit_test.sh"
: > "$master_submit"


for name in "${!jobs[@]}"; do
    exe_file="$output_job_dir/execute_${name}.sh"
    sub_file="$output_job_dir/submit_${name}.sh"

    cat << EOF > "$exe_file"
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd $CMSSW_dir
eval \`scramv1 runtime -sh\`

cd $input_dir
${jobs[$name]}
EOF
    chmod +x "$exe_file"

    cat << EOF > "$sub_file"
universe   = vanilla
executable = $exe_file
getenv     = TRUE
request_memory = 64 GB
log        = $output_job_dir/job_${name}.log
output     = $output_job_dir/job_${name}.out
error      = $output_job_dir/job_${name}.err
notification = never
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
request_runtime = 100000
+MaxRuntime = 100000
queue
EOF

    echo "condor_submit $sub_file" >> "$master_submit"
    echo "Prepared job: $name"
done

chmod +x "$master_submit"
echo "All jobs prepared."
echo "Please submit jobs with: $master_submit"
