#!/bin/bash

special_name="5fold"

run_type="train-only" # Change the run type: "train-test", "train-only", "test-only"

train_region="4b" # Change the train region: "4b", "3b"

test_region="4btest" # Change the test region: "4btest", "3btest", "3bHiggsMW"

script_dir="/data/dust/user/wanghaoy/XtoYH4b/work_scripts/fold5_training.py"

input_dir="/data/dust/user/wanghaoy/XtoYH4b/test_${special_name}"

CMSSW_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/"

output_dir="${input_dir}/OnlyTrainBackgroundEstimation_condor"


mkdir -p "$output_dir"

output_job_dir="${output_dir}/${train_region}/job3"
mkdir -p "$output_job_dir"

cp "$script_dir" "$input_dir"

declare -A jobs

jobs["DNN_${train_region}vs2b_${special_name}_1"]="python3 fold5_training.py --YEAR 2024 --isScaling 1 --isBalanceClass 1 --Model DNN --runType ${run_type} --TrainRegion ${train_region} --TestRegion ${test_region} --foldN 1"

jobs["DNN_${train_region}vs2b_${special_name}_2"]="python3 fold5_training.py --YEAR 2024 --isScaling 1 --isBalanceClass 1 --Model DNN --runType ${run_type} --TrainRegion ${train_region} --TestRegion ${test_region} --foldN 2"

jobs["DNN_${train_region}vs2b_${special_name}_3"]="python3 fold5_training.py --YEAR 2024 --isScaling 1 --isBalanceClass 1 --Model DNN --runType ${run_type} --TrainRegion ${train_region} --TestRegion ${test_region} --foldN 3"

jobs["DNN_${train_region}vs2b_${special_name}_4"]="python3 fold5_training.py --YEAR 2024 --isScaling 1 --isBalanceClass 1 --Model DNN --runType ${run_type} --TrainRegion ${train_region} --TestRegion ${test_region} --foldN 4"    

jobs["DNN_${train_region}vs2b_${special_name}_5"]="python3 fold5_training.py --YEAR 2024 --isScaling 1 --isBalanceClass 1 --Model DNN --runType ${run_type} --TrainRegion ${train_region} --TestRegion ${test_region} --foldN 5"

master_submit="$output_job_dir/condor_submit_${special_name}.sh"
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
