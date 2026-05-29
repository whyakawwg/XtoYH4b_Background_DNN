#!/bin/bash
YEAR="" # No default value!

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -y|--YEAR) YEAR="$2"; shift ;;
        -h|--help) 
            echo "Usage: bash run_4b_train_fold.sh -y <year>"
            echo "Options:"
            echo "  -y, --YEAR <year>    [REQUIRED] Specify the data taking year (e.g., 2024, 2025)."
            exit 0
            ;;
        *) echo "[ERROR] Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$YEAR" ]]; then
    echo "[ERROR] Missing required argument: -y or --YEAR."
    echo "        You must explicitly define the year to prevent accidental job submissions."
    echo "        Example: bash run_4b_train_fold.sh --YEAR 2025"
    exit 1
fi

echo "[INFO] Preparing training jobs for YEAR = ${YEAR}"

n_folds=10

special_name="Background_${YEAR}"

run_type="train-only" # Change the run type: "train-test", "train-only", "test-only"

train_region="4b" # Change the train region: "4b", "3b"

test_region="4btest" # Change the test region: "4btest", "3btest", "3bHiggsMW"

script_dir="/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN/fold_training.py"

input_dir="/data/dust/user/wanghaoy/XtoYH4b/${special_name}"

CMSSW_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/"

output_dir="${input_dir}/Train_BackgroundEstimation_condor"

# If "train-only", isBalanceClass should be 1
# If "train-test", isBalanceClass should be 0

mkdir -p "$output_dir"

output_job_dir="${output_dir}/${train_region}/job3"
mkdir -p "$output_job_dir"
mkdir -p "$output_job_dir"/logs

cp "$script_dir" "$input_dir"

declare -A jobs

for i in $(seq 1 $n_folds); do
    
    job_key="DNN_${train_region}vs2b_${special_name}_Fold${i}"
    
    jobs["$job_key"]="python3 fold_training.py --YEAR ${YEAR} --isScaling 1 --isBalanceClass 1 --Model DNN --runType ${run_type} --TrainRegion ${train_region} --TestRegion ${test_region} --foldN ${i} --Nfold ${n_folds}"
    
done

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
request_memory = 24 GB
request_cpus = 4
request_gpus = 1
log        = $output_job_dir/logs/job_${name}.log
output     = $output_job_dir/logs/job_${name}.out
error      = $output_job_dir/logs/job_${name}.err
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
