#!/bin/bash

YEAR=""
MODE="" # 'train' or 'test'

usage() {
    echo "Usage: bash submit_jobs.sh -y <year> -m <train|test>"
    echo "Options:"
    echo "  -y, --YEAR  <year>    [REQUIRED] Data taking year (e.g., 2024, 2025)"
    echo "  -m, --mode  <mode>    [REQUIRED] Execution mode: 'train' or 'test'"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -y|--YEAR) YEAR="$2"; shift ;;
        -m|--mode) MODE="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "[ERROR] Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Safety Check: Enforce Required Arguments
if [[ -z "$YEAR" || -z "$MODE" ]]; then
    echo "[ERROR] Missing required arguments."
    usage
fi

if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
    echo "[ERROR] Mode must be 'train' or 'test'."
    exit 1
fi

echo "[INFO] Preparing ${MODE}ing jobs for YEAR = ${YEAR}"


n_folds=10
special_name="BackgroundTest_${YEAR}"
train_region="4b" 
test_region="4btest" 

# Base paths
base_script_dir="/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN"
input_dir="/data/dust/user/wanghaoy/XtoYH4b/${special_name}"
CMSSW_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/"


if [[ "$MODE" == "train" ]]; then
    run_type="train-only"
    script_name="fold_training.py"
    output_dir="${input_dir}/Train_BackgroundEstimation_condor"
    output_job_dir="${output_dir}/${train_region}/job3"
    isBalance=1
elif [[ "$MODE" == "test" ]]; then
    run_type="test-only"
    script_name="fold_unroll4b_evaluation_removeempty.py"
    plot_script_name="plot_fold.py"
    output_dir="${input_dir}/${test_region}_evaluation"
    output_job_dir="${output_dir}/job3"
    isBalance=0
else
    echo "[ERROR] Invalid mode: $MODE"
    exit 1
fi

mkdir -p "$output_job_dir/logs"

# Copy scripts to working directory
cp "${base_script_dir}/${script_name}" "$output_dir"
[[ "$MODE" == "test" ]] && cp "${base_script_dir}/${plot_script_name}" "$output_dir"

# ==============================================================================
# Job Preparation
# ==============================================================================
declare -A jobs

if [[ "$MODE" == "train" ]]; then
    for i in $(seq 1 $n_folds); do
        job_key="DNN_${train_region}vs2b_${special_name}_Fold${i}"
        jobs["$job_key"]="python3 ${script_name} --YEAR ${YEAR} --isScaling 1 --isBalanceClass ${isBalance} --Model DNN --runType train-only --TrainRegion ${train_region} --TestRegion ${test_region} --foldN ${i} --Nfold ${n_folds}"
    done
elif [[ "$MODE" == "test" ]]; then
    # Evaluation typically runs as a single job or a specific set
    job_key="Evaluation_${train_region}vs2b_${special_name}"
    jobs["$job_key"]="python3 ${script_name} --YEAR ${YEAR} --isScaling 1 --isBalanceClass ${isBalance} --Model DNN --runType test-only --TrainRegion ${train_region} --TestRegion ${test_region} --Nfold ${n_folds}"
fi

master_submit="$output_job_dir/condor_submit_${MODE}_${special_name}.sh"
: > "$master_submit"

for name in "${!jobs[@]}"; do
    exe_file="$output_job_dir/execute_${name}.sh"
    sub_file="$output_job_dir/submit_${name}.sh"

    cat << EOF > "$exe_file"
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd $CMSSW_dir
eval \`scramv1 runtime -sh\`
cd $output_dir
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
echo "[SUCCESS] All ${MODE}ing jobs prepared."
echo "          Submit with: $master_submit"