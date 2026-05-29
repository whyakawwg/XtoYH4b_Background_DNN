#!/bin/bash

YEAR=""
MODE="" # 'train' or 'test'
REGION="" # '4b' or '3b'
TEST_REGION="" # '3btest', '3bHiggsMW', '4btest', '4bHiggsMW'

usage() {
    echo "Usage: bash submit_jobs.sh -y <year> -m <train|test> -r <4b|3b>"
    echo "Options:"
    echo "  -y, --YEAR  <year>    [REQUIRED] Data taking year (e.g., 2024, 2025)"
    echo "  -m, --mode  <mode>    [REQUIRED] Execution mode: 'train' or 'test'"
    echo "  -r, --region <reg>    [REQUIRED] Analysis region: '4b' or '3b'" 
    echo "  -tr, --testregion <test_reg>    [OPTIONAL] Test region: '3btest', '3bHiggsMW', '4btest', '4bHiggsMW'"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -y|--YEAR) YEAR="$2"; shift ;;
        -m|--mode) MODE="$2"; shift ;;
        -r|--region) REGION="$2"; shift ;;
        -tr|--testregion) TEST_REGION="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "[ERROR] Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Validation
if [[ -z "$YEAR" || -z "$MODE" || -z "$REGION" ]]; then
    echo "[ERROR] Missing required arguments."
    usage
fi

if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
    echo "[ERROR] Mode must be 'train' or 'test'."
    exit 1
fi

echo "[INFO] Preparing ${MODE}ing jobs for YEAR=${YEAR} and REGION=${REGION}"


n_folds=10
special_name="Background_${YEAR}"

if [[ -z "$TEST_REGION" && "$MODE" == "test" ]]; then
    if [[ "$REGION" == "3b" ]]; then
        TEST_REGION="3btest"
    else
        TEST_REGION="4btest"
    fi
    echo "[INFO] No test region specified. Defaulting to ${TEST_REGION} for ${REGION} training."
fi


if [[ "$REGION" == "3b" ]]; then
    if [[ "$TEST_REGION" == "4bHiggsMW" || "$TEST_REGION" == "4btest" ]]; then
        echo "[ERROR] Invalid test region for 3b training. Allowed: '3btest', '3bHiggsMW'."
        exit 1    
    fi
    if [[ "$MODE" == "train" ]]; then
        TEST_REGION="3btest"
        echo "[INFO] Defaulting test region to ${TEST_REGION} for training."
    fi
else
    if [[ "$TEST_REGION" == "3bHiggsMW" || "$TEST_REGION" == "3btest" ]]; then
        echo "[ERROR] Invalid test region for 4b training. Allowed: '4btest', '4bHiggsMW'."
        exit 1
    fi
    if [[ "$MODE" == "train" ]]; then
        TEST_REGION="4btest"
        echo "[INFO] Defaulting test region to ${TEST_REGION} for training."
    fi
fi

base_script_dir="/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN"
input_dir="/data/dust/user/wanghaoy/XtoYH4b/${special_name}"
CMSSW_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/"

if [[ "$MODE" == "train" ]]; then
    script_name="fold_training.py"
    output_dir="${input_dir}/TestTrain_BackgroundEstimation_condor"
    output_job_dir="${output_dir}/${REGION}/job3"
    isBalance=1
else
    script_name="fold_unroll${REGION}_evaluation_removeempty.py"
    plot_script_name="plot_fold.py"
    output_dir="${input_dir}/${TEST_REGION}_evaluation"
    output_job_dir="${output_dir}/job3"
    isBalance=0
fi

mkdir -p "$output_job_dir/logs"
cp "${base_script_dir}/${script_name}" "$output_dir"
[[ "$MODE" == "test" ]] && cp "${base_script_dir}/${plot_script_name}" "$output_dir"


declare -A jobs

if [[ "$MODE" == "train" ]]; then
    # Training logic: Handle 3b split vs standard 4b
    if [[ "$REGION" == "3b" ]]; then
        for split in {0..4}; do
            for i in $(seq 1 $n_folds); do
                job_key="DNN_${REGION}vs2b_${special_name}_Fold${i}_Split${split}"
                jobs["$job_key"]="python3 ${script_name} --YEAR ${YEAR} --isScaling 1 --isBalanceClass 1 --Model DNN --runType train-only --TrainRegion ${REGION} --TestRegion ${TEST_REGION} --foldN ${i} --Nfold ${n_folds} --SplitIndex ${split}"
            done
        done
    else
        for i in $(seq 1 $n_folds); do
            job_key="DNN_${REGION}vs2b_${special_name}_Fold${i}"
            jobs["$job_key"]="python3 ${script_name} --YEAR ${YEAR} --isScaling 1 --isBalanceClass 1 --Model DNN --runType train-only --TrainRegion ${REGION} --TestRegion ${TEST_REGION} --foldN ${i} --Nfold ${n_folds}"
        done
    fi
else

    job_key="Evaluation_${REGION}vs2b_${special_name}"
    jobs["$job_key"]="python3 ${script_name} --YEAR ${YEAR} --isScaling 1 --isBalanceClass ${isBalance} --Model DNN --runType test-only --TrainRegion ${REGION} --TestRegion ${TEST_REGION} --Nfold ${n_folds}"
fi


master_submit="$output_job_dir/condor_submit_${MODE}_${REGION}_${special_name}.sh"
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
    echo "Prepared: $name"
done

chmod +x "$master_submit"
echo "[SUCCESS] All ${MODE}ing jobs for ${REGION} prepared."
echo "          Submit with: $master_submit"