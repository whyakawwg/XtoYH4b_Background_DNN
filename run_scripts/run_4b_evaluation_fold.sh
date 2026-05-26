#!/bin/bash
YEAR="" # No default value!

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -y|--YEAR) YEAR="$2"; shift ;;
        -h|--help) 
            echo "Usage: bash run_4b_evaluation_fold.sh -y <year>"
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
    echo "        Example: bash run_4b_evaluation_fold.sh --YEAR 2025"
    exit 1
fi

echo "[INFO] Preparing training jobs for YEAR = ${YEAR}"

n_folds=10

special_name="Background_${YEAR}"

run_type="test-only" 

train_region="4b" # Change the train region: "4b", "3b"

test_region="4btest" # Change the test region: "4btest", "3btest", "3bHiggsMW"

input_dir="/data/dust/user/wanghaoy/XtoYH4b/${special_name}"

output_dir="${input_dir}/${test_region}_evaluation"

script_dir="/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN/fold_unroll4b_evaluation_removeempty.py"

plot_script_dir="/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN/plot_fold.py"

CMSSW_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/"

# If "train-only", isBalanceClass should be 1
# If "train-test", isBalanceClass should be 0

mkdir -p "$output_dir"
output_job_dir="${output_dir}/job3"
mkdir -p "$output_job_dir"

cp "$script_dir" "$output_dir"
cp "$plot_script_dir" "$output_dir"

declare -A jobs

# jobs["Evaluation_${train_region}vs2b_${special_name}"]="python3 fold_4b_evaluation.py --YEAR 2024 --isScaling 1 --isBalanceClass 0 --Model DNN --runType ${run_type} --TrainRegion ${train_region} --TestRegion ${test_region} --Nfold ${n_folds}"
jobs["Evaluation_${train_region}vs2b_${special_name}"]="python3 fold_unroll4b_evaluation_removeempty.py --YEAR ${YEAR} --isScaling 1 --isBalanceClass 0 --Model DNN --runType ${run_type} --TrainRegion ${train_region} --TestRegion ${test_region} --Nfold ${n_folds}"

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

cd $output_dir
${jobs[$name]}
# python3 plot_fold.py --YEAR ${YEAR} --isScaling 1 --isBalanceClass 0 --Model DNN --runType ${run_type} --TrainRegion ${train_region} --TestRegion ${test_region} --Nfold ${n_folds}
EOF
    chmod +x "$exe_file"

    cat << EOF > "$sub_file"
universe   = vanilla
executable = $exe_file
getenv     = TRUE
request_memory = 24 GB
request_cpus = 4
request_gpus = 1
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
