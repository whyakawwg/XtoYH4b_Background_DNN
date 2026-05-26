#!/bin/bash

n_folds=10

special_name="full_3b"

run_type="test-only" # Change the run type: "train-test", "train-only", "test-only"

train_region="4b" # Change the train region: "4b", "3b"

# test_region="3btest" # Change the test region: "4btest", "3btest", "3bHiggsMW"

input_dir="/data/dust/user/wanghaoy/XtoYH4b/${special_name}"

output_dir="${input_dir}/NonClosure_${train_region}"

script_dir="/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN"

# plot_script_dir="/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN/plot_fold.py"

CMSSW_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/"

# If "train-only", isBalanceClass should be 1
# If "train-test", isBalanceClass should be 0

mkdir -p "$output_dir"
output_job_dir="${output_dir}/job3"
mkdir -p "$output_job_dir"

cp "$script_dir/hist_unroll3b.py" "$output_dir"
cp "$script_dir/convert_to_combine_input.py" "$output_dir"
cp "$script_dir/signal_unroll3b_evaluation_onlyphysical.py" "$output_dir"
cp "$input_dir/3btest_unroll/3btest_OnlyPhysical.root" "$output_dir"
cp "$input_dir/3bHiggsMW_unroll/3bHiggsMW_OnlyPhysical.root" "$output_dir"

# cp "$plot_script_dir" "$output_dir"


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

# python3 hist_unroll3b.py --YEAR "2024" --runType "test-only" --TrainRegion "3b" --TestRegion "3btest" --Nfold 10 --Plot 1 --CreateUncHist 1
# python3 hist_unroll3b.py --YEAR "2024" --runType "test-only" --TrainRegion "3b" --TestRegion "3bHiggsMW" --Nfold 10 --Plot 1 --CreateUncHist 1

# hadd combine_noempty_input.root OnlyPhysical_Signal.root Uncertainty_hists_OnlyPhysical.root

# Make the final Combine input root file, according to the format required by Combine
# python3 convert_to_combine_input.py


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
