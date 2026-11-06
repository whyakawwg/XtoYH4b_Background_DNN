#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/
eval `scramv1 runtime -sh`

set -euo pipefail

gof_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/CombineHarvester/CombineTools/XYHto4b/GoF"

# year="2025"
# split_index="1"
# combined_eras=0

years=("2024" "2025")
split_indices=("0" "1")
year="combined_2024_2025"
split_index="0_1"
split_tag="SplitIndex0_1"

mx="600"
my="150"
run_tag="${year}_SplitIndex${split_index}_MX${mx}_MY${my}"
run_dir="${gof_dir}/results_3b/${run_tag}"
combined_eras=1


mkdir -p "${run_dir}/datacards"
mkdir -p "${run_dir}/workspaces"
mkdir -p "${run_dir}/gof"

cd "$gof_dir"

if [[ $combined_eras -eq 1 ]]; then
    bash combineCards_eras.sh "${years[*]}" "${split_indices[*]}" "$mx" "$my"

else
    python3 write_datacard.py \
        --YEAR "$year" \
        --split-index "$split_index" \
        --MX "$mx" \
        --MY "$my" \
        --output-dir "${run_dir}/datacards"
fi

for card in "${run_dir}"/datacards/*.txt; do
    card_name=$(basename "$card" .txt)

    text2workspace.py "$card" \
        -o "${run_dir}/workspaces/${card_name}.root" \
        --X-allow-no-signal
done



# for i in {1..3}
# do
#     python3 run_postfit.py --YEAR 2024 --combination $i --MX 1000 --MY 150 --fix_signal 1 | tee -a postfit_2024_comb${i}_MX1000_MY150.log
    
#     # python3 run_GoF_test.py --YEAR 2024 --combination $i --MX 1000 --MY 150 --ntoys 400 | tee -a GoF_2024_comb${i}_MX1000_MY150_ntoys400.log
    
#     # test different gof method
#     # python3 run_GoF_test.py --YEAR 2024 --combination $i --MX 1000 --MY 150 --ntoys 400 --algorithm "AD" | tee -a GoF_AD_2024_comb${i}_MX1000_MY150_ntoys400.log
#     # python3 run_GoF_test.py --YEAR 2024 --combination $i --MX 1000 --MY 150 --ntoys 400 --algorithm "KS" | tee -a GoF_KS_2024_comb${i}_MX1000_MY150_ntoys400.log

# done
# python3 postfit_ratio.py

# for i in {1..3}
# do
#     python3 run_GoF_test.py --YEAR 2024 --combination $i --MX 1000 --MY 150 --ntoys 400 | tee -a GoF_2024_comb${i}_MX1000_MY150_ntoys400.log
    
# done


# python3 run_postfit.py --YEAR 2024 --combination 1 --MX 1000 --MY 150 --fix_signal 1 | tee -a postfit_2024_comb1_MX1000_MY150.log
# python3 run_GoF_test.py --YEAR 2024 --combination 1 --MX 1000 --MY 150 --ntoys 400 | tee -a GoF_2024_comb1_MX1000_MY150_ntoys400.log

# same the log to a log file
#python3 run_GoF_test.py --YEAR 2024 --combination 1 --MX 1000 --MY 150 --ntoys 400 | tee -a GoF_2024_comb1_MX1000_MY150_ntoys400.log

# Plot the impacts
# python3 run_impact.py --YEAR 2024 --combination 1 --MX 1000 --MY 150 --expect_signal 0 --ntoys 400 | tee -a Impact_comb1_MX1000_MY150_ntoys400.log

for combination in 1 2 3; do
    result_dir="${run_dir}/gof/combination${combination}"
    mkdir -p "$result_dir"
    cd "$result_dir"
    python3 "${gof_dir}/run_GoF_test.py" --YEAR $year  --split-index $split_index --combination "$combination" --MX $mx --MY $my --ntoys 1000 --algorithm saturated \
        | tee "GoF_${year}_SplitIndex${split_index}_Comb${combination}_MX_${mx}_MY_${my}.log"
done

# bash Run_GoF.sh