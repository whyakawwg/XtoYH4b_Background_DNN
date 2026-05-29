#!/bin/bash
year=$1
suffix=$2
mode=$3

# Validation
if [[ -z "$year" ]]; then
    echo "[ERROR] Missing the Year."
    exit 1
fi

if [[ "$year" != "2024" && "$year" != "2025" && "$year" != "2022" && "$year" != "2022EE" && "$year" != "2023" && "$year" != "2023BPiX" ]]; then
    echo "This is a combined eras: ${year}"
    combined=1
else
    combined=0
fi

if [[ -z "$mode" ]]; then
    echo "[ERROR] Missing the mode. "1" for datacard and workspace creation, "2" for limit calculation, "3" for plotting limits. "check_ws" for checking workspaces."
    exit 1
fi

CombineResults_dir="/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/CombineResults/bkg"
Combine_script_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/CombineHarvester/CombineTools/XYHto4b/"

if [[ "$mode" != "3" ]]; then
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    cd /afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/
    eval `scramv1 runtime -sh`
fi

if [[ "$mode" -eq 1 ]]; then
    echo "Running datacard and workspace creation for year ${year} with suffix ${suffix}..."

    cd $CombineResults_dir

    if [ "$combined" -eq 0 ]; then
        mkdir -p ${year}
        cd ${year}

        mkdir -p datacards${suffix}
        mkdir -p workspace${suffix}
        mkdir -p limits${suffix}

        cd datacards${suffix}
        CreateCardsfull ${year} data_driven
    fi 
    
    cd $Combine_script_dir
    bash run_text2workspace_condor.sh ${year} ${suffix}
    bash $CombineResults_dir/${year}/workspace${suffix}/condor_submit_text2workspace_1.sh

elif [[ "$mode" == "check_ws" ]]; then
    echo "Checking workspaces for year ${year} with suffix ${suffix}... do Run_Limits_FullProcess.sh ${year} ${suffix} 2 after confirming all workspaces are created."
    cd $Combine_script_dir
    python3 check_workspaces.py --YEAR ${year} --SUFFIX ${suffix}

elif [[ "$mode" -eq 2 ]]; then
    echo "Running limit calculation for year ${year} with suffix ${suffix}..."
    cd $Combine_script_dir
    bash run_limit_condor.sh ${year} ${suffix}
    bash $CombineResults_dir/${year}/workspace${suffix}/condor_submit.sh
    echo "Note, need to open a new terminal to do the next step! "

elif [[ "$mode" -eq 3 ]]; then
    echo "Note: Need to deactivate cmsenv for the proper environment setup to plot the limits. You can simply open a new terminal."
    echo "Running plotting limits for year ${year} with suffix ${suffix}..."
    source /data/dust/user/chatterj/environments/bdt/bin/activate
    cd $Combine_script_dir
    python3 plotLimits.py --YEAR ${year} --SUFFIX ${suffix}

else
    echo "[ERROR] Invalid mode: $mode. Please choose 1, 2, check_ws or 3."
    exit 1
fi