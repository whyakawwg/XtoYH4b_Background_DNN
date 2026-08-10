#!/bin/bash

YEAR=""
MODE="" # 'train' or 'test'
REGION="" # '4b' or '3b'
TEST_REGION="" # '3btest', '3bHiggsMW', '4btest', '4bHiggsMW'
PAIR_INDEX="0" # Four-jet pairing index for training

usage() {
    echo "Usage: bash run_fold.sh -y <year> -m <train|test> -r <4b|3b> [-p <0|1|2>]"
    echo "Options:"
    echo "  -y, --YEAR  <year>    [REQUIRED] Data taking year (e.g., 2024, 2025)"
    echo "  -m, --mode  <mode>    [REQUIRED] Execution mode: 'train' or 'test'"
    echo "  -r, --region <reg>    [REQUIRED] Analysis region: '4b' or '3b'" 
    echo "  -tr, --testregion <test_reg>    [OPTIONAL] Test region: '3btest', '3bHiggsMW', '4btest', '4bHiggsMW'"
    echo "  -p, --pair-index <idx> [OPTIONAL] Training pairing index: 0, 1, or 2 (default: 0)"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -y|--YEAR) YEAR="$2"; shift ;;
        -m|--mode) MODE="$2"; shift ;;
        -r|--region) REGION="$2"; shift ;;
        -tr|--testregion) TEST_REGION="$2"; shift ;;
        -p|--pair-index) PAIR_INDEX="$2"; shift ;;
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

if [[ "$REGION" != "3b" && "$REGION" != "4b" ]]; then
    echo "[ERROR] Region must be '3b' or '4b'."
    exit 1
fi

if [[ "$PAIR_INDEX" != "0" && "$PAIR_INDEX" != "1" && "$PAIR_INDEX" != "2" ]]; then
    echo "[ERROR] Pair index must be 0, 1, or 2. Received: ${PAIR_INDEX}"
    exit 1
fi

echo "[INFO] Preparing ${MODE}ing jobs for YEAR=${YEAR} and REGION=${REGION}"
if [[ "$MODE" == "train" ]]; then
    echo "[INFO] Training pairing index: ${PAIR_INDEX} (pair${PAIR_INDEX})"
fi


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
    output_job_dir="${output_dir}/${REGION}/job3/pair${PAIR_INDEX}"
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

load_mass_points() {
    local filename_tree="Tree_Data.root"
    local mapping_year="$YEAR"
    local mapping_file
    local root_output
    local marker mx my point_key
    local -A seen_mass_points=()

    if [[ "$YEAR" == "2024" || "$YEAR" == "2025" ]]; then
        filename_tree="Tree_Data_Parking.root"
    elif [[ "$YEAR" == "2022Full" ]]; then
        mapping_year="2022"
    elif [[ "$YEAR" == "2023Full" ]]; then
        mapping_year="2023"
    fi
    mapping_file="/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/${mapping_year}/${filename_tree}"

    if [[ ! -f "$mapping_file" ]]; then
        echo "[ERROR] Mass-point mapping ROOT file not found: ${mapping_file}"
        return 1
    fi
    if ! command -v root > /dev/null 2>&1; then
        echo "[ERROR] ROOT is required to read the authoritative mass-point mapping."
        return 1
    fi

    if ! root_output=$(MASS_POINT_FILE="$mapping_file" root -l -b -q -e '
        TFile input(gSystem->Getenv("MASS_POINT_FILE"));
        auto tree = (TTree*)input.Get("Tree_SignalGrid");
        if (!tree || tree->GetEntries() < 1) { std::cerr << "Missing Tree_SignalGrid" << std::endl; gSystem->Exit(2); }
        std::vector<int>* mx = nullptr; std::vector<int>* my = nullptr;
        tree->SetBranchAddress("mX_sig", &mx); tree->SetBranchAddress("mY_sig", &my); tree->GetEntry(0);
        if (!mx || !my || mx->size() != 272 || my->size() != 272) { std::cerr << "Mass-point mapping must contain 272 MX/MY entries" << std::endl; gSystem->Exit(3); }
        for (size_t i = 0; i < mx->size(); ++i) std::cout << "MASS_POINT " << mx->at(i) << " " << my->at(i) << std::endl;
    ' 2>&1); then
        echo "[ERROR] Failed to read mass-point mapping from ${mapping_file}:"
        echo "$root_output"
        return 1
    fi

    MASS_POINTS=()
    while read -r marker mx my; do
        [[ "$marker" == "MASS_POINT" ]] || continue
        point_key="${mx}_${my}"
        if [[ -n "${seen_mass_points[$point_key]:-}" ]]; then
            echo "[ERROR] Ambiguous duplicate mass point (MX, MY)=(${mx}, ${my})."
            return 1
        fi
        seen_mass_points["$point_key"]=1
        MASS_POINTS+=("${mx} ${my}")
    done <<< "$root_output"

    if [[ "${#MASS_POINTS[@]}" -ne 272 ]]; then
        echo "[ERROR] Expected 272 mass points, found ${#MASS_POINTS[@]}."
        return 1
    fi
}

if [[ "$MODE" == "train" ]]; then
    # Training logic: Handle 3b split vs standard 4b
    if [[ "$REGION" == "3b" ]]; then
        for split in {0..4}; do
            for i in $(seq 1 $n_folds); do
                job_key="DNN_${REGION}vs2b_${special_name}_Fold${i}_Split${split}_Pair${PAIR_INDEX}"
                jobs["$job_key"]="python3 ${script_name} --YEAR ${YEAR} --isScaling 1 --isBalanceClass 1 --Model DNN --runType train-only --TrainRegion ${REGION} --TestRegion ${TEST_REGION} --foldN ${i} --Nfold ${n_folds} --SplitIndex ${split} --pair-index ${PAIR_INDEX}"
            done
        done
    else
        for i in $(seq 1 $n_folds); do
            job_key="DNN_${REGION}vs2b_${special_name}_Fold${i}_Pair${PAIR_INDEX}"
            jobs["$job_key"]="python3 ${script_name} --YEAR ${YEAR} --isScaling 1 --isBalanceClass 1 --Model DNN --runType train-only --TrainRegion ${REGION} --TestRegion ${TEST_REGION} --foldN ${i} --Nfold ${n_folds} --pair-index ${PAIR_INDEX}"
        done
    fi
else
    declare -a MASS_POINTS
    load_mass_points || exit 1
    echo "[INFO] Creating ${REGION} evaluation jobs for ${#MASS_POINTS[@]} mass points."

    for mass_point in "${MASS_POINTS[@]}"; do
        read -r mx my <<< "$mass_point"
        job_key="Evaluation_${REGION}vs2b_${special_name}_MX${mx}_MY${my}"
        jobs["$job_key"]="python3 ${script_name} --YEAR ${YEAR} --isScaling 1 --isBalanceClass ${isBalance} --Model DNN --runType test-only --TrainRegion ${REGION} --TestRegion ${TEST_REGION} --Nfold ${n_folds} --MX ${mx} --MY ${my}"
    done
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
