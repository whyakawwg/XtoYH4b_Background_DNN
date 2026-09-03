#!/bin/bash

YEAR=""
REGION="" # '4b' or '3b'
SPLIT_INDEX="" # Selected 3b validation split
ADD_3B_HIGGS_NC=0

usage() {
    echo "Usage: bash run_condor_CreateCombineInput.sh -y <year> -r <4b|3b> [-s <0|1|2|3|4>] [--add-3b-higgs-nc <0|1>]"
    echo "Options:"
    echo "  -y, --YEAR  <year>    [REQUIRED] Data taking year (e.g., 2024, 2025)"
    echo "  -r, --region <reg>    [REQUIRED] Analysis region: '4b' or '3b'" 
    echo "  -s, --split-index <idx> [REQUIRED for 3b] Validation split: 0-4"
    echo "      --add-3b-higgs-nc <0|1> [OPTIONAL] Add nc_3b from 3bHiggsMW (default: 0)"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -y|--YEAR) YEAR="$2"; shift ;;
        -r|--region) REGION="$2"; shift ;;
        -s|--split-index) SPLIT_INDEX="$2"; shift ;;
        --add-3b-higgs-nc) ADD_3B_HIGGS_NC="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "[ERROR] Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Validation
if [[ -z "$YEAR" || -z "$REGION" ]]; then
    echo "[ERROR] Missing required arguments."
    usage
fi
if [[ "$REGION" != "3b" && "$REGION" != "4b" ]]; then
    echo "[ERROR] Region must be '3b' or '4b'."
    exit 1
fi
if [[ -n "$SPLIT_INDEX" && ! "$SPLIT_INDEX" =~ ^[0-4]$ ]]; then
    echo "[ERROR] Split index must be 0, 1, 2, 3, or 4. Received: ${SPLIT_INDEX}"
    exit 1
fi
if [[ "$REGION" == "3b" && -z "$SPLIT_INDEX" ]]; then
    echo "[ERROR] --split-index is required for 3b Combine-input jobs."
    exit 1
fi
if [[ "$ADD_3B_HIGGS_NC" != "0" && "$ADD_3B_HIGGS_NC" != "1" ]]; then
    echo "[ERROR] --add-3b-higgs-nc must be 0 or 1."
    exit 1
fi
echo "[INFO] Preparing Combine input for YEAR=${YEAR} and REGION=${REGION}"


special_name="Background_${YEAR}"
CMSSW_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/"
input_dir="/data/dust/user/wanghaoy/XtoYH4b/${special_name}"
output_dir="${input_dir}/CombineInput_${REGION}"
if [[ "$REGION" == "3b" ]]; then
    output_dir="${output_dir}/SplitIndex${SPLIT_INDEX}"
elif [[ "$ADD_3B_HIGGS_NC" == "1" ]]; then
    output_dir="${output_dir}/NC3bAverage5Splits"
fi
script_dir="/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN"
output_job_dir="${output_dir}/jobs_${REGION}"

BkgOutputDir="/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/BackgroundEstimation"

mkdir -p "$output_dir"
mkdir -p "$output_job_dir"
mkdir -p "$output_job_dir/logs"

shared_uncertainty_script="${output_dir}/uncertainty_pipeline.py"
shared_converter_script="${output_dir}/convert_to_combine_input_DecoMX.py"
shared_cleanup_script="${output_dir}/cleanup_combine_masspoint.sh"
cp "$script_dir/uncertainty_pipeline.py" "$shared_uncertainty_script"
cp "$script_dir/convert_to_combine_input_DecoMX.py" "$shared_converter_script"
cp "$script_dir/run_scripts/cleanup_combine_masspoint.sh" "$shared_cleanup_script"

declare -A jobs
declare -A job_output_dirs

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
        if (!mx || !my || mx->size() != 273 || my->size() != 273) { std::cerr << "Mass-point mapping must contain 273 MX/MY entries" << std::endl; gSystem->Exit(3); }
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

    if [[ "${#MASS_POINTS[@]}" -ne 273 ]]; then
        echo "[ERROR] Expected 273 mass points, found ${#MASS_POINTS[@]}."
        return 1
    fi
}


declare -a MASS_POINTS
load_mass_points || exit 1
echo "[INFO] Creating ${REGION} Combine-input jobs for ${#MASS_POINTS[@]} mass points."

if [[ "$REGION" == "3b" ]]; then
    test_region="3btest"
    signal_region="3bHiggsMW"
    signal_region_plot=1
else
    test_region="4btest"
    signal_region="4bHiggsMW"
    signal_region_plot=0
fi

for mass_point in "${MASS_POINTS[@]}"; do
    read -r mx my <<< "$mass_point"
    mass_tag="MX-${mx}_MY-${my}"
    mass_suffix="_MX-${mx}_MY-${my}"
    mass_output_dir="${output_dir}/${mass_tag}"
    name="CombineInput_${REGION}_MX${mx}_MY${my}"

    mkdir -p "$mass_output_dir"

    test_evaluation_file="${input_dir}/${test_region}_evaluation/${test_region}_OnlyPhysical${mass_suffix}.root"
    signal_evaluation_file="${input_dir}/${signal_region}_evaluation/${signal_region}_OnlyPhysical${mass_suffix}.root"
    test_uncertainty_file="${test_region}_Uncertainty_hists_OnlyPhysical${mass_suffix}.root"
    signal_uncertainty_file="${signal_region}_Uncertainty_hists_OnlyPhysical${mass_suffix}.root"
    nonclosure_file="nonclosure_factors${mass_suffix}.root"
    decorrelated_file="combine_addDecorrelatedNC${mass_suffix}.root"
    combined_file="combine_noempty_input${mass_suffix}.root"
    scaled_file="combine_noempty_input_Scaled_${YEAR}${mass_suffix}.root"
    signal_stem="NMSSM-XtoYHto4B_Par-MX-${mx}-MY-${my}_TuneCP5_13p6TeV_madgraph-pythia8"
    final_file="BkgEst_${signal_stem}.root"

    split_args=""
    evaluation_split_dir=""
    evaluation_split_suffix=""
    if [[ "$REGION" == "3b" ]]; then
        split_args="--SplitIndex $SPLIT_INDEX"
        evaluation_split_dir="/SplitIndex${SPLIT_INDEX}"
        evaluation_split_suffix="_SplitIndex${SPLIT_INDEX}"
        test_evaluation_file="${input_dir}/${test_region}_evaluation${evaluation_split_dir}/${test_region}_OnlyPhysical${evaluation_split_suffix}${mass_suffix}.root"
        signal_evaluation_file="${input_dir}/${signal_region}_evaluation${evaluation_split_dir}/${signal_region}_OnlyPhysical${evaluation_split_suffix}${mass_suffix}.root"
    fi

    nc_3b_setup=""
    nc_3b_add_args=""
    if [[ "$ADD_3B_HIGGS_NC" == "1" ]]; then
        for nc_split in {0..4}; do
            nc_3b_evaluation_file="${input_dir}/3bHiggsMW_evaluation/SplitIndex${nc_split}/3bHiggsMW_OnlyPhysical_SplitIndex${nc_split}${mass_suffix}.root"
            nc_3b_setup="${nc_3b_setup}require_file $nc_3b_evaluation_file '3b Higgs-MW SplitIndex${nc_split} nc_3b input'
"
        done
        nc_3b_factor_file="nonclosure_factors_nc_3b_Average5Splits${mass_suffix}.root"
        nc_3b_setup="${nc_3b_setup}python3 $shared_uncertainty_script --YEAR $YEAR --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --Plot 0 --CreateUncHist 0 --function save_nc_3b --MX $mx --MY $my
require_file $nc_3b_factor_file 'pooled five-split nc_3b factor output'"
        nc_3b_add_args="--Add3bHiggsMWNonClosure 1"
    fi

    jobs["$name"]="require_file $test_evaluation_file 'test-region evaluation input'
python3 $shared_uncertainty_script --YEAR $YEAR --runType test-only --TrainRegion $REGION --TestRegion $test_region --Nfold 10 --Plot 1 --CreateUncHist 1 --function create_unc_hists --MX $mx --MY $my $split_args
require_file $test_uncertainty_file 'test-region uncertainty output'
require_file $nonclosure_file 'non-closure output'
$nc_3b_setup
require_file $signal_evaluation_file 'signal-region evaluation input'
python3 $shared_uncertainty_script --YEAR $YEAR --runType test-only --TrainRegion $REGION --TestRegion $signal_region --Nfold 10 --Plot $signal_region_plot --CreateUncHist 1 --function create_unc_hists --MX $mx --MY $my $split_args $nc_3b_add_args
require_file $signal_uncertainty_file 'signal-region uncertainty output'
python3 $shared_uncertainty_script --YEAR $YEAR --runType test-only --TrainRegion $REGION --TestRegion $signal_region --function add_decorrelated_nc_uncertainty --MX $mx --MY $my $split_args
require_file $decorrelated_file 'decorrelated non-closure output'
hadd -f $combined_file $signal_uncertainty_file $decorrelated_file
require_file $combined_file 'combined uncertainty input'
python3 $shared_uncertainty_script --YEAR $YEAR --runType test-only --TrainRegion $REGION --TestRegion $signal_region --function apply_bkg_norm_scalefactor --MX $mx --MY $my $split_args
require_file $scaled_file 'scaled Combine input'
python3 $shared_converter_script --YEAR $YEAR --MX $mx --MY $my
require_file $final_file 'final background output'
mkdir -p $BkgOutputDir/${YEAR}
cp $final_file $BkgOutputDir/${YEAR}/
require_file $BkgOutputDir/${YEAR}/$final_file 'copied final background output'
bash $shared_cleanup_script $mass_output_dir $final_file"
    job_output_dirs["$name"]="$mass_output_dir"
done


master_submit="$output_job_dir/condor_submit_${REGION}.sh"
: > "$master_submit"


for name in "${!jobs[@]}"; do
    exe_file="$output_job_dir/execute_${name}.sh"
    sub_file="$output_job_dir/submit_${name}.sh"
    cat << EOF > "$exe_file"
#!/bin/bash
set -euo pipefail

require_file() {
    local path="\$1"
    local description="\$2"
    if [[ ! -f "\$path" ]]; then
        echo "[ERROR] Missing \$description: \$path" >&2
        exit 1
    fi
}

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd $CMSSW_dir
eval \`scramv1 runtime -sh\`

cd ${job_output_dirs[$name]}
${jobs[$name]}
EOF
    chmod +x "$exe_file"

    cat << EOF > "$sub_file"
universe   = vanilla
executable = $exe_file
getenv     = TRUE

log        = $output_job_dir/logs/job_${name}.log
output     = $output_job_dir/logs/job_${name}.out
error      = $output_job_dir/logs/job_${name}.err
notification = never
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
request_runtime = 10000
+MaxRuntime = 10000
queue
EOF
    echo "condor_submit $sub_file" >> "$master_submit"
    echo "Prepared: $name"
done

chmod +x "$master_submit"
echo "[SUCCESS] ${#jobs[@]} CreateCombineInput job(s) for ${REGION} prepared."
echo "          Submit with: bash $master_submit"
