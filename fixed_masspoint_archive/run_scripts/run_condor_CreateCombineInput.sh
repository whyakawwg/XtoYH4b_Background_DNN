#!/bin/bash
set -euo pipefail

YEAR=""
REGION="" # '4b' or '3b'

usage() {
    echo "Usage: bash run_condor_CreateCombineInput.sh -y <year> -r <4b|3b>"
    echo "Options:"
    echo "  -y, --YEAR  <year>    [REQUIRED] Data taking year (e.g., 2024, 2025)"
    echo "  -r, --region <reg>    [REQUIRED] Analysis region: '4b' or '3b'" 
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -y|--YEAR) YEAR="$2"; shift ;;
        -r|--region) REGION="$2"; shift ;;
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

echo "[INFO] Preparing Combine input for YEAR=${YEAR} and REGION=${REGION}"


special_name="Background_${YEAR}_Fixed"
CMSSW_dir="/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/"
input_dir="/data/dust/user/wanghaoy/XtoYH4b/${special_name}"
output_dir="${input_dir}/CombineInput_Run2_${REGION}"
script_dir="/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN/fixed_masspoint_archive/"
output_job_dir="${output_dir}/jobs_${REGION}"
name="CombineInput_${REGION}"

BkgOutputDir="/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/BackgroundEstimation"

mkdir -p "$output_dir"
mkdir -p "$output_job_dir"
mkdir -p "$output_job_dir/logs"
cp "$script_dir/uncertainty_pipeline.py" "$output_dir"
cp "$script_dir/convert_to_combine_input_DecoMX.py" "$output_dir"


if [[ "$REGION" == "3b" ]]; then
    COMMAND_BLOCK="require_file ../3btest_evaluation/3btest_OnlyPhysical.root '3b test-region evaluation input'
python3 uncertainty_pipeline.py --YEAR $YEAR --runType test-only --TrainRegion 3b --TestRegion 3btest --Nfold 10 --Plot 1 --CreateUncHist 1 --function create_unc_hists
require_file 3btest_Uncertainty_hists_OnlyPhysical.root '3b test-region uncertainty output'
require_file nonclosure_factors.root 'non-closure factors'
require_file ../3bHiggsMW_evaluation/3bHiggsMW_OnlyPhysical.root '3b Higgs-window evaluation input'
python3 uncertainty_pipeline.py --YEAR $YEAR --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --Nfold 10 --Plot 1 --CreateUncHist 1 --function create_unc_hists
require_file 3bHiggsMW_Uncertainty_hists_OnlyPhysical.root '3b Higgs-window uncertainty output'
python3 uncertainty_pipeline.py --YEAR $YEAR --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --function add_decorrelated_nc_uncertainty
require_file combine_addDecorrelatedNC.root 'decorrelated non-closure output'
hadd -f combine_noempty_input.root 3bHiggsMW_Uncertainty_hists_OnlyPhysical.root combine_addDecorrelatedNC.root
require_file combine_noempty_input.root 'combined uncertainty input'
python3 uncertainty_pipeline.py --YEAR $YEAR --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --function apply_bkg_norm_scalefactor
require_file combine_noempty_input_Scaled_${YEAR}.root 'scaled Combine input'"
else
    COMMAND_BLOCK="require_file ../4btest_evaluation/4btest_OnlyPhysical.root '4b test-region evaluation input'
python3 uncertainty_pipeline.py --YEAR $YEAR --runType test-only --TrainRegion 4b --TestRegion 4btest --Nfold 10 --Plot 1 --CreateUncHist 1 --function create_unc_hists
require_file 4btest_Uncertainty_hists_OnlyPhysical.root '4b test-region uncertainty output'
require_file nonclosure_factors.root 'non-closure factors'
require_file ../4bHiggsMW_evaluation/4bHiggsMW_OnlyPhysical.root '4b Higgs-window evaluation input'
python3 uncertainty_pipeline.py --YEAR $YEAR --runType test-only --TrainRegion 4b --TestRegion 4bHiggsMW --Nfold 10 --Plot 0 --CreateUncHist 1 --function create_unc_hists
require_file 4bHiggsMW_Uncertainty_hists_OnlyPhysical.root '4b Higgs-window uncertainty output'
python3 uncertainty_pipeline.py --YEAR $YEAR --runType test-only --TrainRegion 4b --TestRegion 4bHiggsMW --function add_decorrelated_nc_uncertainty
require_file combine_addDecorrelatedNC.root 'decorrelated non-closure output'
hadd -f combine_noempty_input.root 4bHiggsMW_Uncertainty_hists_OnlyPhysical.root combine_addDecorrelatedNC.root
require_file combine_noempty_input.root 'combined uncertainty input'
python3 uncertainty_pipeline.py --YEAR $YEAR --runType test-only --TrainRegion 4b --TestRegion 4bHiggsMW --function apply_bkg_norm_scalefactor
require_file combine_noempty_input_Scaled_${YEAR}.root 'scaled Combine input'"
fi


master_submit="$output_job_dir/condor_submit_${REGION}.sh"
: > "$master_submit"


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

cd $output_dir
$COMMAND_BLOCK
python3 convert_to_combine_input_DecoMX.py --YEAR $YEAR
require_file Output_Background_${YEAR}.root 'final background output'

if [ ! -d "$BkgOutputDir" ]; then
    echo "[ERROR] Background directory '$BkgOutputDir' does not exist!"
    exit 1
else
    mkdir -p $BkgOutputDir/${YEAR}/
    cp Output_Background_${YEAR}.root $BkgOutputDir/${YEAR}/
    echo "[INFO] Output file saved to $BkgOutputDir/${YEAR}/"

fi


EOF
    chmod +x "$exe_file"

    cat << EOF > "$sub_file"
universe   = vanilla
executable = $exe_file
getenv     = TRUE
request_memory = 24 GB
request_cpus = 4
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


chmod +x "$master_submit"
echo "[SUCCESS] CreateCombineInput job for ${REGION} prepared."
echo "          Submit with: bash $master_submit"
echo "          Or simply just run: bash $exe_file"
