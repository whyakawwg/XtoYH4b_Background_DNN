import ROOT
import argparse
import sys

sys.path.insert(0, "/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN")
from fold_functions_ptcut import build_binning_map

parser = argparse.ArgumentParser(description="Convert background histograms to Combine input format")
parser.add_argument("--YEAR", type=str, required=True, help="Data-taking year (e.g., 2024, 2025)")

args = parser.parse_args()
YEAR = args.YEAR

SIGNAL_STEM = (
    f"NMSSM-XtoYHto4B_Par-MX-1000-MY-125_"
    "TuneCP5_13p6TeV_madgraph-pythia8"
)

# According to /afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/CombineHarvester/CombineTools/bin/CreateCards_XYHto4btest.C
# SIGNAL_NAME = "NMSSM_XtoYHto4B_MX-1000_MY-150_TuneCP5_13p6TeV_madgraph-pythia8"
# INPUT_FILE = "combine_inclusive_bkg.root"
INPUT_FILE = f"combine_noempty_input_Scaled_{YEAR}.root"
# OUTPUT_FILE = f"combine_input_XYH4b_{YEAR}_VR.root"
OUTPUT_FILE = f"Output_Background_{YEAR}.root"

HIGGS_WINDOW_LOW = 90.0
HIGGS_WINDOW_HIGH = 150.0
ANALYSIS_BINNING = build_binning_map(njets=4)
MX_BIN_EDGES = ANALYSIS_BINNING["MX"]
MY_BIN_EDGES = ANALYSIS_BINNING["MY"]
N_MX_BINS = len(MX_BIN_EDGES) - 1
TARGET_MY_BINS = [
    index
    for index, (low, high) in enumerate(
        zip(MY_BIN_EDGES[:-1], MY_BIN_EDGES[1:]), start=1
    )
    if low < HIGGS_WINDOW_HIGH and high > HIGGS_WINDOW_LOW
]

MAPPING = {
    "MX":          "h_MaxScore_MX_Comb_3_3_3_2_Inclusive_mHcut", # 3322 for 3b, 3332 for 4b
    "MY":          "h_MaxScore_MY_Comb_3_3_3_2_Inclusive_mHcut",
    "Unrolled_MXMY": "h_MaxScore_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut"
    # "Unrolled_Kinematic_Index": "h_MX_MY_Comb_3_3_2_2_Inclusive_mHcut"
}

def convert():
    old_file = ROOT.TFile.Open(INPUT_FILE, "READ")
    new_file = ROOT.TFile.Open(OUTPUT_FILE, "RECREATE")

    for prefix, dir_name in MAPPING.items():
        print(f">> Creating directory: {dir_name}")
        
        new_dir = new_file.mkdir(dir_name)
        new_dir.cd()

        h_obs = old_file.Get(f"{prefix}_data_4b")
        if h_obs:
            h_obs.SetName("data_obs")
            h_obs.Write()


        h_bkg = old_file.Get(f"{prefix}_2b_w_nominal")
        if h_bkg:
            h_bkg.SetName("Inclusive_Bkg")
            h_bkg.Write()
            

            sys_map = [
                ("_2b_w_sys_up",   f"_Systematic_Uncertainty_{args.YEAR}Up"),
                ("_2b_w_sys_down", f"_Systematic_Uncertainty_{args.YEAR}Down"),
                ("_2b_w_stat_up",  f"_Statistical_Uncertainty_{args.YEAR}Up"),
                ("_2b_w_stat_down", f"_Statistical_Uncertainty_{args.YEAR}Down"),
                ("_2b_w_nc_3b_up", f"_NonClosure3b_Uncertainty_{args.YEAR}Up"),
                ("_2b_w_nc_3b_down", f"_NonClosure3b_Uncertainty_{args.YEAR}Down"),
            ]
            if prefix == "MY":
                for my_bin in TARGET_MY_BINS:
                    sys_map.append((
                        f"_2b_w_mybin_{my_bin}_up",
                        f"_MY125Bin{my_bin}_Uncertainty_{args.YEAR}Up",
                    ))
                    sys_map.append((
                        f"_2b_w_mybin_{my_bin}_down",
                        f"_MY125Bin{my_bin}_Uncertainty_{args.YEAR}Down",
                    ))
                sys_map.append(("_2b_w_modified_nc_up", f"_NonClosure_Uncertainty_{args.YEAR}Up"))
                sys_map.append(("_2b_w_modified_nc_down", f"_NonClosure_Uncertainty_{args.YEAR}Down"))

            if prefix == "MX":
                for i in range(1, N_MX_BINS + 1):
                    sys_map.append((f"_2b_w_nc_bin{i}_up", f"_NonClosure_Bin{i}_Uncertainty_{args.YEAR}Up"))
                    sys_map.append((f"_2b_w_nc_bin{i}_down", f"_NonClosure_Bin{i}_Uncertainty_{args.YEAR}Down"))
            
            if prefix == "Unrolled_MXMY":
                for j in range(1, N_MX_BINS + 1):
                    sys_map.append((f"_2b_w_nc_bin{j}_up", f"_NonClosure_MXBin{j}_Uncertainty_{args.YEAR}Up"))
                    sys_map.append((f"_2b_w_nc_bin{j}_down", f"_NonClosure_MXBin{j}_Uncertainty_{args.YEAR}Down"))
                    for i in TARGET_MY_BINS:
                        sys_map.append((f"_2b_w_mybin_{i}_mx{j}_up", f"_MY125Bin{i}_MXBin{j}_Uncertainty_{args.YEAR}Up"))
                        sys_map.append((f"_2b_w_mybin_{i}_mx{j}_down", f"_MY125Bin{i}_MXBin{j}_Uncertainty_{args.YEAR}Down"))
            
            for old_sfx, new_sfx in sys_map:
                h_sys = old_file.Get(f"{prefix}{old_sfx}")
                if h_sys:
                    h_sys.SetName(f"Inclusive_Bkg{new_sfx}")
                    h_sys.Write()

        # h_sig = old_file.Get(f"{prefix}_hist_signal")
        # if h_sig:
        #     h_sig.SetName(SIGNAL_NAME)
        #     h_sig.Write()

    new_file.Close()
    old_file.Close()
    print(f"\nSuccess! File saved as {OUTPUT_FILE}")


convert()
