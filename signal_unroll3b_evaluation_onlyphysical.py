import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import sys
sys.path.append("/data/dust/user/wanghaoy/XtoYH4b/work_scripts") 
# import fold_functions
# from fold_functions import build_binning_map, processing, fast_fill
import fold_functions_ptcut
from fold_functions_ptcut import build_binning_map, processing, fast_fill

import numpy as np
from tensorflow.keras.models import load_model
import ROOT
import array
import argparse
import os
import joblib


ROOT.gErrorIgnoreLevel = ROOT.kWarning

parser = argparse.ArgumentParser(description="")

parser.add_argument('--YEAR', default="2024", type=str, help="Which era?")
parser.add_argument('--isScaling', default=1, type=int, help = "Standard Scaling")
parser.add_argument('--isBalanceClass', default=1, type=int, help = "Balance class?")
parser.add_argument('--splitfraction', default=0.2, type=float, help = "Fraction of test data")
parser.add_argument('--Model', default="DNN", type=str, help = "Model for training")
parser.add_argument('--runType', default="train-test", choices=["train-test", "train-only", "test-only"], type=str, help = "By default, train-test. Other options: train-only, test-only.")
parser.add_argument('--TrainRegion', default="4b", choices=["4b", "3b"], type=str, help = "Region of training data? Select from: '4b', '3b'. Even test-only, need to specify train region for the model.")
parser.add_argument('--TestRegion', default=None, choices=[None, "4btest", "3btest", "3bHiggsMW"], type=str, help = "Rregion to run the test? Select from: '4btest', '3btest', '3bHiggsMW' or None if train-only.")
parser.add_argument('--isMC', default=0, type=int, help = "MC or Data? Data by default.")
parser.add_argument('--SpecificModelTest', default=None, type=str, help = "Input specific model path for testing.")


parser.add_argument('--Nfold', default=None, type=int, help = "Specify number of folds for training or testing.")

args = parser.parse_args()

isHcand_index_available = False

n_folds = args.Nfold 
if args.Nfold is None:
    print("Please provide the number of folds using --Nfold argument!")
    exit(1)

binning_map = build_binning_map(njets=4)

if args.runType == "train-test" or args.runType == "train-only":
    print("Error: For k-fold, only test-only mode is available currently. Please check!")
    exit(1)

if args.YEAR == "2022":
    data_lumi = 7.98
elif args.YEAR == "2022EE":
    data_lumi = 26.67
elif args.YEAR == "2023":
    data_lumi = 11.24
elif args.YEAR == "2023BPiX":
    data_lumi = 9.45
elif args.YEAR == "2024":
    data_lumi = 108.96
else:
    print("Please select a valid YEAR: '2022', '2022EE', '2023', '2023BPiX', '2024'")
    exit(1)

if args.isScaling == 1:
    Scaling = "Scaling"
else:
    Scaling = "NoScaling"

if args.isBalanceClass == 1:
    BalanceClass = "BalanceClass"
else:
    BalanceClass = "NoBalanceClass"

if args.runType == "test-only":

    # feature_names, features, combined_tree, aux_data = processing(['/data/dust/user/wanghaoy/XtoYH4b/Tree_Data_Parking.root'], args=args)
    feature_names, features, combined_tree, aux_data = processing(['/data/dust/user/wanghaoy/XtoYH4b/Tree_NMSSM-XtoYHto4B_Par-MX-1000-MY-150_TuneCP5_13p6TeV_madgraph-pythia8.root'], args=args)

    features_raw = features.copy()

    BalanceClass  = aux_data["BalanceClass"]
    closure       = aux_data["closure"]
    event_weights = aux_data["event_weights"]
    MH            = aux_data["MH"]
    MY            = aux_data["MY"]
    n_jets_add    = aux_data["n_jets_add"]
    HT_additional = aux_data["HT_additional"]
    dR1_plot      = aux_data["dR1_plot"]
    dR2_plot      = aux_data["dR2_plot"]
    mx            = aux_data["MX"] 

    current_signal = combined_tree["signal"].astype(np.int32)
    if args.isMC == 1 and "Event_weight" in combined_tree:
        current_weights = combined_tree["Event_weight"]
    else:
        current_weights = np.ones(len(features_raw), dtype=float)

    var_data_map = {}
    
    for key in ["MX", "MY", "MH", "n_jets_add", "HT_additional", "dR1_plot", "dR2_plot", 
                "JetAK4_pt_1", "JetAK4_pt_2", "JetAK4_pt_3", "JetAK4_pt_4", "HT_4j"]: 
        if key in aux_data:
            var_data_map[key] = aux_data[key]
        elif key in combined_tree:
            var_data_map[key] = combined_tree[key]

    for i, name in enumerate(feature_names):
        var_data_map[name] = features_raw[:, i]



    mask_3T = current_signal == 1
    mask_2T = current_signal == 0 

    print(f"DEBUG: Total events: {len(current_signal)}")
    print(f"DEBUG: Signal events (mask_3T): {np.sum(mask_3T)}")
    print(f"DEBUG: Background events (mask_2T): {np.sum(mask_2T)}")
    print(f"DEBUG: Features shape: {features_raw.shape}")
    print(f"DEBUG: Unique signal values: {np.unique(current_signal)}")

    

    if "MX" in binning_map and "MY" in binning_map:
        print("Calculating Kinematically Valid Unrolled Index...")

        mx_bins_edge = np.array(binning_map["MX"])
        my_bins_edge = np.array(binning_map["MY"])
        
        n_my_bins = len(my_bins_edge) - 1
        n_mx_bins = len(mx_bins_edge) - 1
        
        # Mapping and the list of valid 2D bins (in bin number)
        valid_2d_to_1d = {}
        unrolled_bin_labels = []
        bin_idx_1d = 0
        
        for my_idx in range(n_my_bins):
            for mx_idx in range(n_mx_bins):
                mx_upper_edge = mx_bins_edge[mx_idx + 1]
                my_lower_edge = my_bins_edge[my_idx]
                # Only Physical bins where MX > MY + 125 
                if mx_upper_edge > (my_lower_edge + 125):
                    valid_2d_to_1d[(my_idx, mx_idx)] = bin_idx_1d
                    
                    # Create a label using the lower edges
                    label = f"MX{int(mx_bins_edge[mx_idx])}_MY{int(my_bins_edge[my_idx])}"
                    unrolled_bin_labels.append(label)
                    
                    bin_idx_1d += 1
                    
        n_valid_bins = bin_idx_1d
        print(f"  -> Total 2D grid bins: {n_mx_bins * n_my_bins}")
        print(f"  -> Valid kinematic bins after cleaning: {n_valid_bins}")

        mx_data = var_data_map["MX"]
        my_data = var_data_map["MY"]
        
        my_indices = np.digitize(my_data, my_bins_edge) - 1
        mx_indices = np.digitize(mx_data, mx_bins_edge) - 1
        
        # Create lookup table
        lookup_table = np.full((n_my_bins, n_mx_bins), -100.0)
        for (my_idx, mx_idx), new_1d_idx in valid_2d_to_1d.items():
            lookup_table[my_idx, mx_idx] = new_1d_idx + 0.5
            
        # Map events instantly
        valid_data_mask = (my_indices >= 0) & (my_indices < n_my_bins) & (mx_indices >= 0) & (mx_indices < n_mx_bins)
        
        unrolled_index = np.full_like(mx_data, -100.0)
        unrolled_index[valid_data_mask] = lookup_table[
            my_indices[valid_data_mask], 
            mx_indices[valid_data_mask]
        ]
        
        var_data_map["Unrolled_MXMY"] = unrolled_index
        binning_map["Unrolled_MXMY"] = list(range(n_valid_bins + 1))

    output_filename = "OnlyPhysical_Signal.root"
    f_out = ROOT.TFile(output_filename, "RECREATE")
    if "Unrolled_MXMY" in binning_map:
        n_valid_bins = len(binning_map["Unrolled_MXMY"]) - 1
        h_labels = ROOT.TH1F("Unrolled_Labels_Metadata_signal", "Metadata only", n_valid_bins, 0, n_valid_bins)
        
        # Attach the labels to this dummy histogram only
        for i, label in enumerate(unrolled_bin_labels):
            h_labels.GetXaxis().SetBinLabel(i + 1, label)
            
        h_labels.Write()
    
    print(f"Generating histograms and saving to {output_filename}...")

    

    ROOT.TH1.SetDefaultSumw2(True)
    for var, data in var_data_map.items():

        if var in binning_map:
            nbins = len(binning_map[var]) - 1
            bins_array = array.array('d', binning_map[var])
            def create_hist(name, title):
                h = ROOT.TH1F(name, title, nbins, bins_array)
                # # Automatically apply labels for the unrolled variable
                # if var == "Unrolled_MXMY":
                #     for i, label in enumerate(unrolled_bin_labels):
                #         # ROOT bins are 1-indexed, so add 1 to 'i'
                #         h.GetXaxis().SetBinLabel(i + 1, label)
                return h
        else:
            nbins, xmin, xmax = 50, np.min(data), np.max(data)
            def create_hist(name, title):
                return ROOT.TH1F(name, title, nbins, xmin, xmax)
            
        d_3T = data[mask_3T]
        w_3T = current_weights[mask_3T]
        
        h_3T = create_hist(f"{var}_hist_signal", f"{var} 3b signal")
        fast_fill(h_3T, d_3T, w_3T) 
        h_3T.Write()

    f_out.Close()
    print("All histograms saved successfully.")


# Command: python3 signal_unroll3b_evaluation.py --YEAR 2024 --isScaling 1 --isBalanceClass 0 --Model DNN --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --Nfold 10
