import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import sys
sys.path.append("/data/dust/user/wanghaoy/XtoYH4b/work_scripts") 
import fold_functions_ptcut
from fold_functions_ptcut import build_binning_map, processing, fast_fill, get_lumi
import numpy as np
from tensorflow.keras.models import load_model
import ROOT
import array
import argparse
import os
import joblib

ROOT.gErrorIgnoreLevel = ROOT.kWarning

OUTPUT_FILENAME_suffix = "OnlyPhysical"

parser = argparse.ArgumentParser(description="")

parser.add_argument('--YEAR', default="2024", type=str, help="Which era?")
parser.add_argument('--isScaling', default=1, type=int, help = "Standard Scaling")
parser.add_argument('--isBalanceClass', default=1, type=int, help = "Balance class?")
parser.add_argument('--splitfraction', default=0.2, type=float, help = "Fraction of test data")
parser.add_argument('--Model', default="DNN", type=str, help = "Model for training")
parser.add_argument('--runType', default="train-test", choices=["test-only"], type=str, help = "Options: test-only.")
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

if args.runType == "train-only":
    print("Error: For k-fold, only test-only mode is available currently. Please check!")
    exit(1)

data_lumi = get_lumi(args.YEAR)

if args.isScaling == 1:
    Scaling = "Scaling"
else:
    Scaling = "NoScaling"

if args.isBalanceClass == 1:
    BalanceClass = "BalanceClass"
else:
    BalanceClass = "NoBalanceClass"

if args.runType == "test-only":
    feature_names, features, combined_tree, aux_data = processing(['/data/dust/user/wanghaoy/XtoYH4b/Tree_Data_Parking.root'], args=args)
    
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

    n_splits = 5 
    
    base_model_dir = f"../{args.YEAR}/{args.TrainRegion}/Models/Model_{args.Model}_{'Scaling' if args.isScaling else 'NoScaling'}_BalanceClass_Nov27/"
    print(f"Starting Grand Ensemble Prediction for {n_splits} splits x {n_folds} folds...")
    
    all_fold_scores = []
    all_fold_weights = []
    
    model_metadata = [] 

    for split in range(n_splits):
        for fold in range(1, n_folds + 1):
            fold_dir = os.path.join(base_model_dir, f"MODEL_{fold}_{split}")
            
            # Load Scalar
            if args.isScaling == 1:
                scaler_path = os.path.join(fold_dir, "scaler.pkl")
                if not os.path.exists(scaler_path):
                    print(f"  Error: Scaler missing for Split {split} Fold {fold} at {scaler_path}")
                    continue
                scaler = joblib.load(scaler_path)
                X_fold = scaler.transform(features_raw)
            else:
                X_fold = features_raw

            # Load Model
            model_path = os.path.join(fold_dir, "model.h5")
            if not os.path.exists(model_path):
                print(f"  Error: Model missing for Split {split} Fold {fold} at {model_path}")
                continue
                
            model = load_model(model_path)
            
            # score = model.predict(X_fold, verbose=0).ravel()
            score = model.predict(X_fold, batch_size=4096, verbose=0).ravel()
            all_fold_scores.append(score)

            epsilon = 1e-10
            fold_weights = score / (1.0 - score + epsilon)
            all_fold_weights.append(fold_weights)
            model_metadata.append( (split, fold) )

            del model
            
            print(f"  -> Split {split} Fold {fold} predicted.")
        import gc
        gc.collect()

    all_fold_scores = np.array(all_fold_scores)
    all_fold_weights = np.array(all_fold_weights)
    
    if len(all_fold_scores) == 0:
        print("Error: No models were loaded successfully.")
        exit(1)

    avg_score = np.mean(all_fold_scores, axis=0)
    avg_weights = np.mean(all_fold_weights, axis=0)

    w_individual_folds = all_fold_weights * current_weights
    w_grand_mean_prediction = avg_weights * current_weights

    mask_3T = current_signal == 1
    mask_2T = current_signal == 0 

    var_data_map["Score"] = avg_score

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

    # output_filename = "OnlyPhysical_Unrolled_50Models.root"
    output_filename = f"{args.TestRegion}_{OUTPUT_FILENAME_suffix}.root"
    f_out = ROOT.TFile(output_filename, "RECREATE")
    
    print(f"Generating histograms and saving to {output_filename}...")

    ROOT.TH1.SetDefaultSumw2(True)

    vars_to_save = ["MX", "MY", "Unrolled_MXMY"]

    for var, data in var_data_map.items():
        
        # For testing Combine, skip other variables here 
        # if var not in vars_to_save:
        #     continue
    
        if var in binning_map:
            nbins = len(binning_map[var]) - 1
            bins_array = array.array('d', binning_map[var])
            def create_hist(name, title):
                h = ROOT.TH1F(name, title, nbins, bins_array)
                return h

        else:
            nbins, xmin, xmax = 50, np.min(data), np.max(data)
            def create_hist(name, title):
                return ROOT.TH1F(name, title, nbins, xmin, xmax)
            
        d_3T = data[mask_3T]
        w_3T = current_weights[mask_3T]
        
        d_2T = data[mask_2T]
        w_2T = current_weights[mask_2T] 
        
        h_3T = create_hist(f"{var}_hist_4b_mean", f"{var} 4b")
        fast_fill(h_3T, d_3T, w_3T) 
        h_3T.Write()

        for i, (split_num, fold_num) in enumerate(model_metadata):
            
            h_fold = create_hist(f"{var}_hist_2b_split{split_num}_fold{fold_num}", f"{var} Split {split_num} Fold {fold_num} 2b")
            
            w_this_fold = w_individual_folds[i][mask_2T]
            fast_fill(h_fold, d_2T, w_this_fold) 
            h_fold.Write()

        h_2b_data = create_hist(f"{var}_hist_2b_mean", f"{var} 2b")
        fast_fill(h_2b_data, d_2T, w_2T) 
        h_2b_data.Write()
        
        h_pred_mean = create_hist(f"{var}_hist_2bw_mean", f"{var} 2b Prediction (Mean)")
        w_mean_pred = w_grand_mean_prediction[mask_2T]
        fast_fill(h_pred_mean, d_2T, w_mean_pred)
        h_pred_mean.Write()

    f_out.Close()
    print("All histograms saved successfully.")

