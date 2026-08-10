import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import sys
sys.path.append("/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN") 
import fold_functions_ptcut
from fold_functions_ptcut import build_binning_map, processing, fast_fill, get_lumi
import numpy as np
from tensorflow.keras.models import load_model
import ROOT
import array
import argparse
import os
import joblib
import uproot
import awkward as ak

ROOT.gErrorIgnoreLevel = ROOT.kWarning

OUTPUT_FILENAME_suffix = "OnlyPhysical"

parser = argparse.ArgumentParser(description="")

parser.add_argument('--YEAR', default="2024", type=str, help="Which era?")
parser.add_argument('--isScaling', default=1, type=int, help = "Standard Scaling")
parser.add_argument('--isBalanceClass', default=1, type=int, help = "Balance class?")
parser.add_argument('--splitfraction', default=0.2, type=float, help = "Fraction of test data")
parser.add_argument('--Model', default="DNN", type=str, help = "Model for training")
parser.add_argument('--runType', default="test-only", choices=["test-only"], type=str, help = "Options: test-only.")
parser.add_argument('--TrainRegion', default="3b", choices=["3b"], type=str, help = "This evaluator uses models trained in the 3b validation region.")
parser.add_argument('--TestRegion', default=None, choices=[None, "4btest", "3btest", "3bHiggsMW"], type=str, help = "Rregion to run the test? Select from: '4btest', '3btest', '3bHiggsMW' or None if train-only.")
parser.add_argument('--isMC', default=0, type=int, help = "MC or Data? Data by default.")
parser.add_argument('--SpecificModelTest', default=None, type=str, help = "Input specific model path for testing.")
parser.add_argument('--MX', required=True, type=int, help="Signal X mass point.")
parser.add_argument('--MY', required=True, type=int, help="Signal Y mass point.")

parser.add_argument('--Nfold', default=None, type=int, help = "Specify number of folds for training or testing.")

args = parser.parse_args()


def mass_point_to_index(file_list, mx, my):
    mappings = []
    for file_path in file_list:
        with uproot.open(file_path) as input_file:
            if "Tree_SignalGrid" not in input_file:
                raise RuntimeError(
                    f"Missing mass-point mapping: Tree_SignalGrid not found in {file_path}."
                )
            grid = input_file["Tree_SignalGrid"]
            missing = [name for name in ("mX_sig", "mY_sig") if name not in grid]
            if missing:
                raise RuntimeError(
                    f"Missing mass-point mapping branches in {file_path}: {', '.join(missing)}."
                )
            mx_entries = ak.to_list(grid["mX_sig"].array(library="ak"))
            my_entries = ak.to_list(grid["mY_sig"].array(library="ak"))
            if len(mx_entries) != len(my_entries) or not mx_entries:
                raise RuntimeError(f"Missing or malformed mass-point mapping in {file_path}.")
            for mx_values, my_values in zip(mx_entries, my_entries):
                if len(mx_values) != len(my_values):
                    raise RuntimeError(f"Ambiguous mass-point mapping in {file_path}: unequal MX/MY lengths.")
                mappings.append(tuple(zip(mx_values, my_values)))

    if not mappings or any(mapping != mappings[0] for mapping in mappings[1:]):
        raise RuntimeError("Missing or ambiguous mass-point mappings across input files/entries.")
    if len(mappings[0]) != 272:
        raise RuntimeError(
            f"Malformed mass-point mapping: expected 272 entries, found {len(mappings[0])}."
        )
    matches = [i for i, point in enumerate(mappings[0]) if point == (mx, my)]
    if len(matches) != 1:
        if not matches:
            raise ValueError(f"Unsupported mass point (MX, MY)=({mx}, {my}).")
        raise RuntimeError(f"Ambiguous mass-point mapping for (MX, MY)=({mx}, {my}).")
    return matches[0]


def load_pairing_inputs(file_list, source_indices, feature_names, mass_index):
    pair_branches = {
        name: f"{name}_pair" for name in feature_names
        if name in fold_functions_ptcut.PAIR_DEPENDENT_COLUMNS
    }
    required_branches = ["best_pair_sig", *pair_branches.values()]
    n_selected = len(source_indices)
    pair_indices = np.empty(n_selected, dtype=np.int8)
    pair_features = {
        name: np.empty((3, n_selected), dtype=np.float32) for name in pair_branches
    }

    file_offset = 0
    for file_path in file_list:
        with uproot.open(file_path) as input_file:
            tree = input_file["Tree_JetInfo"]
            missing = [name for name in required_branches if name not in tree]
            if missing:
                raise RuntimeError(f"Missing pairing branches in {file_path}: {', '.join(missing)}.")
            for start in range(0, tree.num_entries, 100000):
                stop = min(start + 100000, tree.num_entries)
                global_start, global_stop = file_offset + start, file_offset + stop
                positions = np.flatnonzero(
                    (source_indices >= global_start) & (source_indices < global_stop)
                )
                if not len(positions):
                    continue
                arrays = tree.arrays(
                    required_branches, entry_start=start, entry_stop=stop, library="ak"
                )
                local_indices = source_indices[positions] - global_start
                best_pairs = arrays["best_pair_sig"][local_indices]
                lengths = ak.to_numpy(ak.num(best_pairs, axis=1))
                if np.any(lengths < 272):
                    bad = np.flatnonzero(lengths < 272)[0]
                    raise ValueError(
                        f"best_pair_sig must contain at least 272 entries; selected event "
                        f"{positions[bad]} contains {lengths[bad]}."
                    )
                selected_pairs = ak.to_numpy(best_pairs[:, mass_index])
                invalid = ~np.isin(selected_pairs, (0, 1, 2))
                if np.any(invalid):
                    bad = np.flatnonzero(invalid)[0]
                    raise ValueError(
                        f"Invalid pairing value {selected_pairs[bad]} for selected event "
                        f"{positions[bad]}; expected 0, 1, or 2."
                    )
                pair_indices[positions] = selected_pairs

                for name, branch in pair_branches.items():
                    values = arrays[branch][local_indices]
                    lengths = ak.to_numpy(ak.num(values, axis=1))
                    if np.any(lengths < 3):
                        bad = np.flatnonzero(lengths < 3)[0]
                        raise ValueError(
                            f"Pairing branch '{branch}' has only {lengths[bad]} entries for "
                            f"selected event {positions[bad]}; expected at least 3."
                        )
                    pair_features[name][:, positions] = ak.to_numpy(values[:, :3]).T
            file_offset += tree.num_entries

    return pair_indices, pair_features

isHcand_index_available = False

n_folds = args.Nfold 
if args.Nfold is None:
    print("Please provide the number of folds using --Nfold argument!")
    exit(1)
if n_folds != 10:
    parser.error("The split 3b evaluation requires --Nfold 10.")

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
    if args.YEAR == "2024" or args.YEAR == "2025":
        filename_Tree = "Tree_Data_Parking.root"
    else:
        filename_Tree = "Tree_Data.root"

    if args.YEAR == "2022Full":
        input_files = [
            "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/2022/Tree_Data.root",
            "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/2022EE/Tree_Data.root",
        ]
    elif args.YEAR == "2023Full":
        input_files = [
            "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/2023/Tree_Data.root",
            "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/2023BPix/Tree_Data.root",
        ]
    else:
        fulldata_path = f"/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/{args.YEAR}/"
        input_files = [fulldata_path + filename_Tree]

    feature_names, features, combined_tree, aux_data = processing(input_files, args=args)
    
    features_raw = features.copy()
    mass_index = mass_point_to_index(input_files, args.MX, args.MY)
    pair_indices, pair_features = load_pairing_inputs(
        input_files, aux_data["source_indices"], feature_names, mass_index
    )
    features_by_pair = [features_raw.copy() for _ in range(3)]
    for feature_index, feature_name in enumerate(feature_names):
        if feature_name in pair_features:
            for pair_index in range(3):
                features_by_pair[pair_index][:, feature_index] = pair_features[feature_name][pair_index]

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
    
    base_model_dir = f"../{args.YEAR}/{args.TrainRegion}/Models/Model_{args.Model}_{'Scaling' if args.isScaling else 'NoScaling'}_BalanceClass/"
    print(f"Starting Grand Ensemble Prediction for {n_splits} splits x {n_folds} folds...")
    
    all_fold_scores = []
    all_fold_weights = []
    
    model_metadata = [] 

    for split in range(n_splits):
        for fold in range(1, n_folds + 1):
            models = {split: {fold: []}}
            X_folds = []
            for pair_index in range(3):
                fold_dir = os.path.join(
                    base_model_dir, f"MODEL_{fold}_{split}_pair{pair_index}"
                )
                model_path = os.path.join(fold_dir, "model.h5")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"Missing split-fold-pair model for split {split}, fold {fold}, "
                        f"pair {pair_index}: {model_path}"
                    )
                if args.isScaling == 1:
                    scaler_path = os.path.join(fold_dir, "scaler.pkl")
                    if not os.path.exists(scaler_path):
                        raise FileNotFoundError(
                            f"Missing scaler for split {split}, fold {fold}, pair "
                            f"{pair_index}: {scaler_path}"
                        )
                    X_folds.append(
                        joblib.load(scaler_path).transform(features_by_pair[pair_index])
                    )
                else:
                    X_folds.append(features_by_pair[pair_index])
                models[split][fold].append(load_model(model_path))

            pair_scores = np.array([
                models[split][fold][pair_index].predict(
                    X_folds[pair_index], batch_size=4096, verbose=0
                ).ravel()
                for pair_index in range(3)
            ])
            score = pair_scores[pair_indices, np.arange(len(pair_indices))]
            all_fold_scores.append(score)

            epsilon = 1e-10
            fold_weights = score / (1.0 - score + epsilon)
            all_fold_weights.append(fold_weights)
            model_metadata.append( (split, fold) )

            del models
            
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
    output_filename = (
        f"{args.TestRegion}_{OUTPUT_FILENAME_suffix}_MX-{args.MX}_MY-{args.MY}.root"
    )
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
