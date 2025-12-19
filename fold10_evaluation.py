import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import uproot
import numpy as np
import vector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import ROOT
import pandas as pd
import mplhep as hep
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

args = parser.parse_args()

isHcand_index_available = False

def build_binning_map(njets):
    bin_edges      = np.linspace(0, 1, 51)
    mx_bin_edges   = np.array([100,120,140,160,180,200,225,250,275,300,330,360,400,
                               450,500,550,600,650,700,750,800,850,900,950,1000,
                               1100,1200,1300,1400,1500,1600,1800,2000,2250,2500,
                               2750,3000,3500,4000])
    my_bin_edges   = np.linspace(0, 1000, 51)
    mh_bin_edges   = np.linspace(0, 300, 51)
    jet_mass_bin_edges = np.linspace(0, 100, 51)
    njets_add_bin_edges = np.array([0,1,2,3,4,5,6])
    eta_bin_edges  = np.linspace(-5, 5, 51)
    phi_bin_edges  = np.linspace(-3.2, 3.2, 51)
    HT_bin_edges   = np.linspace(0, 2000, 51)
    dr_bin_edges   = np.linspace(0, 6.3, 51)
    pt_bin_edges   = np.linspace(0, 1000, 51)

    bin_map = {
        "MX": mx_bin_edges,
        "MY": my_bin_edges,
        "MH": mh_bin_edges,
        "Score": bin_edges,
        "n_jets_add": njets_add_bin_edges,
        "HT_additional": HT_bin_edges,
        "HT_4j": HT_bin_edges,
        "dR1_plot": dr_bin_edges,
        "dR2_plot": dr_bin_edges,
    }

    # jet AK4 vars
    for i in range(1, njets + 1):
        bin_map[f"JetAK4_mass_{i}"] = jet_mass_bin_edges
        bin_map[f"JetAK4_pt_{i}"]   = pt_bin_edges
        bin_map[f"JetAK4_eta_{i}"]  = eta_bin_edges
        bin_map[f"JetAK4_phi_{i}"]  = phi_bin_edges

    # Higgs candidates
    for v in ["pt", "eta", "phi"]:
        edges = pt_bin_edges if v=="pt" else (eta_bin_edges if v=="eta" else phi_bin_edges)
        bin_map[f"Hcand_1_{v}"] = edges
        bin_map[f"Hcand_2_{v}"] = edges

    # H1/H2 deta/dphi/dR
    for h in ["H1", "H2"]:
        bin_map[f"{h}_b1b2_deta"] = eta_bin_edges
        bin_map[f"{h}_b1b2_dphi"] = phi_bin_edges
        bin_map[f"{h}_b1b2_dR"]   = dr_bin_edges

    # H1H2 system
    bin_map["H1H2_pt"]   = pt_bin_edges
    bin_map["H1H2_eta"]  = eta_bin_edges
    bin_map["H1H2_phi"]  = phi_bin_edges
    bin_map["H1H2_deta"] = eta_bin_edges
    bin_map["H1H2_dphi"] = phi_bin_edges
    bin_map["H1H2_dR"]   = dr_bin_edges

    return bin_map

def processing(file_list):
    """
    Process data from root files
    """

    njets = 4

    columns = ['JetAK4_btag_B_WP_1', 'JetAK4_btag_B_WP_2', 'JetAK4_btag_B_WP_3', 'JetAK4_btag_B_WP_4',
            'JetAK4_pt_1', 'JetAK4_pt_2', 'JetAK4_pt_3', 'JetAK4_pt_4', 
            'JetAK4_eta_1', 'JetAK4_eta_2', 'JetAK4_eta_3', 'JetAK4_eta_4', 
            'JetAK4_phi_1', 'JetAK4_phi_2', 'JetAK4_phi_3', 'JetAK4_phi_4', 
            'JetAK4_mass_1', 'JetAK4_mass_2', 'JetAK4_mass_3', 'JetAK4_mass_4',
            'JetAK4_add_pt', 'JetAK4_add_eta', 'JetAK4_add_phi', 'JetAK4_add_mass', 
            'Hcand_1_pt', 'Hcand_1_eta', 'Hcand_1_phi', 'Hcand_1_mass',
            'Hcand_2_pt', 'Hcand_2_eta', 'Hcand_2_phi', 'Hcand_2_mass',
            'H1_b1b2_deta', 'H1_b1b2_dphi', 'H1_b1b2_dR',
            'H2_b1b2_deta', 'H2_b1b2_dphi', 'H2_b1b2_dR',
            'H1H2_pt', 'H1H2_eta', 'H1H2_phi', #'H1H2_mass',
            'H1H2_deta', 'H1H2_dphi', 'H1H2_dR',
            'HT_4j',
            'njets_add', 'HT_add',
            'Hcand_mass', 'Ycand_mass']

    if args.runType == "train-only":
        tree_arr = uproot.concatenate(
            [f"{f}:Tree_JetInfo" for f in file_list], 
            expressions=columns, 
            library="np"
        )
        n_events = len(tree_arr["JetAK4_pt_1"])

    elif args.runType == "test-only":
        input_file = uproot.open(f"/data/dust/user/wanghaoy/XtoYH4b/Tree_Data_Parking.root")
        tree = input_file["Tree_JetInfo"]
        n_events = tree.num_entries
        tree_arr = tree.arrays(columns, library="np", entry_stop=n_events)

    else:
        print("For k-fold, train-test is not available yet.")
        exit(1)

    wp1 = tree_arr["JetAK4_btag_B_WP_1"]
    wp2 = tree_arr["JetAK4_btag_B_WP_2"]
    wp3 = tree_arr["JetAK4_btag_B_WP_3"]
    wp4 = tree_arr["JetAK4_btag_B_WP_4"]

    H_mass = tree_arr["Hcand_mass"]
    min_mask = (H_mass > 50) & (H_mass < 300)
    common_mask = ((H_mass < 90) | (H_mass > 150)) & min_mask
    bkg_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 < 2) & (wp4 < 2) & common_mask & min_mask

    if args.TrainRegion == "3b":
        sig_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 2) & (wp4 < 2) & common_mask
        if args.TestRegion == "4btest":
            print("Error: If training region is 3b, the test region can only be 3btest or 3bHiggsMW. Please check!")
            exit()

    elif args.TrainRegion == "4b":
        sig_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 3) & (wp4 >= 2) & common_mask 
        if args.TestRegion == "3btest" or args.TestRegion == "3bHiggsMW":
            print("Error: If training region is 4b, the test region can only be 4btest. Please check!")
            exit()

    if args.TestRegion == "3bHiggsMW":
        common_mask = ((H_mass >= 90) & (H_mass <= 150)) # Note this is different from other regions
        sig_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 2) & (wp4 < 2) & common_mask & min_mask
        bkg_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 < 2) & (wp4 < 2) & common_mask & min_mask
        closure = [r"3b_{Higgs\ MW}", "2b", "2b_w"]
    elif args.TestRegion == "3btest":
        sig_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 2) & (wp4 < 2) & common_mask 
        closure = ["3b", "2b", "2b_w"]
    elif args.TestRegion == "4btest":
        sig_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 3) & (wp4 >= 2) & common_mask 
        closure = ["4b", "2b", "2b_w"]
    elif args.TestRegion == None:
        pass

    sig_idx = np.where(sig_mask)[0]
    bkg_idx = np.where(bkg_mask)[0]


    if args.isBalanceClass == 1:
        n_min = min(len(sig_idx), len(bkg_idx))

        sig_idx_bal = sig_idx[:n_min]
        bkg_idx_bal = bkg_idx[:n_min]

        all_idx = np.concatenate([sig_idx_bal, bkg_idx_bal])
        signal_flag = np.concatenate([np.ones(n_min), np.zeros(n_min)])

        BalanceClass = "BalanceClass"
    else:
        all_idx = np.concatenate([sig_idx, bkg_idx])
        signal_flag = np.concatenate([np.ones(len(sig_idx)), np.zeros(len(bkg_idx))])

        BalanceClass = "NoBalanceClass"

    n_events = len(all_idx)

    combined_tree = {
        "signal": signal_flag.astype(np.int32),
        **{key: val[all_idx] for key, val in tree_arr.items()}
    }

    jets = vector.arr({
        "pt":   np.stack([combined_tree[f"JetAK4_pt_{i+1}"]   for i in range(njets)], axis=1),
        "eta":  np.stack([combined_tree[f"JetAK4_eta_{i+1}"]  for i in range(njets)], axis=1),
        "phi":  np.stack([combined_tree[f"JetAK4_phi_{i+1}"]  for i in range(njets)], axis=1),
        "mass": np.stack([combined_tree[f"JetAK4_mass_{i+1}"] for i in range(njets)], axis=1),
    })

    Hcand_index_cols = [f"JetAK4_Hcand_index_{i+1}" for i in range(njets)]

    dR1_arr = np.full(n_events, np.nan)
    dR2_arr = np.full(n_events, np.nan)

    if isHcand_index_available:

        Hcand_index = np.stack([combined_tree[col] for col in Hcand_index_cols], axis=1)

        for i in range(n_events):
            val = Hcand_index[i]
            pair1 = np.where(val == 1)[0]
            pair2 = np.where(val == 2)[0]

            if pair1.size == 2 and pair2.size == 2:
                dR1_arr[i] = jets[i, pair1[0]].deltaR(jets[i, pair1[1]])
                dR2_arr[i] = jets[i, pair2[0]].deltaR(jets[i, pair2[1]])

    else:

        jet_pairs = [[(1, 2), (3, 4)],
                    [(1, 3), (2, 4)],
                    [(1, 4), (2, 3)]]

        for i in range(n_events):
            best_idx   = None
            best_min   = np.inf
            best_dRs   = (np.nan, np.nan)

            for pair_set in jet_pairs:
                dR1 = jets[i, pair_set[0][0]-1].deltaR(jets[i, pair_set[0][1]-1])
                dR2 = jets[i, pair_set[1][0]-1].deltaR(jets[i, pair_set[1][1]-1])

                if min(dR1, dR2) < best_min:
                    best_min = min(dR1, dR2)
                    best_dRs = (dR1, dR2)

            dR1_arr[i], dR2_arr[i] = best_dRs
    if args.isScaling == 1:
        Scaling = "Scaling"
    else:
        Scaling = "NoScaling"

    if args.runType == "train-test" or args.runType == "test-only":
        MH = tree_arr["Hcand_mass"][all_idx]
        MY = tree_arr["Ycand_mass"][all_idx]
        n_jets_add = tree_arr["njets_add"][all_idx]
        HT_additional = tree_arr["HT_add"][all_idx]

    combined_tree["dR_1"] = dR1_arr
    combined_tree["dR_2"] = dR2_arr
    dR1_plot = dR1_arr.copy()
    dR2_plot = dR2_arr.copy()

    if args.isMC == 1:
        event_weights = combined_tree["Event_weight"]
        drop_cols = ["Event_weight"]
    else:
        event_weights = np.ones(int(n_events), dtype=float)
        drop_cols = []

    drop_cols += Hcand_index_cols 
    drop_cols += [f"JetAK4_btag_B_WP_{i+1}" for i in range(njets)]
    drop_cols += ["Hcand_mass", "Ycand_mass"]
    drop_cols += ["dR_1", "dR_2"]
    drop_cols += ["HT_add", "njets_add"]
    drop_cols += [f"JetAK4_add_{var}" for var in ["pt", "eta", "phi", "mass"]]
    drop_cols += ["Hcand_1_mass", "Hcand_2_mass"]

    for col in drop_cols:
        if col in combined_tree:
            del combined_tree[col]

    label_name = "signal"
    feature_names = [col for col in combined_tree.keys() if col != label_name]
    features = np.stack([combined_tree[col] for col in feature_names], axis=1)

    mx = (jets[:, 0] + jets[:, 1] + jets[:, 2] + jets[:, 3]).mass

    # 3. Pack everything into a dictionary
    if args.runType == "test-only":
        aux_data = {
            "jets": jets,             
            "MX": mx,                 
            "MH": MH,
            "MY": MY,
            "n_jets_add": n_jets_add,
            "HT_additional": HT_additional,
            "dR1_plot": dR1_plot,
            "dR2_plot": dR2_plot,
            "event_weights": event_weights,
            "BalanceClass": BalanceClass, 
            "closure": closure            
        }
    else:
        aux_data = {}

    return feature_names, features, combined_tree, aux_data

def fast_fill(hist, data, weights):
    """
    Helper function to speed up ROOT histogram filling using FillN.
    Converts numpy arrays to contiguous float64 for C++ compatibility.
    """
    if len(data) == 0: return
    
    d_arr = np.ascontiguousarray(data, dtype=np.float64)
    w_arr = np.ascontiguousarray(weights, dtype=np.float64)
    
    hist.FillN(len(d_arr), d_arr, w_arr)

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

    feature_names, features, combined_tree, aux_data = processing(['/data/dust/user/wanghaoy/XtoYH4b/Tree_Data_Parking.root'])

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

    n_folds = 10
    base_model_dir = f"../{args.YEAR}/{args.TrainRegion}/Models/Model_{args.Model}_{'Scaling' if args.isScaling else 'NoScaling'}_{aux_data['BalanceClass']}_Nov27/"
    
    print(f"Starting Ensemble Prediction for {n_folds} folds...")
    
    all_fold_scores = []
    all_fold_weights = []

    for fold in range(1, n_folds + 1):
        fold_dir = os.path.join(base_model_dir, f"MODEL_{fold}")
        
        if args.isScaling == 1:
            scaler_path = os.path.join(fold_dir, "scaler.pkl")
            if not os.path.exists(scaler_path):
                print(f"  Error: Scaler missing for fold {fold} at {scaler_path}")
                continue
            scaler = joblib.load(scaler_path)
            X_fold = scaler.transform(features_raw)
        else:
            X_fold = features_raw

        model_path = os.path.join(fold_dir, "model.h5")
        if not os.path.exists(model_path):
            print(f"  Error: Model missing for fold {fold} at {model_path}")
            continue
            
        model = load_model(model_path)
        
        score = model.predict(X_fold, verbose=0).ravel()
        all_fold_scores.append(score)

        epsilon = 1e-10
        fold_weights = score / (1.0 - score + epsilon)
        all_fold_weights.append(fold_weights)

        del model
        print(f"  -> Fold {fold} predicted.")

    all_fold_scores = np.array(all_fold_scores)
    all_fold_weights = np.array(all_fold_weights)
    
    if len(all_fold_scores) == 0:
        print("Error: No models were loaded successfully.")
        exit(1)

    avg_score = np.mean(all_fold_scores, axis=0)
    w_mean_raw = np.mean(all_fold_weights, axis=0)
    q16 = np.percentile(all_fold_weights, 16, axis=0)
    q84 = np.percentile(all_fold_weights, 84, axis=0)

    sys_sigma = (q84 - q16) / 2.0

    w_mean = w_mean_raw * current_weights
    w_up_raw   = w_mean_raw + sys_sigma
    w_down_raw = np.maximum(w_mean_raw - sys_sigma, 0.0)

    w_up   = w_up_raw   * current_weights
    w_down = w_down_raw * current_weights

    w_individual_folds = all_fold_weights * current_weights

    mask_3T = current_signal == 1
    mask_2T = current_signal == 0 

    var_data_map["Score"] = avg_score

    output_filename = "quantiles_Combined_Background_Result.root"
    f_out = ROOT.TFile(output_filename, "RECREATE")
    
    print(f"Generating histograms and saving to {output_filename}...")


    # Define Bin

    for var, data in var_data_map.items():
    
        if var in binning_map:
            nbins = len(binning_map[var]) - 1
            bins_array = array.array('d', binning_map[var])
            def create_hist(name, title):
                return ROOT.TH1F(name, title, nbins, bins_array)
        else:
            # Default binning
            nbins, xmin, xmax = 50, np.min(data), np.max(data)
            def create_hist(name, title):
                return ROOT.TH1F(name, title, nbins, xmin, xmax)
            
        d_3T = data[mask_3T]
        w_3T = current_weights[mask_3T]
        
        d_2T = data[mask_2T]
        w_nom = w_mean[mask_2T]
        w_sys_u = w_up[mask_2T]
        w_sys_d = w_down[mask_2T]
        w_2T = current_weights[mask_2T]

        h_3T = create_hist(f"{var}_hist_4b_mean", f"{var} 4b")
        h_3T.Sumw2()
        fast_fill(h_3T, d_3T, w_3T) 
        h_3T.Write()


        if var == "Score":
            for fold_idx in range(n_folds):
                fold_num = fold_idx + 1
                h_fold = create_hist(f"{var}_hist_2b_fold{fold_num}", f"{var} Fold {fold_num} 2b")
                h_fold.Sumw2()
                
                w_this_fold = w_individual_folds[fold_idx][mask_2T]
                fast_fill(h_fold, d_2T, w_this_fold) 
                h_fold.Write()

        h_nom = create_hist(f"{var}_hist_2b_w_mean", f"{var} 2b_w Mean")
        h_nom.Sumw2()
        fast_fill(h_nom, d_2T, w_nom) 
        h_nom.Write()
        h_up = create_hist(f"{var}_hist_2b_w_sys_up", f"{var} 2b_w Sys Up")
        h_up.Sumw2()
        fast_fill(h_up, d_2T, w_sys_u) 
        h_up.Write()

        h_dn = create_hist(f"{var}_hist_2b_w_sys_down", f"{var} 2b_w Sys Down")
        h_dn.Sumw2()
        fast_fill(h_dn, d_2T, w_sys_d) 
        h_dn.Write()

        h_2T = create_hist(f"{var}_hist_2b_mean", f"{var} 2b")
        h_2T.Sumw2()
        fast_fill(h_2T, d_2T, w_2T) 
        h_2T.Write()
        

    f_out.Close()
    print("All histograms saved successfully.")

