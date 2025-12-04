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

def plot_hist(scores, mask, label, color, linestyle="solid"):
    """Utility to draw one DNN histogram."""
    scores = np.asarray(scores).ravel()
    mask = np.asarray(mask).ravel() 
    plt.hist(
        scores[mask],
        bins=50,
        range=(0, 1),
        histtype='step',
        linewidth=1.5,
        label=label,
        color=color,
        linestyle=linestyle,
        density=True
    )

def plotting(arr_3T, arr_2T, add_bins_=True, bins_=None, var="MX", suffix="reweight", ratio_ylim=(0, 2), label_=["3T", "2T", "2T_w"]):

    if add_bins_:
        if bins_ is None:
            raise ValueError("bins_ must be provided when add_bins_ is True")
        bins = bins_
    else:
        bins = np.linspace(np.min(arr_3T[var]), np.max(arr_3T[var]), 51)

    xlim = [bins[0], bins[-1]]

    ROOT.gStyle.SetOptStat(0)

    n_bins = len(bins) - 1
    bins_array = array.array('d', bins)
    hist_3T = ROOT.TH1F("hist_3T", "", n_bins, bins_array)
    hist_2T = ROOT.TH1F("hist_2T", "", n_bins, bins_array)
    hist_2T_w = ROOT.TH1F("hist_2T_w", "", n_bins, bins_array)

    for i in range(len(arr_3T[var])):
        hist_3T.Fill(arr_3T[var][i], arr_3T["Event_weights"][i])
    for i in range(len(arr_2T[var])):
        hist_2T.Fill(arr_2T[var][i], arr_2T["Event_weights"][i])
        hist_2T_w.Fill(arr_2T[var][i], arr_2T["Combined_weights"][i])

    # normalize
    hist_3T.Scale(1.0 / hist_3T.Integral())
    hist_2T.Scale(1.0 / hist_2T.Integral())
    hist_2T_w.Scale(1.0 / hist_2T_w.Integral())

    # get bin contents and errors
    y_3T = np.array([hist_3T.GetBinContent(i+1) for i in range(n_bins)])
    y_2T = np.array([hist_2T.GetBinContent(i+1) for i in range(n_bins)])
    y_2T_w = np.array([hist_2T_w.GetBinContent(i+1) for i in range(n_bins)])

    err_3T = np.array([hist_3T.GetBinError(i+1) for i in range(n_bins)])
    err_2T = np.array([hist_2T.GetBinError(i+1) for i in range(n_bins)])
    err_2T_w = np.array([hist_2T_w.GetBinError(i+1) for i in range(n_bins)])

    # ratio and uncertainty
    ratio_3T_2T = y_3T / np.where(y_2T > 0, y_2T, 1e-8)
    ratio_3T_2T_w = y_3T / np.where(y_2T_w > 0, y_2T_w, 1e-8)

    err_ratio_3T_2T = ratio_3T_2T * np.sqrt((err_3T / np.where(y_3T != 0, y_3T, 1e-8))**2 +
                                           (err_2T / np.where(y_2T != 0, y_2T, 1e-8))**2)
    err_ratio_3T_2T_w = ratio_3T_2T_w * np.sqrt((err_3T / np.where(y_3T != 0, y_3T, 1e-8))**2 +
                                               (err_2T_w / np.where(y_2T_w != 0, y_2T_w, 1e-8))**2)

    chi2_3T_2T = hist_3T.Chi2Test(hist_2T, "WW P CHI2/NDF")
    chi2_3T_2T_w = hist_3T.Chi2Test(hist_2T_w, "WW P CHI2/NDF")

    # bin centers
    edges = np.array(bins)
    x = 0.5 * (edges[1:] + edges[:-1])

    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)
    hep.cms.label("Preliminary", data=True, lumi=data_lumi, com=13.6, year=2024, ax=ax)

    # plotting
    hep.histplot([y_3T, y_2T, y_2T_w],
                 bins=edges,
                 label=[label_[0], label_[1], label_[2]],
                 color=["orange", "red", "blue"],
                 ax=ax,
                 histtype="step")

    if var in ["MY", "MX", "MH", "JetAK4_pt_1", "JetAK4_pt_2", "JetAK4_pt_3", "JetAK4_pt_4", "HT_additional", "Hcand_1_pt", "Hcand_2_pt", "Hcand_1_mass", "Hcand_2_mass", "HT_4j"]:
        ax.set_yscale("log")
    ax.set_ylabel("Arbitrary units")
    ax.set_xlim(*xlim)
    ax.legend()

    rax.errorbar(x, ratio_3T_2T, yerr=err_ratio_3T_2T, fmt='o', color='red',
                 label=rf"${label_[0]}/{label_[1]}\ \frac{{\chi^2}}{{NDF}}={chi2_3T_2T:.2f}$")
    rax.errorbar(x, ratio_3T_2T_w, yerr=err_ratio_3T_2T_w, fmt='o', color='blue',
                 label=rf"${label_[0]}/{label_[2]}\ \frac{{\chi^2}}{{NDF}}={chi2_3T_2T_w:.2f}$")

    rax.axhline(1.0, color='black', linestyle='--')
    rax.set_ylim(*ratio_ylim)
    rax.set_ylabel("Ratio")
    rax.set_xlabel(var)

    handles_ax, labels_ax = ax.get_legend_handles_labels()
    handles_rax, labels_rax = rax.get_legend_handles_labels()
    ax.legend(handles_ax + handles_rax, labels_ax + labels_rax)

    # outputs
    out_dir = f"{args.YEAR}/{args.TestRegion}/DNN_reweighting_plots/DNN_reweighting_plots_{Scaling}_{BalanceClass}_Nov27"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{var}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/{var}_{suffix}.pdf", bbox_inches="tight")
    plt.close()

    hist_3T.Delete()
    hist_2T.Delete()
    hist_2T_w.Delete()


if args.runType == "train-test":
    if args.TestRegion is None:
        print("Error: For train-test mode, both TrainRegion and TestRegion must be specified. Please check!")
        exit(1)

if args.runType == "train-only":
    if args.TestRegion is not None:
        args.TestRegion = None
        print("Warning: For train-only mode, TestRegion will be ignored.")

if args.runType == "test-only":
    if args.TestRegion is None:
        print("Error: For test-only mode, TestRegion must be specified. Please check!")
        exit(1)
if args.SpecificModelTest is not None:
    if args.runType != "test-only":
        print("Warning: SpecificModelTest is only used in test-only mode. Ignoring it for other modes.")
        args.SpecificModelTest = None
    elif args.TestRegion is None:
        print("Error: For test-only mode with SpecificModelTest, TestRegion must be specified. Please check!") 
        exit(1)
# input_file = uproot.open(f"/data/dust/user/chatterj/XToYHTo4b/SmallNtuples/Histograms/{args.YEAR}/Tree_Data_Parking.root")
input_file = uproot.open(f"/data/dust/user/wanghaoy/XtoYH4b/Tree_Data_Parking.root")

tree = input_file["Tree_JetInfo"]
n_events = tree.num_entries

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

tree_arr = tree.arrays(columns, library="np", entry_stop=n_events)

wp1 = tree_arr["JetAK4_btag_B_WP_1"]
wp2 = tree_arr["JetAK4_btag_B_WP_2"]
wp3 = tree_arr["JetAK4_btag_B_WP_3"]
wp4 = tree_arr["JetAK4_btag_B_WP_4"]

H_mass = tree_arr["Hcand_mass"]
common_mask = ((H_mass < 90) | (H_mass > 150)) 
bkg_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 < 2) & (wp4 < 2) & common_mask

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
    sig_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 2) & (wp4 < 2) & common_mask
    bkg_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 < 2) & (wp4 < 2) & common_mask
    closure = ["3b_HiggsMW", "2b", "2b_w"]
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
# expose plotting-friendly names so plotting(..., var="dR1_plot") works

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

if args.runType == "train-test" or args.runType == "train-only":
    X = np.stack([combined_tree[f] for f in feature_names], axis=1)
    y = combined_tree[label_name]

    plot_dir =  f"{args.YEAR}/{args.TrainRegion}/{args.Model}_plots/{args.Model}_plots_{Scaling}_{BalanceClass}_Nov27/"
    model_dir = f"{args.YEAR}/{args.TrainRegion}/Models/Model_{args.Model}_{Scaling}_{BalanceClass}_Nov27/"

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Plot dir:", plot_dir)
    print("Model dir:", model_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.splitfraction, stratify=y, random_state=42
    )

    if args.isScaling == 1:
        Scaling = "Scaling"

        scaler = StandardScaler()
        scaler.fit(X_train)

        # Scale both train and test
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)

        # ALSO update combined_tree for later closure plotting
        full_scaled = scaler.transform(features)
        for i, name in enumerate(feature_names):
            combined_tree[name] = full_scaled[:, i]

        # Save scaler
        save_dir = model_dir
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(scaler, f"{save_dir}/scaler.pkl")

        corr_matrix = np.corrcoef(full_scaled, rowvar=False)

    else:
        Scaling = "NoScaling"
        corr_matrix = np.corrcoef(features, rowvar=False)

    n_sig = np.sum(y == 1)
    n_bkg = np.sum(y == 0)
    print("Signal events:", n_sig)
    print("Background events:", n_bkg)

    if args.Model == "DNN":

        inputs = Input(shape=(len(X_train[0]),))
        x = Flatten()(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=10, batch_size=64)

        model.save(model_dir+"/model.h5")

        y_score = model.predict(X_test)
        y_pred = (y_score > 0.5).astype(int)

        score_train = model.predict(X_train)
        score_test  = model.predict(X_test)

    elif args.Model == "BDT":

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            #use_label_encoder=False,
            eval_metric="logloss"
        )

        model.fit(X_train, y_train)

        model.save_model(model_dir+"/bdt_model.json")

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        score_train = model.predict_proba(X_train)[:, 1]
        score_test  = model.predict_proba(X_test)[:, 1] 

    else:
        print("This model is unavailable.")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_score))

    plt.figure(figsize=(8, 6))

    # Signal
    plot_hist(score_train, y_train == 1, "Signal (train)", "r")
    plot_hist(score_test, y_test == 1, "Signal (test)", "orange", linestyle="dashed")
    # Background
    plot_hist(score_train, y_train == 0, "Background (train)", "b")
    plot_hist(score_test, y_test == 0, "Background (test)", "green", linestyle="dashed")

    plt.xlabel(f"{args.Model} Score")
    plt.ylabel("Arbitrary units")
    plt.yscale("log")
    plt.legend(loc="best", fontsize=10)
    plt.title(f"{args.Model} Score Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{args.Model}_Score_Distribution.png"), dpi=300)
    plt.close()

    #Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ROC_Curve.png"), dpi=300)
    plt.close()

    # Compute correlation matrix
    #corr_matrix = np.corrcoef(features, rowvar=False)
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, xticklabels=feature_names, yticklabels=feature_names, annot=False, cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.75})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "Feature_Correlation_Matrix.png"), dpi=300)
    plt.close()
elif args.runType == "test-only":
    pass

if args.runType == "train-only":
    print("Training completed. No testing performed in 'train-only' mode.")


elif args.runType == "train-test" or args.runType == "test-only":
    features_raw = features.copy()
    if args.runType == "train-test":
        model_load_path = model_dir
        print(f"Loading model from training: {model_load_path}")

    elif args.runType == "test-only":
        if args.SpecificModelTest is not None:
            model_load_path = args.SpecificModelTest
        else:
            model_load_path = (
                f"{args.YEAR}/{args.TrainRegion}/Models/Model_{args.Model}_{'Scaling' if args.isScaling else 'NoScaling'}_{BalanceClass}_Nov27"
            )
        if not os.path.exists(model_load_path):
            print(f"Error: Model path does not exist: {model_load_path}")
            exit(1)
        print(f"Loading model for testing: {model_load_path}")

    if args.isScaling == 1:
        #scaler_path = os.path.join(os.path.dirname(model_load_path), "scaler.pkl")
        scaler_path = f"{model_load_path}/scaler.pkl"
        if not os.path.exists(scaler_path):
            print(f"Error: Scaler file not found at {scaler_path}")
            exit(1)
        scaler = joblib.load(scaler_path)
        features_scaled = scaler.transform(features_raw)

    else:
        features_scaled = features_raw.copy()

    combined_tree = {
        "signal": signal_flag.astype(np.int32),
        **{name: features_raw[:, i] for i, name in enumerate(feature_names)}
    }

    X = features_scaled      
    y = combined_tree[label_name]

    n_sig = np.sum(combined_tree["signal"] == 1)
    n_bkg = np.sum(combined_tree["signal"] == 0)

    print("Signal events:", n_sig)
    print("Background events:", n_bkg)

    model = load_model(model_load_path+"/model.h5")

    score = model.predict(X).ravel()

    score_weights = score / (1 - score) # DNN(signal_score)/DNN(backgorund_score), 1e-8 avoid division by zero

    combined_weights = score_weights * event_weights

    mx = (jets[:, 0] + jets[:, 1] + jets[:, 2] + jets[:, 3]).mass

    combined_tree["MX"] = mx
    combined_tree["MY"] = MY
    combined_tree["MH"] = MH
    combined_tree["Score"] = score
    combined_tree["Score_weights"] = score_weights
    combined_tree["Event_weights"] = event_weights
    combined_tree["Combined_weights"] = combined_weights
    combined_tree["n_jets_add"] = n_jets_add
    combined_tree["HT_additional"] = HT_additional

    combined_tree["dR1_plot"] = dR1_plot
    combined_tree["dR2_plot"] = dR2_plot

    # 3T and 2T below are not true closure we are looking for, it's just names at my first code
    # for true closure please carefully see 'sig_mask' and 'bkg_mask' and remember to change the label at 'closure' list
    mask_3T = combined_tree["signal"] == 1
    mask_2T = combined_tree["signal"] == 0

    arr_3T, arr_2T = {}, {}

    for k, v in combined_tree.items():
        arr_3T[k] = v[mask_3T]
        arr_2T[k] = v[mask_2T]

    bin_edges = np.linspace(0, 1, 51)
    mx_bin_edges = np.array([100,120,140,160,180,200,225,250,275,300,330,360,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1300,1400,1500,1600,1800,2000,2250,2500,2750,3000,3500,4000,4500])
    my_bin_edges = np.linspace(0,1000,51)
    mh_bin_edges = np.linspace(0,300,51)
    jet_mass_bin_edges = np.linspace(0, 100, 51)
    njets_add_bin_edges = np.array([0,1,2,3,4,5,6])
    suff = "reweight"

    plotting(arr_3T, arr_2T, add_bins_=True, bins_=bin_edges, var="Score", suffix=suff, ratio_ylim=[0.5, 1.5], label_=closure)
    plotting(arr_3T, arr_2T, add_bins_=True, bins_=mx_bin_edges, var="MX", suffix=suff, ratio_ylim=[0.5, 1.5], label_=closure)
    plotting(arr_3T, arr_2T, add_bins_=True, bins_=my_bin_edges, var="MY", suffix=suff, ratio_ylim=[0.5, 1.5], label_=closure)
    plotting(arr_3T, arr_2T, add_bins_=True, bins_=mh_bin_edges, var="MH", suffix=suff, ratio_ylim=[0.5, 1.5], label_=closure)
    plotting(arr_3T, arr_2T, add_bins_=True, bins_=njets_add_bin_edges, var="n_jets_add", suffix=suff, ratio_ylim=[0.5, 1.5], label_=closure)

    plotting(arr_3T, arr_2T, add_bins_=False, bins_=None, var="dR1_plot", suffix=suff, ratio_ylim=[0.5, 1.5], label_=closure)
    plotting(arr_3T, arr_2T, add_bins_=False, bins_=None, var="dR2_plot", suffix=suff, ratio_ylim=[0.5, 1.5], label_=closure)

    except_cols = ["signal", "MX", "MY", "MH", "Score", "Score_weights", "Event_weights", "Combined_weights", "n_jets_add"]
    for i in range(njets):
        except_cols.append(f"JetAK4_mass_{i+1}")
        col = f"JetAK4_mass_{i+1}"
        plotting(arr_3T, arr_2T, add_bins_=True, bins_=jet_mass_bin_edges,var=col, suffix=suff, ratio_ylim=[0.5, 1.5], label_=closure)

    for col in combined_tree.keys():
        if col not in except_cols:
            plotting(arr_3T, arr_2T, add_bins_=False, bins_=None, var=col, suffix=suff, ratio_ylim=[0.5, 1.5], label_=closure)    