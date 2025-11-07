import ROOT
import pandas as pd
import numpy as np
import uproot
import vector
import mplhep as hep
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import array

import os
import argparse


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

    # ratio
    # ratio_3T_2T = hist_3T.Clone("ratio_3T_2T")
    # ratio_3T_2T.Divide(hist_2T)

    # ratio_3T_2T_w = hist_3T.Clone("ratio_3T_2T_w")
    # ratio_3T_2T_w.Divide(hist_2T_w)

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

    pval_3T_2T = hist_3T.KolmogorovTest(hist_2T, "U")
    pval_3T_2T_w = hist_3T.KolmogorovTest(hist_2T_w, "U")

    # bin centers
    edges = np.array(bins)
    x = 0.5 * (edges[1:] + edges[:-1])

    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)
    hep.cms.label("Preliminary", data=True, lumi=9.45, com=13.6, year=2024, ax=ax)

    # plotting
    hep.histplot([y_3T, y_2T, y_2T_w], bins=edges, label=[label_[0], label_[1], label_[2]], ax=ax, histtype="step")
    if var in ["MX", "JetAK4_pt_1", "JetAK4_pt_2", "JetAK4_pt_3", "JetAK4_pt_4"]: ax.set_yscale("log")
    ax.set_ylabel("Entries")
    ax.set_xlim(*xlim)
    ax.legend()

    # plot ratio
    rax.errorbar(x, ratio_3T_2T, yerr=err_ratio_3T_2T, fmt='o', color='red', label=f"{label_[0]}/{label_[1]} p-value={pval_3T_2T:.2e}")
    rax.errorbar(x, ratio_3T_2T_w, yerr=err_ratio_3T_2T_w, fmt='o', color='blue', label=f"{label_[0]}/{label_[2]} p-value={pval_3T_2T_w:.2e}")
    rax.axhline(1.0, color='black', linestyle='--')
    rax.set_ylim(*ratio_ylim)
    rax.set_ylabel("Ratio")
    rax.set_xlabel(var)

    handles_ax, labels_ax = ax.get_legend_handles_labels()
    handles_rax, labels_rax = rax.get_legend_handles_labels()
    ax.legend(handles_ax + handles_rax, labels_ax + labels_rax)

    # outputs
    out_dir = f"{args.YEAR}/4Tvs2T/DNN_reweighting_plots/without_dR/DNN_reweighting_plots_{Scaling}_{BalanceClass}_DATA_DATA_UParTAK4"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{var}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/{var}_{suffix}.pdf", bbox_inches="tight")
    plt.close()

    hist_3T.Delete()
    hist_2T.Delete()
    hist_2T_w.Delete()
    # ratio_3T_2T.Delete()
    # ratio_3T_2T_w.Delete()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--YEAR', default="2024", type=str, help="Which era?")
    #parser.add_argument('--UseWeights', action='store', default=False, type=bool, help = "Use event weights?")
    parser.add_argument('--isScaling', default=1, type=int, help = "Standard Scaling")
    # parser.add_argument('--TrainingScheme', default=0, type=int, help = "Integer for choosing training samples, 0: sum of all samples, 1: high-priority samples, 2: low mass-split samples")
    parser.add_argument('--isBalanceClass', default=1, type=int, help = "Balance class?")
    parser.add_argument('--splitfraction', default=0.2, type=float, help = "Fraction of test data")
    # parser.add_argument('--isExcludedJetMass', default=1, type=int, help = "Balance class?")
    # parser.add_argument('--Model', default="DNN", type=str, help = "Model for training")

    args = parser.parse_args()

    MC = False
    isHcand_index_available = False
    closure = ["4b", "2b", "2b_w"]

    input_f = f"{args.YEAR}/DATA/new/Tree_Data_parking.root"
    print(input_f)

    input_file = uproot.open(input_f)

    tree = input_file["Tree_JetInfo"]
    # n_events = tree.num_entries
    n_events = 40000000
    njets = 4

    columns = ['JetAK4_btag_UParTAK4B_WP_1', 'JetAK4_btag_UParTAK4B_WP_2', 'JetAK4_btag_UParTAK4B_WP_3', 'JetAK4_btag_UParTAK4B_WP_4', 
                'JetAK4_pt_1', 'JetAK4_pt_2', 'JetAK4_pt_3', 'JetAK4_pt_4', 
                'JetAK4_eta_1', 'JetAK4_eta_2', 'JetAK4_eta_3', 'JetAK4_eta_4', 
                'JetAK4_phi_1', 'JetAK4_phi_2', 'JetAK4_phi_3', 'JetAK4_phi_4', 
                'JetAK4_mass_1', 'JetAK4_mass_2', 'JetAK4_mass_3', 'JetAK4_mass_4', 
                # "JetAK4_Hcand_index_1", "JetAK4_Hcand_index_2", "JetAK4_Hcand_index_3", "JetAK4_Hcand_index_4",
                # 'JetAK4_btag_UParTAK4CvB_1', 'JetAK4_btag_UParTAK4CvB_2', 'JetAK4_btag_UParTAK4CvB_4', #'JetAK4_btag_UParTAK4CvB_3'
                # 'JetAK4_btag_UParTAK4CvL_1', 'JetAK4_btag_UParTAK4CvL_2', 'JetAK4_btag_UParTAK4CvL_4', #'JetAK4_btag_UParTAK4CvL_3'
                # 'JetAK4_btag_UParTAK4QG_1', 'JetAK4_btag_UParTAK4QG_2', 'JetAK4_btag_UParTAK4QG_4', #'JetAK4_btag_UParTAK4QG_3'
                'Hcand_mass', 'Ycand_mass'] + (['Event_weight'] if MC else [])

    # columns = ["JetAK4_btag_PNetB_WP_1", "JetAK4_btag_PNetB_WP_2", "JetAK4_btag_PNetB_WP_3", "JetAK4_btag_PNetB_WP_4",
    #             "JetAK4_pt_1", "JetAK4_pt_2", "JetAK4_pt_3", "JetAK4_pt_4",
    #             "JetAK4_eta_1", "JetAK4_eta_2", "JetAK4_eta_3", "JetAK4_eta_4",
    #             "JetAK4_phi_1", "JetAK4_phi_2", "JetAK4_phi_3", "JetAK4_phi_4",
    #             "JetAK4_mass_1", "JetAK4_mass_2", "JetAK4_mass_3", "JetAK4_mass_4",
    #             "JetAK4_Hcand_index_1", "JetAK4_Hcand_index_2", "JetAK4_Hcand_index_3", "JetAK4_Hcand_index_4",
    #             "JetAK4_btag_PNetCvB_1", "JetAK4_btag_PNetCvB_2", "JetAK4_btag_PNetCvB_4", #"JetAK4_btag_PNetCvB_3", 
    #             "JetAK4_btag_PNetCvL_1", "JetAK4_btag_PNetCvL_2", "JetAK4_btag_PNetCvL_4", #"JetAK4_btag_PNetCvL_3", 
    #             "JetAK4_btag_PNetQG_1", "JetAK4_btag_PNetQG_2", "JetAK4_btag_PNetQG_4", #"JetAK4_btag_PNetQG_3", 
    #             "Hcand_mass", "Ycand_mass"]

    tree_arr = tree.arrays(columns, library="np", entry_stop=n_events)

    wp1 = tree_arr["JetAK4_btag_UParTAK4B_WP_1"]
    wp2 = tree_arr["JetAK4_btag_UParTAK4B_WP_2"]
    wp3 = tree_arr["JetAK4_btag_UParTAK4B_WP_3"]
    wp4 = tree_arr["JetAK4_btag_UParTAK4B_WP_4"]

    H_mass = tree_arr["Hcand_mass"]

    # CvB4 = tree_arr["JetAK4_btag_UParTAK4CvB_4"]
    # CvL4 = tree_arr["JetAK4_btag_UParTAK4CvL_4"]
    # QG4 = tree_arr["JetAK4_btag_UParTAK4QG_4"]

    common_mask = ((H_mass < 90) | (H_mass > 150)) #& (CvB4 >= 0) & (CvL4 >= 0) & (QG4 >= 0)

    sig_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 3) & (wp4 >= 2) & common_mask
    bkg_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 < 2) & (wp4 < 2) & common_mask

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

    combined_tree["dR_1"] = dR1_arr
    combined_tree["dR_2"] = dR2_arr

    if MC:
        event_weights = combined_tree["Event_weight"]
        drop_cols = ["Event_weight"]
    else:
        event_weights = np.ones(int(n_events), dtype=float)
        drop_cols = []

    drop_cols += Hcand_index_cols 
    drop_cols += [f"JetAK4_btag_UParTAK4B_WP_{i+1}" for i in range(njets)]
    drop_cols += [f"JetAK4_mass_{i+1}" for i in range(njets)] 
    drop_cols += ["Hcand_mass", "Ycand_mass"]
    drop_cols += ["dR_1", "dR_2"]

    for col in drop_cols:
        if col in combined_tree:
            del combined_tree[col]

    label_name = "signal"
    feature_names = [col for col in combined_tree.keys() if col != label_name]
    features = np.stack([combined_tree[col] for col in feature_names], axis=1)

    if args.isScaling == 1:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        new_combined_tree = {
            "signal": signal_flag.astype(np.int32),
            **{name: features_scaled[:, i] for i, name in enumerate(feature_names)}
        }

        Scaling = "Scaling"

    else:
        new_combined_tree = combined_tree
        Scaling = "NoScaling"

    n_sig = np.sum(new_combined_tree["signal"] == 1)
    n_bkg = np.sum(new_combined_tree["signal"] == 0)

    print("Signal events:", n_sig)
    print("Background events:", n_bkg)

    X = np.stack([new_combined_tree[f] for f in feature_names], axis=1)
    y = new_combined_tree[label_name]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.splitfraction, stratify=y, random_state=42)

    model = load_model("2024/4Tvs2T/Models/without_dR/Model_DNN_Scaling_BalanceClass_HcandSelection_addMoreTagging_remove3rdtagging_UParTAK4_All/model.h5")
        
    score = model.predict(X).ravel()

    score_weights = score / (1 - score) # DNN(signal_score)/DNN(backgorund_score), 1e-8 avoid division by zero

    combined_weights = score_weights * event_weights

    mx = (jets[:, 0] + jets[:, 1] + jets[:, 2] + jets[:, 3]).mass

    combined_tree["MX"] = mx
    combined_tree["Score"] = score
    combined_tree["Score_weights"] = score_weights
    combined_tree["Event_weights"] = event_weights
    combined_tree["Combined_weights"] = combined_weights

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
    suff = "reweight"

    plotting(arr_3T, arr_2T, add_bins_=True, bins_=bin_edges, var="Score", suffix=suff, ratio_ylim=[-2, 4], label_=closure)
    plotting(arr_3T, arr_2T, add_bins_=True, bins_=mx_bin_edges, var="MX", suffix=suff, ratio_ylim=[0, 2], label_=closure)

    except_cols = ["signal", "MX", "Score", "Score_weights", "Event_weights", "Combined_weights"]

    for col in combined_tree.keys():
        if col not in except_cols:
            plotting(arr_3T, arr_2T, add_bins_=False, bins_=None, var=col, suffix=suff, ratio_ylim=[0, 2], label_=closure)
