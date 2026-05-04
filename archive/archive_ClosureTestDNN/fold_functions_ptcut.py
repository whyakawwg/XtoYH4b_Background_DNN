# fold_functions.py
import uproot
import numpy as np
import vector
import matplotlib.pyplot as plt
import seaborn as sns
import ROOT
import pandas as pd
import mplhep as hep
import array
import os

def plot_hist(scores, mask, label, color, linestyle="solid", weights=None):
    """Utility to draw one DNN histogram."""
    scores = np.asarray(scores).ravel()
    mask = np.asarray(mask).ravel() 
    if weights is not None:
        weights = np.asarray(weights).ravel()
        plt.hist(
            scores[mask],
            bins=50,
            range=(0, 1),
            histtype='step',
            linewidth=1.5,
            label=label,
            color=color,
            linestyle=linestyle,
            density=True,
            weights=weights[mask]
        )
    else:
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

def plotting(arr_3T, arr_2T, bins=None, var="MX", suffix="reweight", label_=["3T", "2T", "2T_w"], args=None):
    xlim = [bins[0], bins[-1]]
    ROOT.gStyle.SetOptStat(0)
    ratio_ylim=[0.5, 1.5]

    n_bins = len(bins) - 1
    bins_array = array.array('d', bins)
    hist_3T = ROOT.TH1F("hist_3T", "", n_bins, bins_array)
    hist_2T = ROOT.TH1F("hist_2T", "", n_bins, bins_array)
    hist_2T_w = ROOT.TH1F("hist_2T_w", "", n_bins, bins_array)

    hist_3T.Sumw2()
    hist_2T.Sumw2()
    hist_2T_w.Sumw2()

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

    chi2_3T_2T = hist_3T.Chi2Test(hist_2T, "WW CHI2/NDF")
    chi2_3T_2T_w = hist_3T.Chi2Test(hist_2T_w, "WW CHI2/NDF")

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
    out_dir = f"{args.YEAR}/{args.TestRegion}/DNN_reweighting_plots/DNN_reweighting_plots_{Scaling}_{BalanceClass}_Nov27/MODEL_{foldN}/"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{var}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/{var}_{suffix}.pdf", bbox_inches="tight")
    plt.close()

    hist_dir = f"{out_dir}/HistogramRoot"
    
    os.makedirs(hist_dir, exist_ok=True)

    hist_3T.SaveAs(f"{out_dir}/HistogramRoot/{var}_{suffix}_hist_3T.root")
    hist_2T.SaveAs(f"{out_dir}/HistogramRoot/{var}_{suffix}_hist_2T.root")
    hist_2T_w.SaveAs(f"{out_dir}/HistogramRoot/{var}_{suffix}_hist_2T_w.root")

    hist_3T.Delete()
    hist_2T.Delete()
    hist_2T_w.Delete()

def plot_variable_correlation(df, vars_to_plot, title="Variable Correlation Matrix", args=None): 
    """ 
    df: dataframe containing events 
    vars_to_plot: list of variable names to include 
    """ 
    df = pd.DataFrame(df) 
    vars_available = [v for v in vars_to_plot if v in df.columns]

    corr = df[vars_available].corr()

    plt.figure(figsize=(12,10))
    sns.heatmap(
        corr, 
        xticklabels=vars_to_plot, 
        yticklabels=vars_to_plot, 
        annot=False, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        square=True, 
        cbar_kws={"shrink": 0.75})
    plt.title(title)
    out_dir = f"{args.YEAR}/{args.TestRegion}/DNN_reweighting_plots/DNN_reweighting_plots_{Scaling}_{BalanceClass}_Nov27/MODEL_{foldN}/"
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title.replace(' ', '_')}.png"), dpi=300)
    plt.close()

def plotting_2D(arr_3T, arr_2T, varX="MX", varY="dR1_plot",
                binsX=None, binsY=None,
                suffix="2Dclosure", label_=["4b", "2b_w"], args=None):
    """
    Make 2D closure map: (3T) / (2T weighted).
    """

    if binsX is None:
        binsX = np.linspace(np.min(arr_3T[varX]), np.max(arr_3T[varX]), 41)
    if binsY is None:
        binsY = np.linspace(np.min(arr_3T[varY]), np.max(arr_3T[varY]), 41)

    n_binsX = len(binsX) - 1
    n_binsY = len(binsY) - 1

    binsX_array = array.array('d', binsX)
    binsY_array = array.array('d', binsY)

    ROOT.gStyle.SetOptStat(0)

    # 3T
    h3T = ROOT.TH2F("h3T", "", n_binsX, binsX_array, n_binsY, binsY_array)

    # 2T weighted 
    h2T_w = ROOT.TH2F("h2T_w", "", n_binsX, binsX_array, n_binsY, binsY_array)

    # Fill histograms
    for i in range(len(arr_3T[varX])):
        h3T.Fill(arr_3T[varX][i], arr_3T[varY][i], arr_3T["Event_weights"][i])

    for i in range(len(arr_2T[varX])):
        h2T_w.Fill(arr_2T[varX][i], arr_2T[varY][i], arr_2T["Combined_weights"][i])

    # Convert to numpy
    H3 = np.array([[h3T.GetBinContent(ix+1, iy+1)
                    for iy in range(n_binsY)] for ix in range(n_binsX)])
    H2 = np.array([[h2T_w.GetBinContent(ix+1, iy+1)
                    for iy in range(n_binsY)] for ix in range(n_binsX)])

    # ratio 
    ratio = np.divide(H3, H2, out=np.full_like(H3, np.nan), where=H2 > 0)
    ratio_abs = np.abs(ratio - 1)

    # PLOT
    plt.figure(figsize=(12,10))

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad(color="gray")

    plt.imshow(
        ratio_abs.T,
        origin="lower",
        aspect="auto",
        extent=[binsX[0], binsX[-1], binsY[0], binsY[-1]],
        cmap=cmap,
        vmin=0,
        vmax=0.5
    )


    cbar = plt.colorbar()
    cbar.set_label(f"{label_[0]} / {label_[1]}")

    plt.xlabel(varX)
    plt.ylabel(varY)
    plt.title(f"2D Closure: {varX} vs {varY}")

    out_dir = f"{args.YEAR}/{args.TestRegion}/DNN_reweighting_plots/DNN_reweighting_plots_{Scaling}_{BalanceClass}_Nov27/MODEL_{foldN}/2D_maps"
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(f"{out_dir}/{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/{suffix}.pdf", bbox_inches="tight")
    plt.close()

    h3T.Delete()
    h2T_w.Delete()

def build_binning_map(njets):
    """
    Define bin edges for all variables. Return a dictionary mapping variable names to their bin edges.
    """
    bin_edges      = np.linspace(0, 1, 51)

    # mx_bin_edges   = np.array([110,135,165,200,250,300,375,450,550,675,825,1000,1250,1600,2000,2500,3000,4000,5000])
    mx_bin_edges   = np.array([250,300,375,450,550,675,825,1000,1250,1600,2000,2500,3000,4000,5000])

    my_bin_edges = np.array([30,40,50,60,75,90,110,135,165,200,250,300,375, 
                             450,550,675,825,1000,1250,1600,2000,2500,3000,4000])
    mh_bin_edges   = np.array([30,40,50,60,75,90,110,135,165,200,250,300])
    jet_mass_bin_edges = np.linspace(0, 100, 51)
    njets_add_bin_edges = np.array([0,1,2,3,4,5,6])
    eta_bin_edges  = np.linspace(-5, 5, 51)
    phi_bin_edges  = np.linspace(-3.14, 3.14, 65)
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

def get_5fold_filelists(fold_n, base_path="/data/dust/user/wanghaoy/XtoYH4b/split_rootfile"):
    """
    For 5-fold, each fold uses 2 files for testing and the remaining 8 for training.
    """
    if not (1 <= fold_n <= 5):
        raise ValueError(f"Fold number must be between 1 and 5. Received: {fold_n}")

    all_indices = list(range(0, 10))
    start_num = (fold_n - 1) * 2 
    test_indices = [start_num, start_num + 1]

    train_files = []
    test_files = []

    for i in all_indices:
        # Construct the full file path
        file_path = os.path.join(base_path, f"Tree_Data_Parking_{i}.root")
        
        if i in test_indices:
            test_files.append(file_path)
        else:
            train_files.append(file_path)

    return train_files, test_files

def get_10fold_filelists(fold_n, base_path="/data/dust/user/wanghaoy/XtoYH4b/split_rootfile"):
    """
    For 10-fold, each fold uses 1 file for testing and the remaining 9 for training.
    """
    if not (1 <= fold_n <= 10):
        raise ValueError(f"Fold number must be between 1 and 10. Received: {fold_n}")

    all_indices = list(range(0, 10))
    test_index = fold_n - 1

    train_files = []
    test_files = []

    for i in all_indices:
        # Construct the full file path
        file_path = os.path.join(base_path, f"Tree_Data_Parking_{i}.root")
        
        if i == test_index:
            test_files.append(file_path)
        else:
            train_files.append(file_path)

    return train_files, test_files

def plot_training_results(history, plot_dir, args=None):
    """
    Plot training and validation loss and AUC curves from Keras history object.
    """
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    plt.figure(figsize=(14, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(plot_dir, f"{args.Model}_Training_Validation_Loss.png"), dpi=300)
    # plt.close()

    # AUC Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict['auc'], 'b-', label='Training AUC')
    plt.plot(epochs, history_dict['val_auc'], 'r-', label='Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{args.Model}_Training_Validation_AUC.png"), dpi=300)
    plt.close()

def fast_fill(hist, data, weights):
    """
    Helper function to speed up ROOT histogram filling using FillN.
    Converts numpy arrays to contiguous float64 for C++ compatibility.
    """
    if len(data) == 0: return
    
    d_arr = np.ascontiguousarray(data, dtype=np.float64)
    w_arr = np.ascontiguousarray(weights, dtype=np.float64)
    
    hist.FillN(len(d_arr), d_arr, w_arr)

def calculate_chi2(h_obs, h_exp, h_exp_up, h_exp_dn):
    """
    Calculate chi-squared between observed and expected histograms.
    """
    chi2 = 0.0
    ndf = 0
    
    for i in range(1, h_obs.GetNbinsX() + 1):
        obs = h_obs.GetBinContent(i)
        exp = h_exp.GetBinContent(i)
        if obs == 0 and exp == 0: continue 
            
        sigma_data = h_obs.GetBinError(i)
        sigma_model_stat = h_exp.GetBinError(i)

        if obs >= exp:
            sigma_sys = abs(h_exp_up.GetBinContent(i) - exp)
        else:
            sigma_sys = abs(h_exp_dn.GetBinContent(i) - exp)

        total_variance = sigma_data**2 + sigma_model_stat**2 + sigma_sys**2
        
        if total_variance > 0:
            chi2 += (obs - exp)**2 / total_variance
            ndf += 1
            
    return chi2, max(ndf - 1, 1)

def get_chi2_root_method(h_data, h_nom, h_up, h_dn):
    """
    Use ROOT's built-in Chi2Test method.
    """
    h_total_err = h_nom.Clone("h_total_err_for_chi2")
    
    n_bins = h_nom.GetNbinsX()
    
    for i in range(1, n_bins + 1):

        sigma_stat = h_nom.GetBinError(i)
        
        diff_up = abs(h_up.GetBinContent(i) - h_nom.GetBinContent(i))
        diff_dn = abs(h_dn.GetBinContent(i) - h_nom.GetBinContent(i))
        sigma_sys = max(diff_up, diff_dn)
        
        sigma_total = np.sqrt(sigma_stat**2 + sigma_sys**2)
        h_total_err.SetBinError(i, sigma_total)
        
    chi2 = h_data.Chi2Test(h_total_err, "WW CHI2/NDF")
    
    return chi2

def get_fold_hists(file, var_name, n_folds, scale_factor=1.0):
    """Helper to load individual fold histograms"""
    fold_data = []

    for i in range(1, n_folds + 1):
        h = file.Get(f"{var_name}_hist_2b_fold{i}")
        if not h:
            print(f"Warning: Fold {i} not found for {var_name}")
            continue
        h.Scale(scale_factor)
        n = h.GetNbinsX()
        y = np.array([h.GetBinContent(b) for b in range(1, n+1)])
        fold_data.append(y)
    return fold_data

def get_split_fold_hists(file, var_name, n_splits=5, n_folds=10, scale_factor=1.0):
    """
    Helper to load fold histograms for 5-fold case which has 5 splits. Returns a list of arrays, one per fold, containing the bin contents. Assumes histograms are named like {var_name}_hist_2b_split{split}_fold{fold}.
    """
    fold_data = []

    for split in range(n_splits):
        for fold in range(1, n_folds + 1):
            h_name = f"{var_name}_hist_2b_split{split}_fold{fold}"
            h = file.Get(h_name)
            if not h: 
                print(f"Warning: Fold {fold} Split {split}not found for {var_name}")
                continue
            h.Scale(scale_factor)
            n = h.GetNbinsX()
            y = np.array([h.GetBinContent(b) for b in range(1, n+1)])
            fold_data.append(y)
    return fold_data

def check_h_nomerror_2b_staterror(file, var_name):
    """
    Only used when checking.
    """
    h_nom = file.Get(f"{var_name}_hist_2b_w_mean")
    h_2b  = file.Get(f"{var_name}_hist_2b_mean")

    n = h_nom.GetNbinsX()
    for i in range(1, n+1):
        nom_err = h_nom.GetBinError(i)
        stat_err = h_2b.GetBinError(i)
        if nom_err < stat_err:
            print(f"Warning: For {var_name} bin {i}, nom error {nom_err} < 2b stat error {stat_err}")
        if nom_err == stat_err:
            print(f"Note: For {var_name} bin {i}, nom error {nom_err} == 2b stat error {stat_err}")

def get_fold_errors(file, var_name, n_folds, TrainRegion="4b"):
    """Helper to get statistical errors for each fold histogram"""
    fold_errors = []
    ratio_errs = []

    if TrainRegion == "3b":
        for split in range(5):
            for i in range(1, n_folds + 1):
                h = file.Get(f"{var_name}_hist_2b_split{split}_fold{i}")
                if not h:
                    print(f"Warning: Fold {i} Split {split} not found for {var_name}")
                    continue
                n = h.GetNbinsX()
                errs = np.array([h.GetBinError(b) for b in range(1, n+1)])
                fold_errors.append(errs)
                ratio_err = errs / np.where(np.array([h.GetBinContent(b) for b in range(1, n+1)]) > 0,
                                            np.array([h.GetBinContent(b) for b in range(1, n+1)]),
                                            1e-10)
                ratio_errs.append(ratio_err)

    else:
        for i in range(1, n_folds + 1):
            h = file.Get(f"{var_name}_hist_2b_fold{i}")
            if not h:
                print(f"Warning: Fold {i} not found for {var_name}")
                continue
            n = h.GetNbinsX()
            errs = np.array([h.GetBinError(b) for b in range(1, n+1)])
            fold_errors.append(errs)
            ratio_err = errs / np.where(np.array([h.GetBinContent(b) for b in range(1, n+1)]) > 0,
                                        np.array([h.GetBinContent(b) for b in range(1, n+1)]),
                                        1e-10)
            ratio_errs.append(ratio_err)

    return ratio_errs

def check_4b_2b_errors(file, var_name):
    """Helper to check if 4b errors are smaller than 2b errors, which would be a sign of an issue since 4b has less statistics."""
    h_4b = file.Get(f"{var_name}_hist_4b_mean")
    h_2b = file.Get(f"{var_name}_hist_2b_mean")

    n = h_4b.GetNbinsX()
    err_4b = []
    err_2b = []
    for i in range(1, n+1):
        err_4b.append(h_4b.GetBinError(i))
        err_2b.append(h_2b.GetBinError(i))
    
    ratio_err_4b = np.array(err_4b) / np.where(np.array([h_4b.GetBinContent(b) for b in range(1, n+1)]) > 0,
                                               np.array([h_4b.GetBinContent(b) for b in range(1, n+1)]),
                                               1e-10)
    ratio_err_2b = np.array(err_2b) / np.where(np.array([h_2b.GetBinContent(b) for b in range(1, n+1)]) > 0,
                                               np.array([h_2b.GetBinContent(b) for b in range(1, n+1)]),
                                               1e-10)

    return ratio_err_4b, ratio_err_2b, n

def calculate_error_from_histograms(file, var_name, n_folds, TrainRegion="4b"):
    """
    Calculate the systematic uncertainty from the spread of fold histograms for each bin. Use the same percentile methods.
    """
    if TrainRegion == "3b":
        fold_ys = get_split_fold_hists(file, var_name, n_splits=5, n_folds=n_folds)
    else:
        fold_ys = get_fold_hists(file, var_name, n_folds)
        
    
    n_bins = len(fold_ys[0]) 
    
    sys_sigma = np.zeros(n_bins)
    mean = np.zeros(n_bins) 
    
    for i in range(n_bins):
        bin_values = [fold[i] for fold in fold_ys]   
        mean[i] = np.mean(bin_values)
        q16 = np.percentile(bin_values, 16)
        q84 = np.percentile(bin_values, 84)
        sys_sigma[i] = (q84 - q16) / 2.0
    return mean, sys_sigma

def get_hist_with_total_error(file, var_name, n_folds, normalize=True, TrainRegion="4b", NonClosureFracPath=None):
    """
    Currently the main function to get the errors. 
    Returns:
    - edges: bin edges
    - y_mean: mean bin contents across folds (used as central value for chi2)
    - y_3T: 3T histogram bin contents
    - y_2T: 2T histogram bin contents

    - scale_factor: factor by which histograms were scaled 
    - chi2_val: chi2 value comparing 3T to the mean of folds with total error
    - chi2_2b: chi2 value comparing 3T to the 2T histogram with only statistical error (for comparison)
    - err_stat: array of statistical errors from the 2T histogram
    - err_sys: array of systematic uncertainties derived from fold spread
    - err_tot: total error combining statistical and systematic in quadrature
    - err_nc: non-closure error if NonClosureFracPath is provided (only for 3bHMW)

    """
    # h_3T  = file.Get(f"{var_name}_hist_4b_mean") 
    # h_2T  = file.Get(f"{var_name}_hist_2b_mean")
    # h_3T_ptr = file.Get(f"{var_name}_hist_{TrainRegion}_mean")
    h_3T_ptr = file.Get(f"{var_name}_hist_4b_mean")
    h_2T_ptr = file.Get(f"{var_name}_hist_2b_mean")

    h_3T = h_3T_ptr.Clone()
    h_2T = h_2T_ptr.Clone()

    h_3T.SetDirectory(0) 
    h_2T.SetDirectory(0)

    n_bins = h_3T.GetNbinsX()

    fold_data = []

    if TrainRegion == "3b":
        n_splits = 5         
        for split in range(n_splits):
            for fold in range(1, n_folds + 1):
                h_name = f"{var_name}_hist_2b_split{split}_fold{fold}"
                h_fold = file.Get(h_name)
                if not h_fold:
                    print(f"Warning: Fold {fold} Split {split} not found for {var_name}")
                    continue                
                content = [h_fold.GetBinContent(i) for i in range(1, n_bins + 1)]
                fold_data.append(content)
                
    else:
        for fold_idx in range(n_folds):
            h_name = f"{var_name}_hist_2b_fold{fold_idx+1}"
            h_fold = file.Get(h_name)            
            if not h_fold:
                print(f"Warning: Fold {fold_idx+1} not found for {var_name}")
                continue            
            content = [h_fold.GetBinContent(i) for i in range(1, n_bins + 1)]
            fold_data.append(content)

    fold_array = np.array(fold_data)
    if fold_array.size == 0:
        raise ValueError(
            f"No fold histograms found for '{var_name}' in TrainRegion='{TrainRegion}'. "
            "Check histogram naming and available folds in the input ROOT file."
        )

    y_mean = np.mean(fold_array, axis=0)
    q16 = np.percentile(fold_array, 16, axis=0)
    q84 = np.percentile(fold_array, 84, axis=0)

    err_sys = (q84 - q16) / 2.0

    scale_factor = 1.0

    if normalize:
        if h_3T.Integral() > 0:
            h_3T.Scale(1.0 / h_3T.Integral())

        if h_2T.Integral() > 0:
            h_2T.Scale(1.0 / h_2T.Integral())

        total_integral = np.sum(y_mean)
        
        if total_integral > 0:
            scale_factor = 1.0 / total_integral
            
            y_mean *= scale_factor
            err_sys  *= scale_factor

    err_stat = np.array([h_2T.GetBinError(i) for i in range(1, n_bins+1)])

    if NonClosureFracPath is not None:
        nonclosure_factors = load_nonclosure_factor(var_name, n_bins, NonClosureFracPath)
        err_nc = nonclosure_factors * y_mean
        err_tot = np.sqrt(err_stat**2 + err_sys**2 + err_nc**2)
    else:
        err_nc = np.zeros_like(err_sys)
    
    err_tot = np.sqrt(err_stat**2 + err_sys**2 + err_nc**2)
    err_tot_Nonc = np.sqrt(err_stat**2 + err_sys**2)

    h_total_err_for_chi2 = h_2T.Clone(f"{var_name}_total_err")
    h_Nonc_err_for_chi2 = h_2T.Clone(f"{var_name}_fornonclosure_err")
    for i in range(n_bins):
        bin_idx = i + 1 
        h_total_err_for_chi2.SetBinContent(bin_idx, y_mean[i])
        h_total_err_for_chi2.SetBinError(bin_idx, err_tot[i])
        h_Nonc_err_for_chi2.SetBinContent(bin_idx, y_mean[i])
        h_Nonc_err_for_chi2.SetBinError(bin_idx, err_tot_Nonc[i])

    chi2_val = h_3T.Chi2Test(h_total_err_for_chi2, "WW CHI2/NDF")
    chi2_2b = h_3T.Chi2Test(h_2T, "WW CHI2/NDF")
    chi2_Nonc = h_3T.Chi2Test(h_Nonc_err_for_chi2, "WW CHI2/NDF")

    edges = np.array([h_3T.GetBinLowEdge(i) for i in range(1, n_bins+2)])
    y_3T = np.array([h_3T.GetBinContent(i) for i in range(1, n_bins+1)])
    y_2T = np.array([h_2T.GetBinContent(i) for i in range(1, n_bins+1)])

    # denom_2b_w = np.where(y_mean > 0, y_mean, 1e-10)
    # denom_2b = np.where(y_2T > 0, y_2T, 1e-10)

    # ratio_3b_2b = y_3T/ denom_2b # without transfer factor
    # ratio_3b_2b_w = y_3T/ denom_2b_w # data/prediction

    # ratio_err_tot   = np.divide(err_tot, denom_2b_w, out=np.zeros_like(err_tot), where=denom_2b_w > 0)
    # ratio_err_stat  = np.divide(err_stat, denom_2b_w, out=np.zeros_like(err_stat), where=denom_2b_w > 0)
    # ratio_err_sys   = np.divide(err_sys, denom_2b_w, out=np.zeros_like(err_sys), where=denom_2b_w > 0)

    ratio_3b_2b = np.divide(y_3T, y_2T, out=np.zeros_like(y_3T), where=y_2T > 0) # without transfer factor
    ratio_3b_2b_w = np.divide(y_3T, y_mean, out=np.zeros_like(y_3T), where=y_mean > 0) # data/prediction

    ratio_err_tot   = np.divide(err_tot, y_mean, out=np.ones_like(err_tot), where=y_mean > 0)
    ratio_err_stat  = np.divide(err_stat, y_mean, out=np.ones_like(err_stat), where=y_mean > 0)
    ratio_err_sys   = np.divide(err_sys, y_mean, out=np.ones_like(err_sys), where=y_mean > 0)

    if NonClosureFracPath is not None:
        print(f"Non-closure uncertainty included in total error for chi2 calculation for {var_name}.")
        return edges, y_mean, y_3T, y_2T, err_tot, scale_factor, chi2_val, chi2_2b, err_stat, err_sys, ratio_3b_2b, ratio_3b_2b_w, ratio_err_tot, ratio_err_stat, ratio_err_sys, err_nc, nonclosure_factors, chi2_Nonc
    else:
        return edges, y_mean, y_3T, y_2T, err_tot, scale_factor, chi2_val, chi2_2b, err_stat, err_sys, ratio_3b_2b, ratio_3b_2b_w, ratio_err_tot, ratio_err_stat, ratio_err_sys

def processing(file_list, args=None):
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

        # Construct the matching weight file paths =================================================
        if file_list == ['/data/dust/user/wanghaoy/XtoYH4b/split_rootfile/Tree_Data_Parking_0.root']:
            weights_path = f"TransferFactors_{args.YEAR}_{args.TestRegion}_0.root"

        else:
            weights_path = f"TransferFactors_{args.YEAR}_{args.TestRegion}.root"
    
        try:

            f_friend = uproot.open(weights_path)
            weight_tree = f_friend["Tree_JetInfo"].arrays(["Transfer_Weight"], library="np")

            n_weights = len(weight_tree["Transfer_Weight"])
            if n_weights != n_events:
                print(f"[CRITICAL ERROR] Alignment Mismatch!")
                print(f" -> {n_events} physical events loaded from the 9 files.")
                print(f" -> {n_weights} weights found in {weights_path}.")
                print(" -> Did the file list order or content change between evaluation and training?")
                exit(1)
            else: 
                print(f"[INFO] Alignment Check Passed: {n_events} events and {n_weights} weights.")

            tree_arr["Event_weight"] = weight_tree["Transfer_Weight"]
            print(f"[INFO] Successfully attached Transfer Weights from {weights_path}")
            
        except Exception as e:
            print(f"[WARNING] Could not load Friend Tree ({weights_path}). Defaulting to Event_weight = 1.0")
            print(f" -> Error details: {e}")
            tree_arr["Event_weight"] = np.ones(n_events, dtype=np.float32)
        # =====================================================================================


    elif args.runType == "test-only": 
        #input_file = uproot.open(f"/data/dust/user/wanghaoy/XtoYH4b/Tree_Data_Parking.root")
        input_file = uproot.open(file_list[0])
        tree = input_file["Tree_JetInfo"]
        n_events = tree.num_entries
        tree_arr = tree.arrays(columns, library="np", entry_stop=n_events)

    else:
        print("For k-fold, train-test is not available yet.")
        exit(1)

    # -------------------------------
    n_total_original_rows = len(tree_arr["JetAK4_pt_1"])
    # -------------------------------


    wp1 = tree_arr["JetAK4_btag_B_WP_1"]
    wp2 = tree_arr["JetAK4_btag_B_WP_2"]
    wp3 = tree_arr["JetAK4_btag_B_WP_3"]
    wp4 = tree_arr["JetAK4_btag_B_WP_4"]

    H_mass = tree_arr["Hcand_mass"]
    min_mask = (H_mass > 50) & (H_mass < 300)
    common_mask = ((H_mass < 90) | (H_mass > 150)) & min_mask
    bkg_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 < 2) & (wp4 < 2) & common_mask & min_mask

    pt_cut_mask = (tree_arr["JetAK4_pt_1"] > 50) & (tree_arr["JetAK4_pt_2"] > 50) & \
                  (tree_arr["JetAK4_pt_3"] > 50) & (tree_arr["JetAK4_pt_4"] > 50)

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

    elif args.TrainRegion == "3bHiggsMW":
        sig_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 2) & (wp4 < 2) & ((H_mass >= 90) & (H_mass <= 150)) 

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

    common_mask = common_mask & pt_cut_mask
    sig_mask = sig_mask & pt_cut_mask
    bkg_mask = bkg_mask & pt_cut_mask

    sig_idx = np.where(sig_mask)[0]
    bkg_idx = np.where(bkg_mask)[0]

# split into 5 for 3b region
    if args.runType == "train-only" and args.TrainRegion == "3b":
        
        rng = np.random.default_rng(seed=42)
        
        rng.shuffle(sig_idx)
        
        n_total_3b = len(sig_idx)
        chunk_size = int(n_total_3b / 5)
        
        start_idx = args.SplitIndex * chunk_size
        end_idx   = start_idx + chunk_size
        
        if args.SplitIndex == 4:
            end_idx = n_total_3b
            
        print(f"[INFO] 3b Split Strategy: Using Split {args.SplitIndex}/5")
        print(f"[INFO] Slice Range: {start_idx} to {end_idx} (Total 3b Pool: {n_total_3b})")
        
        sig_idx_subset = sig_idx[start_idx : end_idx]
        sig_idx = sig_idx_subset

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

    isHcand_index_available = False # From training script

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

        event_weights = np.ones(int(n_events), dtype=float)
    drop_cols = []

    combined_tree["dR_1"] = dR1_arr
    combined_tree["dR_2"] = dR2_arr
    dR1_plot = dR1_arr.copy()
    dR2_plot = dR2_arr.copy()


    # if args.isMC == 1:
    #     event_weights = combined_tree["Event_weight"]
    #     drop_cols = ["Event_weight"]
    # else:
    #     event_weights = np.ones(int(n_events), dtype=float)
    #     drop_cols = []

    # event_weights = combined_tree["Event_weight"]
    # drop_cols = ["Event_weight"]
    if "Event_weight" in combined_tree:
        event_weights = combined_tree["Event_weight"]
        drop_cols.append("Event_weight")

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
            "closure": closure,

            # --------------------------- 
            "all_idx": all_idx,         
            # "n_total_events": n_events  
            # ---------------------------         
            "n_total_events": n_total_original_rows  
            # ------------------------   
        }
    else:
        # aux_data = {}
        aux_data = {"event_weights": event_weights}

    return feature_names, features, combined_tree, aux_data


def get_hist_with_total_error_Mass(file, var_name, n_folds, normalize=True, TrainRegion="4b"):
    h_3T_ptr = file.Get(f"{var_name}_hist_4b_mean")
    h_2T_ptr = file.Get(f"{var_name}_hist_2b_mean")

    h_3T = h_3T_ptr.Clone()
    h_2T = h_2T_ptr.Clone()

    h_3T.SetDirectory(0) 
    h_2T.SetDirectory(0)

    n_bins = h_3T.GetNbinsX()

    fold_data = []
    if TrainRegion == "3b":
        n_splits = 5         
        for split in range(n_splits):
            for fold in range(1, n_folds + 1):
                h_name = f"{var_name}_hist_2b_split{split}_fold{fold}"
                h_fold = file.Get(h_name)
                if not h_fold:
                    continue                
                content = [h_fold.GetBinContent(i) for i in range(1, n_bins + 1)]
                fold_data.append(content)       
    else:
        for fold_idx in range(n_folds):
            h_name = f"{var_name}_hist_2b_fold{fold_idx+1}"
            h_fold = file.Get(h_name)            
            if not h_fold:
                continue            
            content = [h_fold.GetBinContent(i) for i in range(1, n_bins + 1)]
            fold_data.append(content)

    fold_array = np.array(fold_data)
    y_mean = np.mean(fold_array, axis=0)
    q16 = np.percentile(fold_array, 16, axis=0)
    q84 = np.percentile(fold_array, 84, axis=0)
    y_sys = (q84 - q16) / 2.0

    scale_factor = 1.0

    if normalize:
        if h_3T.Integral() > 0:
            h_3T.Scale(1.0 / h_3T.Integral())
        if h_2T.Integral() > 0:
            h_2T.Scale(1.0 / h_2T.Integral())

        total_integral = np.sum(y_mean)
        if total_integral > 0:
            scale_factor = 1.0 / total_integral
            y_mean *= scale_factor
            y_sys  *= scale_factor

    err_stat = np.array([h_2T.GetBinError(i) for i in range(1, n_bins+1)])
    err_tot = np.sqrt(err_stat**2 + y_sys**2)

    h_total_err_for_chi2 = h_2T.Clone(f"{var_name}_total_err")
    for i in range(n_bins):
        bin_idx = i + 1 
        h_total_err_for_chi2.SetBinContent(bin_idx, y_mean[i])
        h_total_err_for_chi2.SetBinError(bin_idx, err_tot[i])

    chi2_val = h_3T.Chi2Test(h_total_err_for_chi2, "WW CHI2/NDF")
    chi2_2b = h_3T.Chi2Test(h_2T, "WW CHI2/NDF")

    edges = np.array([h_3T.GetBinLowEdge(i) for i in range(1, n_bins+2)])
    widths = np.array([h_3T.GetBinWidth(i) for i in range(1, n_bins+1)])

    y_3T = np.array([h_3T.GetBinContent(i) for i in range(1, n_bins+1)])
    y_2T = np.array([h_2T.GetBinContent(i) for i in range(1, n_bins+1)])

    y_mean   = y_mean / widths
    y_3T     = y_3T / widths
    y_2T     = y_2T / widths
    err_tot  = err_tot / widths
    err_stat = err_stat / widths
    y_sys    = y_sys / widths

    return edges, y_mean, y_3T, y_2T, err_tot, scale_factor, chi2_val, chi2_2b, err_stat, y_sys


def get_label_name(var):
    """
    Returns a formatted label string for a given variable name.
    Explicitly mentions b-tag ordering to avoid confusion with pT ordering.
    """
    njets = 4 

    sort_label = " (ordered by b-tag)" 

    for i in range(1, njets + 1):
        if var == f"JetAK4_mass_{i}":
            return f"Jet {i} mass{sort_label}"
        
        if var == f"JetAK4_pt_{i}":
            return f"Jet {i} pt{sort_label}"
            
        if var == f"JetAK4_eta_{i}":
            return f"Jet {i} $\eta${sort_label}"
            
        if var == f"JetAK4_phi_{i}":
            return f"Jet {i} $\phi${sort_label}"

    if var == "Unrolled_MXMY":
        return "Unrolled (MX, MY) value"
    #     return "Higgs Candidate Mass [GeV]"
        
    # Default return if no match
    return var




def error_bands(nominal, error, prevent_negative=True):
    """
    Create error bands for plotting.
    """
    low = nominal - error
    if prevent_negative:
        low = np.maximum(low, 1e-10) 
        
    band_low  = np.append(low, low[-1])
    band_high = np.append(nominal + error, (nominal + error)[-1])

    return band_low, band_high

def load_nonclosure_factor(var_name, n_bins, in_filename="nonclosure_factors.root"):
    """Loads the saved fractional uncertainty to apply to a new region."""
    f_in = ROOT.TFile.Open(in_filename, "READ")
    if not f_in or f_in.IsZombie():
        return None # Return None if file doesn't exist yet
        
    h_name = f"{var_name}_nonclosure_factor"
    h_nonclosure = f_in.Get(h_name)
    if not h_nonclosure:
        f_in.Close()
        return None
        
    frac_array = np.array([h_nonclosure.GetBinContent(i) for i in range(1, n_bins + 1)])
    f_in.Close()
    return frac_array

def make_hist(suffix, values):
    h_name = f"{var}_{suffix}"
    h = ROOT.TH1F(h_name, h_name, nbins, edges_array)
    for i in range(nbins):
        h.SetBinContent(i+1, values[i])
        h.SetBinError(i+1, 1e-6) 
    return h

def get_lumi(year):
    lumi_map = {"2022": 7.98, "2022EE": 26.67, "2023": 11.24, "2023BPiX": 9.45, "2024": 109}
    if year not in lumi_map:
        print(f"Warning: Invalid year {year}. Defaulting to 109 (2024).")
        return 109.0
    return lumi_map[year]


def get_unrolled_bins_for_my(target_my_bins, mx_bin_edges=None, my_bin_edges=None):
    """
    Given a list of target MY bins (1-based index), returns a dictionary mapping 
    each MY bin to its corresponding 1D unrolled bin indices.
    
    Args:
        target_my_bins (list of int): e.g., [6, 7, 8]
        mx_bin_edges (np.array, optional): Edges for MX.
        my_bin_edges (np.array, optional): Edges for MY.
        
    Returns:
        dict: Format {my_bin_index: [list_of_unrolled_1d_indices]}
    """
    
    # Default edges if none are provided
    if mx_bin_edges is None:
        mx_bin_edges = np.array([250,300,375,450,550,675,825,1000,1250,1600,2000,2500,3000,4000,5000])
    if my_bin_edges is None:
        my_bin_edges = np.array([30,40,50,60,75,90,110,135,165,200,250,300,375,450,550,675,825,1000,1250,1600,2000,2500,3000,4000])

    n_my_bins = len(my_bin_edges) - 1
    n_mx_bins = len(mx_bin_edges) - 1

    # Initialize a dictionary for the target bins
    mapping = {my_bin: [] for my_bin in target_my_bins}
    
    current_1d_bin = 1

    # Apply physical unrolled bin logic
    for my_idx in range(n_my_bins):
        for mx_idx in range(n_mx_bins):
            mx_upper = mx_bin_edges[mx_idx + 1]
            my_lower = my_bin_edges[my_idx]

            # The physical cut
            if mx_upper > (my_lower + 125):
                my_bin_1d = my_idx + 1  # 1-based ROOT bin index
                
                # If this is one of the bins we want, save the unrolled ID
                if my_bin_1d in target_my_bins:
                    mapping[my_bin_1d].append(current_1d_bin)
                
                current_1d_bin += 1

    return mapping

def get_all_bin_mappings(mx_bin_edges=None, my_bin_edges=None):
    """
    Builds a complete two-way mapping for 2D MX-MY bins to 1D unrolled bins.
    
    Returns a dictionary containing 5 lookup tables:
      - 'my_to_unrolled': dict[int, list] -> {MY_bin: [unrolled_bins]}
      - 'mx_to_unrolled': dict[int, list] -> {MX_bin: [unrolled_bins]}
      - 'unrolled_to_my': dict[int, int]  -> {unrolled_bin: MY_bin}
      - 'unrolled_to_mx': dict[int, int]  -> {unrolled_bin: MX_bin}
      - 'unrolled_to_label': dict[int, str] -> {unrolled_bin: "MX..._MY..."}
    """
    
    # Default edges
    if mx_bin_edges is None:
        mx_bin_edges = np.array([250,300,375,450,550,675,825,1000,1250,1600,2000,2500,3000,4000,5000])
    if my_bin_edges is None:
        my_bin_edges = np.array([30,40,50,60,75,90,110,135,165,200,250,300,375,450,550,675,825,1000,1250,1600,2000,2500,3000,4000])

    n_my_bins = len(my_bin_edges) - 1
    n_mx_bins = len(mx_bin_edges) - 1

    # Initialize the dictionaries
    my_to_unrolled = {i: [] for i in range(1, n_my_bins + 1)}
    mx_to_unrolled = {i: [] for i in range(1, n_mx_bins + 1)}
    unrolled_to_my = {}
    unrolled_to_mx = {}
    unrolled_to_label = {}
    
    current_1d_bin = 1

    # Apply physical unrolled bin logic
    for my_idx in range(n_my_bins):
        for mx_idx in range(n_mx_bins):
            mx_upper = mx_bin_edges[mx_idx + 1]
            mx_lower = mx_bin_edges[mx_idx]
            my_lower = my_bin_edges[my_idx]

            # The physical cut
            if mx_upper > (my_lower + 125):
                my_bin_1d = my_idx + 1  # 1-based ROOT index
                mx_bin_1d = mx_idx + 1  # 1-based ROOT index
                
                # Forward mappings (1D to Unrolled)
                my_to_unrolled[my_bin_1d].append(current_1d_bin)
                mx_to_unrolled[mx_bin_1d].append(current_1d_bin)
                
                # Reverse mappings (Unrolled to 1D)
                unrolled_to_my[current_1d_bin] = my_bin_1d
                unrolled_to_mx[current_1d_bin] = mx_bin_1d
                unrolled_to_label[current_1d_bin] = f"MX{int(mx_lower)}_MY{int(my_lower)}"
                
                current_1d_bin += 1

    # Return everything bundled in one master dictionary
    return {
        "my_to_unrolled": my_to_unrolled,
        "mx_to_unrolled": mx_to_unrolled,
        "unrolled_to_my": unrolled_to_my,
        "unrolled_to_mx": unrolled_to_mx,
        "unrolled_to_label": unrolled_to_label
    }