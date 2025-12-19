import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

input_file = "quantiles_Combined_Background_Result.root"

output_dir = "Plots_Evaluation_fold5"
os.makedirs(output_dir, exist_ok=True)

log_filename1 = os.path.join(output_dir, "chi2_results.txt")

log_filename2 = os.path.join(output_dir, "ratio_error.txt")

normalize = True

data_lumi = 108.96 
year_label = 2024
ratio_ylim = [0.5, 1.5]
labels = ["4b", "2b", "2b_w"]

ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Style
hep.style.use("CMS")

def calculate_chi2(h_obs, h_exp, h_exp_up, h_exp_dn):
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


def get_hist_with_total_error(file, var_name):
    # Load Hists
    h_nom = file.Get(f"{var_name}_hist_2b_w_mean")
    h_up  = file.Get(f"{var_name}_hist_2b_w_sys_up")
    h_dn  = file.Get(f"{var_name}_hist_2b_w_sys_down")
    h_3T  = file.Get(f"{var_name}_hist_4b_mean") 
    h_2T  = file.Get(f"{var_name}_hist_2b_mean")

    if normalize:
        if h_3T.Integral() > 0:
            h_3T.Scale(1.0 / h_3T.Integral())

        if h_2T.Integral() > 0:
            h_2T.Scale(1.0 / h_2T.Integral())

        if h_nom.Integral() > 0:
            scale_factor = 1.0 / h_nom.Integral()

            h_nom.Scale(scale_factor)
            h_up.Scale(scale_factor)
            h_dn.Scale(scale_factor)

    chi2, ndf = calculate_chi2(h_3T, h_nom, h_up, h_dn)
    chi2_ndf_func = chi2 / ndf if ndf > 0 else 0

    chi2_ndf_root = get_chi2_root_method(h_3T, h_nom, h_up, h_dn)

    chi2_ndf_without = h_3T.Chi2Test(h_nom, "WW CHI2/NDF")
    result_line = (f"{var_name}: Chi2/NDF (w/ total err) = {chi2_ndf_func:.3f} | "
                     f"Chi2/NDF (ROOT method) = {chi2_ndf_root:.3f} | "
                        f"Chi2/NDF (w/o sys err) = {chi2_ndf_without:.3f}")
    
    f_log1.write(result_line + "\n") # Write to file

    chi2_2b = h_3T.Chi2Test(h_2T, "WW CHI2/NDF")

    # Extract Bin Contents (y)
    n = h_nom.GetNbinsX()
    y_nom = np.array([h_nom.GetBinContent(i) for i in range(1, n+1)])
    y_up  = np.array([h_up.GetBinContent(i)  for i in range(1, n+1)])
    y_dn  = np.array([h_dn.GetBinContent(i)  for i in range(1, n+1)])
    y_3T = np.array([h_3T.GetBinContent(i)  for i in range(1, n+1)])
    y_2T = np.array([h_2T.GetBinContent(i)  for i in range(1, n+1)])
    
    edges = np.array([h_nom.GetBinLowEdge(i) for i in range(1, n+2)])
    
    err_stat = np.array([h_nom.GetBinError(i) for i in range(1, n+1)])
    
    err_sys_up   = np.abs(y_up - y_nom)
    err_sys_down = np.abs(y_dn - y_nom)

    err_tot_up   = np.sqrt(err_stat**2 + err_sys_up**2)
    err_tot_down = np.sqrt(err_stat**2 + err_sys_down**2)

    return edges, y_nom, y_3T, y_2T, err_tot_up, err_tot_down, scale_factor, chi2_ndf_func, chi2_2b, err_stat, err_sys_up, err_sys_down

def get_fold_hists(file, var_name, n_folds=5, scale_factor=1.0):
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

def check_h_nomerror_2b_staterror(file, var_name):
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

def get_fold_errors(file, var_name, n_folds=5):
    """Helper to get statistical errors for each fold histogram"""
    fold_errors = []
    ratio_errs = []

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

def calculate_error_from_histograms(file, var_name, n_folds=5):
    """
    Calculate the systematic uncertainty from the spread of fold histograms for each bin. Use the same percentile methods.
    """
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

# --- Main Plotting Loop ---
f = ROOT.TFile(input_file)
if not f:
    print(f"Error: Could not open {input_file}")
    exit()
# vars_to_plot = ["Score"]
#check_h_nomerror_2b_staterror(f, "Score")
vars_to_plot = ["MX", "MY", "MH", "Score", "n_jets_add", "HT_additional", "dR1_plot", "dR2_plot", 
                "JetAK4_pt_1", "JetAK4_pt_2", "JetAK4_pt_3", "JetAK4_pt_4", 
                "JetAK4_eta_1", "JetAK4_eta_2", "JetAK4_eta_3", "JetAK4_eta_4", 
                "JetAK4_phi_1", "JetAK4_phi_2", "JetAK4_phi_3", "JetAK4_phi_4", 
                "JetAK4_mass_1", "JetAK4_mass_2", "JetAK4_mass_3", "JetAK4_mass_4",
                #"JetAK4_add_pt", "JetAK4_add_eta", "JetAK4_add_phi", "JetAK4_add_mass", 
                "Hcand_1_pt", "Hcand_1_eta", "Hcand_1_phi", # "Hcand_1_mass",
                "Hcand_2_pt", "Hcand_2_eta", "Hcand_2_phi", # "Hcand_2_mass",
                "H1_b1b2_deta", "H1_b1b2_dphi", "H1_b1b2_dR",
                "H2_b1b2_deta", "H2_b1b2_dphi", "H2_b1b2_dR",
                "H1H2_pt", "H1H2_eta", "H1H2_phi", 
                "H1H2_deta", "H1H2_dphi", "H1H2_dR",
                "HT_4j"]

with open(log_filename1, "w") as f_log1:
    with open(log_filename2, "w") as f_log2:
        for var in vars_to_plot:

            edges, y_model, y_4b, y_2b, err_up, err_down, scale_factor, chi2_ndf_func, chi2_2b, err_stat, err_sys_up, err_sys_down = get_hist_with_total_error(f, var)

            # Histograms
            #fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}, sharex=True)
            fig, (ax, rax) = plt.subplots(
                2, 1, 
                gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), 
                sharex=True,
                figsize=(10, 10)
            )

            hep.cms.label("Preliminary", data=True, lumi=data_lumi, com=13.6, year=year_label, ax=ax)

            hep.histplot(y_4b, bins=edges, ax=ax, color="orange", label="4b")
            hep.histplot(y_2b, bins=edges, ax=ax, color="red", label="2b")
            hep.histplot(y_model, bins=edges, ax=ax, color="blue", label="2b_w")
            
            band_low = y_model - err_down
            band_high = y_model + err_up

            band_low_step  = np.append(band_low, band_low[-1])
            band_high_step = np.append(band_high, band_high[-1])

            stat_low = y_model - err_stat
            stat_high = y_model + err_stat
            stat_low_step  = np.append(stat_low, stat_low[-1])
            stat_high_step = np.append(stat_high, stat_high[-1])
            
            ax.fill_between(
                edges, 
                band_low_step, 
                band_high_step, 
                step='post', 
                color='gray', 
                alpha=0.15, 
                label="Total Uncertainty")

            ax.fill_between(
                edges,
                stat_low_step,
                stat_high_step,
                step='post',
                color='green',
                alpha=0.2,
                label="Statistical Uncertainty")


            if var in ["MY", "MX", "MH", "JetAK4_pt_1", "JetAK4_pt_2", "JetAK4_pt_3", "JetAK4_pt_4", "HT_additional", "Hcand_1_pt", "Hcand_2_pt", "Hcand_1_mass", "Hcand_2_mass", "HT_4j"]:
                ax.set_yscale("log")

            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylabel("Arbitrary units")# if norm_3T != 1 else "Events")

            x_centers = 0.5 * (edges[:-1] + edges[1:])
            rax.axhline(1.0, color='black', linestyle='--')
            
            denom = np.where(y_model > 0, y_model, 1e-10)
            ratio = y_4b / denom
            
            demon_2b = np.where(y_2b > 0, y_2b, 1e-10)
            r_raw = y_4b / demon_2b

            rel_err_up   = err_up / denom
            rel_err_down = err_down / denom

            # Center the band at 1.0
            r_band_low  = np.append(1.0 - rel_err_down, (1.0 - rel_err_down)[-1])
            r_band_high = np.append(1.0 + rel_err_up,   (1.0 + rel_err_up)[-1])

            rel_err_stat_up   = err_stat / denom
            rel_err_stat_down = err_stat / denom

            r_band_stat_low  = np.append(1.0 - rel_err_stat_down, (1.0 - rel_err_stat_down)[-1])
            r_band_stat_high = np.append(1.0 + rel_err_stat_up,   (1.0 + rel_err_stat_up)[-1])

            rel_err_sys_up   = err_sys_up / denom
            rel_err_sys_down = err_sys_down / denom
            
            rax.fill_between(edges, r_band_low, r_band_high, step='post', color='gray', alpha=0.15)
            #rax.errorbar(x_centers, ratio, yerr=np.sqrt(y_4b)/denom, fmt='o', color='blue', alpha=0.5, label="4b/2b_w")
            rax.errorbar(x_centers, r_raw, fmt='o', color='red', label=rf"4b/2b $\frac{{\chi^2}}{{NDF}}={chi2_2b:.2f}$")
            rax.errorbar(x_centers, ratio, fmt='o', color='blue', label=rf"4b/2b_w $\frac{{\chi^2}}{{NDF}}={chi2_ndf_func:.2f}$")

            rax.fill_between(edges, r_band_stat_low, r_band_stat_high, step='post', color='green', alpha=0.2)

            
            rax.set_ylim(0.5, 1.5)
            rax.set_ylabel("Ratio")
            rax.set_xlabel(var)

            handles_ax, labels_ax = ax.get_legend_handles_labels()
            handles_rax, labels_rax = rax.get_legend_handles_labels()
            ax.legend(handles_ax + handles_rax, labels_ax + labels_rax, loc='best', ncol=2, fontsize='small')


            plt.savefig(f"{output_dir}/{var}_evaluation.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{output_dir}/{var}_evaluation.pdf", bbox_inches="tight")
            #print(f"  -> Saved plots to {output_dir}")
            plt.close()

            if var == "Score":
                #print("  -> Generating extra 5-fold comparison plot for Score...")
                
                # Load the individual fold data
                folds_y = get_fold_hists(f, var, n_folds=5, scale_factor=scale_factor)
                
                # Plot 5-fold comparison
                fig5, (ax5, rax5) = plt.subplots(
                    2, 1, 
                    gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), 
                    sharex=True,
                    figsize=(10, 10)
                )
                
                hep.cms.label("Preliminary", data=True, lumi=data_lumi, com=13.6, year=year_label, ax=ax5)


                hep.histplot(y_4b, bins=edges, ax=ax5, color="orange",linestyle='-', label="4b")
                hep.histplot(y_2b, bins=edges, ax=ax5, color="red", linestyle='-', label="2b")
                hep.histplot(y_model, bins=edges, ax=ax5, color="blue", linestyle='-', linewidth=2, label="2b_w Mean")
            
                for i, y_fold in enumerate(folds_y):
                    hep.histplot(y_fold, bins=edges, ax=ax5, color="blue", linewidth=1, linestyle='--', alpha=0.3, label=f"2b_w Fold {i+1}")

                # Add Uncertainty Band (Mean based)
                ax5.fill_between(edges, band_low_step, band_high_step, step='post', color='gray', alpha=0.15, label="Mean Uncertainty")

                ax5.set_xlim(edges[0], edges[-1])
                ax5.set_ylabel("Arbitrary units")

                rax5.axhline(1.0, color='black', linestyle='--')
                
                rax5.fill_between(edges, r_band_low, r_band_high, step='post', color='blue', alpha=0.15)
                
                for i, y_fold in enumerate(folds_y):
                    denom_f = np.where(y_fold > 0, y_fold, 1e-10)
                    ratio_f = y_4b / denom_f
                    
                    hep.histplot(ratio_f, bins=edges, ax=rax5, color="blue", linewidth=0.8, linestyle='--', alpha=0.6, histtype='step')

                demon = np.where(y_model > 0, y_model, 1e-10)
                ratio = y_4b / demon
                #rax5.errorbar(x_centers, ratio, yerr=np.sqrt(y_4b)/denom, fmt='o', color='blue', markersize=4, label="4b/2b_w Mean")
                rax5.errorbar(x_centers, ratio, fmt='o', color='blue', markersize=4, label="4b/2b_w Mean")

                rax5.set_ylim(0.5, 1.5)
                rax5.set_ylabel("Ratio")
                rax5.set_xlabel("Score")

                handles_ax5, labels_ax5 = ax5.get_legend_handles_labels()
                handles_rax5, labels_rax5 = rax5.get_legend_handles_labels()
                new_handles = []
                new_labels = []
                
                for h, l in zip(handles_ax5, labels_ax5):
                    if "Fold" not in l:
                        new_handles.append(h)
                        new_labels.append(l)

                ax5.legend(new_handles + handles_rax5, new_labels + labels_rax5, loc='best', ncol=2, fontsize='small')
                
                
                plt.savefig(f"{output_dir}/{var}_5Fold_Comparison.png", dpi=300, bbox_inches="tight")
                plt.savefig(f"{output_dir}/{var}_5Fold_Comparison.pdf", bbox_inches="tight")
                #print(f"  -> Saved plots to {output_dir}")
                plt.close()



                # Plot 5-fold comparison with error bands
                fig5, (ax5w, rax5w, rax5ww) = plt.subplots(
                    3, 1,
                    gridspec_kw=dict(height_ratios=[3, 1, 1], hspace=0.1), 
                    sharex=True,
                    figsize=(10, 12.5)
                )
                
                hep.cms.label("Preliminary", data=True, lumi=data_lumi, com=13.6, year=year_label, ax=ax5w)

                hep.histplot(y_model, bins=edges, ax=ax5w, color="blue", linestyle='-', linewidth=2, label="2b_w Mean")
            
                for i, y_fold in enumerate(folds_y):
                    hep.histplot(y_fold, bins=edges, ax=ax5w, color="blue", linewidth=1, linestyle='--', alpha=0.3, label=f"2b_w Fold {i+1}")

                # Histograms
                ax5w.fill_between(edges, band_low_step, band_high_step, step='post', color='gray', alpha=0.15, label="Mean Uncertainty")
                ax5w.fill_between(edges, stat_low_step, stat_high_step, step='post', color='green', alpha=0.2, label="Mean Statistical Uncertainty")

                ax5w.set_xlim(edges[0], edges[-1])
                ax5w.set_ylabel("Arbitrary units")




                # Ratio 2: ww

                rax5ww.axhline(1.0, color='blue', linestyle='--', linewidth=3, label="Mean(Events)")
                rax5ww.fill_between(edges, r_band_low, r_band_high, step='post', color='gray', alpha=0.15)
                rax5ww.fill_between(edges, r_band_stat_low, r_band_stat_high, step='post', color='green', alpha=0.2)


                color_list = ["darkred", "chocolate", "darkorange", "olivedrab", "indigo"]
                hatch_list = ['///', '\\\\\\', '|||', '---', 'xxx']
                #plot error band for each model for each fold


                # Calculate uncertainties from per bins
                mean_hist, sys_errs_hist, = calculate_error_from_histograms(f, var, n_folds=5)
                sys_errs_up_hist   = sys_errs_hist
                sys_errs_down_hist = sys_errs_hist
                rel_sys_errs_up_hist   = sys_errs_up_hist / np.where(mean_hist > 0, mean_hist, 1e-10)
                rel_sys_errs_down_hist = sys_errs_down_hist / np.where(mean_hist > 0, mean_hist, 1e-10)
                

                fold_errors_2bw = get_fold_errors(f, var, n_folds=5) # purely statistical errors from each fold



                # check if 2b error same as fold error and save to f_log
        
                err_4b, err_2b, n_bins = check_4b_2b_errors(f, var)
                # for j in range(5):
                #     f_log.write(f"Fold {j+1} errors for {var}:\n")

                #     for i in range(n_bins):
                #         f_log.write(f"Stat errors for {var}:\n")
                #         f_log.write(f"Stat 4b error: {err_4b[i]}, 2b error: {err_2b[i]}\n")

                #         f_log.write(f"Fold {j+1} bin {i+1} error: {fold_errors_2bw[j][i]}\n")
                        
                #         f_log.write(f"2b_w nom error: {rel_err_stat_up[i]}\n")
                #         f_log.write("Systematic error:\n")
                #         f_log.write(f"Sys up: {rel_err_sys_up[i]}, Sys down: {rel_err_sys_down[i]}\n")
                #         f_log.write("-----\n")


                

                for i in range(n_bins):
                    f_log2.write(f"Stat errors for {var}:\n")
                    f_log2.write(f"Stat 4b error: {err_4b[i]}, 2b error: {err_2b[i]}\n")
                    for j in range(5):
                        f_log2.write(f"Fold {j+1} bin {i+1} error: {fold_errors_2bw[j][i]}\n")
                        
                    f_log2.write(f"2b_w nom error: {rel_err_stat_up[i]}\n")
                    f_log2.write("Systematic error:\n")
                    f_log2.write(f"Sys up: {rel_err_sys_up[i]}, Sys down: {rel_err_sys_down[i]}\n")
                    f_log2.write("Systematic error from histograms:\n")
                    f_log2.write(f"Sys up: {rel_sys_errs_up_hist[i]}, Sys down: {rel_sys_errs_down_hist[i]}\n")
                    f_log2.write("-----\n")

                # Ratio plot 1: w
                # assuming stat error is the same for hist and bins

                err_tot_hist_up   = np.sqrt(err_stat**2 + sys_errs_up_hist**2)
                err_tot_hist_down = np.sqrt(err_stat**2 + sys_errs_down_hist**2)

                rel_err_tot_hist_up   = err_tot_hist_up / mean_hist
                rel_err_tot_hist_down = err_tot_hist_down / mean_hist

                r_band_low_hist  = np.append(1.0 - rel_err_tot_hist_down, (1.0 - rel_err_tot_hist_down)[-1])
                r_band_high_hist = np.append(1.0 + rel_err_tot_hist_up,   (1.0 + rel_err_tot_hist_up)[-1])

                rel_err_stat_up_hist   = err_stat / mean_hist
                rel_err_stat_down_hist = err_stat / mean_hist

                r_band_stat_low_hist  = np.append(1.0 - rel_err_stat_down_hist, (1.0 - rel_err_stat_down_hist)[-1])
                r_band_stat_high_hist = np.append(1.0 + rel_err_stat_up_hist,   (1.0 + rel_err_stat_up_hist)[-1])
                #r_band_sys_low  = np.append(1.0 - rel_sys_errs_down_hist, (1.0 - rel_sys_errs_down_hist)[-1])
                #r_band_sys_high = np.append(1.0 + rel_sys_errs_up_hist,   (1.0 + rel_sys_errs_up_hist)[-1])

                rax5w.axhline(1.0, color='magenta', linestyle='--', linewidth=3)
                rax5w.fill_between(edges, r_band_low_hist, r_band_high_hist, step='post', color='gray', alpha=0.15)
                rax5w.fill_between(edges, r_band_stat_low_hist, r_band_stat_high_hist, step='post', color='green', alpha=0.2)



                # for i, fold_err in enumerate(fold_errors_2bw):
                #     rel_fold_err = fold_err #/ np.where(folds_y[i] > 0, folds_y[i], 1e-10)
                #     r_fold_low  = np.append(1.0 - rel_fold_err, (1.0 - rel_fold_err)[-1])
                #     r_fold_high = np.append(1.0 + rel_fold_err, (1.0 + rel_fold_err)[-1])
                    
                #     #rax5ww.fill_between(edges, r_fold_low, r_fold_high, step='post', color=color_list[i], alpha=0.1, label=f"Fold {i+1} Stat. Unc.")    
                #     rax5ww.fill_between(
                #         edges, 
                #         r_fold_low, 
                #         r_fold_high, 
                #         step='post', 
                        
                #         # KEY SETTINGS:
                #         facecolor="none",        # Transparent background (prevents muddy mixing)
                #         edgecolor=color_list[i], # The color applies to the hatch lines
                #         hatch=hatch_list[i],     # Assign distinct pattern
                #         linewidth=0,             # Optional: set to 0 to hide the outer border, or 1 to show it
                #         alpha=0.7,               # Keep alpha high so the thin lines are visible
                        
                #         label=f"Fold {i+1} Stat. Unc."
                #     )
                
                for i, y_fold,  in enumerate(folds_y):
                    #print(f"Fold {i+1} stat errors: {fold_err}")

                    demon_model = np.where(y_model > 0, y_model, 1e-10)
                    ratio_w = y_fold / demon_model

                    ratio_hist = y_fold / mean_hist
                    
                    #hep.histplot(ratio_hist, bins=edges, ax=rax5w, color=color_list[i], linewidth=0.8, linestyle='--', alpha=0.6, histtype='step')
                    #hep.histplot(ratio_f, bins=edges, ax=rax5ww, color="blue", linewidth=0.8, linestyle='--', alpha=0.6, histtype='step')
                    #hep.histplot(ratio_w, bins=edges, ax=rax5ww, color=color_list[i], linewidth=0.8, linestyle='--', alpha=0.6, histtype='step')
                    
                    rax5w.errorbar(x_centers, ratio_hist, fmt='o', color=color_list[i], markersize=4)#, label="Fold/Mean_hist")
                    rax5ww.errorbar(x_centers, ratio_w, fmt='o', color=color_list[i], markersize=4)#, label="Fold/Mean_model")


                #demon = np.where(y_model > 0, y_model, 1e-10)
                #ratio = y_4b / demon
                #rax5w.errorbar(x_centers, ratio, yerr=np.sqrt(y_4b)/denom, fmt='o', color='blue', markersize=4, label="4b/2b_w Mean")
                #rax5w.errorbar(x_centers, ratio, fmt='o', color='blue', markersize=4, label="4b/2b_w Mean")
                

                rax5w.set_ylim(0.5, 1.5)
                rax5w.set_ylabel("Ratio_hist")
                #rax5ww.set_xlabel("Score")


                rax5ww.set_ylabel("Ratio_model")

                rax5ww.set_ylim(0.5, 1.5)
                rax5ww.set_xlabel("Score")


                # Add hist calculation to histogram

                hep.histplot(mean_hist, bins=edges, ax=ax5w, color="magenta", linestyle='-', linewidth=2, label="2b_w Mean_hist")
                band_low_hist = mean_hist - err_tot_hist_down
                band_high_hist = mean_hist + err_tot_hist_up

                band_low_step_hist  = np.append(band_low_hist, band_low_hist[-1])
                band_high_step_hist = np.append(band_high_hist, band_high_hist[-1])
            
                # Histograms
                ax5w.fill_between(edges, band_low_step_hist, band_high_step_hist, step='post', color='pink', alpha=0.3, label="Mean_hist Uncertainty")

                handles_ax5w, labels_ax5w = ax5w.get_legend_handles_labels()
                handles_rax5w, labels_rax5w = rax5w.get_legend_handles_labels()
                new_handles = []
                new_labels = []

                for h, l in zip(handles_ax5w, labels_ax5w):
                    if "Fold" not in l:
                        new_handles.append(h)
                        new_labels.append(l)

                ax5w.legend(new_handles + handles_rax5w, new_labels + labels_rax5w, loc='best', ncol=2, fontsize='small')
                
                
                plt.savefig(f"{output_dir}/{var}_5Fold_Comparison_w.png", dpi=300, bbox_inches="tight")
                plt.savefig(f"{output_dir}/{var}_5Fold_Comparison_w.pdf", bbox_inches="tight")
                #print(f"  -> Saved plots to {output_dir}")
                plt.close()

print(f"  -> Saved plots to {output_dir}")
f.Close()