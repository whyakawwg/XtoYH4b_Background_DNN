import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import sys
sys.path.append("/data/dust/user/wanghaoy/XtoYH4b/work_scripts")
# import fold_functions
# from fold_functions import get_hist_with_total_error, get_fold_hists, check_h_nomerror_2b_staterror, get_fold_errors, check_4b_2b_errors, calculate_error_from_histograms
import fold_functions_ptcut
from fold_functions_ptcut import get_hist_with_total_error, get_fold_hists, check_h_nomerror_2b_staterror, get_fold_errors, check_4b_2b_errors, calculate_error_from_histograms

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import argparse
import os

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

parser.add_argument('--Nfold', default=None, type=int, help = "Specify fold number for training or testing.")

args = parser.parse_args()

n_folds = args.Nfold
year_label = args.YEAR

input_file = "quantiles_Combined_Background_Result.root"

output_dir = f"Plots_Evaluation_fold{n_folds}"
os.makedirs(output_dir, exist_ok=True)

log_filename1 = os.path.join(output_dir, "chi2_results.txt")

log_filename2 = os.path.join(output_dir, "ratio_error.txt")

normalize = True

data_lumi = 109 
ratio_ylim = [0.5, 1.5]
# labels = ["4b", "2b", "2b_w"]
# labels = ["3b", "2b", "2b_w"]
labels = [f"{args.TrainRegion}", "2b", "2b_w"]


ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Style
hep.style.use("CMS")

# --- Main Plotting Loop ---
f = ROOT.TFile(input_file)
if not f:
    print(f"Error: Could not open {input_file}")
    exit()
#vars_to_plot = ["Score"]
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

            edges, y_model, y_4b, y_2b, err_tot, scale_factor, chi2_val, chi2_2b, err_stat, err_sys = get_hist_with_total_error(f, var, n_folds, normalize=normalize)

            result_line = (f"{var}: Chi2/NDF (w/ total err) = {chi2_val:.3f} | ")
            f_log1.write(result_line + "\n")

            fig, (ax, rax) = plt.subplots(
                2, 1, 
                gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), 
                sharex=True,
                figsize=(10, 10)
            )

            hep.cms.label("Preliminary", data=True, lumi=data_lumi, com=13.6, year=year_label, ax=ax)

            hep.histplot(y_4b, bins=edges, ax=ax, color="orange", label=labels[0])
            hep.histplot(y_2b, bins=edges, ax=ax, color="red", label=labels[1])
            hep.histplot(y_model, bins=edges, ax=ax, color="blue", label=labels[2])
            
            band_low = y_model - err_tot
            band_high = y_model + err_tot

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
                alpha=0.3, 
                label="Total Uncertainty")

            ax.fill_between(
                edges,
                stat_low_step,
                stat_high_step,
                step='post',
                #color='green',
                facecolor='none',
                edgecolor='green',
                hatch='////',
                alpha=0.5,
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

            rel_err_tot   = err_tot / denom

            # Center the band at 1.0
            r_band_low  = np.append(1.0 - rel_err_tot, (1.0 - rel_err_tot)[-1])
            r_band_high = np.append(1.0 + rel_err_tot,   (1.0 + rel_err_tot)[-1])

            rel_err_stat   = err_stat / denom

            r_band_stat_low  = np.append(1.0 - rel_err_stat, (1.0 - rel_err_stat)[-1])
            r_band_stat_high = np.append(1.0 + rel_err_stat,   (1.0 + rel_err_stat)[-1])

            rel_err_sys   = err_sys / denom

            rax.fill_between(edges, r_band_low, r_band_high, step='post', color='gray', alpha=0.3)# , label="Total Uncertainty")
            rax.errorbar(x_centers, r_raw, fmt='o', color='red', label=rf"{labels[0]}/{labels[1]} $\frac{{\chi^2}}{{NDF}}={chi2_2b:.2f}$")
            rax.errorbar(x_centers, ratio, fmt='o', color='blue', label=rf"{labels[0]}/{labels[2]} $\frac{{\chi^2}}{{NDF}}={chi2_val:.2f}$")
            rax.fill_between(edges, r_band_stat_low, r_band_stat_high, step='post', facecolor='none', edgecolor='green', hatch='////', alpha=0.5)# , label="Statistical Uncertainty")

            rax.set_ylim(0.5, 1.5)
            rax.set_ylabel("Ratio")
            rax.set_xlabel(var)

            handles_ax, labels_ax = ax.get_legend_handles_labels()
            handles_rax, labels_rax = rax.get_legend_handles_labels()
            ax.legend(handles_ax + handles_rax, labels_ax + labels_rax, loc='best', ncol=1, fontsize='x-small')

            plt.savefig(f"{output_dir}/{var}_{args.TestRegion}_evaluation.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{output_dir}/{var}_{args.TestRegion}_evaluation.pdf", bbox_inches="tight")
            #print(f"  -> Saved plots to {output_dir}")
            plt.close()


            if var == "Score":
                folds_y = get_fold_hists(f, var, n_folds, scale_factor=scale_factor)
                fig5, (ax5w, rax5w) = plt.subplots(
                    2, 1,
                    gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), 
                    sharex=True,
                    figsize=(10, 10)
                )
                
                if n_folds == 10:   
                    color_list = ["darkred", "chocolate", "darkorange", "olivedrab", "indigo", "teal", "slateblue", "crimson", "darkgreen", "saddlebrown"]
                elif n_folds == 5:
                    color_list = ["darkred", "darkorange", "olivedrab", "indigo", "teal"]

                hep.cms.label("Preliminary", data=True, lumi=data_lumi, com=13.6, year=year_label, ax=ax5w)

                hep.histplot(y_model, bins=edges, ax=ax5w, color="blue", linestyle='-', linewidth=2, label="<Weighted 2b>")
            
                for i, y_fold in enumerate(folds_y):
                    hep.histplot(y_fold, bins=edges, ax=ax5w, color=color_list[i], linewidth=1, linestyle='--', alpha=0.7) # , label=f"2b_w Fold {i+1}")

                # Histograms
                ax5w.fill_between(edges, band_low_step, band_high_step, step='post', color='gray', alpha=0.3, label="Total Uncertainty")
                ax5w.fill_between(edges, stat_low_step, stat_high_step, step='post', facecolor='none', edgecolor='green', hatch='////', alpha=0.5, label="Statistical Uncertainty")
                ax5w.set_xlim(edges[0], edges[-1])
                ax5w.set_ylabel("Arbitrary units")

                # Calculate uncertainties from per bins
                mean_hist, sys_errs_hist, = calculate_error_from_histograms(f, var, n_folds)
                rel_sys_errs_hist   = sys_errs_hist / np.where(mean_hist > 0, mean_hist, 1e-10)

                fold_errors_2bw = get_fold_errors(f, var, n_folds) # purely statistical errors from each fold

                # check if 2b error same as fold error and save to f_log
        
                err_4b, err_2b, n_bins = check_4b_2b_errors(f, var)

                for i in range(n_bins):
                    f_log2.write(f"Stat errors for {var}:\n")
                    f_log2.write(f"Stat {labels[0]} error: {err_4b[i]}, {labels[1]} error: {err_2b[i]}\n")
                    for j in range(5):
                        f_log2.write(f"Fold {j+1} bin {i+1} error: {fold_errors_2bw[j][i]}\n")
                        
                    f_log2.write(f"{labels[2]} nom error: {rel_err_stat[i]}\n")
                    f_log2.write("Systematic error:\n")
                    f_log2.write(f"Sys error from : {rel_err_sys[i]}\n")
                    f_log2.write(f"Sys error from histogram : {rel_sys_errs_hist[i]}\n")
                    f_log2.write("-----\n")

                rax5w.axhline(1.0, color='blue', linestyle='-', linewidth=2)
                rax5w.fill_between(edges, r_band_low, r_band_high, step='post', color='gray', alpha=0.3)
                rax5w.fill_between(edges, r_band_stat_low, r_band_stat_high, step='post',  facecolor='none', edgecolor='green', hatch='////', alpha=0.5)

        
                for i, y_fold,  in enumerate(folds_y):
                    demon_model = np.where(y_model > 0, y_model, 1e-10)
                    ratio_w = y_fold / demon_model
                    #rax5w.errorbar(x_centers, ratio_w, fmt='o', color=color_list[i], markersize=4)#, label="Fold/Mean_model")
                    hep.histplot(ratio_w, bins=edges, ax=rax5w, color=color_list[i], linestyle='--', linewidth=1, alpha=1) # , label=f"Fold {i+1}/Mean")
                rax5w.set_ylim(0.5, 1.5)
                rax5w.set_ylabel("Ratio")
                rax5w.set_xlabel("Score")

                handles_ax5w, labels_ax5w = ax5w.get_legend_handles_labels()
                handles_rax5w, labels_rax5w = rax5w.get_legend_handles_labels()
                new_handles = []
                new_labels = []

                for h, l in zip(handles_ax5w, labels_ax5w):
                    if "Fold" not in l:
                        new_handles.append(h)
                        new_labels.append(l)

                ax5w.legend(new_handles + handles_rax5w, new_labels + labels_rax5w, loc='best', ncol=1, fontsize='x-small')
                
                
                plt.savefig(f"{output_dir}/{var}_{args.TestRegion}_5Fold_Comparison_w.png", dpi=300, bbox_inches="tight")
                plt.savefig(f"{output_dir}/{var}_{args.TestRegion}_5Fold_Comparison_w.pdf", bbox_inches="tight")
                #print(f"  -> Saved plots to {output_dir}")
                plt.close()


print(f"  -> Saved plots to {output_dir}")
f.Close()