import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import sys
import os
import argparse

import numpy as np
import uproot
import vector
import matplotlib.pyplot as plt
import mplhep as hep
import ROOT

# Import tools from your fold_functions_ptcut
sys.path.append("/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN") 
from fold_functions_ptcut import build_binning_map, get_lumi, get_label_name

ROOT.gErrorIgnoreLevel = ROOT.kWarning

parser = argparse.ArgumentParser(description="Plotting Pre-Training Data Distributions Across Years")
parser.add_argument('--YEAR_NUM', default="2024", type=str, help="Numerator era (e.g., 2024)")
parser.add_argument('--YEAR_DEN', default="2025", type=str, help="Denominator era (e.g., 2025)")
parser.add_argument('--isMC', default=0, type=int, help="MC or Data? Data by default.")
parser.add_argument('--isBalanceClass', default=0, type=int, help="Balance class? (0 for plotting raw shapes)")
parser.add_argument('--isScaling', default=1, type=int, help="Standard Scaling")

args, unknown = parser.parse_known_args()

binning_map = build_binning_map(njets=4)
# Use numerator year's lumi for the top label, or note that it's a comparison
data_lumi = get_lumi(args.YEAR_NUM) 

def calculate_chi2(h_obs, h_exp, err_obs, err_exp):
    """
    Calculate chi-squared between observed and expected arrays.
    Treating the two years as independent samples with summed variance.
    """
    chi2 = 0.0
    ndf = 0
    if not (len(h_obs) == len(h_exp) == len(err_obs) == len(err_exp)):
        raise ValueError("All input arrays must have the same length.")

    for obs, exp, e_obs, e_exp in zip(h_obs, h_exp, err_obs, err_exp):
        variance = (e_obs ** 2) + (e_exp ** 2)
        if variance > 0:
            chi2 += ((obs - exp) ** 2) / variance
            ndf += 1
    chi2_per_ndf = chi2 / ndf if ndf > 0 else float('inf')

    return chi2_per_ndf, chi2, ndf

def processing(file_list, args):
    """
    Process data from root files. Returns the variable map and region masks.
    """
    njets = 4
    columns = [
        'JetAK4_btag_B_WP_1', 'JetAK4_btag_B_WP_2', 'JetAK4_btag_B_WP_3', 'JetAK4_btag_B_WP_4',
        'JetAK4_pt_1', 'JetAK4_pt_2', 'JetAK4_pt_3', 'JetAK4_pt_4', 
        'JetAK4_eta_1', 'JetAK4_eta_2', 'JetAK4_eta_3', 'JetAK4_eta_4', 
        'JetAK4_phi_1', 'JetAK4_phi_2', 'JetAK4_phi_3', 'JetAK4_phi_4', 
        'JetAK4_mass_1', 'JetAK4_mass_2', 'JetAK4_mass_3', 'JetAK4_mass_4',
        'Hcand_mass', 'Ycand_mass', 'njets_add', 'HT_add'
    ]
    if args.isMC == 1:
        columns.append("Event_weight")

    input_file = uproot.open(file_list[0])
    tree = input_file["Tree_JetInfo"]
    tree_arr = tree.arrays(columns, library="np")

    wp1 = tree_arr["JetAK4_btag_B_WP_1"]
    wp2 = tree_arr["JetAK4_btag_B_WP_2"]
    wp3 = tree_arr["JetAK4_btag_B_WP_3"]
    wp4 = tree_arr["JetAK4_btag_B_WP_4"]

    # Off-Mass Definition
    H_mass = tree_arr["Hcand_mass"]
    min_mask = (H_mass > 50) & (H_mass < 300)
    common_mask = ((H_mass < 90) | (H_mass > 150)) & min_mask
    
    pt_cut_mask = (tree_arr["JetAK4_pt_1"] > 50) & (tree_arr["JetAK4_pt_2"] > 50) & \
                  (tree_arr["JetAK4_pt_3"] > 50) & (tree_arr["JetAK4_pt_4"] > 50)
                  
    # overall_mask = common_mask & pt_cut_mask
    overall_mask = pt_cut_mask

    # 4b, 3b, 2b definitions
    mask_4b = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 3) & (wp4 >= 2) & overall_mask
    mask_3b = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 2) & (wp4 < 2) & overall_mask
    mask_2b = (wp1 >= 3) & (wp2 >= 3) & (wp3 < 2) &  (wp4 < 2) & overall_mask

    idx_4b = np.where(mask_4b)[0]
    idx_3b = np.where(mask_3b)[0]
    idx_2b = np.where(mask_2b)[0]
    all_idx = np.concatenate([idx_4b, idx_3b, idx_2b])
    
    region_flag = np.concatenate([
        np.full(len(idx_4b), 4), 
        np.full(len(idx_3b), 3), 
        np.full(len(idx_2b), 2)
    ])

    combined_tree = {
        "region": region_flag.astype(np.int32),
        **{key: val[all_idx] for key, val in tree_arr.items()}
    }

    jets = vector.arr({
        "pt":   np.stack([combined_tree[f"JetAK4_pt_{i+1}"]   for i in range(njets)], axis=1),
        "eta":  np.stack([combined_tree[f"JetAK4_eta_{i+1}"]  for i in range(njets)], axis=1),
        "phi":  np.stack([combined_tree[f"JetAK4_phi_{i+1}"]  for i in range(njets)], axis=1),
        "mass": np.stack([combined_tree[f"JetAK4_mass_{i+1}"] for i in range(njets)], axis=1),
    })

    # Delta R Logic
    dR1_c1, dR2_c1 = jets[:, 0].deltaR(jets[:, 1]), jets[:, 2].deltaR(jets[:, 3])
    dR1_c2, dR2_c2 = jets[:, 0].deltaR(jets[:, 2]), jets[:, 1].deltaR(jets[:, 3])
    dR1_c3, dR2_c3 = jets[:, 0].deltaR(jets[:, 3]), jets[:, 1].deltaR(jets[:, 2])
    
    min_c1 = np.minimum(dR1_c1, dR2_c1)
    min_c2 = np.minimum(dR1_c2, dR2_c2)
    min_c3 = np.minimum(dR1_c3, dR2_c3)
    
    best_comb_idx = np.argmin(np.stack([min_c1, min_c2, min_c3], axis=1), axis=1)
    row_indices = np.arange(len(all_idx))
    
    dR1_arr = np.stack([dR1_c1, dR1_c2, dR1_c3], axis=1)[row_indices, best_comb_idx]
    dR2_arr = np.stack([dR2_c1, dR2_c2, dR2_c3], axis=1)[row_indices, best_comb_idx]

    if args.isMC == 1:
        event_weights = combined_tree["Event_weight"]
    else:
        event_weights = np.ones(len(all_idx), dtype=float)

    mx = (jets[:, 0] + jets[:, 1] + jets[:, 2] + jets[:, 3]).mass

    # Compile the final mapping of variables for this specific era
    var_map = {
        "MX": mx,
        "MY": combined_tree["Ycand_mass"],
        "MH": combined_tree["Hcand_mass"],
        "n_jets_add": combined_tree["njets_add"],
        "HT_additional": combined_tree["HT_add"],
        "dR1_plot": dR1_arr,
        "dR2_plot": dR2_arr,
        "region": combined_tree["region"],
        "weights": event_weights
    }
    
    # Add pure tree variables (kinematics)
    for col in combined_tree.keys():
        if "JetAK4" in col or "Hcand" in col:
            var_map[col] = combined_tree[col]

    return var_map


# Structure to hold datasets for both years
era_data = {}
for year in [args.YEAR_NUM, args.YEAR_DEN]:
    file_path = f"/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/{year}/Tree_Data_Parking.root"
    print(f"Reading from: {file_path}")
    era_data[year] = processing([file_path], args)

output_dir = f"Plots_EraComparison_{args.YEAR_NUM}_vs_{args.YEAR_DEN}"
os.makedirs(output_dir, exist_ok=True)
hep.style.use("CMS")

print(f"Creating era comparison plots in: {output_dir}")

# vars_to_plot = ["MX", "MY"]
vars_to_plot = ["MX", "MY", "MH", "n_jets_add", "HT_additional", "dR1_plot", "dR2_plot", # "Unrolled_MXMY",
                "JetAK4_pt_1", "JetAK4_pt_2", "JetAK4_pt_3", "JetAK4_pt_4", 
                "JetAK4_eta_1", "JetAK4_eta_2", "JetAK4_eta_3", "JetAK4_eta_4", 
                "JetAK4_phi_1", "JetAK4_phi_2", "JetAK4_phi_3", "JetAK4_phi_4", 
                "JetAK4_mass_1", "JetAK4_mass_2", "JetAK4_mass_3", "JetAK4_mass_4",
                "Hcand_1_pt", "Hcand_1_eta", "Hcand_1_phi", 
                "Hcand_2_pt", "Hcand_2_eta", "Hcand_2_phi", 
                "H1_b1b2_deta", "H1_b1b2_dphi", "H1_b1b2_dR",
                "H2_b1b2_deta", "H2_b1b2_dphi", "H2_b1b2_dR",
                "H1H2_pt", "H1H2_eta", "H1H2_phi", 
                "H1H2_deta", "H1H2_dphi", "H1H2_dR",
                "HT_4j"]

for var in vars_to_plot:
    if var not in era_data[args.YEAR_NUM] or var not in era_data[args.YEAR_DEN]:
        continue
        
    fig, (ax, rax) = plt.subplots(
        2, 1, 
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), 
        sharex=True,
        figsize=(10, 10)
    )

    # hep.cms.label("Preliminary", data=(args.isMC==0), lumi=data_lumi, com=13.6, year=f"{args.YEAR_NUM} vs {args.YEAR_DEN}", ax=ax)
    hep.cms.label("Preliminary", data=(args.isMC==0), com=13.6, year=f"{args.YEAR_NUM}vs{args.YEAR_DEN}", ax=ax)
    # Establish Binning based on Numerator (to ensure identical binning across years)
    if var in binning_map:
        edges = np.array(binning_map[var], dtype=float)
    else:
        min_val, max_val = np.percentile(era_data[args.YEAR_NUM][var], [1, 99]) 
        edges = np.linspace(min_val, max_val, 51)
        
    bin_widths = edges[1:] - edges[:-1]
    x_centers = 0.5 * (edges[:-1] + edges[1:])
    
    colors = {4: "orange", 3: "green", 2: "red"}
    labels = {4: "4b", 3: "3b", 2: "2b"}

    # Process and plot each region
    for region_code in [4, 3, 2]:
        color = colors[region_code]
        label = labels[region_code]
        
        # Extract data for this specific region
        mask_num = era_data[args.YEAR_NUM]["region"] == region_code
        mask_den = era_data[args.YEAR_DEN]["region"] == region_code
        
        data_num = era_data[args.YEAR_NUM][var][mask_num]
        w_num = era_data[args.YEAR_NUM]["weights"][mask_num]
        
        data_den = era_data[args.YEAR_DEN][var][mask_den]
        w_den = era_data[args.YEAR_DEN]["weights"][mask_den]

        # Histograms
        y_num, _ = np.histogram(data_num, bins=edges, weights=w_num)
        y_den, _ = np.histogram(data_den, bins=edges, weights=w_den)
        
        err_num = np.sqrt(np.histogram(data_num, bins=edges, weights=w_num**2)[0])
        err_den = np.sqrt(np.histogram(data_den, bins=edges, weights=w_den**2)[0])

        # Normalize to Area = 1.0
        scale_num = (np.sum(y_num) or 1.0) * bin_widths
        scale_den = (np.sum(y_den) or 1.0) * bin_widths

        y_num_norm, err_num_norm = y_num / scale_num, err_num / scale_num
        y_den_norm, err_den_norm = y_den / scale_den, err_den / scale_den

        # Calculate Chi2 between years
        chi2_val, _, _ = calculate_chi2(y_num_norm, y_den_norm, err_num_norm, err_den_norm)

        # Plot Main Densities
        hep.histplot(y_num_norm, bins=edges, ax=ax, color=color, linestyle='-', label=f"{label} {args.YEAR_NUM}")
        hep.histplot(y_den_norm, bins=edges, ax=ax, color=color, linestyle='--', label=f"{label} {args.YEAR_DEN}")

        # Ratio Calculation
        denom_safe = np.where(y_den_norm > 0, y_den_norm, 1e-10)
        ratio = y_num_norm / denom_safe
        
        # Proper Error Propagation for Ratio: R * sqrt((dNum/Num)^2 + (dDen/Den)^2)
        rel_err_num = err_num_norm / np.where(y_num_norm > 0, y_num_norm, 1e-10)
        rel_err_den = err_den_norm / denom_safe
        ratio_err = ratio * np.sqrt(rel_err_num**2 + rel_err_den**2)

        # Plot Ratio
        rax.errorbar(x_centers, ratio, yerr=ratio_err, fmt='o', color=color, 
                     label=rf"{label} ({args.YEAR_NUM}/{args.YEAR_DEN}) $\chi^2/NDF={chi2_val:.2f}$")

    # Axes styling
    if var in ["MY", "MX", "MH"]:
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(bottom=1e-5) 
    else:
        ax.set_ylim(bottom=0.0)

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylabel("Arbitrary Units")

    rax.axhline(1.0, color='black', linestyle='--')
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel(f"{args.YEAR_NUM} / {args.YEAR_DEN}")
    rax.set_xlabel(get_label_name(var))

    # Unified Legends
    handles_ax, labels_ax = ax.get_legend_handles_labels()
    handles_rax, labels_rax = rax.get_legend_handles_labels()
    ax.legend(handles_ax + handles_rax, labels_ax + labels_rax, loc='best', ncol=1, fontsize='x-small')

    # Save outputs
    plt.savefig(f"{output_dir}/{var}_EraComparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/{var}_EraComparison.pdf", bbox_inches="tight")
    plt.close()

print(f"\n[Success] Era shape comparison plots saved to '{output_dir}'.")