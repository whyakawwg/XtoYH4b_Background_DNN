import ROOT
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import argparse
import re
import sys
import math
import matplotlib.lines
import matplotlib.patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import FancyArrowPatch 


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

parser = argparse.ArgumentParser(description="Arguments: YEAR")
parser.add_argument('--YEAR', default="2022", type=str, help="Which era?")
parser.add_argument('--SUFFIX', default="", type=str, help="Suffix for workspace and limit directories")
parser.add_argument('--NoSys', action='store', default=False, type=bool, help = "No systematic uncs?")

args = parser.parse_args()

# Luminosity mapping
lumi_map = {
    "2022": 7.98,
    "2022EE": 22.1, 
    "2023": 11.2,
    "2023BPiX": 9.45,
    "2024": 109.0,
    "2025": 111.0,  
    "combined_2024_2025": 220.0
}
data_lumi = lumi_map.get(args.YEAR)

suffix = args.SUFFIX

def plot1D_Brazilian(group, mx, scenario, y_axis_title, outdir, filename, legend_label):
    """
    Generates a 1D 'Brazilian flag' limit plot as a function of MY for a fixed MX.
    """
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    group = group.sort_values(by="MY")

    my_vals = group["MY"].values
    expected = group["expected"].values
    sigma_1_up = group["sigma_1_up"].values
    sigma_1_dn = group["sigma_1_dn"].values
    sigma_2_up = group["sigma_2_up"].values
    sigma_2_dn = group["sigma_2_dn"].values
    
    if "observed" in group.columns:
        observed = group["observed"].values
    else:
        observed = None

    ax.fill_between(my_vals, sigma_2_dn, sigma_2_up, color="#FFCC00", label="95% expected")
    ax.fill_between(my_vals, sigma_1_dn, sigma_1_up, color="#228b22", label="68% expected")
    ax.plot(my_vals, expected, color="black", linestyle="--", linewidth=2, label="Median expected")

    # if observed is not None:
    #     ax.plot(my_vals, observed, color="black", linestyle="-", marker="o", markersize=4, linewidth=2, label="Observed")

    # Formatting Axes
    ax.set_xlabel("$m_Y$ [GeV]", loc="right")
    ax.set_ylabel(y_axis_title, loc="top")
    ax.set_yscale("log")
    
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    min_y = np.min(sigma_2_dn)
    max_y = np.max(sigma_2_up)
    ax.set_ylim(min_y * 0.5, max_y * 10)

    text_info = f"$m_X = {mx}$ GeV\n{legend_label}"
    ax.text(
        0.05, 0.95,                   
        text_info,                  
        transform=ax.transAxes,       
        ha="left",                  
        va="top",                  
        fontsize=16,
        color="black",
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
    )

    hep.cms.label("Preliminary", data=True, lumi=data_lumi, com=13.6, year=int(args.YEAR), ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right", frameon=False, fontsize=14)

    out_file_name = f"{args.YEAR}_{filename}.png"
    plt.savefig(os.path.join(outdir, out_file_name), dpi=300, bbox_inches="tight")
    plt.close(fig)


# def plot1D_Grid(df_scenario, scenario, y_axis_title, outdir, filename, legend_label):
#     """
#     Generates a compact grid plot containing all available MX mass points for a given scenario.
#     """
#     plt.style.use(hep.style.CMS)
    
#     mx_values = sorted(df_scenario["MX"].unique())
#     n_plots = len(mx_values) - 1
    
#     if n_plots == 0:
#         return

#     ncols = 3
#     nrows = math.ceil(n_plots / ncols)

#     fig, axes = plt.subplots(
#         nrows=nrows, ncols=ncols, 
#         figsize=(8 * ncols, 6 * nrows), 
#         sharex=True, sharey=False, 
#         gridspec_kw={'hspace': 0.1, 'wspace': 0.35} 
#     )
    
#     axes = axes.flatten() if n_plots > 1 else [axes]

#     for idx, mx in enumerate(mx_values[:-1]):
#         ax = axes[idx]
#         group = df_scenario[df_scenario["MX"] == mx].sort_values(by="MY")

#         if len(group) < 2:
#             ax.set_visible(False)
#             continue

#         my_vals = group["MY"].values
#         expected = group["expected"].values
#         sigma_1_up = group["sigma_1_up"].values
#         sigma_1_dn = group["sigma_1_dn"].values
#         sigma_2_up = group["sigma_2_up"].values
#         sigma_2_dn = group["sigma_2_dn"].values
        
#         observed = group["observed"].values if "observed" in group.columns else None

#         ax.fill_between(my_vals, sigma_2_dn, sigma_2_up, color="#FFCC00")
#         ax.fill_between(my_vals, sigma_1_dn, sigma_1_up, color="#228b22")
#         ax.plot(my_vals, expected, color="black", linestyle="--", linewidth=2)
        
#         if observed is not None:
#             ax.plot(my_vals, observed, color="black", linestyle="-", marker="o", markersize=4, linewidth=2)

#         ax.set_yscale("log")
#         ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


#         ax.tick_params(axis="y", labelleft=True)
#         min_y = np.min(sigma_2_dn)
#         max_y = np.max(sigma_2_up)
#         ax.set_ylim(min_y * 0.5, max_y * 15)

#         ax.text(0.05, 0.95, f"$m_X = {mx}$ GeV", transform=ax.transAxes, ha="left", va="top", fontsize=18,
#                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

#     for idx in range(n_plots, len(axes)):
#         axes[idx].set_visible(False)
#         # If the plot is hidden in the bottom row, force the X-axis labels on for the plot above it
#         if idx - ncols >= 0:
#             axes[idx - ncols].tick_params(axis="x", labelbottom=True)

#     hep.cms.text("Preliminary", loc=0, ax=axes[0])
    
#     # Put the Luminosity cleanly over the Top-Right plot (index: ncols - 1)
#     top_right_ax = axes[ncols - 1] if len(axes) >= ncols else axes[-1]
#     hep.cms.lumitext(f"{data_lumi} fb$^{{-1}}$ (13.6 TeV)", ax=top_right_ax)

#     legend_elements = [
#         Line2D([0], [0], color='black', linestyle='--', lw=2, label='Median expected'),
#         Patch(facecolor='#228b22', label='68% expected'),
#         Patch(facecolor='#FFCC00', label='95% expected')
#     ]
#     if "observed" in df_scenario.columns:
#         legend_elements.insert(0, Line2D([0], [0], color='black', linestyle='-', marker='o', label='Observed'))

#     # Shift layout cleanly
#     plt.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.96)
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=28, ncol=4)

#     # X-axis Global arrow
#     arrow_x = FancyArrowPatch((0.12, 0.05), (0.96, 0.05), transform=fig.transFigure, mutation_scale=30, arrowstyle='-|>', color='black', lw=2.5, clip_on=False)
#     fig.patches.append(arrow_x)
#     fig.text(0.54, 0.01, r'$m_{\mathrm{Y}} \text{ [GeV]}$', horizontalalignment='center', verticalalignment='bottom', fontsize=30)

#     # Y-axis Global arrow
#     arrow_y = FancyArrowPatch((0.05, 0.12), (0.05, 0.92), transform=fig.transFigure, mutation_scale=30, arrowstyle='-|>', color='black', lw=2.5, clip_on=False)
#     fig.patches.append(arrow_y)
#     fig.text(0.01, 0.52, y_axis_title, rotation=90, horizontalalignment='left', verticalalignment='center', fontsize=30)

#     out_file_name = f"{args.YEAR}_{filename}.png"
#     plt.savefig(os.path.join(outdir, out_file_name), dpi=300, bbox_inches="tight")
#     plt.close(fig)

# def plot1D_Grid(df_scenario, scenario, y_axis_title, outdir, filename, legend_label):
#     """
#     Generates a compact grid plot containing all available MX mass points for a given scenario.
#     """
#     plt.style.use(hep.style.CMS)
    
#     mx_values = sorted(df_scenario["MX"].unique())
#     n_plots = len(mx_values) - 1 # Excludes the last MX point
    
#     if n_plots == 0:
#         return

#     ncols = 3
#     nrows = math.ceil(n_plots / ncols)

#     # 1. Smaller overall figsize makes the fonts appear relatively LARGER
#     # 2. Large wspace and hspace provide room for individual axes
#     fig, axes = plt.subplots(
#         nrows=nrows, ncols=ncols, 
#         figsize=(18, 5 * nrows), 
#         sharex=False, sharey=False, 
#         gridspec_kw={'hspace': 0.35, 'wspace': 0.35} 
#     )
    
#     axes = axes.flatten() if n_plots > 1 else [axes]

#     for idx, mx in enumerate(mx_values[:-1]):
#         ax = axes[idx]
#         group = df_scenario[df_scenario["MX"] == mx].sort_values(by="MY")

#         if len(group) < 2:
#             ax.set_visible(False)
#             continue

#         my_vals = group["MY"].values
#         expected = group["expected"].values
#         sigma_1_up = group["sigma_1_up"].values
#         sigma_1_dn = group["sigma_1_dn"].values
#         sigma_2_up = group["sigma_2_up"].values
#         sigma_2_dn = group["sigma_2_dn"].values
        
#         observed = group["observed"].values if "observed" in group.columns else None

#         ax.fill_between(my_vals, sigma_2_dn, sigma_2_up, color="#FFCC00")
#         ax.fill_between(my_vals, sigma_1_dn, sigma_1_up, color="#228b22")
#         ax.plot(my_vals, expected, color="black", linestyle="--", linewidth=2)
        
#         if observed is not None:
#             ax.plot(my_vals, observed, color="black", linestyle="-", marker="o", markersize=4, linewidth=2)

#         ax.set_yscale("log")
        
#         # 3. FORCE TICKS ON EVERY SUBPLOT FOR INDEPENDENCE
#         ax.tick_params(axis="both", which="major", labelsize=18, direction="in", top=True, right=True)
#         ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)
        
#         # Show numbers on the left and bottom of EVERY plot
#         ax.tick_params(axis="y", labelleft=True)
#         ax.tick_params(axis="x", labelbottom=True)

#         # 4. SAFE LOG SCALING
#         # Filter out 0 or negative values to prevent log-scale crashes
#         valid_min = sigma_2_dn[sigma_2_dn > 0]
#         min_y = np.nanmin(valid_min) if len(valid_min) > 0 else 0.1
#         max_y = np.nanmax(sigma_2_up)
        
#         # Multiply max_y by 100 to leave a clear gap for the text box
#         ax.set_ylim(min_y * 0.5, max_y * 100)

#         # Add mass text box inside the plot
#         ax.text(0.05, 0.95, f"$m_X = {mx}$ GeV", transform=ax.transAxes, ha="left", va="top", fontsize=20,
#                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))

#     # Hide unused subplots
#     for idx in range(n_plots, len(axes)):
#         axes[idx].set_visible(False)

#     # 5. HEADER PLACEMENT (Larger Fonts)
#     hep.cms.text("Preliminary", loc=0, ax=axes[0], fontsize=28)
    
#     top_right_ax = axes[ncols - 1] if len(axes) >= ncols else axes[-1]
#     hep.cms.lumitext(f"{data_lumi} fb$^{{-1}}$ (13.6 TeV)", ax=top_right_ax, fontsize=24)

#     # 6. GLOBAL LEGEND (Larger Font)
#     legend_elements = [
#         Line2D([0], [0], color='black', linestyle='--', lw=2, label='Median expected'),
#         Patch(facecolor='#228b22', label='68% expected'),
#         Patch(facecolor='#FFCC00', label='95% expected')
#     ]
#     if "observed" in df_scenario.columns:
#         legend_elements.insert(0, Line2D([0], [0], color='black', linestyle='-', marker='o', label='Observed'))

#     # Shift layout cleanly (right=0.95 fixes the right margin cutoff)
#     plt.subplots_adjust(top=0.90, bottom=0.12, left=0.10, right=0.95)
    
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=24, ncol=4)

#     # 7. GLOBAL ARROWS & TEXT (Massive Fonts for Readability)
#     arrow_x = FancyArrowPatch((0.10, 0.05), (0.95, 0.05), transform=fig.transFigure, mutation_scale=30, arrowstyle='-|>', color='black', lw=2.5, clip_on=False)
#     fig.patches.append(arrow_x)
#     fig.text(0.525, 0.01, r'$m_{\mathrm{Y}} \text{ [GeV]}$', horizontalalignment='center', verticalalignment='bottom', fontsize=32)

#     arrow_y = FancyArrowPatch((0.03, 0.12), (0.03, 0.90), transform=fig.transFigure, mutation_scale=30, arrowstyle='-|>', color='black', lw=2.5, clip_on=False)
#     fig.patches.append(arrow_y)
#     fig.text(0.01, 0.51, y_axis_title, rotation=90, horizontalalignment='left', verticalalignment='center', fontsize=32)

#     out_file_name = f"{args.YEAR}_{filename}.png"
#     plt.savefig(os.path.join(outdir, out_file_name), dpi=300, bbox_inches="tight")
#     plt.close(fig)

def plot1D_Grid(df_scenario, scenario, y_axis_title, outdir, filename, legend_label):
    """
    Generates a compact grid plot containing all available MX mass points for a given scenario.
    """
    plt.style.use(hep.style.CMS)
    
    mx_values = sorted(df_scenario["MX"].unique())
    n_plots = len(mx_values) - 1 # Excludes the last MX point
    
    if n_plots == 0:
        return

    ncols = 3
    nrows = math.ceil(n_plots / ncols)

    # Increased width and height slightly to accommodate massive fonts
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, 
        figsize=(25, 6.5 * nrows), 
        sharex=False, sharey=False, 
        gridspec_kw={'hspace': 0.35, 'wspace': 0.35} 
    )
    
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, mx in enumerate(mx_values[:-1]):
        ax = axes[idx]
        group = df_scenario[df_scenario["MX"] == mx].sort_values(by="MY")

        if len(group) < 2:
            ax.set_visible(False)
            continue

        my_vals = group["MY"].values
        expected = group["expected"].values
        sigma_1_up = group["sigma_1_up"].values
        sigma_1_dn = group["sigma_1_dn"].values
        sigma_2_up = group["sigma_2_up"].values
        sigma_2_dn = group["sigma_2_dn"].values
        
        observed = group["observed"].values if "observed" in group.columns else None

        ax.fill_between(my_vals, sigma_2_dn, sigma_2_up, color="#FFCC00")
        ax.fill_between(my_vals, sigma_1_dn, sigma_1_up, color="#228b22")
        ax.plot(my_vals, expected, color="black", linestyle="--", linewidth=2)
        
        # if observed is not None:
        #     ax.plot(my_vals, observed, color="black", linestyle="-", marker="o", markersize=4, linewidth=2)

        ax.set_yscale("log")
        
        # Bigger tick labels
        ax.tick_params(axis="both", which="major", labelsize=22, direction="in", top=True, right=True)
        ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)
        
        ax.tick_params(axis="y", labelleft=True)
        ax.tick_params(axis="x", labelbottom=True)

        # Safe Log Scaling
        valid_min = sigma_2_dn[sigma_2_dn > 0]
        min_y = np.nanmin(valid_min) if len(valid_min) > 0 else 0.1
        max_y = np.nanmax(sigma_2_up)
        
        # ax.set_ylim(min_y * 0.5, max_y * 10)
        # set maximum of 10e3, no minimum to allow for better scaling of the y-axis
        ax.set_ylim(bottom=min_y * 0.5, top=1e4)

        # Bigger internal MX label
        ax.text(0.05, 0.95, f"$m_X = {mx}$ GeV", transform=ax.transAxes, ha="left", va="top", fontsize=35,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    # Bigger Header
    hep.cms.text("Preliminary", loc=0, ax=axes[0], fontsize=42)
    top_right_ax = axes[ncols - 1] if len(axes) >= ncols else axes[-1]
    hep.cms.lumitext(f"{data_lumi} fb$^{{-1}}$ (13.6 TeV)", ax=top_right_ax, fontsize=38)

    # BIGGER LEGEND
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', lw=3, label='Median expected'),
        Patch(facecolor='#228b22', label='68% expected'),
        Patch(facecolor='#FFCC00', label='95% expected')
    ]
    # if "observed" in df_scenario.columns:
    #     legend_elements.insert(0, Line2D([0], [0], color='black', linestyle='-', marker='o', lw=3, markersize=8, label='Observed'))

    # 1. MASSIVE MARGINS: left=0.14 and bottom=0.12 carve out plenty of empty space for the arrows/text
    plt.subplots_adjust(top=0.90, bottom=0.14, left=0.14, right=0.95)
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.54, 0.98), frameon=False, fontsize=40, ncol=4)

    # 2. SEPARATED GLOBAL ARROWS & TEXT
    # X-axis Arrow (Starts at 0.14 to match the plot left edge)
    arrow_x = FancyArrowPatch((0.14, 0.1), (0.95, 0.1), transform=fig.transFigure, mutation_scale=45, arrowstyle='-|>', color='black', lw=3.5, clip_on=False)
    fig.patches.append(arrow_x)
    fig.text(0.545, 0.06, r'$m_{\mathrm{Y}} \text{ [GeV]}$', horizontalalignment='center', verticalalignment='bottom', fontsize=40)

    # Y-axis Arrow (Drawn at x=0.07, safely away from both the plots and the text)
    arrow_y = FancyArrowPatch((0.1, 0.14), (0.1, 0.90), transform=fig.transFigure, mutation_scale=45, arrowstyle='-|>', color='black', lw=3.5, clip_on=False)
    fig.patches.append(arrow_y)
    
    # Y-axis Label (Pushed far left to x=0.02)
    fig.text(0.06, 0.52, y_axis_title, rotation=90, horizontalalignment='left', verticalalignment='center', fontsize=40)

    out_file_name = f"{args.YEAR}_{filename}.png"
    plt.savefig(os.path.join(outdir, out_file_name), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plotLimits(log=False, inDir="", outDir="", shapes=[], legends=[]):
    os.makedirs(outDir, exist_ok=True)
    
    file_list = []
    print("inDir", inDir)
    for template in shapes:
        files = glob.glob(os.path.join(inDir, f"higgsCombine_Comb_{template}*"))
        file_list.extend([f for f in files])

    data = []
    failed_files = []

    for root_file in file_list:
        file_name = os.path.basename(root_file)
        pattern = r"Comb_(\d+)_MX_(\d+)_MY_(\d+)"
        match = re.search(pattern, file_name)
        
        if match:
            scenario, MX, MY = map(int, match.groups())
        else:
            continue

        file = ROOT.TFile.Open(root_file)
        if not file or file.IsZombie():
            print(f"Warning: Could not open {root_file}. Skipping.")
            continue
            
        tree = file.Get("limit")
        if not tree:
            print(f"Warning: No 'limit' tree found in {root_file}. Skipping.")
            continue

        limits = [confidence_level.limit * 1000 for confidence_level in tree] 

        if len(limits) == 0:
            print(f"Warning: No limits found in {root_file}. Skipping.")
            continue

        if len(limits) != 6: 
            expected = limits[-1] 
            failed_files.append(file_name)
        else:
            expected = limits[2]     
            sigma_1_dn = limits[1]   
            sigma_1_up = limits[3]   
            sigma_2_dn = limits[0]   
            sigma_2_up = limits[4]   
            observed = limits[5]     

        data.append([scenario, MX, MY, expected, sigma_1_up, sigma_1_dn, sigma_2_up, sigma_2_dn, observed])

    df_org = pd.DataFrame(data, columns=["scenario", "MX", "MY", "expected","sigma_1_up","sigma_1_dn","sigma_2_up","sigma_2_dn", "observed"])
    
    df = df_org[(df_org["MX"] > 350) & (df_org["MX"] <= 2200)]

    y_axis_title = r"$\sigma(pp \rightarrow X)\mathcal{B}(X \rightarrow YH \rightarrow b\bar{b}b\bar{b})$ [fb]"

    for scenario, group_scen in df.groupby("scenario"):
        legend_idx = scenario - 1 if (scenario - 1) < len(legends) else 0
        current_legend = legends[legend_idx]
        
        for mx, group_mx in group_scen.groupby("MX"):
            if len(group_mx) > 1:
                plot_filename = f"Brazilian_limit_MX_{mx}_Scen_{scenario}"
                plot1D_Brazilian(group_mx, mx, scenario, y_axis_title, outDir, plot_filename, current_legend)
            else:
                print(f"Info: Not enough MY points to plot 1D limit for MX={mx}, Scenario={scenario}")

        grid_filename = f"Grid_Brazilian_limit_Scen_{scenario}"
        plot1D_Grid(group_scen, scenario, y_axis_title, outDir, grid_filename, current_legend)

    return failed_files


if __name__ == "__main__":
    path_dir="/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/CombineResults/bkg/"
    
    if args.NoSys:
        input_dir = os.path.join(path_dir, args.YEAR, "limits_v5_nosys")
        output_dir = os.path.join(os.getcwd(), "MXMY_LimitsPlots", args.YEAR, "plots_nosys")
    else:
        input_dir = os.path.join(path_dir, args.YEAR, f"limits{suffix}")
        output_dir = os.path.join(os.getcwd(), "MXMY_LimitsPlots", args.YEAR, f"plots{suffix}")

    os.makedirs(output_dir, exist_ok=True)
    log_file_name = f"PlotLimits_log_{args.YEAR}{args.SUFFIX}.txt"
    sys.stdout = Logger(os.path.join(output_dir, log_file_name))
    print(f"Logging all output to {log_file_name}")

    templates = [1]
    label_list = [">=T,>=T,>=T,>=M"]

    failed_files = plotLimits(log=False, inDir=input_dir, outDir=output_dir, shapes=templates, legends=label_list)

    if failed_files:
        print(f"\n{len(failed_files)} Failed files (did not contain 5 expected + 1 observed):")
        for f in failed_files:
            print(f" - {f}")

    print("\nPlotting sequence complete.")
    print(f"Plots saved to: {output_dir}")