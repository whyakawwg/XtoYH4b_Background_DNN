import os
import json
import argparse
import uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# def error_bands(y, err):
#     """Returns the lower and upper bounds for the uncertainty band."""
#     return y - err, y + err

def variable_comb(variable):
    if variable == "MX":
        return 2
    elif variable == "MY":
        return 3
    elif variable == "MX_MY":
        return 1


def get_histogram_data(file_path, directory, hist_name):
    """Safely retrieves values, edges, and errors from an uproot file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input ROOT file not found: {file_path}")
        
    with uproot.open(file_path) as root_file:
        if directory is None:
            if hist_name not in root_file:
                raise KeyError(f"Histogram '{hist_name}' not found in {file_path}")
            full_key = hist_name
        else:
            if directory not in root_file:
                raise KeyError(f"Directory '{directory}' not found in {file_path}")
            dir_obj = root_file[directory]
            full_key = f"{directory}/{hist_name}"
        
        if full_key not in root_file:
            matched_key = None
            for k in dir_obj.keys(cycle=False):
                if k.startswith(hist_name):
                    matched_key = f"{directory}/{k}"
                    break
            if matched_key:
                full_key = matched_key
            else:
                raise KeyError(f"Histogram starting with '{hist_name}' not found in '{directory}'")
        
        hist = root_file[full_key]
        values = hist.values()
        edges = hist.axis().edges()
        # errors = hist.errors()
        
    return values, edges# , errors

def convert_to_events_per_gev(y, edges, err=None):
    """Divides bin contents by bin widths to ensure multi-binning compatibility."""
    widths = np.diff(edges)
    y_gev = y / widths
    if err is not None:
        err_gev = err / widths
        return y_gev, err_gev
    return y_gev

def plot_overlay(args, variable="MX"):
    # Fixed production input paths
    base_group_dir = "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples"
    file_background = base_group_dir + f"/BackgroundEstimation/{args.YEAR}/Output_Background_{args.YEAR}.root"
    file_MX600_MY150 = base_group_dir + f"/Histograms/{args.YEAR}/Histogram_NMSSM-XtoYHto4B_Par-MX-600-MY-150_TuneCP5_13p6TeV_madgraph-pythia8.root"
    file_MX1000_MY300 = base_group_dir + f"/Histograms/{args.YEAR}/Histogram_NMSSM-XtoYHto4B_Par-MX-1000-MY-300_TuneCP5_13p6TeV_madgraph-pythia8.root"
    file_MX900_MY100 = base_group_dir + f"/Histograms/{args.YEAR}/Histogram_NMSSM-XtoYHto4B_Par-MX-1800-MY-600_TuneCP5_13p6TeV_madgraph-pythia8.root"
    file_HH = base_group_dir + f"/Histograms/{args.YEAR}/Output_DoubleH.root"
    file_H = base_group_dir + f"/Histograms/{args.YEAR}/Output_SingleH.root"
    
    comb_numb = variable_comb(variable)

    hist_name = f"h_{variable}_Comb_3_3_3_2_Inclusive_mHcut"
    if variable == "MX_MY":
        hist_name_sig = "h_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut"
    else:
        hist_name_sig = hist_name
    
    y_bkg,  edges_bkg  = get_histogram_data(file_background, hist_name, "Inclusive_Bkg")
    y_sig1, edges_sig1 = get_histogram_data(file_MX600_MY150, None, hist_name_sig)
    y_sig2, edges_sig2 = get_histogram_data(file_MX1000_MY300, None, hist_name_sig)
    y_sig3, edges_sig3 = get_histogram_data(file_MX900_MY100, None, hist_name_sig)
    y_HH,   edges_HH   = get_histogram_data(file_HH, None, hist_name)
    y_H,    edges_H    = get_histogram_data(file_H, None, hist_name)

    # scale the signal
    y_sig1 *= data_lumi_scale 
    y_sig2 *= data_lumi_scale
    y_sig3 *= data_lumi_scale

    # Convert to (Events / GeV)
    y_bkg_gev= convert_to_events_per_gev(y_bkg, edges_bkg)
    y_sig1_gev = convert_to_events_per_gev(y_sig1, edges_sig1)
    y_sig2_gev = convert_to_events_per_gev(y_sig2, edges_sig2)
    y_sig3_gev = convert_to_events_per_gev(y_sig3, edges_sig3)
    y_HH_gev   = convert_to_events_per_gev(y_HH, edges_HH)
    y_H_gev    = convert_to_events_per_gev(y_H, edges_H)

    y_sig1_gev *= 10.0
    y_sig2_gev *= 10.0
    y_sig3_gev *= 10.0

    y_HH_gev *= 10000.0
    y_H_gev *= 10000.0


    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Render official CMS branding header
    hep.cms.label("Preliminary", data=True, lumi=data_lumi, com=13.6, year=int(args.YEAR), ax=ax)

    hep.histplot(y_bkg_gev,  bins=edges_bkg, ax=ax, color="black", lw=2.5, label="Background")
    hep.histplot(y_sig1_gev, bins=edges_sig1, ax=ax, color="crimson", lw=2.0, label=r"$m_{X}=600\text{ GeV}, m_{Y}=150\text{ GeV}$")#\ (\times 10)$")
    hep.histplot(y_sig2_gev, bins=edges_sig2, ax=ax, color="royalblue", lw=2.0, label=r"$m_{X}=1000\text{ GeV}, m_{Y}=300\text{ GeV}$")#\ (\times 10)$")
    hep.histplot(y_sig3_gev, bins=edges_sig3, ax=ax, color="forestgreen", lw=2.0, label=r"$m_{X}=1800\text{ GeV}, m_{Y}=600\text{ GeV}$")#\ (\times 10)$")
    if variable == "MX" or variable == "MY":

        hep.histplot(y_HH_gev,   bins=edges_HH, ax=ax, color="orange", lw=2.0, label=r"Double H $(\times 10^4)$")
        hep.histplot(y_H_gev,    bins=edges_H, ax=ax, color="purple", lw=2.0, label=r"Single H $(\times 10^4)$")

    # band_low, band_high = error_bands(y_bkg_gev, err_bkg_gev)
    # band_low_padded = np.append(band_low, band_low[-1])
    # band_high_padded = np.append(band_high, band_high[-1])
    
    # ax.fill_between(
    #     edges_bkg, band_low_padded, band_high_padded, 
    #     step="post", color="gray", alpha=0.3, label="Bkg Uncertainty"
    # )

    if args.logX and (variable == "MX" or variable == "MY"):
        ax.set_xscale("log")
        # if variable == "MX":
        #     # explicit_ticks = [400, 600, 650, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000] 
        #     explicit_ticks = [400, 600, 900, 1000, 2000] 

        # elif variable == "MY":
        #     # explicit_ticks = [60, 70, 80, 90, 95, 100, 125, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800]
        #     explicit_ticks = [60, 90, 100, 125, 150, 400, 1000]
        # ax.set_xticks(explicit_ticks)
        # ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    if variable == "MX_MY":
        # ax.set_yscale("log")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3) 
        # ax.set_ylim(0, 100000) 
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3) 
    # ax.set_xlim(edges_bkg[0], edges_bkg[-1])
    if variable == "MX":
        ax.set_xlim(edges_bkg[0], 2500)
    if variable == "MY":
        ax.set_xlim(edges_bkg[0], 1000)
    

    if variable == "MX":
        sub_script = "$m_{X}$ [GeV]"
    elif variable == "MY":
        sub_script = "$m_{Y}$ [GeV]"
    if variable == "MX_MY":
        sub_script = "Unrolled $m_{X}$-$m_{Y}$ Bin Index"
    ax.set_xlabel(f"{sub_script}", fontsize='small')
    if variable == "MX_MY":
        ax.set_ylabel("Events", fontsize='small')
    else:
        ax.set_ylabel("Events / GeV", fontsize='small')
    
    ax.legend(loc="best", ncol=1, fontsize=16, frameon=False)

    output_dirname = f"Signal_Bkg_Plots/{args.YEAR}"
    os.makedirs(output_dirname, exist_ok=True)
    
    out_base = f"{output_dirname}/Sig_Bkg_{variable}_{args.YEAR}_PhysicalSpacing"
    
    # if variable == "MX_MY":
    #     out_base += "_noscale"
    if args.logX and (variable == "MX" or variable == "MY"):
        out_base += "_logX"

        
    plt.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    # plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
    print(f"Plots saved to:\n -> {out_base}.png\n -> {out_base}.pdf")
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Absolute Yield Overlay Engine")
    parser.add_argument("--YEAR", type=str, required=True, help="Data era string (e.g., 2022, 2023, 2024)")
    # parser.add_argument("--TrainRegion", type=str, required=True, choices=["3b", "4b"], help="Training classification sideband region")
    parser.add_argument("--logX", action="store_true", help="Set horizontal scale axis to logarithmic scaling")
    
    args = parser.parse_args()
    if args.YEAR=="2023BPiX":
        data_lumi = 9.45
        data_lumi_scale = 9.45
    elif args.YEAR=="2023":
        data_lumi = 11.2
        data_lumi_scale = 11.24
    elif args.YEAR=="2022EE":
        data_lumi = 22.1
        data_lumi_scale = 22.1
    elif args.YEAR=="2024":
        data_lumi = 109
        data_lumi_scale = 108.96
    elif args.YEAR=="2025":
        data_lumi = 111
        data_lumi_scale = 110.73
    elif args.YEAR=="2022":
        data_lumi = 7.98
        data_lumi_scale = 7.98
    # else:
    #     data_lumi = 111+109
    #     data_lumi_scale = 110.73+108.96

    # plot_overlay(args, variable="MX")
    # plot_overlay(args, variable="MY")
    plot_overlay(args, variable="MX_MY")