import warnings

warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import sys

sys.path.append("/data/dust/user/wanghaoy/XtoYH4b/work_scripts")
import fold_functions_ptcut
from fold_functions_ptcut import (
    get_hist_with_total_error,
    get_hist_with_total_error_Mass,
    get_fold_hists,
    get_split_fold_hists,
    get_label_name,
    build_binning_map,
    get_lumi,
    error_bands,
    load_nonclosure_factor,
    make_hist,
    get_all_bin_mappings,
)

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import argparse
import os
import array

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

dir_suffix = ""
NONCLOSURE_FACTORS_FILE = f"{dir_suffix}nonclosure_factors.root"

MX_BIN_EDGES = np.array([250, 300, 375, 450, 550, 675, 825, 1000, 1250, 1600, 2000, 2500, 3000, 4000, 5000])
MY_BIN_EDGES = np.array([30, 40, 50, 60, 75, 90, 110, 135, 165, 200, 250, 300, 375, 450, 550, 675, 825, 1000, 1250, 1600, 2000, 2500, 3000, 4000])
N_MX_BINS = len(MX_BIN_EDGES) - 1  # 14
TARGET_MY_BINS = [6, 7, 8]         # MY bins with lower edges 90, 110, 135

LOG_VARS = [
    "MY", "MX", "MH", "JetAK4_pt_1", "JetAK4_pt_2", "JetAK4_pt_3",
    "JetAK4_pt_4", "HT_additional", "Hcand_1_pt", "Hcand_2_pt",
    "Hcand_1_mass", "Hcand_2_mass", "HT_4j",
]


def _get_hist_safe(tfile, name):
    """Return histogram from TFile, or None if missing / wrong type."""
    h = tfile.Get(name)
    if not h or (isinstance(h, ROOT.TObject) and h.ClassName() == "TObject"):
        return None
    return h


def _make_hist_arrays(h_nom):
    """Return (n_bins, edges, x_centers) for a ROOT TH1."""
    n = h_nom.GetNbinsX()
    edges = np.array(
        [h_nom.GetBinLowEdge(i + 1) for i in range(n)]
        + [h_nom.GetBinLowEdge(n) + h_nom.GetBinWidth(n)]
    )
    return n, edges, 0.5 * (edges[:-1] + edges[1:])


def _th1_to_array(h, n_bins):
    return np.array([h.GetBinContent(i + 1) for i in range(n_bins)])


def _build_th1(name, values, edges_array, err_per_bin=None, epsilon=1e-6):
    """Create a TH1F from a numpy array.  err_per_bin defaults to 1e-6."""
    nbins = len(values)
    h = ROOT.TH1F(name, name, nbins, edges_array)
    for i, v in enumerate(values):
        h.SetBinContent(i + 1, max(float(v), epsilon))
        h.SetBinError(i + 1, err_per_bin[i] if err_per_bin is not None else 1e-6)
    return h


# Non-closure factor
def save_non_closure_factor(
    var_name, edges, ratio_array, out_filename=NONCLOSURE_FACTORS_FILE, exclude_my_bins=None
):
    """Save |ratio-1| as a TH1F to a ROOT file (0 for bins in exclude_my_bins)."""
    if exclude_my_bins is None:
        exclude_my_bins = []

    bins_to_exclude = []
    if exclude_my_bins:
        if var_name == "Unrolled_MXMY":
            maps = get_all_bin_mappings()
            for mb in exclude_my_bins:
                bins_to_exclude.extend(maps["my_to_unrolled"].get(mb, []))
        elif var_name == "MY":
            bins_to_exclude = list(exclude_my_bins)

    f_out = ROOT.TFile(out_filename, "UPDATE")
    nbins = len(edges) - 1
    edges_array = array.array("d", edges)

    h_name = f"{var_name}_nonclosure_factor"
    h = ROOT.TH1F(h_name, f"Non-closure fraction for {var_name}", nbins, edges_array)
    for i in range(nbins):
        root_bin = i + 1

        if root_bin in bins_to_exclude:
            frac = 0.0
        else:
            frac = abs(ratio_array[i] - 1)
            if frac >1.0:
                frac = 1.0
        h.SetBinContent(root_bin, frac)
        h.SetBinError(root_bin, 0)

    f_out.cd()
    h.Write(h_name, ROOT.TObject.kOverwrite)
    f_out.Close()
    print(f"Saved non-closure factors for {var_name} to {out_filename} (excluded bins: {bins_to_exclude})")


# Plot the closure plots (Now with correct total uncertainty)
def plot_evaluation(
    var, args, edges,
    y_4b, y_2b, y_model, err_tot, err_stat,
    ratio_4b_2b, ratio_4b_2b_w, ratio_err_tot, ratio_err_stat,
    chi2_val, chi2_2b,
    output_dirname=f"{dir_suffix}Closure_Plots",
    normalize_shapes=True,
    x_scale_log=False,
):
    if normalize_shapes:
        int_4b  = np.sum(y_4b)  or 1e-10
        int_2b  = np.sum(y_2b)  or 1e-10
        int_2bw = np.sum(y_model) or 1e-10
        y_4b   = y_4b  / int_4b
        y_2b   = y_2b  / int_2b
        y_model = y_model / int_2bw
        err_tot = err_tot / int_2bw
        err_stat = err_stat / int_2bw
        ratio_4b_2b   = y_4b / np.where(y_2b   > 0, y_2b,   1e-10)
        ratio_4b_2b_w = y_4b / np.where(y_model > 0, y_model, 1e-10)

    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(
        2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1),
        sharex=True, figsize=(10, 10),
    )
    lumi = get_lumi(args.YEAR)
    hep.cms.label("Preliminary", data=True, lumi=lumi, com=13.6, year=args.YEAR, ax=ax)

    labels = [f"{args.TrainRegion}", "2b", "2b_w"]
    hep.histplot(y_4b,   bins=edges, ax=ax, color="orange", label=labels[0])
    hep.histplot(y_2b,   bins=edges, ax=ax, color="red",    label=labels[1])
    hep.histplot(y_model, bins=edges, ax=ax, color="blue",   label=labels[2])

    band_low, band_high   = error_bands(y_model, err_tot)
    stat_low, stat_high   = error_bands(y_model, err_stat)
    ax.fill_between(edges, band_low, band_high, step="post", color="gray",  alpha=0.3, label="Total Uncertainty")
    ax.fill_between(edges, stat_low, stat_high, step="post", facecolor="none",
                    edgecolor="green", hatch="////", alpha=0.5, label="Stat Uncertainty")

    if var in LOG_VARS:
        ax.set_yscale("log")
    if x_scale_log and var in ["MX", "MY"]:
        ax.set_xscale("log")
        rax.set_xscale("log")

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylabel("Arbitrary units")

    x_centers = 0.5 * (edges[:-1] + edges[1:])
    rax.axhline(1.0, color="black", linestyle="--")

    ones = np.ones_like(y_model)
    r_band_low, r_band_high = error_bands(ones, ratio_err_tot)
    r_band_stat_low, r_band_stat_high = error_bands(ones, ratio_err_stat)
    rax.fill_between(edges, r_band_low, r_band_high, step="post", color="gray", alpha=0.3)
    rax.errorbar(x_centers, ratio_4b_2b,   fmt="o", color="red",
                 label=rf"{labels[0]}/{labels[1]} $\chi^2/NDF={chi2_2b:.2f}$")
    rax.errorbar(x_centers, ratio_4b_2b_w, fmt="o", color="blue",
                 label=rf"{labels[0]}/{labels[2]} $\chi^2/NDF={chi2_val:.2f}$")
    rax.fill_between(edges, r_band_stat_low, r_band_stat_high, step="post",
                     facecolor="none", edgecolor="green", hatch="////", alpha=0.5)

    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("Ratio")
    rax.set_xlabel(var)

    handles_ax,  labels_ax  = ax.get_legend_handles_labels()
    handles_rax, labels_rax = rax.get_legend_handles_labels()
    ax.legend(handles_ax + handles_rax, labels_ax + labels_rax,
              loc="best", ncol=1, fontsize="x-small")

    outdir = f"{output_dirname}_{args.TestRegion}"
    os.makedirs(outdir, exist_ok=True)
    outname = f"{outdir}/{var}_BkgEstimation" + ("_xlog" if x_scale_log else "")
    plt.savefig(f"{outname}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outname}.pdf", bbox_inches="tight")
    plt.close()


def plot_ratio_uncertainty(
    var, args, edges,
    ratio_err_tot, ratio_err_stat, chi2_val, chi2_2b,
    ratio_err_sys, ratio_err_nonclosure, chi2_Nonc,
    output_dirname=f"{dir_suffix}Uncertainty_Ratio",
    normalize_shapes=True,
    x_scale_log=False,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    lumi = get_lumi(args.YEAR)
    hep.cms.label(ax=ax, exp="", label="Private work (CMS data)",
                  data=True, year=f"{args.YEAR}", lumi=lumi, com=13.6)

    if args.TestRegion == "3bHiggsMW":
        hep.histplot(ratio_err_nonclosure, bins=edges, ax=ax, color="purple",
                     label=f"Error_NC, WITHOUT NC $\\chi^2/NDF={chi2_Nonc:.2f}$")
    hep.histplot(ratio_err_sys,  bins=edges, ax=ax, color="green",  label="Error_Sys")
    hep.histplot(ratio_err_stat, bins=edges, ax=ax, color="orange", label="Error_Stat")
    hep.histplot(ratio_err_tot,  bins=edges, ax=ax, color="black",
                 label=f"Error_Total $\\chi^2/NDF={chi2_val:.2f}$")

    ax.set_ylabel("Uncertainty Ratio (Error/Prediction)")
    ax.set_xlabel(var)
    if x_scale_log and var in ["MX", "MY"]:
        ax.set_xscale("log")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best", ncol=1, fontsize="x-small")

    outdir = f"{output_dirname}_{args.TestRegion}"
    os.makedirs(outdir, exist_ok=True)
    outname = f"{outdir}/{var}_Ratio_Uncertainty" + ("_xlog" if x_scale_log else "")
    plt.savefig(f"{outname}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outname}.pdf", bbox_inches="tight")
    plt.close()


def _plot_unc_histograms_impl(
    uncertainty_filename, args, vars_to_plot,
    year_label, data_lumi, output_dirname, normalize, x_scale_log,
    extra_hists_fn=None,          # optional callable(f_in, var) → (hists, labels, colors, alphas, linestyles)
):
    """
    Core implementation for plotting uncertainty histograms.
    extra_hists_fn is used by the 'separateMY' variant to inject the per-bin MY histograms.
    """
    f_in = ROOT.TFile(uncertainty_filename, "READ")
    if f_in is None or f_in.IsZombie():
        print(f"Error: Could not open {uncertainty_filename}")
        return

    output_dir = f"{output_dirname}_{args.TestRegion}"
    os.makedirs(output_dir, exist_ok=True)
    kwargs = {"binwnorm": True} if normalize else {}

    for var in vars_to_plot:
        print(f"Plotting histograms for {var}...")

        h_nominal = _get_hist_safe(f_in, f"{var}_2b_w_nominal")
        h_data    = _get_hist_safe(f_in, f"{var}_data_4b")
        if h_nominal is None:
            print(f"  -> WARNING: '{var}_2b_w_nominal' not found! Skipping."); continue
        if h_data is None:
            print(f"  -> WARNING: '{var}_data_4b' not found! Skipping."); continue

        n_bins, edges, x_centers = _make_hist_arrays(h_nominal)

        # Base histograms
        hist_list    = [h_data,   h_nominal,
                        f_in.Get(f"{var}_2b_w_stat_up"),   f_in.Get(f"{var}_2b_w_stat_down"),
                        f_in.Get(f"{var}_2b_w_sys_up"),    f_in.Get(f"{var}_2b_w_sys_down"),
                        f_in.Get(f"{var}_2b_w_total_up"),  f_in.Get(f"{var}_2b_w_total_down")]
        hist_labels  = [f"{args.TrainRegion} data", "2b_w_nominal",
                        "2b_w_stat_up", "2b_w_stat_down",
                        "2b_w_sys_up",  "2b_w_sys_down",
                        "2b_w_total_up","2b_w_total_down"]
        color_list       = ["black","blue","orange","orange","green","green","grey","grey"]
        line_style_list  = ["-","-",":",":","--","--","--","--"]
        alpha_list       = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]

        if args.TestRegion == "3bHiggsMW" or args.TestRegion == "4bHiggsMW":
            for var in ["MX", "Unrolled_MXMY"]:
                for suffix, c in [("nc_up","purple"), ("nc_down","purple")]:
                    h = f_in.Get(f"{var}_2b_w_{suffix}")
                    if h:
                        hist_list.append(h);       hist_labels.append(f"2b_w_{suffix}")
                        color_list.append(c);      alpha_list.append(0.5)
                    line_style_list.append("-.")
            for var in ["MY"]:
                for suffix, c in [("modified_nc_up","purple"), ("modified_nc_down","purple")]:
                    h = f_in.Get(f"{var}_2b_w_{suffix}")
                    if h:
                        hist_list.append(h);       hist_labels.append(f"2b_w_{suffix}")
                        color_list.append(c);      alpha_list.append(0.5)
                    line_style_list.append("-.")

        # Caller-supplied extra histograms (e.g. per-bin MY uncertainties)
        if extra_hists_fn is not None:
            extra = extra_hists_fn(f_in, var)
            if extra:
                ex_hists, ex_labels, ex_colors, ex_alphas, ex_ls = extra
                hist_list       += ex_hists
                hist_labels     += ex_labels
                color_list      += ex_colors
                alpha_list      += ex_alphas
                line_style_list += ex_ls

        hep.style.use("CMS")
        figsize = (20, 20) if var == "Unrolled_MXMY" else (15, 15)
        fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1),
                                       sharex=True, figsize=figsize)

        hep.cms.label("Preliminary", data=True, lumi=data_lumi, com=13.6, year=year_label, ax=ax)

        hep.histplot(hist_list, histtype="step", yerr=False, alpha=alpha_list,
                     linestyle=line_style_list, label=hist_labels, color=color_list, ax=ax, **kwargs)
        hep.histplot(h_data, ax=ax, color="black", yerr=False, histtype="errorbar",
                     label=f"{args.TrainRegion} data", linewidth=0.5, **kwargs)

        data_hist    = _th1_to_array(h_data,    n_bins)
        nominal_hist = _th1_to_array(h_nominal, n_bins)

        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylabel("Events")
        if var in LOG_VARS:
            ax.set_yscale("log")
        if x_scale_log and var in ["MX", "MY"]:
            ax.set_xscale("log")
            rax.set_xscale("log")

        # Ratio pad
        rax.axhline(1.0, color="black", linestyle="--")
        nominal_ratio = np.divide(data_hist, nominal_hist, out=np.zeros_like(data_hist), where=nominal_hist != 0)
        rax.errorbar(x_centers, nominal_ratio, yerr=0, fmt="o", color="blue")

        for i, h in enumerate(hist_list):
            if h is h_nominal or h is h_data:
                continue
            h_arr = _th1_to_array(h, n_bins)
            ratio = np.divide(data_hist, h_arr, out=np.zeros_like(h_arr), where=h_arr != 0)
            rax.plot(x_centers, ratio, drawstyle="steps-mid",
                     color=color_list[i], linestyle=line_style_list[i])

        rax.set_ylim(0.5, 1.5)
        rax.set_ylabel("Ratio")
        rax.set_xlabel(var)

        handles_ax,  labels_ax  = ax.get_legend_handles_labels()
        handles_rax, labels_rax = rax.get_legend_handles_labels()
        ax.legend(handles_ax + handles_rax, labels_ax + labels_rax,
                  loc="best", ncol=1, fontsize="x-small")

        outname = f"{output_dir}/{var}_Hist_Uncertainty" + ("_xlog" if x_scale_log else "")
        plt.savefig(f"{outname}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{outname}.pdf", bbox_inches="tight")
        plt.close()

    f_in.Close()


def plot_uncertainty_histograms(uncertainty_filename, args, vars_to_plot,
                                year_label="2024", data_lumi=109,
                                output_dirname=f"{dir_suffix}Uncertainty_Histograms",
                                normalize=True, x_scale_log=False):
    _plot_unc_histograms_impl(uncertainty_filename, args, vars_to_plot,
                              year_label, data_lumi, output_dirname, normalize, x_scale_log)


def plot_separateMY_uncertainty_histograms(uncertainty_filename, args, vars_to_plot,
                                           year_label="2024", data_lumi=109,
                                           output_dirname=f"{dir_suffix}Uncertainty_input_Histograms",
                                           normalize=True, x_scale_log=False):
    """Like plot_uncertainty_histograms but also overlays the individual MY-bin shapes."""

    def extra_hists_fn(f_in, var):
        if var not in ("MY", "Unrolled_MXMY"):
            return None
        hists, labels, colors, alphas, ls = [], [], [], [], []
        style_map = {6: "cyan", 7: "olive", 8: "brown"}
        for mb in TARGET_MY_BINS:
            for direction in ("up", "down"):
                name = f"{var}_2b_w_mybin_{mb}_{direction}"
                h = f_in.Get(name)
                if h:
                    hists.append(h)
                    labels.append(f"2b_w_mybin_{mb}_{direction}")
                    colors.append(style_map[mb])
                    alphas.append(0.5)
                    ls.append("-.")
        return (hists, labels, colors, alphas, ls) if hists else None

    _plot_unc_histograms_impl(uncertainty_filename, args, vars_to_plot,
                              year_label, data_lumi, output_dirname, normalize, x_scale_log,
                              extra_hists_fn=extra_hists_fn)


def add_decorrelated_nc_uncertainty(
    input_filename="Uncertainty_hists_OnlyPhysical.root",
    output_filename="combine_addDecorrelatedNC.root",
    target_my_bins=None,
):
    """
    Creates fully decorrelated uncertainty histograms for all three variables.

    MY distribution
    ---------------
    • h_nc_up / h_nc_down  — NC variation with target MY bins zeroed (set to nominal),
      written under the *same* histogram names so they replace the originals when hadded.
    • {var}_2b_w_mybin_{6,7,8}_{up,down}  — 100 % up/down per target MY bin (same
      naming as add_separate_my_125_uncertainty for consistency).

    MX distribution
    ---------------
    • {var}_2b_w_nc_bin{1..14}_{up,down}  — NC variation decorrelated by MX bin:
      only the target bin takes the NC value; all others remain at nominal.

    Unrolled_MXMY distribution
    --------------------------
    • {var}_2b_w_nc_bin{1..14}_{up,down}  — MX-decorrelated NC (same naming as MX);
      unrolled bins correlated to target MY bins are kept at nominal.
    • {var}_2b_w_mybin_{6,7,8}_mx{1..14}_{up,down}  — 100 % MY-bin uncertainty
      decorrelated further by MX bin:  only the (my_bin × mx_bin) intersection varies.
    """
    if target_my_bins is None:
        target_my_bins = TARGET_MY_BINS

    maps        = get_all_bin_mappings()
    my_to_unr   = maps["my_to_unrolled"]
    unr_to_mx   = maps["unrolled_to_mx"]

    # All unrolled bins that belong to any target MY bin
    target_unrolled = set()
    for mb in target_my_bins:
        target_unrolled.update(my_to_unr.get(mb, []))

    f_in  = ROOT.TFile(input_filename, "READ")
    f_out = ROOT.TFile(output_filename, "RECREATE")
    if f_in is None or f_in.IsZombie():
        print(f"Error: Could not open {input_filename}"); return

    epsilon = 1e-6

    # ── 1. MY ──────────────────────────────────────────────────────────────
    var = "MY"
    h_nom        = _get_hist_safe(f_in, f"{var}_2b_w_nominal")
    h_nc_up_orig = _get_hist_safe(f_in, f"{var}_2b_w_nc_up")
    h_nc_dn_orig = _get_hist_safe(f_in, f"{var}_2b_w_nc_down")

    if h_nom and h_nc_up_orig and h_nc_dn_orig:
        n_bins = h_nom.GetNbinsX()

        # NC with target bins set to nominal
        h_nc_up_mod = h_nc_up_orig.Clone(f"{var}_2b_w_modified_nc_up")
        h_nc_dn_mod = h_nc_dn_orig.Clone(f"{var}_2b_w_modified_nc_down")
        for b in target_my_bins:
            nom_val = h_nom.GetBinContent(b)
            h_nc_up_mod.SetBinContent(b, nom_val);  h_nc_up_mod.SetBinError(b, 0)
            h_nc_dn_mod.SetBinContent(b, nom_val);  h_nc_dn_mod.SetBinError(b, 0)
        f_out.cd()
        h_nc_up_mod.Write()
        h_nc_dn_mod.Write()

        # 100 % up/down per target MY bin (same naming as add_separate_my_125_uncertainty)
        for mb in target_my_bins:
            unc_name = f"mybin_{mb}"
            h_up   = h_nom.Clone(f"{var}_2b_w_{unc_name}_up")
            h_down = h_nom.Clone(f"{var}_2b_w_{unc_name}_down")
            h_up.SetTitle(f"{var}_2b_w_{unc_name}_up")
            h_down.SetTitle(f"{var}_2b_w_{unc_name}_down")
            for b in range(1, n_bins + 1):
                nom = h_nom.GetBinContent(b)
                if b == mb:
                    h_up.SetBinContent(b, nom * 2.0)
                    h_down.SetBinContent(b, epsilon)
                h_up.SetBinError(b, 0);  h_down.SetBinError(b, 0)
            f_out.cd()
            h_up.Write();  h_down.Write()
            print(f"  [MY] created: {unc_name}")

        print(f"[MY] Done — modified NC + {len(target_my_bins)} individual mybin uncertainties.")
    else:
        print("[MY] WARNING: nominal or NC histograms not found. Skipping.")

    # ── 2. MX : individual NC per bin ─────────────────────────────────────
    var = "MX"
    h_nom_mx    = _get_hist_safe(f_in, f"{var}_2b_w_nominal")
    h_nc_up_mx  = _get_hist_safe(f_in, f"{var}_2b_w_nc_up")
    h_nc_dn_mx  = _get_hist_safe(f_in, f"{var}_2b_w_nc_down")

    if h_nom_mx and h_nc_up_mx and h_nc_dn_mx:
        n_bins_mx = h_nom_mx.GetNbinsX()

        for mx_bin in range(1, n_bins_mx + 1):
            unc_name = f"nc_bin{mx_bin}"
            # Clone nominal so all other bins automatically stay at nominal
            h_up   = h_nom_mx.Clone(f"{var}_2b_w_{unc_name}_up")
            h_down = h_nom_mx.Clone(f"{var}_2b_w_{unc_name}_down")
            h_up.SetTitle(f"{var}_2b_w_{unc_name}_up")
            h_down.SetTitle(f"{var}_2b_w_{unc_name}_down")
            # Only vary this specific MX bin
            h_up.SetBinContent(mx_bin,  h_nc_up_mx.GetBinContent(mx_bin))
            h_down.SetBinContent(mx_bin, h_nc_dn_mx.GetBinContent(mx_bin))
            for b in range(1, n_bins_mx + 1):
                h_up.SetBinError(b, 0);  h_down.SetBinError(b, 0)
            f_out.cd()
            h_up.Write();  h_down.Write()

        print(f"[MX] Done — {n_bins_mx} individual NC-bin uncertainties.")
    else:
        print("[MX] WARNING: nominal or NC histograms not found. Skipping.")

    # ── 3. Unrolled_MXMY ───────────────────────────────────────────────────
    var = "Unrolled_MXMY"
    h_nom_unr   = _get_hist_safe(f_in, f"{var}_2b_w_nominal")
    h_nc_up_unr = _get_hist_safe(f_in, f"{var}_2b_w_nc_up")
    h_nc_dn_unr = _get_hist_safe(f_in, f"{var}_2b_w_nc_down")

    if h_nom_unr and h_nc_up_unr and h_nc_dn_unr:
        n_bins_unr = h_nom_unr.GetNbinsX()

        # 3a. MX-decorrelated NC (14 histograms)
        #     Bins belonging to target MY slices → held at nominal (not NC)
        for mx_bin in range(1, N_MX_BINS + 1):
            unc_name = f"nc_bin{mx_bin}"
            h_up   = h_nom_unr.Clone(f"{var}_2b_w_{unc_name}_up")
            h_down = h_nom_unr.Clone(f"{var}_2b_w_{unc_name}_down")
            h_up.SetTitle(f"{var}_2b_w_{unc_name}_up")
            h_down.SetTitle(f"{var}_2b_w_{unc_name}_down")

            for b in range(1, n_bins_unr + 1):
                is_target_my = b in target_unrolled
                is_this_mx   = (unr_to_mx.get(b) == mx_bin)

                if not is_target_my and is_this_mx:
                    # Apply NC variation only for this MX slice (excluding MY 6/7/8)
                    h_up.SetBinContent(b,   h_nc_up_unr.GetBinContent(b))
                    h_down.SetBinContent(b, h_nc_dn_unr.GetBinContent(b))
                # else: keep cloned nominal (handles both target-MY and other-MX bins)
                h_up.SetBinError(b, 0);  h_down.SetBinError(b, 0)

            f_out.cd()
            h_up.Write();  h_down.Write()

        print(f"[Unrolled] Done — {N_MX_BINS} MX-decorrelated NC uncertainties "
              f"(MY bins {target_my_bins} held at nominal).")

        # 3b. 3 × 14 = 42 100 % MY-bin uncertainties, decorrelated by MX bin
        #     For each (my_bin, mx_bin) pair, only the intersection moves.
        for mb in target_my_bins:
            unrolled_for_my = my_to_unr.get(mb, [])

            # Group those unrolled bins by their MX bin
            mx_to_unrolled_slice: dict[int, list] = {}
            for unr_bin in unrolled_for_my:
                mx = unr_to_mx.get(unr_bin)
                if mx is not None:
                    mx_to_unrolled_slice.setdefault(mx, []).append(unr_bin)

            for mx_bin in range(1, N_MX_BINS + 1):
                unc_name    = f"mybin_{mb}_mx{mx_bin}"
                bins_to_vary = mx_to_unrolled_slice.get(mx_bin, [])

                h_up   = h_nom_unr.Clone(f"{var}_2b_w_{unc_name}_up")
                h_down = h_nom_unr.Clone(f"{var}_2b_w_{unc_name}_down")
                h_up.SetTitle(f"{var}_2b_w_{unc_name}_up")
                h_down.SetTitle(f"{var}_2b_w_{unc_name}_down")

                for b in bins_to_vary:
                    nom = h_nom_unr.GetBinContent(b)
                    h_up.SetBinContent(b, nom * 2.0)
                    h_down.SetBinContent(b, epsilon)
                for b in range(1, n_bins_unr + 1):
                    h_up.SetBinError(b, 0);  h_down.SetBinError(b, 0)

                f_out.cd()
                h_up.Write();  h_down.Write()

        print(f"[Unrolled] Done — {len(target_my_bins)}×{N_MX_BINS} "
              f"MY×MX decorrelated 100 % uncertainties.")
    else:
        print("[Unrolled_MXMY] WARNING: nominal or NC histograms not found. Skipping.")

    f_out.Close(); f_in.Close()
    print(f"\nAll decorrelated uncertainty histograms written to {output_filename}.")


# Processing function: 
def run_create_uncertainty_histograms(args):
    """
    Compute fold-based uncertainty histograms and optionally plot them.
    Mirrors the original main() from hist_unroll3b.py.
    """
    n_folds = args.Nfold
    if n_folds is None:
        print("Please provide the number of folds using --Nfold!"); return

    vars_to_plot = ["MX", "MY", "Unrolled_MXMY"]

    # signal_file = args.SignalFile
    # if not os.path.exists(signal_file):
    #     print(f"Signal file not found: {signal_file}. Please provide a valid path using --SignalFile."); return 

    if args.TestRegion == "3bHiggsMW" or args.TestRegion == "4bHiggsMW":
        NonClosureFactor_path = NONCLOSURE_FACTORS_FILE
        SaveNonClosure = False
        input_file  = f"{args.TestRegion}_OnlyPhysical.root"
        output_file = f"{args.TestRegion}_Uncertainty_hists_OnlyPhysical.root"
        if not os.path.exists(NonClosureFactor_path):
            print(f"Missing non-closure file: {NonClosureFactor_path}. Run 3btest first."); return
    elif args.TestRegion == "3btest" or args.TestRegion == "4btest":
        SaveNonClosure = True
        NonClosureFactor_path = None
        input_file  = f"{args.TestRegion}_OnlyPhysical.root"
        output_file = f"{args.TestRegion}_Uncertainty_hists_OnlyPhysical.root"
    else:
        print(f"Unsupported TestRegion: {args.TestRegion}"); return

    # f_signal = ROOT.TFile(signal_file, "READ")
    f_in  = ROOT.TFile(input_file,  "READ")
    f_out = ROOT.TFile(output_file, "RECREATE")

    if args.TestRegion in ["3bHiggsMW", "4bHiggsMW"]:
        error_normalization = False
    elif args.TestRegion in ["3btest", "4btest"]:
        error_normalization = True

    for var in vars_to_plot:
        result_hist = get_hist_with_total_error(
            f_in, var, n_folds,
            normalize=error_normalization,
            TrainRegion=args.TrainRegion,
            NonClosureFracPath=NonClosureFactor_path,
        )
        if result_hist is None:
            print(f"Failed to retrieve data for {var}"); continue

        if SaveNonClosure:
            (edges, y_mean, y_3T, y_2T, err_tot, scale, chi2_val, chi2_2b,
             err_stat, err_sys, ratio_3b_2b, ratio_3b_2b_w,
             ratio_err_tot, ratio_err_stat, ratio_err_sys) = result_hist
            save_non_closure_factor(var, edges, ratio_3b_2b_w,
                                    out_filename=NONCLOSURE_FACTORS_FILE,
                                    exclude_my_bins=None)
            ratio_err_nonclosure = np.zeros_like(ratio_err_tot)
            chi2_Nonc = 0.0
        else:
            (edges, y_mean, y_3T, y_2T, err_tot, scale, chi2_val, chi2_2b,
             err_stat, err_sys, ratio_3b_2b, ratio_3b_2b_w,
             ratio_err_tot, ratio_err_stat, ratio_err_sys,
             err_nc, ratio_err_nonclosure, chi2_Nonc) = result_hist

        nbins       = len(edges) - 1
        edges_array = array.array("d", edges)
        epsilon     = 1e-6

        if args.CreateUncHist == 1:
            def _mh(suffix, values, err_per_bin=None):
                return _build_th1(f"{var}_{suffix}", values, edges_array,
                                  err_per_bin=err_per_bin, epsilon=epsilon)

            h_data    = _build_th1(f"{var}_data_4b", y_3T, edges_array,
                                  err_per_bin=None, epsilon=0.0)
            h_nominal = _mh("2b_w_nominal", np.maximum(y_mean, epsilon), err_per_bin=err_stat)
            h_stat_up   = _mh("2b_w_stat_up",   np.maximum(y_mean + err_stat, epsilon))
            h_stat_down = _mh("2b_w_stat_down",  np.maximum(y_mean - err_stat, epsilon))
            h_sys_up    = _mh("2b_w_sys_up",     np.maximum(y_mean + err_sys,  epsilon))
            h_sys_down  = _mh("2b_w_sys_down",   np.maximum(y_mean - err_sys,  epsilon))
            h_tot_up    = _mh("2b_w_total_up",   np.maximum(y_mean + err_tot,  epsilon))
            h_tot_down  = _mh("2b_w_total_down", np.maximum(y_mean - err_tot,  epsilon))
            

            f_out.cd()
            for h in [h_data, h_nominal, h_stat_up, h_stat_down,
                      h_sys_up, h_sys_down, h_tot_up, h_tot_down]:
                h.Write()

            if not SaveNonClosure:
                h_nc_up   = _mh("2b_w_nc_up",   np.maximum(y_mean + err_nc, epsilon))
                h_nc_down = _mh("2b_w_nc_down",  np.maximum(y_mean - err_nc, epsilon))
                h_nc_up.Write(); h_nc_down.Write()

                # direclty clone signal from the h_signal
                # h_signal = f_signal.Get(f"{var}_hist_signal")
                # if h_signal:
                #     h_signal.SetName(f"{var}_hist_signal")
                #     h_signal.Write()

        if args.Plot == 1:
            for x_log in [False, True]:
                plot_evaluation(var, args, edges, y_3T, y_2T, y_mean,
                                err_tot, err_stat, ratio_3b_2b, ratio_3b_2b_w,
                                ratio_err_tot, ratio_err_stat, chi2_val, chi2_2b,
                                normalize_shapes=True, x_scale_log=x_log)
                plot_ratio_uncertainty(var, args, edges, ratio_err_tot, ratio_err_stat,
                                       chi2_val, chi2_2b, ratio_err_sys,
                                       ratio_err_nonclosure, chi2_Nonc,
                                       normalize_shapes=True, x_scale_log=x_log)

            if not SaveNonClosure:
                print(f"Chi2 for {var}: {chi2_val:.3f}  |  "
                      f"Chi2 without NC uncertainty: {chi2_Nonc:.3f}")


    # f_signal.Close(); 
    f_in.Close(); f_out.Close()

def apply_bkg_norm_scalefactor(args):
    """
    Reads the base uncertainty histograms, applies the global yield scale factor 
    to all 2b proxy histograms (nominal + systematic variations), and writes a new file.
    """
    input_file  = f"combine_noempty_input.root"
    output_file = f"combine_noempty_input_Scaled.root"

    f_in  = ROOT.TFile(input_file, "READ")
    if f_in is None or f_in.IsZombie():
        print(f"Error: Could not open {input_file}. Run create_unc_hists first!"); return

    f_out = ROOT.TFile(output_file, "RECREATE")
    
    if args.TrainRegion == "3b":
        scale = 0.1780
    elif args.TrainRegion == "4b":
        scale = 0.2331
    print(f"Applying global scale factor {scale} to '{input_file}'...")

    for key in f_in.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TH1"):
            f_out.cd()
            h_clone = obj.Clone()
            
            # CRITICAL: Only scale the estimated 2b background!
            if "2b_w_" in h_clone.GetName():
                h_clone.Scale(scale)
                
            h_clone.Write()

    f_in.Close()
    f_out.Close()
    print(f"Success! Scaled histograms saved to {output_file}")



def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified uncertainty pipeline for XtoYH4b background estimation."
    )
    parser.add_argument("--YEAR",        default="2024", type=str)
    parser.add_argument("--runType",     default="train-test",
                        choices=["train-test", "train-only", "test-only"])
    parser.add_argument("--TrainRegion", default="4b", choices=["4b", "3b"])
    parser.add_argument("--TestRegion",  default=None,
                        choices=[None, "4btest", "3btest", "3bHiggsMW", "4bHiggsMW"],)
    parser.add_argument("--Nfold",       default=None, type=int,
                        help="Number of folds (required for create_unc_hists).")
    
    # parser.add_argument("--SignalFile", default=None, type=str, # Now the signal is added with Hist2Comb.py
    #                     help="ROOT file containing the signal histogram to be included in Combine input.")

    parser.add_argument(
        "--Plot",
        default=1,
        type=int,
        choices=[0, 1],
        help="Set to 1 to create plots, 0 to skip plotting.",
    )
    parser.add_argument(
        "--CreateUncHist",
        default=1,
        type=int,
        choices=[0, 1],
        help="Set to 1 to create uncertainty histograms, 0 to skip writing them.",
    )
    parser.add_argument(
        "--function",
        default="add_decorrelated_nc_uncertainty",
        choices=[
            "create_unc_hists",               # hist_unroll3b main()
            "add_MY_binuncertainty",           # combined MY 100% up/down
            "add_separate_my_125_uncertainty", # per-bin MY 100% up/down
            "add_decorrelated_nc_uncertainty", # NEW: full decorrelation
            "plot_uncertainty_histograms",     # standard plot
            "plot_separateMY_histograms",      # plot with per-MY-bin overlays
            "apply_bkg_norm_scalefactor"
        ],
    )
    return parser


def main():
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning

    args   = build_parser().parse_args()
    _build = build_binning_map(njets=4)  

    vars_to_plot = ["MX", "MY", "Unrolled_MXMY"]

    if args.function == "create_unc_hists":
        if args.runType != "test-only":
            print("Error: create_unc_hists supports test-only mode only."); return
        run_create_uncertainty_histograms(args)

    elif args.function == "add_decorrelated_nc_uncertainty":
        add_decorrelated_nc_uncertainty(
            f"{args.TestRegion}_Uncertainty_hists_OnlyPhysical.root",
            "combine_addDecorrelatedNC.root",
        )

    elif args.function == "plot_uncertainty_histograms":
        for x_log in [False, True]:
            plot_uncertainty_histograms(
                "combine_noempty_input.root", args, vars_to_plot,
                year_label=args.YEAR, data_lumi=get_lumi(args.YEAR), normalize=True, x_scale_log=x_log,
            )

    elif args.function == "plot_separateMY_histograms":
        for x_log in [False, True]:
            plot_separateMY_uncertainty_histograms(
                "combine_noempty_input.root", args, vars_to_plot,
                year_label=args.YEAR, data_lumi=get_lumi(args.YEAR), normalize=True, x_scale_log=x_log,
            )
        # Also plot just MX / MY with x-log
        plot_separateMY_uncertainty_histograms(
            "combine_noempty_input.root", args, vars_to_plot=["MX", "MY"],
            year_label=args.YEAR, data_lumi=get_lumi(args.YEAR), normalize=True, x_scale_log=True,
        )
    elif args.function == "apply_bkg_norm_scalefactor":
        apply_bkg_norm_scalefactor(args)


main()


# ─────────────────────────────────────────────────────────────────────────────
# Usage guide
# ─────────────────────────────────────────────────────────────────────────────
#
# Step 1 – compute fold-based uncertainty histograms (3btest first, then 3bHiggsMW)
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 3b --TestRegion 3btest --Nfold 10 --Plot 1 --CreateUncHist 1 --function create_unc_hists

# Step 2 - add non-closure uncertainty and signal
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --Nfold 10 --SignalFile OnlyPhysical_Signal.root --Plot 1 --CreateUncHist 1 --function create_unc_hists

# Step 3 – modify NC uncertainties (add MY-3bin uncertainty and modify old NC, deccorelate NC by MX bin, Unroll)
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --function add_decorrelated_nc_uncertainty

# Step 4 – hadd signal, original uncertainty histograms, and new decorrelated NC histograms into a single Combine input file
#   hadd combine_noempty_input.root Uncertainty_hists_OnlyPhysical.root combine_addDecorrelatedNC.root

# Step 5 – produce final Combine input
#   python3 convert_to_combine_input_DecoMX.py 




# Step 1 – compute fold-based uncertainty histograms (4btest first, then 4bHiggsMW)
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 4b --TestRegion 4btest --Nfold 10 --Plot 1 --CreateUncHist 1 --function create_unc_hists

# Step 2 - add non-closure uncertainty
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 4b --TestRegion 4bHiggsMW --Nfold 10 --Plot 0 --CreateUncHist 1 --function create_unc_hists

# Step 3 – modify NC uncertainties (add MY-3bin uncertainty and modify old NC, deccorelate NC by MX bin, Unroll)
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 4b --TestRegion 4bHiggsMW --function add_decorrelated_nc_uncertainty

# Step 4 – hadd original uncertainty histograms, and new decorrelated NC histograms into a single Combine input file
#   hadd combine_noempty_input.root 4bHiggsMW_Uncertainty_hists_OnlyPhysical.root combine_addDecorrelatedNC.root

# Step 5 – apply background normalization scale factor
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 4b --TestRegion 4bHiggsMW --function apply_bkg_norm_scalefactor