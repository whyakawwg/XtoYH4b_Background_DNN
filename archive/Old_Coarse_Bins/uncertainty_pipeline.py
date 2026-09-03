import warnings

warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import sys

sys.path.append("/data/dust/user/wanghaoy/XtoYH4b/XtoYH4b_Background_DNN")
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
    get_binning_mappings,
)

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import argparse
import os
import array
import json

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

dir_suffix = ""

MX_BIN_EDGES = np.array([250, 300, 375, 450, 550, 675, 825, 1000, 1250, 1600, 2000, 2500, 3000, 4000, 5000])
MY_BIN_EDGES = np.array([30, 40, 50, 60, 75, 90, 110, 135, 165, 200, 250, 300, 375, 450, 550, 675, 825, 1000, 1250, 1600, 2000, 2500, 3000, 4000])
N_MX_BINS = len(MX_BIN_EDGES) - 1  # 14
TARGET_MY_BINS = [6, 7, 8]         # MY bins with lower edges 90, 110, 135
# nc_3b is intentionally disabled in these MY slices for now.  Keeping this
# separate makes it straightforward to enable them for nc_3b in the future.
NC_3B_EXCLUDED_MY_BINS = list(TARGET_MY_BINS)

LOG_VARS = [
    "MY", "MX", "MH", "JetAK4_pt_1", "JetAK4_pt_2", "JetAK4_pt_3",
    "JetAK4_pt_4", "HT_additional", "Hcand_1_pt", "Hcand_2_pt",
    "Hcand_1_mass", "Hcand_2_mass", "HT_4j",
]


def mass_suffix(args):
    """Return the mass-point suffix used by pairing-aware evaluation products."""
    if args.MX is not None:
        return f"_MX-{args.MX}_MY-{args.MY}"
    return ""


def evaluation_split_parts(args):
    """Return split-specific directory and filename components for 3b inputs."""
    if args.TrainRegion != "3b":
        return "", ""
    if args.SplitIndex is None:
        raise ValueError("--SplitIndex is required for 3b uncertainty processing.")
    split_tag = f"SplitIndex{args.SplitIndex}"
    return f"/{split_tag}", f"_{split_tag}"


def nonclosure_factors_file(args):
    return f"{dir_suffix}nonclosure_factors{mass_suffix(args)}.root"


def nc_3b_factors_file(args):
    """Return the five-split pooled nc_3b factor file used by the final output."""
    return f"{dir_suffix}nonclosure_factors_nc_3b_Average5Splits{mass_suffix(args)}.root"


def nc_3b_split_factors_file(args, split_index):
    """Return a diagnostic factor file for one individual 3b split."""
    return (
        f"{dir_suffix}nonclosure_factors_nc_3b_SplitIndex{split_index}"
        f"{mass_suffix(args)}.root"
    )


def nc_3b_evaluation_input_file(args, split_index):
    split_tag = f"SplitIndex{split_index}"
    return (
        f"/data/dust/user/wanghaoy/XtoYH4b/Background_{args.YEAR}/"
        f"3bHiggsMW_evaluation/{split_tag}/"
        f"3bHiggsMW_OnlyPhysical_{split_tag}{mass_suffix(args)}.root"
    )


def evaluation_input_file(args):
    split_dir, split_suffix = evaluation_split_parts(args)
    return (
        f"/data/dust/user/wanghaoy/XtoYH4b/Background_{args.YEAR}/"
        f"{args.TestRegion}_evaluation{split_dir}/"
        f"{args.TestRegion}_OnlyPhysical{split_suffix}{mass_suffix(args)}.root"
    )


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


def _chi2_per_ndf(observed, expected, observed_error, expected_error, normalized):
    """Calculate chi2/NDF for the arrays exactly as displayed."""
    variance = observed_error**2 + expected_error**2
    valid = np.isfinite(observed) & np.isfinite(expected) & (variance > 0)
    ndf = max(int(np.count_nonzero(valid)) - int(normalized), 1)
    return float(np.sum((observed[valid] - expected[valid])**2 / variance[valid]) / ndf)


# Non-closure factor
def save_non_closure_factor(
    var_name, edges, ratio_array, out_filename="nonclosure_factors.root", exclude_my_bins=None
):
    """Save |ratio-1| as a TH1F to a ROOT file (0 for bins in exclude_my_bins)."""
    if exclude_my_bins is None:
        exclude_my_bins = []

    bins_to_exclude = []
    if exclude_my_bins:
        if var_name == "Unrolled_MXMY":
            maps = get_binning_mappings()
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


def _histogram_values_and_edges(histogram):
    n_bins = histogram.GetNbinsX()
    values = np.array(
        [histogram.GetBinContent(i) for i in range(1, n_bins + 1)], dtype=float
    )
    edges = np.array(
        [histogram.GetBinLowEdge(i) for i in range(1, n_bins + 2)], dtype=float
    )
    return values, edges


def _normalized_shape(values, label):
    integral = np.sum(values)
    if integral <= 0:
        raise ValueError(f"Cannot normalize {label}: integral is {integral}.")
    return values / integral


def save_3b_higgsmw_nonclosure(args):
    """Save per-split and pooled |3b/2b_w - 1| Higgs-MW non-closures."""
    if args.TrainRegion != "3b" or args.TestRegion != "3bHiggsMW":
        raise ValueError(
            "save_nc_3b requires --TrainRegion 3b --TestRegion 3bHiggsMW."
        )

    variables = ["MX", "MY", "Unrolled_MXMY"]
    summed_3b = {var: None for var in variables}
    summed_2b_w = {var: None for var in variables}
    reference_edges = {}

    for split_index in range(5):
        input_file = nc_3b_evaluation_input_file(args, split_index)
        f_in = ROOT.TFile(input_file, "READ")
        if f_in is None or f_in.IsZombie():
            raise FileNotFoundError(
                f"Could not open 3b Higgs-MW SplitIndex{split_index} input: {input_file}"
            )

        split_output = nc_3b_split_factors_file(args, split_index)
        for var in variables:
            h_3b = _get_hist_safe(f_in, f"{var}_hist_4b_mean")
            h_2b_w = _get_hist_safe(f_in, f"{var}_hist_2bw_mean")
            if not h_3b or not h_2b_w:
                f_in.Close()
                raise KeyError(
                    f"Missing target or prediction histogram for {var} in {input_file}."
                )

            values_3b, edges_3b = _histogram_values_and_edges(h_3b)
            values_2b_w, edges_2b_w = _histogram_values_and_edges(h_2b_w)
            if not np.array_equal(edges_3b, edges_2b_w):
                f_in.Close()
                raise ValueError(f"3b and 2b_w binning differ for {var} in {input_file}.")
            if var in reference_edges and not np.array_equal(reference_edges[var], edges_3b):
                f_in.Close()
                raise ValueError(f"Split binning differs for {var} in {input_file}.")
            reference_edges[var] = edges_3b

            ratio_split = np.divide(
                _normalized_shape(values_3b, f"SplitIndex{split_index} {var} 3b"),
                _normalized_shape(values_2b_w, f"SplitIndex{split_index} {var} 2b_w"),
                out=np.ones_like(values_3b),
                where=values_2b_w > 0,
            )
            save_non_closure_factor(
                var,
                edges_3b,
                ratio_split,
                out_filename=split_output,
                exclude_my_bins=NC_3B_EXCLUDED_MY_BINS,
            )

            if summed_3b[var] is None:
                summed_3b[var] = values_3b.copy()
                summed_2b_w[var] = values_2b_w.copy()
            else:
                summed_3b[var] += values_3b
                summed_2b_w[var] += values_2b_w

        f_in.Close()
        print(f"Saved SplitIndex{split_index} nc_3b diagnostics to {split_output}")

    output_file = nc_3b_factors_file(args)
    for var in variables:
        ratio_pooled = np.divide(
            _normalized_shape(summed_3b[var], f"pooled {var} 3b"),
            _normalized_shape(summed_2b_w[var], f"pooled {var} 2b_w"),
            out=np.ones_like(summed_3b[var]),
            where=summed_2b_w[var] > 0,
        )
        save_non_closure_factor(
            var,
            reference_edges[var],
            ratio_pooled,
            out_filename=output_file,
            exclude_my_bins=NC_3B_EXCLUDED_MY_BINS,
        )

    print(
        "Saved pooled nc_3b = |normalized(sum 3b) / normalized(sum 2b_w) - 1| "
        f"to {output_file}"
    )


# Plot the closure plots (Now with correct total uncertainty)
def plot_evaluation(
    var, args, edges,
    y_4b, y_2b, y_model, err_tot, err_stat, err_stat_2b,
    ratio_4b_2b, ratio_4b_2b_w, ratio_err_tot, ratio_err_stat,
    chi2_val, chi2_2b, err_stat_4b=None, 
    output_dirname=f"{dir_suffix}Closure_Plots",
    normalize_shapes=True,
    x_scale_log=False,
):
    if err_stat_4b is None:
        raise ValueError("err_stat_4b is required for closure-plot ratio errors")

    if normalize_shapes:
        int_4b  = np.sum(y_4b)  or 1e-10
        int_2b  = np.sum(y_2b)  or 1e-10
        int_2bw = np.sum(y_model) or 1e-10
        y_4b   = y_4b  / int_4b
        y_2b   = y_2b  / int_2b
        y_model = y_model / int_2bw
        err_tot = err_tot / int_2bw
        err_stat = err_stat / int_2bw
        err_stat_2b = err_stat_2b / int_2b
        err_stat_4b = err_stat_4b / int_4b

    bin_widths = np.diff(edges)
    if np.any(bin_widths <= 0):
        raise ValueError(f"Non-positive bin width found for {var}: {bin_widths}")

    y_4b = y_4b / bin_widths
    y_2b = y_2b / bin_widths
    y_model = y_model / bin_widths
    err_tot = err_tot / bin_widths
    err_stat = err_stat / bin_widths
    err_stat_2b = err_stat_2b / bin_widths
    err_stat_4b = err_stat_4b / bin_widths

    ratio_4b_2b = np.divide(
        y_4b, y_2b, out=np.zeros_like(y_4b), where=y_2b > 0
    )
    ratio_4b_2b_w = np.divide(
        y_4b, y_model, out=np.zeros_like(y_4b), where=y_model > 0
    )
    ratio_err_4b2b = np.sqrt(
        np.divide(err_stat_4b, y_2b, out=np.zeros_like(err_stat_4b), where=y_2b > 0)**2
        + np.divide(y_4b * err_stat_2b, y_2b**2,
                    out=np.zeros_like(y_4b), where=y_2b > 0)**2
    )
    ratio_err_4b2b_w = np.sqrt(
        np.divide(err_stat_4b, y_model, out=np.zeros_like(err_stat_4b), where=y_model > 0)**2
        + np.divide(y_4b * err_stat, y_model**2,
                    out=np.zeros_like(y_4b), where=y_model > 0)**2
    )
    ratio_err_tot = np.divide(
        err_tot, y_model, out=np.zeros_like(err_tot), where=y_model > 0
    )
    ratio_err_stat = np.divide(
        err_stat, y_model, out=np.zeros_like(err_stat), where=y_model > 0
    )
    chi2_2b = _chi2_per_ndf(
        y_4b, y_2b, err_stat_4b, err_stat_2b, normalize_shapes
    )
    chi2_val = _chi2_per_ndf(
        y_4b, y_model, err_stat_4b, err_tot, normalize_shapes
    )

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
    ax.set_ylabel("Arbitrary units / bin width")

    x_centers = 0.5 * (edges[:-1] + edges[1:])
    rax.axhline(1.0, color="black", linestyle="--")

    ones = np.ones_like(y_model)
    r_band_low, r_band_high = error_bands(ones, ratio_err_tot)
    r_band_stat_low, r_band_stat_high = error_bands(ones, ratio_err_stat)
    rax.fill_between(edges, r_band_low, r_band_high, step="post", color="gray", alpha=0.3)
    rax.errorbar(x_centers, ratio_4b_2b, yerr=ratio_err_4b2b, fmt="o", color="red",
                 label=rf"{labels[0]}/{labels[1]} $\chi^2/NDF={chi2_2b:.2f}$")
    rax.errorbar(x_centers, ratio_4b_2b_w, yerr=ratio_err_4b2b_w, fmt="o", color="blue",
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

    outdir = f"{output_dirname}_{args.TestRegion}{mass_suffix(args)}"
    os.makedirs(outdir, exist_ok=True)
    outname = f"{outdir}/{var}_BkgEstimation" + ("_xlog" if x_scale_log else "")
    plt.savefig(f"{outname}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outname}.pdf", bbox_inches="tight")
    plt.close()

def plot_evaluation_signal_region(
    var, args, edges,
    y_4b, y_2b, y_model, err_tot, err_stat,
    ratio_4b_2b, ratio_4b_2b_w, ratio_err_tot, ratio_err_stat,
    chi2_val, chi2_2b,
    output_dirname=f"{dir_suffix}Closure_Plots",
    normalize_shapes=True,
    x_scale_log=False,
):
    """Same as plot_evaluation but for the signal region, blinded, so without data points and ratios."""
    if normalize_shapes:
        int_2b  = np.sum(y_2b)  or 1e-10
        int_2bw = np.sum(y_model) or 1e-10
        y_2b   = y_2b  / int_2b
        y_model = y_model / int_2bw
        err_tot = err_tot / int_2bw
        err_stat = err_stat / int_2bw

    bin_widths = np.diff(edges)
    if np.any(bin_widths <= 0):
        raise ValueError(f"Non-positive bin width found for {var}: {bin_widths}")

    y_2b = y_2b / bin_widths
    y_model = y_model / bin_widths
    err_tot = err_tot / bin_widths
    err_stat = err_stat / bin_widths

    hep.style.use("CMS")

    fig, ax = plt.subplots(figsize=(15, 8))
    lumi = get_lumi(args.YEAR)
    hep.cms.label("Preliminary", data=True, lumi=lumi, com=13.6, year=args.YEAR, ax=ax)

    labels = ["2b", "2b_w"]
    hep.histplot(y_2b,   bins=edges, ax=ax, color="red",    label=labels[0])
    hep.histplot(y_model, bins=edges, ax=ax, color="blue",   label=labels[1])

    band_low, band_high   = error_bands(y_model, err_tot)
    stat_low, stat_high   = error_bands(y_model, err_stat)
    ax.fill_between(edges, band_low, band_high, step="post", color="gray",  alpha=0.3, label="Total Uncertainty")
    ax.fill_between(edges, stat_low, stat_high, step="post", facecolor="none",
                    edgecolor="green", hatch="////", alpha=0.5, label="Stat Uncertainty")

    if var in LOG_VARS:
        ax.set_yscale("log")
    if x_scale_log and var in ["MX", "MY"]:
        ax.set_xscale("log")

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylabel("Arbitrary units / bin width")

    ax.legend(loc="best", ncol=1, fontsize="x-small")

    outdir = f"{output_dirname}_{args.TestRegion}{mass_suffix(args)}"
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

    outdir = f"{output_dirname}_{args.TestRegion}{mass_suffix(args)}"
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

    output_dir = f"{output_dirname}_{args.TestRegion}{mass_suffix(args)}"
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

    maps        = get_binning_mappings()
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
        NonClosureFactor_path = nonclosure_factors_file(args)
        SaveNonClosure = False
        if not os.path.exists(NonClosureFactor_path):
            print(f"Missing non-closure file: {NonClosureFactor_path}. Run 3btest first."); return
    
    elif args.TestRegion == "3btest" or args.TestRegion == "4btest":
        SaveNonClosure = True
        NonClosureFactor_path = None

    else:
        print(f"Unsupported TestRegion: {args.TestRegion}"); return
    
    suffix = mass_suffix(args)
    input_file = evaluation_input_file(args)
    output_file = f"{args.TestRegion}_Uncertainty_hists_OnlyPhysical{suffix}.root"

    nc_3b_path = None
    if args.Add3bHiggsMWNonClosure:
        if args.TestRegion not in ("3bHiggsMW", "4bHiggsMW"):
            raise ValueError(
                "--Add3bHiggsMWNonClosure is only valid for a Higgs-MW TestRegion."
            )
        nc_3b_path = nc_3b_factors_file(args)
        if not os.path.exists(nc_3b_path):
            raise FileNotFoundError(
                f"Missing nc_3b factors: {nc_3b_path}. Run --function save_nc_3b first."
            )

    # f_signal = ROOT.TFile(signal_file, "READ")
    f_in  = ROOT.TFile(input_file,  "READ")
    f_out = ROOT.TFile(output_file, "RECREATE")
    if f_in is None or f_in.IsZombie():
        f_out.Close()
        raise FileNotFoundError(f"Could not open evaluation input file: {input_file}")

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
            SplitIndex=args.SplitIndex,
        )
        if result_hist is None:
            print(f"Failed to retrieve data for {var}"); continue

        if SaveNonClosure:
            (edges, y_mean, y_3T, y_2T, err_tot, scale, chi2_val, chi2_2b,
             err_stat, err_sys, ratio_3b_2b, ratio_3b_2b_w,
             ratio_err_tot, ratio_err_stat, ratio_err_sys, err_stat_4b) = result_hist
            save_non_closure_factor(var, edges, ratio_3b_2b_w,
                                    out_filename=nonclosure_factors_file(args),
                                    exclude_my_bins=None)
            ratio_err_nonclosure = np.zeros_like(ratio_err_tot)
            chi2_Nonc = 0.0
        else:
            (edges, y_mean, y_3T, y_2T, err_tot, scale, chi2_val, chi2_2b,
             err_stat, err_sys, ratio_3b_2b, ratio_3b_2b_w,
             ratio_err_tot, ratio_err_stat, ratio_err_sys,
             err_nc, ratio_err_nonclosure, chi2_Nonc, err_stat_4b) = result_hist

        err_nc_3b = None
        if nc_3b_path is not None:
            nc_3b_factors = load_nonclosure_factor(var, len(edges) - 1, nc_3b_path)
            if nc_3b_factors is None:
                raise KeyError(
                    f"Missing {var}_nonclosure_factor in nc_3b file {nc_3b_path}."
                )
            err_nc_3b = nc_3b_factors * y_mean
            err_tot = np.sqrt(err_tot**2 + err_nc_3b**2)
            ratio_err_tot = np.divide(
                err_tot, y_mean, out=np.ones_like(err_tot), where=y_mean > 0
            )
            chi2_val = _chi2_per_ndf(
                y_3T, y_mean, err_stat_4b, err_tot, error_normalization
            )

        h_2b_stat = f_in.Get(f"{var}_hist_2b_mean")
        if not h_2b_stat:
            raise ValueError(f"Unweighted 2b histogram not found for '{var}'")
        err_stat_2b = np.array([
            h_2b_stat.GetBinError(i) for i in range(1, h_2b_stat.GetNbinsX() + 1)
        ])
        if error_normalization and h_2b_stat.Integral() > 0:
            err_stat_2b /= h_2b_stat.Integral()

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

            if nc_3b_path is not None:
                h_nc_3b_up = _mh(
                    "2b_w_nc_3b_up", np.maximum(y_mean + err_nc_3b, epsilon)
                )
                h_nc_3b_down = _mh(
                    "2b_w_nc_3b_down", np.maximum(y_mean - err_nc_3b, epsilon)
                )
                h_nc_3b_up.Write(); h_nc_3b_down.Write()
                if var in ("MY", "Unrolled_MXMY"):
                    print(
                        f"Added nc_3b variations for {var}; MY bins "
                        f"{NC_3B_EXCLUDED_MY_BINS} remain nominal."
                    )
                else:
                    print(f"Added nc_3b variations for {var}.")

                # direclty clone signal from the h_signal
                # h_signal = f_signal.Get(f"{var}_hist_signal")
                # if h_signal:
                #     h_signal.SetName(f"{var}_hist_signal")
                #     h_signal.Write()

        if args.Plot == 1:
            for x_log in [False, True]:
                if args.TestRegion == "4bHiggsMW":
                    plot_evaluation_signal_region(var, args, edges, y_3T, y_2T, y_mean,
                                                  err_tot, err_stat, ratio_3b_2b, ratio_3b_2b_w,
                                                  ratio_err_tot, ratio_err_stat, chi2_val, chi2_2b,
                                                  normalize_shapes=True, x_scale_log=x_log)
                    plot_ratio_uncertainty(var, args, edges, ratio_err_tot, ratio_err_stat,
                                           chi2_val, chi2_2b, ratio_err_sys,
                                           ratio_err_nonclosure, chi2_Nonc,
                                           normalize_shapes=True, x_scale_log=x_log)
                else:
                    plot_evaluation(var, args, edges, y_3T, y_2T, y_mean,
                                    err_tot, err_stat, err_stat_2b,
                                    ratio_3b_2b, ratio_3b_2b_w,
                                    ratio_err_tot, ratio_err_stat, chi2_val, chi2_2b,
                                    err_stat_4b=err_stat_4b,
                                    normalize_shapes=True, x_scale_log=x_log)
                    plot_ratio_uncertainty(var, args, edges, ratio_err_tot, ratio_err_stat,
                                        chi2_val, chi2_2b, ratio_err_sys,
                                        ratio_err_nonclosure, chi2_Nonc,
                                        normalize_shapes=True, x_scale_log=x_log)

            # if not SaveNonClosure:
                # print(f"Chi2 for {var}: {chi2_val:.3f}  |  "
                #       f"Chi2 without NC uncertainty: {chi2_Nonc:.3f}")


    # f_signal.Close(); 
    f_in.Close(); f_out.Close()

def apply_bkg_norm_scalefactor(args):
    """
    Reads the base uncertainty histograms, applies the global yield scale factor 
    to all 2b proxy histograms (nominal + systematic variations), and writes a new file.
    """
    suffix = mass_suffix(args)
    input_file  = f"combine_noempty_input{suffix}.root"
    output_file = f"combine_noempty_input_Scaled_{args.YEAR}{suffix}.root"

    f_in  = ROOT.TFile(input_file, "READ")
    if f_in is None or f_in.IsZombie():
        print(f"Error: Could not open {input_file}. Run create_unc_hists first!"); return

    f_out = ROOT.TFile(output_file, "RECREATE")
  
    with open(f"/data/dust/user/wanghaoy/XtoYH4b/Bkg_10fold_datafile/{args.YEAR}/metadata_{args.YEAR}.json") as f:
        metadata = json.load(f)

    # Use it based on your argparse input:
    if args.TrainRegion == "3b":
        norm_scale_factor = metadata["normalization_scale_factor_3b"]
    elif args.TrainRegion == "4b":
        norm_scale_factor = metadata["normalization_scale_factor_4b"]

    print(f"Applying global scale factor {norm_scale_factor} to '{input_file}'...")

    for key in f_in.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TH1"):
            f_out.cd()
            h_clone = obj.Clone()
            
            # Only scale the estimated 2b background
            if "2b_w_" in h_clone.GetName():
                h_clone.Scale(norm_scale_factor)
                
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
    parser.add_argument("--MX",          default=None, type=int,
                        help="Signal X mass point (required for uncertainty processing).")
    parser.add_argument("--MY",          default=None, type=int,
                        help="Signal Y mass point (required for uncertainty processing).")
    parser.add_argument("--SplitIndex",  default=None, type=int, choices=range(5),
                        help="Selected 3b validation split: 0-4.")
    parser.add_argument(
        "--Add3bHiggsMWNonClosure",
        default=0,
        type=int,
        choices=[0, 1],
        help="Add the pooled five-split 3b Higgs-MW non-closure as nc_3b.",
    )
    
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
            "save_nc_3b",                     # derive nc_3b from 3bHiggsMW
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
    if (args.MX is None) != (args.MY is None):
        raise ValueError("--MX and --MY must be provided together.")
    if args.MX is None:
        raise ValueError("--MX and --MY are required for uncertainty processing.")
    if (
        args.TrainRegion == "3b"
        and args.function != "save_nc_3b"
        and args.SplitIndex is None
    ):
        raise ValueError("--SplitIndex is required for 3b uncertainty processing.")
    _build = build_binning_map(njets=4)  

    vars_to_plot = ["MX", "MY", "Unrolled_MXMY"]

    if args.function == "create_unc_hists":
        if args.runType != "test-only":
            print("Error: create_unc_hists supports test-only mode only."); return
        run_create_uncertainty_histograms(args)

    elif args.function == "save_nc_3b":
        if args.runType != "test-only":
            raise ValueError("save_nc_3b supports test-only mode only.")
        save_3b_higgsmw_nonclosure(args)

    elif args.function == "add_decorrelated_nc_uncertainty":
        suffix = mass_suffix(args)
        add_decorrelated_nc_uncertainty(
            f"{args.TestRegion}_Uncertainty_hists_OnlyPhysical{suffix}.root",
            f"combine_addDecorrelatedNC{suffix}.root",
        )

    elif args.function == "plot_uncertainty_histograms":
        for x_log in [False, True]:
            plot_uncertainty_histograms(
                f"combine_noempty_input{mass_suffix(args)}.root", args, vars_to_plot,
                year_label=args.YEAR, data_lumi=get_lumi(args.YEAR), normalize=True, x_scale_log=x_log,
            )

    elif args.function == "plot_separateMY_histograms":
        for x_log in [False, True]:
            plot_separateMY_uncertainty_histograms(
                f"combine_noempty_input{mass_suffix(args)}.root", args, vars_to_plot,
                year_label=args.YEAR, data_lumi=get_lumi(args.YEAR), normalize=True, x_scale_log=x_log,
            )
        # Also plot just MX / MY with x-log
        plot_separateMY_uncertainty_histograms(
            f"combine_noempty_input{mass_suffix(args)}.root", args, vars_to_plot=["MX", "MY"],
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
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 3b --TestRegion 3btest --Nfold 10 --Plot 1 --CreateUncHist 1 --function create_unc_hists --MX 1000 --MY 150

# Step 2 - add non-closure uncertainty and signal
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --Nfold 10 --Plot 1 --CreateUncHist 1 --function create_unc_hists --MX 1000 --MY 150

# Step 3 – modify NC uncertainties (add MY-3bin uncertainty and modify old NC, deccorelate NC by MX bin, Unroll)
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --function add_decorrelated_nc_uncertainty --MX 1000 --MY 150

# Step 4 – hadd signal, original uncertainty histograms, and new decorrelated NC histograms into a single Combine input file
#   hadd combine_noempty_input_MX-1000_MY-150.root 3bHiggsMW_Uncertainty_hists_OnlyPhysical_MX-1000_MY-150.root combine_addDecorrelatedNC_MX-1000_MY-150.root

# Step 5 – produce final Combine input
#   python3 convert_to_combine_input_DecoMX.py --YEAR 2024 --MX 1000 --MY 150




# Step 1 – compute fold-based uncertainty histograms (4btest first, then 4bHiggsMW)
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 4b --TestRegion 4btest --Nfold 10 --Plot 1 --CreateUncHist 1 --function create_unc_hists --MX 1000 --MY 150

# Step 2 - add non-closure uncertainty
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 4b --TestRegion 4bHiggsMW --Nfold 10 --Plot 0 --CreateUncHist 1 --function create_unc_hists --MX 1000 --MY 150

# Step 3 – modify NC uncertainties (add MY-3bin uncertainty and modify old NC, deccorelate NC by MX bin, Unroll)
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 4b --TestRegion 4bHiggsMW --function add_decorrelated_nc_uncertainty --MX 1000 --MY 150

# Step 4 – hadd original uncertainty histograms, and new decorrelated NC histograms into a single Combine input file
#   hadd combine_noempty_input_MX-1000_MY-150.root 4bHiggsMW_Uncertainty_hists_OnlyPhysical_MX-1000_MY-150.root combine_addDecorrelatedNC_MX-1000_MY-150.root

# Step 5 – apply background normalization scale factor
#   python3 uncertainty_pipeline.py --YEAR 2024 --runType test-only --TrainRegion 4b --TestRegion 4bHiggsMW --function apply_bkg_norm_scalefactor --MX 1000 --MY 150

#   python3 convert_to_combine_input_DecoMX.py --YEAR 2024 --MX 1000 --MY 150
