import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os

hep.style.use(hep.style.CMS)
ROOT.gROOT.SetBatch(True) # Prevent X11 window popups in SSH

variable_names = ["Unrolled_MXMY", "MX", "MY"]





def plot_result_ratio(i, plot_fit):
    if plot_fit == "PostFit":
        root_directory_name = "postfit"
    if plot_fit == "PreFit":
        root_directory_name = "prefit"

    input_file = f"PostFitResults_XYH4b_2024_Comb-{i}_MX-1000_MY-150_Signal_0_Bonly.root"
    directory_name = f"XYH_4b_{i}_13p6TeV_2024_{root_directory_name}"
    variable_name = variable_names[i-1]

    output_name = f"{plot_fit}_Ratio_{variable_name}_MX1000_MY150.png"

    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return

    f = ROOT.TFile.Open(input_file)
    dir_postfit = f.Get(directory_name)
    
    if not dir_postfit:
        print(f"Error: Directory {directory_name} not found in file.")
        return

    h_data = dir_postfit.Get("data_obs")
    h_bkg = dir_postfit.Get("TotalBkg")

    if not h_data or not h_bkg:
        print("Error: Could not find data_obs or TotalBkg histograms.")
        return

    nbins = h_bkg.GetNbinsX()
    edges = np.array([h_bkg.GetBinLowEdge(i) for i in range(1, nbins + 2)])
    centers = np.array([h_bkg.GetBinCenter(i) for i in range(1, nbins + 1)])

    data_val = np.array([h_data.GetBinContent(i) for i in range(1, nbins + 1)])
    data_err = np.array([h_data.GetBinError(i) for i in range(1, nbins + 1)])

    bkg_val = np.array([h_bkg.GetBinContent(i) for i in range(1, nbins + 1)])
    bkg_err = np.array([h_bkg.GetBinError(i) for i in range(1, nbins + 1)])

    # Calculate the chi2
    chi2 = h_data.Chi2Test(h_bkg, "WW CHI2/NDF")
    # print(f"Chi2/NDF using ROOT method: {chi2:.2f}")
    def calculate_chi2(h_exp, h_obs):
        chi2 = 0.0
        ndf = 0
        n_bins = h_obs.GetNbinsX()

        for i in range(1, n_bins + 1):
            obs = h_obs.GetBinContent(i)
            exp = h_exp.GetBinContent(i)
            
            # Total error in quadrature
            err_obs = h_obs.GetBinError(i)
            err_exp = h_exp.GetBinError(i)
            err_sq = err_obs**2 + err_exp**2

            # Avoid division by zero
            if err_sq > 0:
                chi2 += (obs - exp)**2 / err_sq
                ndf += 1
                
        return chi2, ndf
    chi2_calc, ndf = calculate_chi2(h_data, h_bkg)
    # print(f"Calculate Chi2/NDF: {chi2_calc:.2f} / {ndf} = {chi2_calc/ndf:.2f}")

    def get_chi2_root_method(h_data, h_nom):
        """
        Use ROOT's built-in Chi2Test method.
        """
        h_total_err = h_nom.Clone("h_total_err_for_chi2")
        
        n_bins = h_nom.GetNbinsX()
        
        for i in range(1, n_bins + 1):

            sigma_stat = h_nom.GetBinError(i)
            
            #sigma_total = np.sqrt(sigma_stat**2 + sigma_sys**2)
            h_total_err.SetBinError(i, sigma_stat)
            
        chi2 = h_data.Chi2Test(h_total_err, "WW CHI2/NDF")
        
        return chi2
    chi2_root = get_chi2_root_method(h_data, h_bkg)
    #  print(f"Chi2/NDF using ROOT function method: {chi2_root:.2f}")


    mask = bkg_val > 0  # Only compute ratio where background is non-zero

    # print if the bakground is zero in any bin
    if np.any(~mask):
        zero_bins = np.where(~mask)[0] + 1  # Bin numbers are 1-indexed in ROOT
        print(f"Warning: Background is zero in bins {zero_bins}. Ratio will be undefined in these bins.")

    ratio = np.full_like(data_val, np.nan)
    ratio_err = np.full_like(data_err, np.nan)
    # ratio_err = np.zeros_like(data_err)
    rel_bkg_err = np.zeros_like(bkg_val)

    ratio[mask] = data_val[mask] / bkg_val[mask]
    ratio_err[mask] = data_err[mask] / bkg_val[mask]
    
    rel_bkg_err[mask] = bkg_err[mask] / bkg_val[mask]

    fig, ax = plt.subplots(figsize=(12, 8))

## Plot
    x_fill = edges
    y_fill_up = np.append(1.0 + rel_bkg_err, (1.0 + rel_bkg_err)[-1])
    y_fill_dn = np.append(1.0 - rel_bkg_err, (1.0 - rel_bkg_err)[-1])

    ax.fill_between(x_fill, y_fill_dn, y_fill_up, step='post', 
                    facecolor='none', edgecolor='gray', hatch='////', 
                    alpha=0.5, label=f'{plot_fit} Bkg. Uncertainty')
    
    ax.errorbar(centers, ratio, yerr=ratio_err, fmt='ko', 
                markersize=4, capsize=0, label=f'Data / Total Bkg $\chi^2/NDF = {chi2_calc/ndf:.2f}$')

    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_ylim(0.8, 1.2)  
    ax.set_xlim(edges[0], edges[-1])

    ax.set_xlabel(f"{variable_name} Bins", fontsize=18)
    ax.set_ylabel("Data / Bkg", fontsize=18)
    
    ax.legend(loc='upper right', fontsize=18, frameon=False)

    hep.cms.label(
        ax=ax,
        exp="",
        label="Private work (CMS data)",
        data=True, 
        year="2024",
        lumi=109,
        com=13.6,
        fontsize=24
    )

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.savefig(output_name.replace('.png', '.pdf'))

# plot the histogram directly
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    hep.histplot(bkg_val, bins=edges, histtype='step', color='blue', label=f'{plot_fit} Total Bkg', ax=ax2)
    hep.histplot(data_val, bins=edges, histtype='step', linestyle='--', color='red', label=f'Data $\chi^2/NDF = {chi2_calc/ndf:.2f}$', ax=ax2)
    ax2.set_xlabel(f"{variable_name} Bins", fontsize=18)
    ax2.set_ylabel("Events", fontsize=18)
    ax2.legend(loc='upper right', fontsize=18, frameon=False)
    hep.cms.label(
        ax=ax2,
        exp="",
        label="Private work (CMS data)",
        data=True,
        year="2024",
        lumi=109,
        com=13.6,
        fontsize=24
    )
    
    plt.tight_layout()
    output_name_hist = f"{plot_fit}_Hist_{variable_name}_MX1000_MY150.png"
    plt.savefig(output_name_hist, dpi=300)
    plt.savefig(output_name_hist.replace('.png', '.pdf'))


    print(f"Successfully saved plot to {output_name}")

    difference = [data_val[i] - bkg_val[i] for i in range(nbins)]
    sum_diff = sum(difference)
    # print(f"Sum of differences between data and background across all bins: {sum_diff:.2f}")

    with open(f"{plot_fit}_Results_{variable_name}_MX1000_MY150.txt", "w") as f_out:
        f_out.write(f"Chi2/NDF use root function: {chi2:.2f}\n")
        f_out.write(f"Calculate Chi2/NDF: {chi2_calc/ndf:.2f}\n")
        f_out.write(f"Chi2/NDF using ROOT function method: {chi2_root:.2f}\n")
        f_out.write("\n")
        f_out.write(f"Sum of differences between data and background across all bins: {sum_diff:.2f}\n")
        f_out.write("\n")
        for i in range(1, nbins + 1):
            f_out.write(f"Bin {i}: Data = {data_val[i-1]:.2f} ± {data_err[i-1]:.2f}, Bkg = {bkg_val[i-1]:.2f} ± {bkg_err[i-1]:.2f}, Ratio = {ratio[i-1]:.2f} ± {ratio_err[i-1]:.2f}\n")
            f_out.write(f"    Relative Bkg Uncertainty: {rel_bkg_err[i-1]:.2f}\n")
            f_out.write("\n")
            f_out.write(f"Difference between data and bkg: {data_val[i-1] - bkg_val[i-1]:.2f}\n")
            f_out.write(f"\n")



for i in range(1, 4):
    plot_result_ratio(i, "PostFit")
    plot_result_ratio(i, "PreFit")