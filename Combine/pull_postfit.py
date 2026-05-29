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



def post_pull(i):

    input_file = f"PostFitResults_XYH4b_2024_Comb-{i}_MX-1000_MY-150_Signal_0_Bonly.root"
    directory_name = f"XYH_4b_{i}_13p6TeV_2024_postfit"
    variable_name = variable_names[i-1]

    output_name = f"Pull_PostFit_{variable_name}_MX1000_MY150.png"

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

    total_err = np.sqrt(data_err**2 + bkg_err**2)
    pull = (data_val - bkg_val) / total_err

    pull_err = np.zeros_like(pull)
    mask = total_err > 0
    pull_err[mask] = data_err[mask] / total_err[mask]


    # print(f"Pull values for {variable_name}: {pull}")
    with open(f"Pull_PostFit_{variable_name}_MX1000_MY150.txt", "w") as f_out:
        f_out.write(f"Pull values for {variable_name}:\n")
        for i in range(nbins):
            f_out.write(f"Bin {i+1}: Pull = {pull[i]:.2f},\n Data value = {data_val[i]:.2f}, Bkg value = {bkg_val[i]:.2f},\n Data error = {data_err[i]:.2f}, Bkg error = {bkg_err[i]:.2f}\n")
            f_out.write("\n")


    # plot

    plt.figure(figsize=(10, 8))
    plt.errorbar(centers, pull, yerr=pull_err, fmt='o', label='Pull (Data - Bkg) / Uncertainty')
    plt.axhline(0, color='red', linestyle='--')
    hep.cms.label(exp="", label="Private work (CMS data)", data=True, lumi=109, com=13.6, year=2024, fontsize=24)
    plt.xlabel(variable_name)
    plt.ylabel('PostFit Pull')
    # plt.title(f'Pull Distribution for {variable_name} (Post-Fit)')
    plt.legend()
    plt.grid()
    plt.savefig(output_name)

for i in range(1, 4):
    post_pull(i)