#######################################################

# This script extracts expected limits from Combine output ROOT files for various signal scenarios
# and plots the expected limits vs MX for each MY. 
# It saves both linear and log-scale plots. 

# command to run this scripts:
# python3 plotLimits.py
# please check "input_dir", "output_dir", "templates", "unavailable" and "label_list" before running

#######################################################

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

data_lumi = 1.0
if args.YEAR=="2023BPiX":
    data_lumi = 9.45
elif args.YEAR=="2023":
    data_lumi = 11.2
elif args.YEAR=="2022EE":
    data_lumi = 22.1
elif args.YEAR=="2024":
    data_lumi = 109
elif args.YEAR=="2025":
    data_lumi = 111
elif args.YEAR=="2022":
    data_lumi = 7.98
else:
    data_lumi = 111+109

suffix = args.SUFFIX

def plot2D(group,variable_to_plot,z_axis_title,outdir,filename,cmap_style='OrRd',add_Text=False,Text_to_add="", vmax=None):

    plt.style.use(hep.style.CMS)

    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14

    pivot = group.pivot(index="MY", columns="MX", values=variable_to_plot)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    print("Pivot table for plotting:")
    print(pivot)


    imshow_kwargs = {
        "origin": "lower",
        "aspect": "auto",
        "cmap": cmap_style
    }
    if vmax is not None:
        imshow_kwargs["vmax"] = vmax
    
    im = ax.imshow(pivot.values, **imshow_kwargs)

    ax.set_xlabel("$m_X$ [GeV]",loc="right", labelpad=8)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    #ax.set_xticks(pivot.columns)

    ax.set_ylabel("$m_Y$ [GeV]",loc="top", labelpad=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    #ax.set_yticks(pivot.index)

    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=0.5)

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    #plt.xlim(350,1950)
    #plt.ylim(50,1500)
    #ax.set_xlim([350, 1950])

    fig.colorbar(im, ax=ax, label=z_axis_title)
    

    if add_Text:
        ax.text(
            0.2, 0.875,                   # x, y in axes coordinates (0-1)
            Text_to_add,                  # text
            transform=ax.transAxes,       # coordinates relative to axes
            ha="center",                  # horizontal alignment
            va="bottom",                  # vertical alignment
            fontsize=12,
            color="black"
        )

    
    for i, y in enumerate(pivot.index):
        for j, x in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)
                #ax.text(x, y, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)
    
    hep.cms.text("Simulation", loc=0, ax=ax)
    ax.text(1.0, 1.02, "L = "+str(data_lumi)+" fb$^{-1}$ (13.6 TeV)",transform=ax.transAxes, ha="right", va="bottom", fontsize=18)

    out_file_name =  f"{outdir}/{filename}.png"

    plt.savefig(os.path.join(outdir, out_file_name), dpi=1000, bbox_inches="tight")


def plotLimits(log=False,inDir="",outDir="",shapes=[],legends=[]):

    #input_dir = "/afs/desy.de/user/c/chokepra/private/XtoYH4b/CMSSW_14_2_1/src/CombineHarvester/CombineTools/XYHto4b/workspace_v3"
    #output_dir ="/afs/desy.de/user/c/chokepra/private/XtoYH4b/CMSSW_14_2_1/src/CombineHarvester/CombineTools/XYHto4b/ExpLimits_v3"

    os.makedirs(outDir, exist_ok=True)

    # v1, v2, v3 (no unavailable)
    unavailable = []

    unavailable_df = pd.DataFrame(unavailable, columns=["Scenario", "MX", "MY"])

    file_list = []
    print("inDir",inDir)
    for template in shapes:
        files = glob.glob(os.path.join(inDir, f"higgsCombine_Comb_{template}*"))
        file_list.extend([f for f in files])

    data = []
    mx_list = []
    my_list = []

    failed_files = []

    # for root_file in args.input
    for root_file in file_list:

        file_name = os.path.basename(root_file)

        pattern = r"Comb_(\d+)_MX_(\d+)_MY_(\d+)"
        match = re.search(pattern, file_name)
        
        if match:
            scenario, MX, MY = map(int, match.groups())
            print(f"Comb = {scenario}, MX = {MX}, MY = {MY}")

        #scenario, MX, MY = file_name.split("higgsCombine")[1].split(".")[0].split("_")

        if scenario==6:
            mx_list.append(MX)
            my_list.append(MY)

        file = ROOT.TFile.Open(root_file)
        tree = file.Get("limit")

        limits = [confidence_level.limit for confidence_level in tree]

        sigma_1_up, sigma_1_dn, sigma_2_up, sigma_2_dn = 0, 0, 0, 0

        #changing units from pb to fb
        for lx in range(len(limits)):
            limits[lx] *= 1000

        if len(limits) == 0:
            print(f"Warning: No limits found in {root_file}. Skipping.")
            continue
            

        if len(limits) != 6: # 6 -> 5 expected + 1 observed
            observed = limits[-1] # -1 = observed
            expected = observed
            failed_files.append(file_name)
        else:
            expected = limits[2] # 2 = median

            sigma_1_up, sigma_1_dn, sigma_2_up, sigma_2_dn = limits[3], limits[1], limits[4], limits[0]

        data.append([scenario, MX, MY, expected, sigma_1_up, sigma_1_dn, sigma_2_up, sigma_2_dn])

        print("expected limit",expected)

    unique_MX = sorted(set(mx_list))
    unique_MY = sorted(set(my_list))

    df_org = pd.DataFrame(data, columns=["scenario", "MX", "MY", "expected","sigma_1_up","sigma_1_dn","sigma_2_up","sigma_2_dn"])

    df = df_org[(df_org["MX"]>350) & (df_org["MX"]<=2200)]

    # plotting limits for each scenario

    for scenario, group in df.groupby("scenario"):
      
        plot2D(group,"expected","Expected limits at 95% CL [fb]",outDir,f"Exp_limits_MX_MY_{scenario}",'OrRd',True,legends[scenario-1])
        plot2D(group,"expected","Expected limits at 95% CL [fb]",outDir,f"Exp_limits_MX_MY_{scenario}_vmax200",'OrRd',True,legends[scenario-1], vmax=200)

    # ratio between combine and inclusive signal regions #

    # df_s5 = df[df["scenario"] == 5]
    # df_s6 = df[df["scenario"] == 6]

    # df_merged = pd.merge(
    #     df_s5[["MX", "MY", "expected"]],
    #     df_s6[["MX", "MY", "expected"]],
    #     on=["MX", "MY"],
    #     suffixes=("_s5", "_s6")
    # )

    # df_merged["ratio"] = df_merged["expected_s5"] / df_merged["expected_s6"]

    # plot2D(df_merged,"ratio","Ratio of expected limits",outDir,"Ratios_new",'coolwarm')
    return failed_files

if __name__ == "__main__":

    
    path_dir="/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/CombineResults/bkg/"
    # if args.YEAR=="2024":
    #     path_dir+="SignalExtraction/"
    if args.NoSys:
        input_dir= path_dir+args.YEAR+"/limits_v5_nosys"
    else:
        input_dir= path_dir+args.YEAR+f"/limits{suffix}"

    if args.NoSys:
        output_dir = os.getcwd()+"/LimitsPlots/"+args.YEAR+"/plots_nosys/"
    else:
        output_dir = os.getcwd()+"/LimitsPlots/"+args.YEAR+"/plots"+suffix+"/"

    os.makedirs(output_dir, exist_ok=True)
    log_file_name = f"PlotLimits_log_{args.YEAR}{args.SUFFIX}.txt"
    sys.stdout = Logger(f"{output_dir}{log_file_name}")
    print(f"Logging all output to {log_file_name}")

    #templates = [1, 2, 3, 4, 5, 6]
    #label_list = ["XXT,XXT,>=XT,>=XT", "XXT,XXT,XT,XT", "XXT,XXT,XXT,XT", "XXT,XXT,XXT,XXT", "combined", ">=T,>=T,>=T,>=M"]

    templates = [1]
    label_list = [">=T,>=T,>=T,>=M"] # , "MX", "MY"]
    # label_list = ["XXT,XXT,[>=XT,>=XT]", ">=T,>=T,>=T,>=M"]

    failed_files = plotLimits(log=False,inDir=input_dir,outDir=output_dir,shapes=templates,legends=label_list)

    number_of_failed_files = len(failed_files)
    if number_of_failed_files > 0:
        print(f"{number_of_failed_files} Failed files (not 5 expected + 1 observed):")
        print(failed_files)

    print("See the limits plots in ",output_dir)
    print("See the log file for details: ",output_dir+log_file_name)

    #plotLimits(log=True)
