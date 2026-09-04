import ROOT
import os
import argparse
import multiprocessing
import getpass

parser = argparse.ArgumentParser(description="Arguments: YEAR")

parser.add_argument('--YEAR', default="2022", type=str, help="Which era?")
parser.add_argument('--Btag_SF', default="WP", type=str, choices=["WP", "Shape"], help="Which b-tagging scale factor?")
parser.add_argument('--add-3b-higgs-nc', default=0, type=int, choices=[0, 1], help="Require nc_3b input templates? (default: 0)")

args = parser.parse_args()

ROOT.gROOT.SetBatch(True)

def read_sample_names(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Signal list does not exist: {file_path}")
    with open(file_path, 'r') as f:
        sample_names = [line.strip() for line in f.readlines() if line.strip()]
    return sample_names


def get_histogram_from_root(file_path, histogram_name):
    file = ROOT.TFile.Open(file_path, "READ")
    if not file or file.IsZombie():
        print(f"Error: ROOT file {file_path} could not be opened!")
        return None
    histogram = file.Get(histogram_name)
    if not histogram:
        print(f"Error: Histogram {histogram_name} not found in {file_path}!")
        file.Close()
        return None
    histogram.SetDirectory(0)
    file.Close()
    return histogram


def get_histogram_from_BkgHist(file_path, histogram_name):
    file = ROOT.TFile.Open(file_path, "READ")
    if not file or file.IsZombie():
        print(f"Error: ROOT file {file_path} could not be opened!")
        return None

    directory = file.Get(histogram_name)
    if not directory:
        print(f"Error: Directory '{histogram_name}' not found in {file_path}!")
        file.Close()
        return None

    histograms_bkg = []
    histogram_names = set()
    for key in directory.GetListOfKeys():
        obj_name = key.GetName()

        if "NMSSM_" in obj_name or "NMSSM-" in obj_name:
            continue
        obj = key.ReadObj()

        if isinstance(obj, ROOT.TH1):
            obj.SetDirectory(0)
            histograms_bkg.append(obj)
            histogram_names.add(obj.GetName())

    required_histograms = {
        "data_obs",
        "Inclusive_Bkg",
        f"Inclusive_Bkg_Statistical_Uncertainty_{args.YEAR}Up",
        f"Inclusive_Bkg_Statistical_Uncertainty_{args.YEAR}Down",
        f"Inclusive_Bkg_Systematic_Uncertainty_{args.YEAR}Up",
        f"Inclusive_Bkg_Systematic_Uncertainty_{args.YEAR}Down",
    }
    if args.add_3b_higgs_nc == 1:
        required_histograms.update({
            f"Inclusive_Bkg_NonClosure3b_Uncertainty_{args.YEAR}Up",
            f"Inclusive_Bkg_NonClosure3b_Uncertainty_{args.YEAR}Down",
        })

    missing_histograms = sorted(required_histograms - histogram_names)
    file.Close()
    if missing_histograms:
        raise RuntimeError(
            f"{file_path}:{histogram_name} is missing required 4b shapes: "
            + ", ".join(missing_histograms)
        )

    return histograms_bkg


def write_outputfile(output_file, input_dir, signals, backgrounds, data,
                     histogram_name="hx", signal_systematics=[], lumi=1.,
                     bkg_file_path=None):

    print("Running for", histogram_name)

    processes = []  # Store process names: signal + backgrounds
    hists = []

    # Process signal
    for sig in signals:

        sig_file_path = input_dir + "Histogram_" + sig + ".root"
        print(sig_file_path)
        sig_file = ROOT.TFile.Open(sig_file_path, "READ")
        if not sig_file or sig_file.IsZombie():
            raise RuntimeError(f"Signal ROOT file could not be opened: {sig_file_path}")

        hist = sig_file.Get(histogram_name)
        if not hist:
            sig_file.Close()
            raise RuntimeError(f"Histogram {histogram_name} not found in {sig_file_path}")

        hist.SetDirectory(0)
        hist.SetName(sig)
        hists.append(hist)
        processes.append(sig)

        for syst in signal_systematics:
            # up
            histogram_name_sys = histogram_name + "_Sys_" + syst + "_up"
            hist_up = sig_file.Get(histogram_name_sys)
            if hist_up:
                hist_up.SetDirectory(0)
                if syst in signal_systematic_eras_decorrelated:
                    hist_up.SetName(sig + "_" + syst + f"_{args.YEAR}Up")
                else:
                    hist_up.SetName(sig + "_" + syst + "Up")
                hists.append(hist_up)
            else:
                print(f"[WARNING] Missing signal shape {histogram_name_sys} in {sig_file_path}")

            # down
            histogram_name_sys = histogram_name + "_Sys_" + syst + "_down"
            hist_dn = sig_file.Get(histogram_name_sys)
            if hist_dn:
                hist_dn.SetDirectory(0)
                if syst in signal_systematic_eras_decorrelated:
                    hist_dn.SetName(sig + "_" + syst + f"_{args.YEAR}Down")
                else:
                    hist_dn.SetName(sig + "_" + syst + "Down")
                hists.append(hist_dn)
            else:
                print(f"[WARNING] Missing signal shape {histogram_name_sys} in {sig_file_path}")

        sig_file.Close()

    # The mass-specific BkgEst file contains both data_obs and Inclusive_Bkg.
    if bkg_file_path is None:
        raise ValueError("A mass-specific bkg_file_path is required")
    print(bkg_file_path)
    bkg_hists = get_histogram_from_BkgHist(bkg_file_path, histogram_name)
    if not bkg_hists:
        raise RuntimeError(
            f"No background histograms found in {bkg_file_path}:{histogram_name}"
        )

    for hist in bkg_hists:
        proc_name = hist.GetName()
        processes.append(proc_name)
        hists.append(hist)

    # Write the histograms to file
    output_file.cd()
    dirc = output_file.mkdir(histogram_name)
    dirc.cd()

    for ih, hist in enumerate(hists):
        if "Inclusive" not in str(hist.GetName()) and "data_obs" not in str(hist.GetName()):
            hist.Scale(lumi)
        hist.Write()


def process_histogram(histogram_name, output_file_name, input_dir, signals,
                      backgrounds, data, signal_systematic_uncs,
                      bkg_file_path):
    # Re-open the output file
    output_file = ROOT.TFile.Open(output_file_name, "UPDATE")

    write_outputfile(output_file, input_dir, signals, backgrounds, data,
                     histogram_name=histogram_name,
                     signal_systematics=signal_systematic_uncs,
                     bkg_file_path=bkg_file_path)

    output_file.Close()


# Input signal and background processes here

input_dir = f"/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/{args.YEAR}/SIGNAL/HLT_HT250/"

bkg_dir = f"/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/BackgroundEstimation/{args.YEAR}/"
input_dir = input_dir.rstrip("/") + "/"
bkg_dir = bkg_dir.rstrip("/") + "/"

# Change this to your XtoYH4b directory
XtoYH4b_dir = "/afs/desy.de/user/" + getpass.getuser()[0] + "/" + getpass.getuser() + "/private/work/CMSSW_14_2_1/src/XtoYH4b"

if(args.YEAR=="2022"):
    signals = read_sample_names(XtoYH4b_dir+"/SIGNAL_names.txt") # Can coppy this to group area? 
    data_lumi = 7.98
elif(args.YEAR=="2022EE"):
    signals = read_sample_names(XtoYH4b_dir+"/SIGNAL_names.txt")
    data_lumi = 26.67
elif(args.YEAR=="2023"):
    signals = read_sample_names(XtoYH4b_dir+"/SIGNAL_names_2023.txt")
    data_lumi = 11.24
elif(args.YEAR=="2023BPiX"):
    signals = read_sample_names(XtoYH4b_dir+"/SIGNAL_names_2023BPIX.txt")
    data_lumi = 9.45
elif(args.YEAR=="2024"):
    signals = read_sample_names(XtoYH4b_dir+"/SIGNAL_names_2024.txt")
    data_lumi = 108.96
elif(args.YEAR=="2025"):
    signals = read_sample_names(XtoYH4b_dir+"/SIGNAL_names_2024.txt") # For now, use 2024 signals for 2025!!
    data_lumi = 110.73
else:
    raise ValueError(f"Unsupported YEAR/ERA: {args.YEAR}")

print("Signals:")
for sig in signals:
    print(sig)

# If some signals need to be excluded
exclude_signals = []
signals_filtered = [sig for sig in signals if sig not in exclude_signals]
signals = signals_filtered

backgrounds = ["TT", "ST", "Zto2Q", "Wto2Q", "Diboson", "QCD", "SingleH", "DoubleH"]

signal_systematic_uncs = [
	 "JER",    
	 "JES_AbsoluteStat", "JES_AbsoluteScale","JES_AbsoluteMPFBias", 
	 "JES_FlavorQCD", "JES_Fragmentation", 
	 "JES_PileUpDataMC",  "JES_PileUpPtBB", "JES_PileUpPtEC1", "JES_PileUpPtEC2", 
	 "JES_PileUpPtRef",
	 "JES_RelativeFSR", "JES_RelativeJEREC1", "JES_RelativeJEREC2", 
	 "JES_RelativePtBB", "JES_RelativePtEC1", "JES_RelativePtEC2", 
	 "JES_RelativeBal", "JES_RelativeSample", "JES_RelativeStatEC", "JES_RelativeStatFSR", 
	 "JES_SinglePionECAL", "JES_SinglePionHCAL","JES_TimePtEta",
	#  "JES_Total",

	 "PU",
	 "LHEScale","LHEPDF","LHEAlphaS","PS_ISR","PS_FSR",
    #  "LHEScale_muR","LHEScale_muF",
]

signal_systematic_eras_decorrelated = [
    "JER",    
    "JES_AbsoluteStat", "JES_RelativeJEREC1", "JES_RelativeJEREC2",
    "JES_RelativePtEC1", "JES_RelativePtEC2", 
    "JES_RelativeSample", "JES_RelativeStatEC", "JES_RelativeStatFSR",
    "JES_TimePtEta",
]

if args.Btag_SF == "WP":
    signal_systematic_uncs.extend(["Btag_WP_SF_correlated", "Btag_WP_SF_uncorrelated"])
    signal_systematic_eras_decorrelated.extend(["Btag_WP_SF_uncorrelated"])
elif args.Btag_SF == "Shape":
    signal_systematic_uncs.extend(["Btag_SF_jes","Btag_SF_lf","Btag_SF_lfstats1","Btag_SF_lfstats2",
                                    "Btag_SF_hf","Btag_SF_hfstats1","Btag_SF_hfstats2","Btag_SF_cferr1","Btag_SF_cferr2",
	                                "Btag_SF_correction"])

data = "Data"
if args.YEAR in ["2023", "2023BPiX", "2024", "2025"]:
    data = "Data_Parking"


output_dir = "/afs/desy.de/user/" + getpass.getuser()[0] + "/" + getpass.getuser() + "/private/work/CMSSW_14_2_1/src/CombineHarvester/CombineTools/bin/InputFiles/"

if not os.path.exists(output_dir):
    print(f"Input file directory '{output_dir}' does not exist. Creating it.")
    os.makedirs(output_dir)

histograms_to_process = [
    "h_MaxScore_MX_Comb_3_3_3_2_Inclusive_mHcut",
    "h_MaxScore_MY_Comb_3_3_3_2_Inclusive_mHcut",
    "h_MaxScore_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut",
]

# Each signal mass point has its own DNN background estimate and Combine file.
for sig in signals:
    sig_file_path = input_dir + "Histogram_" + sig + ".root"
    bkg_file_path = bkg_dir + "BkgEst_" + sig + ".root"
    output_filename = os.path.join(
        output_dir, "combine_input_XYH4b_" + args.YEAR + "_" + sig + ".root"
    )

    for required_file in [sig_file_path, bkg_file_path]:
        if not os.path.isfile(required_file):
            raise FileNotFoundError(f"Required input does not exist: {required_file}")

    # if os.path.exists(output_filename):
    #     print(f"Output file '{output_filename}' already exists. It will be overwritten.")
    if os.path.exists(output_filename):
        print(f"Output file '{output_filename}' already exists. Skipping...")
        continue



    output_file = ROOT.TFile.Open(output_filename, "RECREATE")
    if not output_file or output_file.IsZombie():
        raise RuntimeError(f"Output ROOT file could not be created: {output_filename}")

    for histo in histograms_to_process:
        write_outputfile(
            output_file, input_dir, [sig], backgrounds, data,
            histogram_name=histo,
            signal_systematics=signal_systematic_uncs,
            lumi=data_lumi,
            bkg_file_path=bkg_file_path,
        )

    output_file.Close()

print(f"Created {len(signals)} mass-specific Combine input files.")
