import ROOT
import os
import argparse
import multiprocessing

parser = argparse.ArgumentParser(description="Arguments: YEAR")

parser.add_argument('--YEAR', default="2022", type=str, help="Which era?")
parser.add_argument('--Btag_SF', default="WP", type=str, choices=["WP", "Shape"], help="Which b-tagging scale factor?")

args = parser.parse_args()

def read_sample_names(file_path):
    with open(file_path, 'r') as f:
        sample_names = [line.strip() for line in f.readlines() if line.strip()]
    return sample_names

def get_histogram_from_root(file_path, histogram_name):
    file = ROOT.TFile.Open(file_path, "READ")
    if not file:
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
    if not file:
        print(f"Error: ROOT file {file_path} could not be opened!")
        return None
    
    if "h_MX_MY_Comb_3_3_3_2_mHcut" in histogram_name:
        return None
        
    if histogram_name == "h_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut":
        histogram_name = "h_MX_MY_Comb_3_3_3_2_Inclusive_mHcut"
    directory = file.Get(histogram_name)
    if not directory:
        print(f"Error: Directory '{histogram_name}' not found in {file_path}!")
        file.Close()
        return None
    
    histograms_bkg = []
    for key in directory.GetListOfKeys():
        obj_name = key.GetName()

        # if "NMSSM_" in obj_name or "data_obs" in obj_name:
        if "NMSSM_" in obj_name:
            continue
        obj = key.ReadObj()

        if isinstance(obj, ROOT.TH1):
            obj.SetDirectory(0)  
            histograms_bkg.append(obj)
            
    file.Close()
    
    return histograms_bkg


def write_outputfile(output_file, input_dir, signals, backgrounds, data, histogram_name="hx",signal_systematics=[],lumi=1.):
   
    print("Running for",histogram_name)

    # Add process information
    processes = []  # Store process names: signal + backgrounds
    hists = []

    # Process signal
    for sig in signals:

        sig_file_path = input_dir+"Histogram_"+sig+".root"
        print(sig_file_path)
        sig_file = ROOT.TFile.Open(sig_file_path, "READ")
        #hist = get_histogram_from_root(sig_file, histogram_name)
        hist = sig_file.Get(histogram_name)
        
        if hist:
            hist.SetDirectory(0)
            hist.SetName(sig)
            hists.append(hist)
            processes.append(sig)  # Signal process

            for syst in signal_systematics:
                #up
                histogram_name_sys = histogram_name + "_Sys_" + syst + "_up"
                hist_up = sig_file.Get(histogram_name_sys)
                if hist_up:
                    hist_up.SetDirectory(0)
                    if syst in signal_systematic_eras_decorrelated:
                        hist_up.SetName(sig+"_"+syst+f"_{args.YEAR}Up")
                    else:
                        hist_up.SetName(sig+"_"+syst+"Up")
                    hists.append(hist_up)
                #down
                histogram_name_sys = histogram_name + "_Sys_" + syst + "_down"
                hist_dn = sig_file.Get(histogram_name_sys)
                if hist_dn:
                    hist_dn.SetDirectory(0)
                    if syst in signal_systematic_eras_decorrelated:
                        hist_dn.SetName(sig+"_"+syst+f"_{args.YEAR}Down")
                    else:
                        hist_dn.SetName(sig+"_"+syst+f"Down")
                    hists.append(hist_dn)

            #close file
            sig_file.Close()

    # Process background files
    # for bkg in backgrounds:
    #     bkg_file = input_dir+"Output_"+bkg+".root"
    #     print(bkg_file)
    #     hist = get_histogram_from_root(bkg_file, histogram_name)
    #     if hist:
    #         processes.append(bkg)  # Background process
    #         hist.SetName(bkg)
    #         hists.append(hist)

    # data_file = input_dir+"Output_"+data+".root"
    # hist = get_histogram_from_root(data_file,histogram_name)
    # if hist:
    #     processes.append(data)
    #     hist.SetName("data_obs")
    #     hists.append(hist)

    # Process background estimation files
    bkg_file = f"{bkg_dir}Output_Background_{args.YEAR}.root"
    print(bkg_file)
    bkg_hists = get_histogram_from_BkgHist(bkg_file, histogram_name)
    if bkg_hists:
        for hist in bkg_hists:
            proc_name = hist.GetName()    
            processes.append(proc_name)   
            hists.append(hist)   

    # Write the histogram to file

    output_file.cd()
    dirc = output_file.mkdir(histogram_name)
    dirc.cd()

    for ih, hist in enumerate(hists):
        # if ih<(len(hists)-1) or "Inclusive" in str(hist.GetName()):
        if "Inclusive" not in str(hist.GetName()) and "data_obs" not in str(hist.GetName()):
            hist.Scale(lumi)
        hist.Write()

def process_histogram(histogram_name, output_file_name, input_dir, signals, backgrounds, data, signal_systematic_uncs):
    # Re-open the output file 
    ROOT.TFile.Open(output_file_name, "UPDATE") # Use UPDATE to avoid overwriting
    
    # Call the main writing function (using the optimized version from section 1)
    write_outputfile(output_file, input_dir, signals, backgrounds, data, 
                     histogram_name=histogram_name, 
                     signal_systematics=signal_systematic_uncs)
    
    output_file.Close() # Close the file 

#Input signal and background processes here

input_dir = "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/"+args.YEAR+"/"
bkg_dir = "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/BackgroundEstimation/"+args.YEAR+"/"

# Change this to your XtoYH4b directory
XtoYH4b_dir = "/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b"

signals = read_sample_names(XtoYH4b_dir+"/SIGNAL_names.txt") # Can coppy this to group area? 

data_lumi = float(7.98)
if(args.YEAR=="2023"):
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

print("Signals:")
for sig in signals:
    print(sig)

#if some signals need to be excluded
exclude_signals = []
#if args.YEAR=="2023BPiX":
#    exclude_signals = ["NMSSM_XtoYHto4B_MX-600_MY-60_TuneCP5_13p6TeV_madgraph-pythia8","NMSSM_XtoYHto4B_MX-1600_MY-800_TuneCP5_13p6TeV_madgraph-pythia8"]
signals_filtered = [sig for sig in signals if sig not in exclude_signals]
signals = signals_filtered

backgrounds = ["TT","ST","Zto2Q","Wto2Q","Diboson","QCD","SingleH","DoubleH" ]

signal_systematic_uncs = [
	 "JES_AbsoluteStat", "JES_AbsoluteScale","JES_AbsoluteMPFBias", 
	 "JES_FlavorQCD", "JES_Fragmentation", 
	 "JES_PileUpDataMC",  "JES_PileUpPtBB", "JES_PileUpPtEC1", "JES_PileUpPtEC2", 
	 "JES_PileUpPtRef",
	 "JES_RelativeFSR", "JES_RelativeJEREC1", "JES_RelativeJEREC2", 
	 "JES_RelativePtBB", "JES_RelativePtEC1", "JES_RelativePtEC2", 
	 "JES_RelativeBal", "JES_RelativeSample", "JES_RelativeStatEC", "JES_RelativeStatFSR", 
	 "JES_SinglePionECAL", "JES_SinglePionHCAL","JES_TimePtEta",
	#  "JES_Total",
	 "JER",
	 "PU",
	 "LHEScale","LHEPDF","LHEAlphaS","PS_ISR","PS_FSR"
    #  "LHEScale_muR","LHEScale_muF",
]

signal_systematic_eras_decorrelated = [
    "JER", "JES_AbsoluteMPFBias", "JES_AbsoluteScale", "JES_FlavorQCD", "JES_Fragmentation",
    "JES_PileUpDataMC",  "JES_PileUpPtBB", "JES_PileUpPtEC1", "JES_PileUpPtEC2", 
    "JES_RelativeFSR", "JES_RelativePtBB", "JES_RelativeBal", 
    "JES_SinglePionECAL", "JES_SinglePionHCAL"
]





if args.Btag_SF == "WP":
    signal_systematic_uncs.extend(["Btag_WP_SF_correlated","Btag_WP_SF_uncorrelated"])
elif args.Btag_SF == "Shape":
    signal_systematic_uncs.extend(["Btag_SF_jes","Btag_SF_lf","Btag_SF_lfstats1","Btag_SF_lfstats2",
                                    "Btag_SF_hf","Btag_SF_hfstats1","Btag_SF_hfstats2","Btag_SF_cferr1","Btag_SF_cferr2",
	                                "Btag_SF_correction"])

data = "Data"
if args.YEAR=="2023" or args.YEAR=="2023BPiX" or args.YEAR=="2024" or args.YEAR=="2025":
    data = "Data_Parking"

# Output directory & file

output_dir = "/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/CombineHarvester/CombineTools/bin/InputFiles/"

if not os.path.exists(output_dir):
    print(f"Input file directory '{output_dir}' does not exist. Creating it.")
    os.makedirs(output_dir)

output_filename = os.path.join(output_dir, "combine_input_XYH4b_"+args.YEAR+".root")
if os.path.exists(output_filename):
    print(f"Output file '{output_filename}' already exists. It will be overwritten.")
    os.remove(output_filename)
output_file = ROOT.TFile.Open(output_filename, "RECREATE")

histograms_to_process = [
    ##v1/v2##
    #"h_MX_Comb_5_5_4_4_Inclusive",
    #"h_MX_Comb_5_5_4_4",
    #"h_MX_Comb_5_5_5_4",
    #"h_MX_Comb_5_5_5_5",
    #"h_MX_Comb_3_3_3_2_Inclusive",
    ##v3##
    #"h_MX_Comb_5_5_4_4_Inclusive_mHcut",
    #"h_MX_Comb_5_5_4_4_mHcut",
    #"h_MX_Comb_5_5_5_4_mHcut",
    #"h_MX_Comb_5_5_5_5_mHcut",
    #"h_MX_Comb_3_3_3_2_Inclusive_mHcut",
    ##v4##
    #"h_MX_MY_Comb_5_5_4_4_Inclusive",
    #"h_MX_MY_Comb_5_5_4_4",
    #"h_MX_MY_Comb_5_5_5_4",
    #"h_MX_MY_Comb_5_5_5_5",
    #"h_MX_MY_Comb_3_3_3_2_Inclusive",
    ##v5##
    # "h_MX_MY_Comb_5_5_4_4_Inclusive_mHcut",
    # "h_MX_MY_Comb_5_5_4_4_mHcut",
    # "h_MX_MY_Comb_5_5_5_4_mHcut",
    # "h_MX_MY_Comb_5_5_5_5_mHcut",
    # "h_MX_MY_Comb_3_3_3_2_Inclusive_mHcut",
    ##v6## or not?
    # "h_MX_MY_Comb_3_3_3_2_Inclusive",
    # "h_MX_Comb_3_3_3_2_Inclusive",
    # "h_MY_Comb_3_3_3_2_Inclusive",
    # "h_MX_MY_index_Comb_3_3_2_2_Inclusive",
]
# histograms for BDT score based selection #
if args.YEAR=="2023BPiX" or args.YEAR=="2024" or args.YEAR=="2025":
    histograms_to_process.extend([
        # "h_MaxScore_MX_MY_Comb_5_5_4_4_Inclusive_mHcut",
        # "h_MaxScore_MX_MY_Comb_5_5_4_4_mHcut",
        # "h_MaxScore_MX_MY_Comb_5_5_5_4_mHcut",
        # "h_MaxScore_MX_MY_Comb_5_5_5_5_mHcut",
        # "h_MaxScore_MX_MY_Comb_3_3_3_2_Inclusive_mHcut",
        # "h_MX_MY_Comb_3_3_3_2_Inclusive",
        # "h_MX_Comb_3_3_3_2_Inclusive",
        # "h_MY_Comb_3_3_3_2_Inclusive",
        # "h_MX_MY_index_Comb_3_3_2_2_Inclusive",
        "h_MX_MY_Comb_3_3_3_2_Inclusive_mHcut",
        "h_MX_Comb_3_3_3_2_Inclusive_mHcut",
        "h_MY_Comb_3_3_3_2_Inclusive_mHcut",
        "h_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut",
    ])

# Write to the output file

# try with only one signal
# signals1 = ["NMSSM-XtoYHto4B_Par-MX-1000-MY-150_TuneCP5_13p6TeV_madgraph-pythia8"]
# for histo in histograms_to_process:
#     write_outputfile(output_file, input_dir, signals1, backgrounds, data, histogram_name=histo, signal_systematics=signal_systematic_uncs, lumi=data_lumi)
for histo in histograms_to_process:
    write_outputfile(output_file, input_dir, signals, backgrounds, data, histogram_name=histo, signal_systematics=signal_systematic_uncs, lumi=data_lumi)

#output_file.Write()
#output_file.Close()
'''
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=10) # Adjust number of processes based on your CPU cores
    
    # tasks = [(hist_name, output_filename, input_dir, signals1, backgrounds, data, signal_systematic_uncs) 
    #          for hist_name in histograms_to_process]
    tasks = [(hist_name, output_filename, input_dir, signals, backgrounds, data, signal_systematic_uncs) 
             for hist_name in histograms_to_process]

    pool.starmap(process_histogram, tasks)
    
    pool.close()
    pool.join()
'''
