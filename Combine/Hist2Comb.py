import ROOT
import os

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

def write_outputfile(output_file, input_dir, signals, backgrounds, data, histogram_name="hx"):
    
    # Add process information
    processes = []  # Store process names: signal + backgrounds
    hists = []

    # Process signal
    for sig in signals:
        sig_file = input_dir+"Histogram_"+sig+".root"
        print(sig_file)
        hist = get_histogram_from_root(sig_file, histogram_name)
        if hist:
            processes.append(sig)  # Signal process
            hist.SetName(sig)
            hists.append(hist)

    # Process background files
    for bkg in backgrounds:
        bkg_file = input_dir+"Output_"+bkg+".root"
        print(bkg_file)
        hist = get_histogram_from_root(bkg_file, histogram_name)
        if hist:
            processes.append(bkg)  # Background process
            hist.SetName(bkg)
            hists.append(hist)

    data_file = input_dir+"Output_"+data+".root"
    hist = get_histogram_from_root(data_file,histogram_name)
    if hist:
        processes.append(data)
        hist.SetName("data_obs")
        hists.append(hist)

    # Write the histogram to file

    output_file.cd()
    dirc = output_file.mkdir(histogram_name)
    dirc.cd()

    for hist in hists:
        hist.Write()

#Input signal and background processes here

input_dir = "/data/dust/user/chatterj/XToYHTo4b/SmallNtuples/Histograms/2022/"

signals = [
        "NMSSM_XtoYHto4B_MX-4000_MY-400_TuneCP5_13p6TeV_madgraph-pythia8",
        "NMSSM_XtoYHto4B_MX-4000_MY-600_TuneCP5_13p6TeV_madgraph-pythia8"
        ]

backgrounds = ["TT","ST","Zto2Q","Wto2Q","Diboson","QCD" ]

data = "Data"

# Output directory & file

output_dir = "datacards"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_filename = os.path.join(output_dir, "combine_input_XYH4b.root")
output_file = ROOT.TFile.Open(output_filename, "RECREATE")

# Write to the output file
write_outputfile(output_file, input_dir, signals, backgrounds, data, histogram_name="h_MX_Comb_5_5_4_4_Inclusive")
write_outputfile(output_file, input_dir, signals, backgrounds, data, histogram_name="h_MX_Comb_5_5_4_4")
write_outputfile(output_file, input_dir, signals, backgrounds, data, histogram_name="h_MX_Comb_5_5_5_4")
write_outputfile(output_file, input_dir, signals, backgrounds, data, histogram_name="h_MX_Comb_5_5_5_5")
write_outputfile(output_file, input_dir, signals, backgrounds, data, histogram_name="h_MX_Comb_3_3_3_2_Inclusive")

output_file.Write()
output_file.Close()
