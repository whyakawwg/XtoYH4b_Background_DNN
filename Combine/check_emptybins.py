import ROOT

f = ROOT.TFile("/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/CombineHarvester/CombineTools/bin/InputFiles/combine_input_XYH4b_2024_VR.root")
d = f.Get("h_MX_MY_Comb_3_3_2_2_Inclusive")

for key in d.GetListOfKeys():
    h = key.ReadObj()
    
    if isinstance(h, ROOT.TH1):
        if h.GetName() in ["data_obs", "NMSSM_XtoYHto4B_MX-1000_MY-150_TuneCP5_13p6TeV_madgraph-pythia8"]:
            continue
        for i in range(1, h.GetNbinsX() + 1):
            content = h.GetBinContent(i)
            
            if content <= 0:
                print(f"Hist: {h.GetName()} | Bin: {i} | Content: {content}")

f.Close()