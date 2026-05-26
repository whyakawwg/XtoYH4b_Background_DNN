import ROOT
import random
import os

YEAR = 2025

input_dir = f"/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/{YEAR}/"  # adjust this path
input_file = ROOT.TFile.Open(input_dir + "Tree_Data_Parking.root")
tree = input_file.Get("Tree_JetInfo")   
output_dir = f"/data/dust/user/wanghaoy/XtoYH4b/split_output/{YEAR}"  # adjust this path

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Shuffle entry indices once
n = tree.GetEntries()
indices = list(range(n))
random.shuffle(indices)

# Split into 10 folds (no overlap!)
k = 10
folds = [indices[i::k] for i in range(k)]

# Sort inside each fold to speed up ROOT I/O
folds = [sorted(fold) for fold in folds]

# Create 10 output files
for i, fold in enumerate(folds):
    print(f"Writing fold {i} with {len(fold)} entries...")

    outfile = ROOT.TFile(f"{output_dir}/Tree_Data_Parking_{YEAR}_{i}.root", "RECREATE")
    outtree = tree.CloneTree(0)  

    for idx in fold:
        tree.GetEntry(idx)
        outtree.Fill()

    outtree.Write()
    outfile.Close()

input_file.Close()
print("Done!")
