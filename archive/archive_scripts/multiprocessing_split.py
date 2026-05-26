import ROOT
import random
import os
import multiprocessing as mp



YEAR = 2024

input_dir = f"/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/{YEAR}/"  # adjust this path
input_file = ROOT.TFile.Open(input_dir + "Tree_Data_Parking.root")
tree = input_file.Get("Tree_JetInfo")   
output_dir = f"/data/dust/user/wanghaoy/XtoYH4b/Bkg_10fold_datafile/{YEAR}"  # adjust this path

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

K = 10

def write_fold(args):
    fold_id, indices = args

    infile = ROOT.TFile.Open(input_dir + "Tree_Data_Parking.root")
    tree = infile.Get("Tree_JetInfo")

    outfile = ROOT.TFile(f"{output_dir}/Tree_Data_Parking_{YEAR}_{fold_id}.root", "RECREATE")
    outtree = tree.CloneTree(0)

    for idx in indices:
        tree.GetEntry(idx)
        outtree.Fill()

    outtree.Write()
    outfile.Close()
    infile.Close()

    print(f"Finished fold {fold_id} with {len(indices)} events.")
    return True


if __name__ == "__main__":
    # Count entries & create random fold assignment
    infile = ROOT.TFile.Open(input_dir + "Tree_Data_Parking.root")
    tree = infile.Get("Tree_JetInfo")
    n = tree.GetEntries()
    infile.Close()

    print(f"Processing {YEAR} data:")
    print(f"Total entries = {n}")  

    # Random shuffle
    indices = list(range(n))
    random.shuffle(indices)

    # Split into K folds (guaranteed non-overlapping)
    folds = [sorted(indices[i::K]) for i in range(K)]

    print("Starting parallel processing...")

    # Use all CPU cores
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(write_fold, [(i, folds[i]) for i in range(K)])

    print("All folds completed!")