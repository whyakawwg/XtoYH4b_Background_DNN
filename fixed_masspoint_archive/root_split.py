import ROOT
import random
import os
import multiprocessing as mp
import argparse

def write_fold(args):
    fold_id, indices, file_paths, output_dir, year = args

    chain = ROOT.TChain("Tree_JetInfo")
    for path in file_paths:
        chain.Add(path)

    outfile = ROOT.TFile(f"{output_dir}/Tree_Data_Parking_{year}_{fold_id}.root", "RECREATE")
    
    outtree = chain.CloneTree(0)

    for idx in indices:
        chain.GetEntry(idx)
        outtree.Fill()

    outtree.Write()
    outfile.Close()

    print(f"Finished fold {fold_id} with {len(indices)} events.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into 10 folds for background estimation.")
    parser.add_argument('--YEAR', default="2024", type=str, help="Which era?")
    args = parser.parse_args()
    YEAR = args.YEAR

    era_mapping = {
        "2022": ["2022", "2022EE"],
        "2023": ["2023", "2023BPiX"]
    }
    subdirs = era_mapping.get(YEAR, [YEAR])

    base_input_dir = "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/"
    if YEAR == "2022" or YEAR == "2023":
        filenname = "Tree_Data.root"
    else:
        filenname = "Tree_Data_Parking.root"

    file_paths = [f"{base_input_dir}{subdir}/{filenname}" for subdir in subdirs]

    output_dir = f"/data/dust/user/wanghaoy/XtoYH4b/Bkg_10fold_datafile/{YEAR}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chain = ROOT.TChain("Tree_JetInfo")
    
    print(f"Processing {YEAR} data across {len(file_paths)} file(s):")
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")
        print(f" - {path}")
        chain.Add(path)

    n = chain.GetEntries()
    print(f"Total combined entries = {n}")  

    K = 10
    indices = list(range(n))
    random.shuffle(indices)

    folds = [sorted(indices[i::K]) for i in range(K)]

    print("Starting parallel processing...")

    pool_args = [(i, folds[i], file_paths, output_dir, YEAR) for i in range(K)]

    # Use all CPU cores
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(write_fold, pool_args)

    print("All folds completed!")