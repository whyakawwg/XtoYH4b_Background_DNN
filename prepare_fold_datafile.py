import ROOT
import random
import os
import multiprocessing as mp
import uproot
import numpy as np
import json
import argparse

def calculate_and_save_norm_scale_metadata(file_paths, output_json, year_label):
    """
    Blazing fast calculation of the global normalization scale factors.
    Calculates both 3b and 4b in a single pass over the arrays.
    """
    print(f"[INFO] Calculating normalization scale factors for {year_label} via uproot...")
    
    essential_columns = [
        'JetAK4_btag_B_WP_1', 'JetAK4_btag_B_WP_2', 'JetAK4_btag_B_WP_3', 'JetAK4_btag_B_WP_4',
        'JetAK4_pt_1', 'JetAK4_pt_2', 'JetAK4_pt_3', 'JetAK4_pt_4', 'Hcand_mass'
    ]

    with uproot.open(file_path) as f:
        tree = f["Tree_JetInfo"]
        arr = tree.arrays(essential_columns, library="np")

    h_mass = arr["Hcand_mass"]
    min_mask = (h_mass > 50) & (h_mass < 300)
    pt_mask = (arr["JetAK4_pt_1"] > 50) & (arr["JetAK4_pt_2"] > 50) & \
              (arr["JetAK4_pt_3"] > 50) & (arr["JetAK4_pt_4"] > 50)

    # Only use off-mass window, Higgs MW data is blinded!
    mass_window = (h_mass < 90) | (h_mass > 150)

    common_mask = min_mask & pt_mask & mass_window

    wp1, wp2 = arr['JetAK4_btag_B_WP_1'], arr['JetAK4_btag_B_WP_2']
    wp3, wp4 = arr['JetAK4_btag_B_WP_3'], arr['JetAK4_btag_B_WP_4']

    # Define region masks
    mask_2b = common_mask & (wp1 >= 3) & (wp2 >= 3) & (wp3 < 2) & (wp4 < 2)
    mask_3b = common_mask & (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 2) & (wp4 < 2)
    mask_4b = common_mask & (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 3) & (wp4 >= 2)

    n_bkg = np.sum(mask_2b)
    n_sig_3b = np.sum(mask_3b)
    n_sig_4b = np.sum(mask_4b)

    if n_bkg == 0:
        print("[WARNING] Zero background events found. Scale factor cannot be computed.")
        exit(1)

    # Note: 3b training utilizes 1/5 of the data pool
    sf_3b = float(int(n_sig_3b / 5)) / float(n_bkg)
    sf_4b = float(n_sig_4b) / float(n_bkg)

    print(f"[INFO] 2b Background Yield: {n_bkg}")
    print(f"[INFO] 3b Normalization Scale Factor: {sf_3b:.5f} (using {int(n_sig_3b/5)} events)")
    print(f"[INFO] 4b Normalization Scale Factor: {sf_4b:.5f} (using {n_sig_4b} events)")

    metadata = {
        "year": year_label,
        "n_events_total": len(h_mass),
        "yield_2b_off_mass": int(n_bkg),
        "yield_3b_off_mass": int(n_sig_3b),
        "yield_4b_off_mass": int(n_sig_4b),
        "normalization_scale_factor_3b": round(sf_3b, 5),
        "normalization_scale_factor_4b": round(sf_4b, 5)
    }

    with open(output_json, "w") as jf:
        json.dump(metadata, jf, indent=4)
        
    print(f"[INFO] Metadata saved to {output_json}\n")


def write_fold(args):
    fold_id, indices, file_paths, output_dir, year = args
    
    ROOT.gErrorIgnoreLevel = ROOT.kWarning 

    chain = ROOT.TChain("Tree_JetInfo")
    for path in file_paths:
        chain.Add(path)

    out_path = f"{output_dir}/Tree_Data_Parking_{year}_{fold_id}.root"
    outfile = ROOT.TFile(out_path, "RECREATE")
    outtree = chain.CloneTree(0)

    for idx in indices:
        chain.GetEntry(idx)
        outtree.Fill()

    outtree.Write()
    outfile.Close()

    print(f"[INFO] Finished fold {fold_id} with {len(indices)} events.")
    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare K-Fold splits and normalization metadata for XtoYH4b background estimation.")
    parser.add_argument("--YEAR", type=str, required=True, help="Data taking year (e.g., 2024, 2022Full)")
    parser.add_argument("--K", type=int, default=10, help="Number of folds (default: 10)")
    args = parser.parse_args()

    YEAR = args.YEAR
    K_FOLDS = args.K

    # Era Mapping Logic
    era_mapping = {
        "2022Full": ["2022", "2022EE"],
        "2023Full": ["2023", "2023BPiX"]
    }
    subdirs = era_mapping.get(YEAR, [YEAR])

    base_input_dir = "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/"
    file_paths = [f"{base_input_dir}{subdir}/Tree_Data_Parking.root" for subdir in subdirs]
    
    OUTPUT_DIR = f"/data/dust/user/wanghaoy/XtoYH4b/Bkg_10fold_datafile/{YEAR}/"

    print(f"=== Starting Data Preparation Pipeline for {YEAR} ===")
    for p in file_paths:
        print(f" - Input file: {p}")
    print(f"Output dir: {OUTPUT_DIR}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    json_path = os.path.join(OUTPUT_DIR, f"metadata_{YEAR}.json")
    calculate_and_save_norm_scale_metadata(file_paths, json_path, YEAR)

    chain = ROOT.TChain("Tree_JetInfo")
    for path in file_paths:
        chain.Add(path)
    
    n = chain.GetEntries()
    print(f"[INFO] Total combined events to fold: {n}")  

    rng = random.Random(42) 
    indices = list(range(n))
    rng.shuffle(indices)

    folds = [sorted(indices[i::K_FOLDS]) for i in range(K_FOLDS)]

    print(f"[INFO] Starting parallel processing with {min(mp.cpu_count(), K_FOLDS)} cores...")

    pool_args = [(i, folds[i], file_paths, OUTPUT_DIR, YEAR) for i in range(K_FOLDS)]

    with mp.Pool(processes=min(mp.cpu_count(), K_FOLDS)) as pool:
        pool.map(write_fold, pool_args)

    print(f"=== All processing completed for {YEAR} ===")