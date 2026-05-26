import ROOT
import random
import os
import multiprocessing as mp
import uproot
import numpy as np
import json
import argparse

def calculate_and_save_norm_scale_metadata(file_path, output_json):
    """
    Blazing fast calculation of the global normalization scale factors.
    Calculates both 3b and 4b in a single pass over the arrays.
    """
    print("[INFO] Calculating normalization scale factors via uproot...")
    
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
        "year": YEAR,
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
    fold_id, indices = args
    
    # Suppress PyROOT welcome message per-thread
    ROOT.gErrorIgnoreLevel = ROOT.kWarning 

    infile = ROOT.TFile.Open(INPUT_FILE)
    tree = infile.Get("Tree_JetInfo")

    out_path = f"{OUTPUT_DIR}/Tree_Data_Parking_{YEAR}_{fold_id}.root"
    outfile = ROOT.TFile(out_path, "RECREATE")
    outtree = tree.CloneTree(0)

    for idx in indices:
        tree.GetEntry(idx)
        outtree.Fill()

    outtree.Write()
    outfile.Close()
    infile.Close()

    print(f"[INFO] Finished fold {fold_id} with {len(indices)} events.")
    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare K-Fold splits and normalization metadata for XtoYH4b background estimation.")
    parser.add_argument("--YEAR", type=int, required=True, help="Data taking year (e.g., 2024, 2025)")
    parser.add_argument("--K", type=int, default=10, help="Number of folds (default: 10)")
    args = parser.parse_args()

    # Dynamic path configuration based on argparse
    INPUT_DIR = f"/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/{args.YEAR}/"
    INPUT_FILE = INPUT_DIR + "Tree_Data_Parking.root"
    OUTPUT_DIR = f"/data/dust/user/wanghaoy/XtoYH4b/Bkg_10fold_datafile/{args.YEAR}/"
    K_FOLDS = args.K
    YEAR = args.YEAR

    print(f"=== Starting Data Preparation Pipeline for {YEAR} ===")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output dir: {OUTPUT_DIR}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    
    json_path = os.path.join(OUTPUT_DIR, f"metadata_{YEAR}.json")
    calculate_and_save_norm_scale_metadata(INPUT_FILE, json_path)

    infile = ROOT.TFile.Open(INPUT_FILE)
    tree = infile.Get("Tree_JetInfo")
    n = tree.GetEntries()
    infile.Close()

    print(f"[INFO] Total events to fold: {n}")  

    rng = random.Random(42 + YEAR) 
    indices = list(range(n))
    rng.shuffle(indices)

    folds = [sorted(indices[i::K_FOLDS]) for i in range(K_FOLDS)]

    print(f"[INFO] Starting parallel processing with {mp.cpu_count()} cores...")

    with mp.Pool(processes=min(mp.cpu_count(), K_FOLDS)) as pool:
        pool.map(write_fold, [(i, folds[i]) for i in range(K_FOLDS)])

    print(f"=== All processing completed for {YEAR} ===")