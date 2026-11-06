import argparse
import os

import ROOT


HISTOGRAMS = [
    "h_MaxScore_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut",
    "h_MaxScore_MX_Comb_3_3_3_2_Inclusive_mHcut",
    "h_MaxScore_MY_Comb_3_3_3_2_Inclusive_mHcut",
]


def require_histogram(root_file, histogram_path):
    if not root_file.Get(histogram_path):
        raise RuntimeError(f"Missing required histogram: {histogram_path}")


def create_datacard(output_dir, bkg_file, year, split_index, mx, my,
                    combination, histogram_name):
    bin_name = f"XYH_3b_{combination}_13p6TeV_{year}"
    nuisances = [
        f"Statistical_Uncertainty_{year}",
        f"Systematic_Uncertainty_{year}",
    ]

    root_file = ROOT.TFile.Open(bkg_file, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open background input: {bkg_file}")

    require_histogram(root_file, f"{histogram_name}/data_obs")
    require_histogram(root_file, f"{histogram_name}/Inclusive_Bkg")
    for nuisance in nuisances:
        require_histogram(
            root_file, f"{histogram_name}/Inclusive_Bkg_{nuisance}Up"
        )
        require_histogram(
            root_file, f"{histogram_name}/Inclusive_Bkg_{nuisance}Down"
        )
    root_file.Close()

    os.makedirs(output_dir, exist_ok=True)
    datacard_filename = os.path.join(
        output_dir,
        f"XYH_3b_{combination}_13p6TeV_{year}_"
        f"XtoYHto4B_Par-MX-{mx}-MY-{my}_SplitIndex{split_index}.txt",
    )

    lines = [
        "imax 1",
        "jmax *",
        "kmax *",
        "------------",
        f"shapes * {bin_name} {os.path.abspath(bkg_file)} "
        f"{histogram_name}/$PROCESS "
        f"{histogram_name}/$PROCESS_$SYSTEMATIC",
        "------------",
        f"bin {bin_name}",
        "observation -1",
        "------------",
        f"bin {bin_name}",
        "process Inclusive_Bkg",
        "process 1",
        "rate -1",
        "------------",
    ]
    lines.extend(f"{nuisance} shape 1" for nuisance in nuisances)

    with open(datacard_filename, "w") as datacard:
        datacard.write("\n".join(lines) + "\n")
    print(f"Datacard written to {datacard_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Create background-only 3b GoF datacards."
    )
    parser.add_argument("--YEAR", required=True, help="Data-taking year")
    parser.add_argument(
        "--split-index", required=True, type=int, choices=range(5)
    )
    parser.add_argument("--MX", required=True, type=int)
    parser.add_argument("--MY", required=True, type=int)
    parser.add_argument("--output-dir", default="datacards_3b")
    args = parser.parse_args()

    signal_stem = (
        f"NMSSM-XtoYHto4B_Par-MX-{args.MX}-MY-{args.MY}_"
        "TuneCP5_13p6TeV_madgraph-pythia8"
    )
    bkg_file = os.path.join(
        "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples",
        "BackgroundEstimation",
        args.YEAR,
        "3b",
        f"SplitIndex{args.split_index}",
        f"BkgEst_{signal_stem}_SplitIndex{args.split_index}.root",
    )
    if not os.path.isfile(bkg_file):
        raise FileNotFoundError(f"Background input does not exist: {bkg_file}")

    for combination, histogram_name in enumerate(HISTOGRAMS, start=1):
        create_datacard(
            args.output_dir,
            bkg_file,
            args.YEAR,
            args.split_index,
            args.MX,
            args.MY,
            combination,
            histogram_name,
        )


if __name__ == "__main__":
    main()
