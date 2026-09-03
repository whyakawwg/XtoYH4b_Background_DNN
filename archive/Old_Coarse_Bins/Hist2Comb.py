import argparse
import os

import ROOT


parser = argparse.ArgumentParser(
    description="Create one 4b Combine input file per signal mass point."
)
parser.add_argument("--YEAR", default="2022", help="Data-taking era")
parser.add_argument(
    "--Btag_SF",
    default="WP",
    choices=["WP", "Shape"],
    help="Signal b-tagging scale-factor scheme",
)
parser.add_argument(
    "--add-3b-higgs-nc",
    type=int,
    choices=[0, 1],
    default=0,
    help="Require nc_3b input templates (default: 0, disabled)",
)
parser.add_argument(
    "--signal-dir",
    default=None,
    help="Directory containing Histogram_<signal>.root files",
)
parser.add_argument(
    "--background-dir",
    default=None,
    help="Directory containing BkgEst_<signal>.root files",
)
parser.add_argument(
    "--output-dir",
    default=None,
    help="CombineHarvester bin/InputFiles directory",
)
parser.add_argument(
    "--signal-list",
    default=None,
    help="Optional explicit SIGNAL_names text file",
)
args = parser.parse_args()

ROOT.gROOT.SetBatch(True)


HISTOGRAMS = [
    "h_MaxScore_MX_Comb_3_3_3_2_Inclusive_mHcut",
    "h_MaxScore_MY_Comb_3_3_3_2_Inclusive_mHcut",
    "h_MaxScore_MX_MY_index_Comb_3_3_3_2_Inclusive_mHcut",
]

SIGNAL_SYSTEMATICS = [
    "JER",
    "JES_AbsoluteStat", "JES_AbsoluteScale", "JES_AbsoluteMPFBias",
    "JES_FlavorQCD", "JES_Fragmentation",
    "JES_PileUpDataMC", "JES_PileUpPtBB", "JES_PileUpPtEC1",
    "JES_PileUpPtEC2", "JES_PileUpPtRef",
    "JES_RelativeFSR", "JES_RelativeJEREC1", "JES_RelativeJEREC2",
    "JES_RelativePtBB", "JES_RelativePtEC1", "JES_RelativePtEC2",
    "JES_RelativeBal", "JES_RelativeSample", "JES_RelativeStatEC",
    "JES_RelativeStatFSR", "JES_SinglePionECAL", "JES_SinglePionHCAL",
    "JES_TimePtEta",
    "PU", "LHEScale", "LHEPDF", "LHEAlphaS", "PS_ISR", "PS_FSR",
]

ERA_DECORRELATED_SIGNAL_SYSTEMATICS = {
    "JER", "JES_AbsoluteStat", "JES_RelativeJEREC1",
    "JES_RelativeJEREC2", "JES_RelativePtEC1", "JES_RelativePtEC2",
    "JES_RelativeSample", "JES_RelativeStatEC", "JES_RelativeStatFSR",
    "JES_TimePtEta",
}

LUMINOSITY = {
    "2022": 7.98,
    "2022EE": 7.98,
    "2023": 11.24,
    "2023BPiX": 9.45,
    "2024": 108.96,
    "2025": 110.73,
}


def read_sample_names(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Signal list does not exist: {path}")
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def clone_histogram(root_file, name, source_path):
    histogram = root_file.Get(name)
    if not histogram or not histogram.InheritsFrom("TH1"):
        raise RuntimeError(f"Missing histogram '{name}' in {source_path}")
    histogram = histogram.Clone()
    histogram.SetDirectory(0)
    return histogram


def copy_background_directory(background_file, histogram_name, source_path, output_file):
    source_dir = background_file.Get(histogram_name)
    if not source_dir or not source_dir.InheritsFrom("TDirectory"):
        raise RuntimeError(f"Missing directory '{histogram_name}' in {source_path}")

    objects = {}
    for key in source_dir.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TH1"):
            obj.SetDirectory(0)
            objects[obj.GetName()] = obj

    required = {
        "data_obs",
        "Inclusive_Bkg",
        f"Inclusive_Bkg_Statistical_Uncertainty_{args.YEAR}Up",
        f"Inclusive_Bkg_Statistical_Uncertainty_{args.YEAR}Down",
        f"Inclusive_Bkg_Systematic_Uncertainty_{args.YEAR}Up",
        f"Inclusive_Bkg_Systematic_Uncertainty_{args.YEAR}Down",
    }
    if args.add_3b_higgs_nc:
        required.update({
            f"Inclusive_Bkg_NonClosure3b_Uncertainty_{args.YEAR}Up",
            f"Inclusive_Bkg_NonClosure3b_Uncertainty_{args.YEAR}Down",
        })
    missing = sorted(required.difference(objects))
    if missing:
        raise RuntimeError(
            f"{source_path}:{histogram_name} is missing required 4b shapes: "
            + ", ".join(missing)
        )

    output_dir = output_file.mkdir(histogram_name)
    output_dir.cd()
    for obj in objects.values():
        # BkgEst files are already normalized to data; never luminosity-scale them.
        obj.Write()
    return output_dir


def copy_signal(signal_file, signal, histogram_name, source_path, output_dir, lumi):
    output_dir.cd()
    nominal = clone_histogram(signal_file, histogram_name, source_path)
    nominal.SetName(signal)
    nominal.Scale(lumi)
    nominal.Write()

    for systematic in SIGNAL_SYSTEMATICS:
        output_syst = (
            f"{systematic}_{args.YEAR}"
            if systematic in ERA_DECORRELATED_SIGNAL_SYSTEMATICS
            else systematic
        )
        for source_direction, output_direction in (("up", "Up"), ("down", "Down")):
            source_name = f"{histogram_name}_Sys_{systematic}_{source_direction}"
            variation = signal_file.Get(source_name)
            if not variation:
                print(f"[WARNING] Missing signal shape {source_name} in {source_path}")
                continue
            variation = variation.Clone()
            variation.SetDirectory(0)
            variation.SetName(f"{signal}_{output_syst}{output_direction}")
            variation.Scale(lumi)
            variation.Write()


def default_signal_list(base_dir):
    if args.YEAR in ("2022", "2022EE"):
        filename = "SIGNAL_names.txt"
    elif args.YEAR == "2023BPiX":
        filename = "SIGNAL_names_2023BPIX.txt"
    elif args.YEAR == "2025":
        filename = "SIGNAL_names_2024.txt"
    else:
        filename = f"SIGNAL_names_{args.YEAR}.txt"
    return os.path.join(base_dir, filename)


def main():
    xtoyh4b_dir = "/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b"
    signal_dir = args.signal_dir or (
        "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/Histograms/"
        + args.YEAR
    )
    background_dir = args.background_dir or (
        "/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/"
        f"BackgroundEstimation/{args.YEAR}"
    )
    output_dir = args.output_dir or (
        "/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/"
        "CombineHarvester/CombineTools/bin/InputFiles"
    )
    signal_list = args.signal_list or default_signal_list(xtoyh4b_dir)

    if args.Btag_SF == "WP":
        SIGNAL_SYSTEMATICS.extend(["Btag_WP_SF_correlated", "Btag_WP_SF_uncorrelated"])
    else:
        SIGNAL_SYSTEMATICS.extend([
            "Btag_SF_jes", "Btag_SF_lf", "Btag_SF_lfstats1",
            "Btag_SF_lfstats2", "Btag_SF_hf", "Btag_SF_hfstats1",
            "Btag_SF_hfstats2", "Btag_SF_cferr1", "Btag_SF_cferr2",
            "Btag_SF_correction",
        ])

    signals = read_sample_names(signal_list)
    if not signals:
        raise RuntimeError(f"No signals found in {signal_list}")
    os.makedirs(output_dir, exist_ok=True)
    lumi = LUMINOSITY[args.YEAR]

    for signal in signals:
        signal_path = os.path.join(signal_dir, f"Histogram_{signal}.root")
        background_path = os.path.join(background_dir, f"BkgEst_{signal}.root")
        output_path = os.path.join(
            output_dir, f"combine_input_XYH4b_{args.YEAR}_{signal}.root"
        )

        for required_path in (signal_path, background_path):
            if not os.path.isfile(required_path):
                raise FileNotFoundError(f"Required input does not exist: {required_path}")

        print(f"[INFO] Signal:     {signal_path}")
        print(f"[INFO] Background: {background_path}")
        print(f"[INFO] Output:     {output_path}")

        signal_file = ROOT.TFile.Open(signal_path, "READ")
        background_file = ROOT.TFile.Open(background_path, "READ")
        output_file = ROOT.TFile.Open(output_path, "RECREATE")
        if any(not root_file or root_file.IsZombie() for root_file in
               (signal_file, background_file, output_file)):
            raise RuntimeError(f"Failed to open a ROOT file while processing {signal}")

        try:
            for histogram_name in HISTOGRAMS:
                output_subdir = copy_background_directory(
                    background_file, histogram_name, background_path, output_file
                )
                copy_signal(
                    signal_file, signal, histogram_name, signal_path,
                    output_subdir, lumi,
                )
        finally:
            output_file.Close()
            background_file.Close()
            signal_file.Close()

    print(f"[INFO] Created {len(signals)} mass-specific Combine input files.")


if __name__ == "__main__":
    main()
