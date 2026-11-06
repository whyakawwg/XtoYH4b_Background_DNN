import argparse
import os

# Note: this should be run after the workspace has been created, e.g. after running Limits

def main():
    parser = argparse.ArgumentParser(
        description="Find a ROOT file path based on combination, MX, and MY values."
    )
    parser.add_argument('--YEAR', default="2022", type=str, help="Which era?")
    parser.add_argument('--suffix', default="", type=str, help="Suffix for workspace file name")
    parser.add_argument("--combination", type=int, help="Combination number (e.g., 5)")
    parser.add_argument("--MX", type=int, help="MX value (e.g., 2000)")
    parser.add_argument("--MY", type=int, help="MY value (e.g., 60)")
    parser.add_argument('--expect_signal',               action='store',      default='1',   type=str,           help="expected signal strength for impact plot?")
    parser.add_argument('--ntoys_impact',               action='store',      default='-1',   type=str,           help="# of toys for impact plot?")

    args = parser.parse_args()

    workspace = f"/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/CombineResults/bkg/{args.YEAR}/workspace{args.suffix}/workspace_XYH_4b_{args.combination}_13p6TeV_{args.YEAR}_XtoYHto4B_Par-MX-{args.MX}-MY-{args.MY}_TuneCP5_13p6TeV_madgraph-pythia8.root"

    os.makedirs(f"ImpactPlots/{args.YEAR}/MX_{args.MX}_MY_{args.MY}", exist_ok=True)

    out_file = "ImpactPlots/"+args.YEAR+"/MX_"+str(args.MX)+"_MY_"+str(args.MY)+"/impacts_XYH4b_"+args.YEAR+"_Comb-"+str(args.combination)+"_MX-"+str(args.MX)+"_MY-"+str(args.MY)+"_ExpSignal_"+args.expect_signal

    os.system("combineTool.py -M Impacts -d "+workspace+" -m 125 --robustFit 1 --doInitialFit --rMax 10 --rMin -10 --expectSignal "+args.expect_signal+" -t "+args.ntoys_impact)
    os.system("combineTool.py -M Impacts -d "+workspace+" -m 125 --robustFit 1 --doFits --rMax 10 --rMin -10 --expectSignal "+args.expect_signal+" -t "+args.ntoys_impact)
    os.system("combineTool.py -M Impacts -d "+workspace+" -m 125 --robustFit 1 --rMax 10 --rMin -10 -o "+out_file+".json")
    os.system("plotImpacts.py -i "+out_file+".json -o "+out_file)

if __name__ == "__main__":
    main()

# python3 run_impact.py --YEAR 2024 --suffix _v1_mass --combination 1 --MX 1000 --MY 125 