import argparse
import os

def find_workspace(file_path, combination, mx, my):
    """
    Reads a text file and returns the matching ROOT file path based on
    Combination (e.g., 5), MX (e.g., 2000), and MY (e.g., 60).
    """
    combo_pattern = f"4b_{combination}_"
    mx_pattern = f"MX-{mx}_"
    my_pattern = f"MY-{my}_"

    with open(file_path, "r") as f:
        for line in f:
            if combo_pattern in line and mx_pattern in line and my_pattern in line:
                return line.strip()

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Find a ROOT file path based on combination, MX, and MY values."
    )
    parser.add_argument('--YEAR', default="2022", type=str, help="Which era?")
    parser.add_argument("--combination", type=int, help="Combination number (e.g., 5)")
    parser.add_argument("--MX", type=int, help="MX value (e.g., 2000)")
    parser.add_argument("--MY", type=int, help="MY value (e.g., 60)")
    parser.add_argument('--fix_signal', action='store',      default=False,   type=bool,           help="Fix signal strength?")
    parser.add_argument("--signal", type=int, default=0, help="Signal strength (in integer) if using `--fix_signal`: Default: 0 (bkg-only fit)")

    args = parser.parse_args()

    file_path = "CombineResults/"+args.YEAR+"/workspaces.txt"
  

    workspace = find_workspace(file_path, args.combination, args.MX, args.MY)

    if workspace:
        print("Matching workspace found:")
        print(workspace)
    else:
        print("No matching workspace found for the given parameters.")

    Output = "XYH4b_"+args.YEAR+"_Comb-"+str(args.combination)+"_MX-"+str(args.MX)+"_MY-"+str(args.MY)

    if args.fix_signal:

        out_file = "FitResults_"+Output+"_Signal_"+str(args.signal)
        #running fit
        # os.system("combineTool.py -M MultiDimFit -d "+workspace+" -m 125 --rMax 10 --rMin -1 --setParameters r="+str(args.signal)+" --freezeParameters r  --saveShapes --saveWithUncertainties -n "+out_file)
        
        os.system("combineTool.py -M FitDiagnostics -d "+workspace+" -m 125 --rMax 10 --rMin -1 --setParameters r="+str(args.signal)+" --freezeParameters r  --saveShapes --saveWithUncertainties -n "+out_file)

        # os.system('combineTool.py -M FitDiagnostics -d ' + workspace + ' -m 125 --rMax 10 --rMin 1 --X-rtd HIPPOCRATE=1 --cminDefaultMinimizerStrategy 0 --cminFallbackAlgo "Minuit2,Migrad,0:0.2" --setParameters r=' + str(args.signal) + ' --freezeParameters r --saveShapes --saveWithUncertainties -n ' + out_file)

        
        # os.system("combineTool.py -M FitDiagnostics -d "+workspace+" -m 125 --rMax 10 --rMin -1 --setParameters r="+str(args.signal)+" --freezeParameters r -n FitResults_Debug -v 3 --robustFit 1 --robustHesse 1 2>&1")
        
        # os.system("combine -M FitDiagnostics -d "+workspace+" -m 125 --rMax 10 --rMin 0 --robustFit 1 --cminDefaultMinimizerStrategy 0 --saveShapes --saveWithUncertainties -n --verbose 3 "+out_file)
        #producing plots
        os.system("PostFitShapesFromWorkspace -w "+workspace+" -m 125 --output PostFitResults_"+Output+"_Signal_"+str(args.signal)+"_Bonly.root -f fitDiagnostics"+out_file+".root:fit_b --postfit --sampling --print")
        # os.system("PostFitShapesFromWorkspace -w "+workspace+" -m 125 --output PostFitResults_"+Output+"_Signal_"+str(args.signal)+"_SplusB.root -f fitDiagnostics"+out_file+".root:fit_s --postfit --sampling --print")

    else:

        out_file = "FitResults_XYH4b_"+Output
        #running fit
        # os.system("combineTool.py -M FitDiagnostics -d "+workspace+" -m 125 --rMax 10 --rMin 0  --saveShapes --saveWithUncertainties -n "+out_file)
        os.system("combine -M FitDiagnostics -d "+workspace+" -m 125 --rMax 10 --rMin 0 --robustFit 1 --cminDefaultMinimizerStrategy 0 --saveShapes --saveWithUncertainties -n "+out_file)
        #producing plots
        os.system("PostFitShapesFromWorkspace -w "+workspace+" -m 125 --output PostFitResults_XYH4b_"+Output+"_Bonly.root -f fitDiagnostics"+out_file+".root:fit_b --postfit --sampling --print")
        # os.system("PostFitShapesFromWorkspace -w "+workspace+" -m 125 --output PostFitResults_XYH4b_"+Output+"_SplusB.root -f fitDiagnostics"+out_file+".root:fit_s --postfit --sampling --print")
        
if __name__ == "__main__":
    main()
