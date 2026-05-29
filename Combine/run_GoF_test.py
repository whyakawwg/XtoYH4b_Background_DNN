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
    parser.add_argument("--ntoys", type=int, default=500, help="Number of toys: Default: 500")
    parser.add_argument('--float_signal', action='store',      default=False,   type=bool,           help="Float signal strength?  Default: False")
    parser.add_argument('--algorithm', default="saturated", type=str, help="Algorithm: Default: saturated; Other options: KS, AD")

    args = parser.parse_args()

    file_path = "CombineResults/"+args.YEAR+"/workspaces.txt"

    workspace = find_workspace(file_path, args.combination, args.MX, args.MY)

    if workspace:
        print("Matching workspace found:")
        print(workspace)
    else:
        print("No matching workspace found for the given parameters.")

    Output = "XYH4b_"+args.YEAR+"_Comb-"+str(args.combination)+"_MX-"+str(args.MX)+"_MY-"+str(args.MY)

    #First running on data

    out_file_data = ".GoFResults_"+Output+"_Data"
    # os.system("combineTool.py -M GoodnessOfFit "+workspace+" -m 125 --algo="+args.algorithm+" -n "+out_file_data)
    os.system("combineTool.py -M GoodnessOfFit "+workspace+" -m 125 --algo="+args.algorithm+" -n "+out_file_data+" --fixedSignalStrength 0")
    
    #Then running with toys  

    out_file = ".GoFResults_XYH4b_Bonly_"+Output

    if args.float_signal:

        out_file = ".GoFResults_XYH4b_"+Output
        os.system("combineTool.py -M GoodnessOfFit "+workspace+" -m 125 --algo="+args.algorithm+" -n "+out_file+" -t "+str(args.ntoys)+" --toysFrequentist --bypassFrequentistFit")
        #Computing p-value
        # os.system("combineTool.py -M CollectGoodnessOfFit --input higgsCombine"+out_file_data+".GoodnessOfFit.mH125.root higgsCombine"+out_file+".GoodnessOfFit.mH125.*.root -o gof_"+Output+".json")
        #Computing p-value (Notice the Toy file .* is now FIRST, and Data is SECOND)
        os.system("combineTool.py -M CollectGoodnessOfFit --input higgsCombine"+out_file+".GoodnessOfFit.mH125.*.root higgsCombine"+out_file_data+".GoodnessOfFit.mH125.root -o gof_"+Output+".json")
        
    else:

        #background-only fit (r=0)
        # os.system("combineTool.py -M GoodnessOfFit "+workspace+" -m 125 --algo="+args.algorithm+" -n "+out_file+" --fixedSignalStrength 0 -t "+str(args.ntoys)+" --toysFrequentist --bypassFrequentistFit")
        os.system("combineTool.py -M GoodnessOfFit "+workspace+" -m 125 --algo="+args.algorithm+" -n "+out_file+" --fixedSignalStrength 0 -t "+str(args.ntoys)+" --toysFrequentist")


    #Computing p-value
    # os.system("combineTool.py -M CollectGoodnessOfFit --input higgsCombine"+out_file_data+".GoodnessOfFit.mH125.root higgsCombine"+out_file+".GoodnessOfFit.mH125.*.root -o gof_"+Output+".json")
    #Computing p-value (Notice the Toy file .* is now FIRST, and Data is SECOND)
    os.system("combineTool.py -M CollectGoodnessOfFit --input higgsCombine"+out_file+".GoodnessOfFit.mH125.*.root higgsCombine"+out_file_data+".GoodnessOfFit.mH125.root -o gof_"+Output+".json")
    #Making plot
    os.system("plotGof.py gof_"+Output+".json --statistic "+args.algorithm+" --mass 125.0 -o gof_plot_"+Output)#+" --title-right=")

if __name__ == "__main__":
    main()
