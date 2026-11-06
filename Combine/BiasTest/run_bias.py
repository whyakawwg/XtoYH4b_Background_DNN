import argparse
import shlex
import subprocess
from pathlib import Path


def run_command(command, log_path, cwd):
    print(f"Running: {shlex.join(command)}")
    with log_path.open("a") as log:
        log.write(f"\n$ {shlex.join(command)}\n")
        log.flush()
        try:
            subprocess.run(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"Command failed; see {log_path}") from error


def main():
    parser = argparse.ArgumentParser(
        description="Run blinded bias tests at r=0 and at the expected limit."
    )
    parser.add_argument("--YEAR", default="2022", help="Which era?")
    parser.add_argument("--suffix", default="", help="Suffix for workspace directory")
    parser.add_argument("--combination", required=True, type=int, help="Combination number")
    parser.add_argument("--MX", required=True, type=int, help="MX value")
    parser.add_argument("--MY", required=True, type=int, help="MY value")
    parser.add_argument(
        "--expect_signal",
        required=True,
        type=float,
        help="Expected signal strength; tests are run for r=0 and this value",
    )
    parser.add_argument("--ntoys", default=1000, type=int, help="Number of toys")
    parser.add_argument("--verbose", default=0, type=int, help="Combine verbosity")
    args = parser.parse_args()

    workspace = Path(
        f"/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/CombineResults/bkg/"
        f"{args.YEAR}/workspace{args.suffix}/workspace_XYH_4b_{args.combination}_"
        f"13p6TeV_{args.YEAR}_XtoYHto4B_Par-MX-{args.MX}-MY-{args.MY}_"
        "TuneCP5_13p6TeV_madgraph-pythia8.root"
    )
    if not workspace.is_file():
        parser.error(f"Workspace does not exist: {workspace}")

    output_dir = (Path.cwd() / "BiasTestResults" / args.YEAR / f"MX_{args.MX}_MY_{args.MY}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for expected_signal in (0.0, args.expect_signal):
        signal_label = f"{expected_signal}"
        safe_signal_label = signal_label.replace("-", "m").replace(".", "p")
        tag = (
            f"XYH4b_{args.YEAR}_Comb{args.combination}_MX{args.MX}_MY{args.MY}_"
            f"r{safe_signal_label}"
        )
        output_stem = (
            f"bias_XYH4b_{args.YEAR}_Comb-{args.combination}_MX-{args.MX}_"
            f"MY-{args.MY}_ExpSignal_{signal_label}"
        )
        log_path = output_dir / f"{output_stem}.log"
        log_path.write_text("")

        generate_name = f"_{tag}_Generate"
        fit_name = f"_{tag}_Fit"
        toys_file = output_dir / f"higgsCombine{generate_name}.GenerateOnly.mH125.123456.root"

        generate_command = [
            "combine", "-M", "GenerateOnly", "-d", str(workspace), "-m", "125",
            "--expectSignal", signal_label, "--saveToys", "--toysFrequentist",
            "--bypassFrequentistFit", "-t", str(args.ntoys), "-n", generate_name,
            "--rMax", "10",
        ]
        fit_command = [
            "combine", "-M", "MultiDimFit", "-d", str(workspace), "-m", "125",
            "-t", str(args.ntoys), "-n", fit_name, "--toysFile", str(toys_file),
            "--algo", "singles", "--toysFrequentist", "--rMin", "-10", "--rMax", "10",
        ]
        if args.verbose:
            generate_command.extend(["--verbose", str(args.verbose)])
            fit_command.extend(["--verbose", str(args.verbose)])

        run_command(generate_command, log_path, output_dir)
        run_command(fit_command, log_path, output_dir)


if __name__ == "__main__":
    main()


# Example:
# python3 run_bias.py --YEAR 2024 --suffix _Nov5 --combination 1 --MX 1000 --MY 300 --expect_signal 0.0112
# python3 run_bias.py --YEAR 2024 --suffix _Nov5 --combination 1 --MX 1800 --MY 600 --expect_signal 0.0073
