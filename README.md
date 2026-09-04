# XtoYH4b_Background_DNN
Scripts that train and test DNN model for background estimation in the analysis of $X \rightarrow YH \rightarrow 4b$.

## Prepare the root files for 10 fold
Use the scripts from: [text](https://github.com/whyakawwg/XtoYH4b_Background_DNN/tree/main)

In this background estimation method, the 10-fold emsembling method is used. It's more convinient for the model training to prepare the randomly splitted 10-fold data files:
```
python3 prepare_fold_datafile.py --YEAR 2024
```
The 10 root files and the metadata json file will be created at :`/data/dust/user/wanghaoy/XtoYH4b/Bkg_10fold_datafile/YEAR`

The normalization scale factors, i.e. $\frac{N(CR_{3b})}{N(CR_{2b})}$ and $\frac{N(CR_{4b})}{N(CR_{2b})}$ are saved here. This normalization scale factor will be used in the final step of creating combine input files.

## Train DNN Models
Use the bash script to create and submit condor jobs for training:
```
cd run_scripts
for pair in 0 1 2; do
    bash run_fold.sh -y 2024 -m train -r 4b -p ${pair}
    bash run_fold.sh -y 2024 -m train -r 3b -p ${pair}
done
```
The 4b training creates 10 fold models per pairing. The 3b validation training keeps five statistical splits and creates 50 models per pairing. Submit the pair-specific master scripts under `/data/dust/user/wanghaoy/XtoYH4b/Background_2024/TestTrain_BackgroundEstimation_condor/<region>/job3/pair<pair>/`.
The models and corresponding training plots will be saved at `/data/dust/user/wanghaoy/XtoYH4b/Background_2024/2024/4b/`

## Estimate the background with DNN models
Use the bash script to create and submit condor jobs for evaluation:
```
cd run_scripts
bash run_fold.sh -y 2024 -m test -r 4b -tr 4btest
bash run_fold.sh -y 2024 -m test -r 4b -tr 4bHiggsMW
bash run_fold.sh -y 2024 -m test -r 3b -tr 3btest -s 0
bash run_fold.sh -y 2024 -m test -r 3b -tr 3bHiggsMW -s 0 # qill nwws to s
```
Each evaluation command creates 273 Condor jobs, one for every prepared `(MX, MY)` signal mass point. Submit the generated script from the corresponding `<TestRegion>_evaluation/job3/` directory (or `<TestRegion>_evaluation/SplitIndex<k>/job3/` for 3b). The 4b evaluator averages 10 fold predictions. A 3b evaluation uses the selected split's 10 fold models and the matching deterministic one-fifth 3b target; its 2b source remains unsplit.

Each job writes a mass-point-specific ROOT file, for example:
```
/data/dust/user/wanghaoy/XtoYH4b/Background_2024/4btest_evaluation/4btest_OnlyPhysical_MX-1000_MY-150.root
/data/dust/user/wanghaoy/XtoYH4b/Background_2024/3btest_evaluation/SplitIndex0/3btest_OnlyPhysical_SplitIndex0_MX-1000_MY-150.root
```

To plot one 4b mass point after its evaluation job finishes:
```
cd /data/dust/user/wanghaoy/XtoYH4b/Background_2024/4btest_evaluation
python3 plot_fold.py --YEAR 2024 --isScaling 1 --isBalanceClass 0 --Model DNN --runType test-only --TrainRegion 4b --TestRegion 4btest --Nfold 10 --MX 1000 --MY 150
python3 plot_fold.py --YEAR 2024 --isScaling 1 --isBalanceClass 0 --Model DNN --runType test-only --TrainRegion 3b --TestRegion 3btest --Nfold 10 --SplitIndex 0 --MX 1000 --MY 150
```
Plots are written to `Plots_Evaluation_fold10_MX-1000_MY-150/`.

## Create Combine input
```
cd run_scripts
bash run_condor_CreateCombineInput.sh -y 2024 -r 4b
# For a split-specific 3b validation product:
bash run_condor_CreateCombineInput.sh -y 2024 -r 3b -s 0
# Optionally include an independent non-closure derived from 3bHiggsMW:
bash run_condor_CreateCombineInput.sh -y 2024 -r 4b --add-3b-higgs-nc 1
```
For either 3b or 4b, this creates 273 isolated Combine-input jobs using the same authoritative mass-point ordering as the evaluators. Each job uses the matching mass-point files from the test and Higgs-mass-window regions, derives a mass-specific non-closure uncertainty, and runs in:
```
/data/dust/user/wanghaoy/XtoYH4b/Background_2024/CombineInput_Run2_4b/MX-<MX>_MY-<MY>/
```
The two Python processing scripts and the cleanup utility are copied only once to the Combine-input directory and shared by all jobs. Every executable uses fail-fast shell settings and checks each required input and generated output. After the final background file has been successfully copied to the group directory, intermediate ROOT files are removed from that mass-point directory and only the signal-matched background file is retained.
Submit all generated jobs with:
```
bash /data/dust/user/wanghaoy/XtoYH4b/Background_2024/CombineInput_Run2_4b/jobs_4b/condor_submit_4b.sh
```
The final mass-point-specific files are copied to `/data/dust/group/cms/higgs-bb-desy/XToYHTo4b/SmallNtuples/BackgroundEstimation/2024/`. The year is carried by the directory, while each filename mirrors its signal counterpart. For example:
```
Histogram_NMSSM-XtoYHto4B_Par-MX-1000-MY-300_TuneCP5_13p6TeV_madgraph-pythia8.root
Background_2_NMSSM-XtoYHto4B_Par-MX-1000-MY-300_TuneCP5_13p6TeV_madgraph-pythia8.root
```

With `--add-3b-higgs-nc 1`, each job reads all five `3bHiggsMW` split products.
It saves five diagnostic files named
`nonclosure_factors_nc_3b_SplitIndex<k>_MX-<MX>_MY-<MY>.root`, then saves the
nominal source as `nonclosure_factors_nc_3b_Average5Splits_MX-<MX>_MY-<MY>.root`.
The nominal factor is calculated from the normalized ratio of the summed raw 3b
histograms to the summed raw `2b_w` histograms, rather than by averaging five
absolute non-closure values. Independent `nc_3b` up/down shapes are then added to
the signal-region uncertainty file. MY bins 6 through 11, together with their
unrolled counterparts, remain nominal for this source.

Now go to the combine directory `/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/CombineHarvester/CombineTools/bin/`. Run the `Hist2Comb.py` script to create input files, which will be at `/Inputfiles`:
```
python3 Hist2Comb.py --YEAR 2024
```
Check the datacard creation script `CreateCards_XYHto4b_full.C`, if everything is correct (e.g. uncertainties), compile
```
scram b -j10
```

## Calculate and plot the limits
Go to the directory `/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/CombineHarvester/CombineTools/XYHto4b`. Use the scripts from [text](https://github.com/whyakawwg/XtoYH4b_Background_DNN/tree/main/Combine)

Input the era of data and a customerised suffix. In the end use function `1` for datacards and workspace creation:
```
bash Run_Limits_FullProcess.sh 2024 _v1 1

```
Apart from `condor_q`, you can also check if all the workspaces are ther by using funtion `check_ws`:
```
bash Run_Limits_FullProcess.sh 2024 _v1 check_ws

```
If all the workspaces are ready, run the limit calculation with function `2`:
```
bash Run_Limits_FullProcess.sh 2024 _v1 2

```
After all theh jobs are finished, open a new terminal to run the command below (function `3`) for plotting the limits: 
```
# Need to deactivate cmsenv! Open a new terminal
bash Run_Limits_FullProcess.sh 2024 _v1 3

```

The output limits plot's directory will be printed out, as well as the logs. 

Another option is to plot the 1D limits :
```
python3 plotLimits_MYMX.py --YEAR 2024 --SUFFIX _v1
```

## Combined eras
For combined eras (assuming already done the above process for e.g. 2024 and 2025, and used consistent suffix), create the combined data cards first:


```
bash combineCards_eras.sh "2024 2025" _v1
```

Then follow the same procedure, but use `combined_2024_2025` as the `$year` input:
```
# Create the workspaces
bash Run_Limits_FullProcess.sh combined_2024_2025 _v1 1

# Check if the workspaces creation jobs are finished
bash Run_Limits_FullProcess.sh combined_2024_2025 _v1 check_ws

# Caculate the limits
bash Run_Limits_FullProcess.sh combined_2024_2025 _v1 2


# Plots the limits. Need to deactivate cmsenv! Open a new terminal
bash Run_Limits_FullProcess.sh combined_2024_2025 _v1 3

```


Note: now the previous scripts are still available (background estimated with signal MX-1000 MY-125, used in all mass points)