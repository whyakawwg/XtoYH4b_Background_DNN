# XtoYH4b_Background_DNN
Scripts that train and test DNN model for background estimation in the analysis of $X \rightarrow YH \rightarrow 4b$.

## Prepare the root files for 10 fold
Use the scripts from: [text](https://github.com/whyakawwg/XtoYH4b_Background_DNN/tree/main)

In this background estimation method, the 10-fold emsembling method is used. It's more convinient for the model training to prepare the randomly splitted 10-fold data files:
```
python3 prepare_fold_datafile --YEAR 2024
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
bash run_fold.sh -y 2024 -m test -r 3b -tr 3btest
bash run_fold.sh -y 2024 -m test -r 3b -tr 3bHiggsMW
```
Each evaluation command creates 272 Condor jobs, one for every prepared `(MX, MY)` signal mass point. Submit the generated script from the corresponding `<TestRegion>_evaluation/job3/` directory. The 4b evaluator averages 10 fold predictions; the 3b evaluator preserves the five validation splits and averages 5 splits × 10 folds.

Each job writes a mass-point-specific ROOT file, for example:
```
/data/dust/user/wanghaoy/XtoYH4b/Background_2024/4btest_evaluation/4btest_OnlyPhysical_MX-1000_MY-150.root
```

To plot one 4b mass point after its evaluation job finishes:
```
cd /data/dust/user/wanghaoy/XtoYH4b/Background_2024/4btest_evaluation
python3 plot_fold.py --YEAR 2024 --isScaling 1 --isBalanceClass 0 --Model DNN --runType test-only --TrainRegion 4b --TestRegion 4btest --Nfold 10 --MX 1000 --MY 150
```
Plots are written to `Plots_Evaluation_fold10_MX-1000_MY-150/`.

## Create Combine input
```
bash run_condor_CreateCombineInput.sh -y 2024 -r 4b
```
This will create condor submit file as usual, but since it will not take a long time, it is also suggested to run the script locally `bash /data/dust/user/wanghaoy/XtoYH4b/Background_2024/CombineInput_4b/jobs_4b/execute_CombineInput_4b.sh`
But note that, in case something wrong, try to delete the directory or the outputed root files to rerun everything. There is one command in 'uncertainty_pipeline.py' does `"UPDATE"` instead of `"RECREATE"`, which could potentially cause problem.
If you wanted to investigate, you can also run the commands from the excute file interactively. 

Now go to the combine directory `/afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/CombineHarvester/CombineTools/bin/`. Run the `Hist2Comb.py` script to create input files, which will be at `/Inputfiles`:
```
python3 Hist2Comb.py --YEAR 2024
```
Check the datacard creation script `CreateCards_XYHto4b_full.C`, if everything is correct (e.g. uncertainties), compile
```
scram b j10
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
