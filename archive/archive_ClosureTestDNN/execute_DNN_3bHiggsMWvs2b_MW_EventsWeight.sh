#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/
eval `scramv1 runtime -sh`

cd /data/dust/user/wanghaoy/XtoYH4b/GPU_MWnew
python3 fold_training_Weights.py --YEAR 2024 --isScaling 1 --isBalanceClass 1 --Model DNN --runType train-only --TrainRegion 3bHiggsMW --TestRegion 3bHiggsMW 
