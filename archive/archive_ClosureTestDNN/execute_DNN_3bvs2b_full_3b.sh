#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /afs/desy.de/user/w/wanghaoy/private/work/CMSSW_14_2_1/src/XtoYH4b/
eval `scramv1 runtime -sh`

cd /data/dust/user/wanghaoy/XtoYH4b/full_3b/Save_EventsWeight
# python3 fold_unroll3b_evaluation_SaveWeights10fold.py  --YEAR 2024 --isScaling 1 --isBalanceClass 1 --Model DNN --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --Nfold 10
# python3 plot_fold.py --YEAR 2024 --isScaling 1 --isBalanceClass 0 --Model DNN --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --Nfold 10
python3 fold_unroll3b_evaluation_SaveWeights10fold.py  --YEAR 2024 --isScaling 1 --isBalanceClass 0 --Model DNN --runType test-only --TrainRegion 3b --TestRegion 3bHiggsMW --Nfold 10
