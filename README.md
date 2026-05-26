# XtoYH4b_Background_DNN
Scripts that train and test DNN model for background estimation in the analysis of $X \rightarrow YH \rightarrow 4b$.

## Prepare the root files for 10 fold
In this background estimation method, the 10-fold emsembling method is used. It's more convinient for the model training to prepare the randomly splitted 10-fold data files:
`python3 multiprocessing_split.py --YEAR 2024`

The normalization scale factors, i.e. \frac{N(CR_{3b})}{N(CR_{2b})} and \frac{N(CR_{4b})}{N(CR_{2b})} are saved here. This normalization scale factor will be used in the final step of creating combine input files.

## Train DNN Models
Use the bash script to create and submit condor jobs for training:
``