
import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import sys
sys.path.append("/data/dust/user/wanghaoy/XtoYH4b/work_scripts")
# import fold_functions
# from fold_functions import *
import fold_functions_ptcut
from fold_functions_ptcut import *
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import ROOT
import array
import argparse
import os
import joblib

parser = argparse.ArgumentParser(description="")

parser.add_argument('--YEAR', default="2024", type=str, help="Which era?")
parser.add_argument('--isScaling', default=1, type=int, help = "Standard Scaling")
parser.add_argument('--isBalanceClass', default=1, type=int, help = "Balance class?")
parser.add_argument('--splitfraction', default=0.2, type=float, help = "Fraction of test data")
parser.add_argument('--Model', default="DNN", type=str, help = "Model for training")
parser.add_argument('--runType', default="train-test", choices=["train-test", "train-only", "test-only"], type=str, help = "By default, train-test. Other options: train-only, test-only.")
parser.add_argument('--TrainRegion', default="4b", choices=["4b", "3b"], type=str, help = "Region of training data? Select from: '4b', '3b'. Even test-only, need to specify train region for the model.")
parser.add_argument('--TestRegion', default=None, choices=[None, "4btest", "3btest", "3bHiggsMW"], type=str, help = "Rregion to run the test? Select from: '4btest', '3btest', '3bHiggsMW' or None if train-only.")
parser.add_argument('--isMC', default=0, type=int, help = "MC or Data? Data by default.")
parser.add_argument('--SpecificModelTest', default=None, type=str, help = "Input specific model path for testing.")
parser.add_argument('--foldN', default=0, type=int)
parser.add_argument('--Nfold', default=None, type=int, help = "Specify number of folds for training or testing.")
parser.add_argument('--SplitIndex', default=None, type=int, help = "Specify split number for 3b training: 0-9.")

args = parser.parse_args()

isHcand_index_available = False

binning_map = build_binning_map(njets=4)

if args.YEAR == "2022":
    data_lumi = 7.98
elif args.YEAR == "2022EE":
    data_lumi = 26.67
elif args.YEAR == "2023":
    data_lumi = 11.24
elif args.YEAR == "2023BPiX":
    data_lumi = 9.45
elif args.YEAR == "2024":
    data_lumi = 108.96
else:
    print("Please select a valid YEAR: '2022', '2022EE', '2023', '2023BPiX', '2024'")
    exit(1)

if args.isScaling == 1:
    Scaling = "Scaling"
else:
    Scaling = "NoScaling"

if args.isBalanceClass == 1:
    BalanceClass = "BalanceClass"
else:
    BalanceClass = "NoBalanceClass"

foldN = args.foldN

n_folds = args.Nfold 
if args.Nfold is None:
    print("Please provide the number of folds using --Nfold argument!")
    exit(1)

if n_folds == 10:
    get_fold_filelists = get_10fold_filelists
elif n_folds == 5:
    get_fold_filelists = get_5fold_filelists
else:
    print("Currently only 5-fold and 10-fold are supported.")
    exit(1)

YEAR = args.YEAR
TrainRegion = args.TrainRegion
TestRegion = args.TestRegion
runType = args.runType

if args.runType == "train-test" or args.runType == "train-only":
    if args.TrainRegion == "3b":
        SplitIndex = args.SplitIndex
        plot_dir =  f"{args.YEAR}/{args.TrainRegion}/{args.Model}_plots/{args.Model}_plots_{Scaling}_{BalanceClass}_Nov27/MODEL_{foldN}_{SplitIndex}/"
        model_dir = f"{args.YEAR}/{args.TrainRegion}/Models/Model_{args.Model}_{Scaling}_{BalanceClass}_Nov27/MODEL_{foldN}_{SplitIndex}/"
    else:
        plot_dir =  f"{args.YEAR}/{args.TrainRegion}/{args.Model}_plots/{args.Model}_plots_{Scaling}_{BalanceClass}_Nov27/MODEL_{foldN}/"
        model_dir = f"{args.YEAR}/{args.TrainRegion}/Models/Model_{args.Model}_{Scaling}_{BalanceClass}_Nov27/MODEL_{foldN}/"

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Plot dir:", plot_dir)
    print("Model dir:", model_dir)

    train_list, test_list = get_fold_filelists(foldN)
    feature_names_train, features_train, combined_tree_train, aux_data_train = processing(train_list, args=args)
    feature_names_test, features_test, combined_tree_test, aux_data_test = processing(test_list, args=args)
    label_name = "signal"
    X_train = np.stack([combined_tree_train[f] for f in feature_names_train], axis=1)
    y_train = combined_tree_train[label_name]

    X_test = np.stack([combined_tree_test[f] for f in feature_names_test], axis=1)
    y_test = combined_tree_test[label_name]    

    if args.isScaling == 1:
        Scaling = "Scaling"

        scaler = StandardScaler()
        scaler.fit(X_train)

        # Scale both train and test
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)

        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        # Save scaler
        save_dir = model_dir
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(scaler, f"{save_dir}/scaler.pkl")
        
        full_scaled = scaler.transform(features_train)

        corr_matrix = np.corrcoef(full_scaled, rowvar=False)

    else:
        Scaling = "NoScaling"
        corr_matrix = np.corrcoef(features_train, rowvar=False)

    if args.Model == "DNN":

        inputs = Input(shape=(len(X_train[0]),))
        x = Flatten()(inputs)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.AUC()])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)  

        history = model.fit(X_train, y_train, 
              epochs=100, 
              batch_size=512, 
              validation_split=0.1, 
              callbacks=[early_stop],
              verbose=1)

        plot_training_results(history, plot_dir, args=args)

        model.save(model_dir+"/model.h5")

        y_score = model.predict(X_test)
        y_pred = (y_score > 0.5).astype(int)

        score_train = model.predict(X_train)
        score_test  = model.predict(X_test)

    elif args.Model == "BDT":

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            #use_label_encoder=False,
            eval_metric="logloss"
        )

        model.fit(X_train, y_train)

        model.save_model(model_dir+"/bdt_model.json")

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        score_train = model.predict_proba(X_train)[:, 1]
        score_test  = model.predict_proba(X_test)[:, 1] 

    else:
        print("This model is unavailable.")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_score))

    plt.figure(figsize=(8, 6))

    # Signal
    plot_hist(score_train, y_train == 1, "Signal (train)", "r")
    plot_hist(score_test, y_test == 1, "Signal (test)", "orange", linestyle="dashed")
    # Background
    plot_hist(score_train, y_train == 0, "Background (train)", "b")
    plot_hist(score_test, y_test == 0, "Background (test)", "green", linestyle="dashed")

    plt.xlabel(f"{args.Model} Score")
    plt.ylabel("Arbitrary units")
    plt.yscale("log")
    plt.legend(loc="best", fontsize=10)
    plt.title(f"{args.Model} Score Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{args.Model}_Score_Distribution.png"), dpi=300)
    plt.close()

    #Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ROC_Curve.png"), dpi=300)
    plt.close()

    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, xticklabels=feature_names_train, yticklabels=feature_names_train, annot=False, cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.75})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "Feature_Correlation_Matrix.png"), dpi=300)
    plt.close()
elif args.runType == "test-only":
    pass

