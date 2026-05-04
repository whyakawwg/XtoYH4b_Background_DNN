import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
import sys
sys.path.append("/data/dust/user/wanghaoy/XtoYH4b/work_scripts")
import fold_functions_ptcut
from fold_functions_ptcut import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
parser.add_argument('--TrainRegion', default="4b", choices=["4b", "3b", "3bHiggsMW"], type=str, help = "Region of training data? Select from: '4b', '3b'. Even test-only, need to specify train region for the model.")
parser.add_argument('--TestRegion', default=None, choices=[None, "4btest", "3btest", "3bHiggsMW"], type=str, help = "Rregion to run the test? Select from: '4btest', '3btest', '3bHiggsMW' or None if train-only.")
parser.add_argument('--isMC', default=0, type=int, help = "MC or Data? Data by default.")
parser.add_argument('--SpecificModelTest', default=None, type=str, help = "Input specific model path for testing.")
# parser.add_argument('--foldN', default=0, type=int)
# parser.add_argument('--Nfold', default=None, type=int, help = "Specify number of folds for training or testing.")
# parser.add_argument('--SplitIndex', default=None, type=int, help = "Specify split number for 3b training: 0-9.")

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
        plot_dir =  f"{args.YEAR}/{args.TrainRegion}/{args.Model}_plots/{args.Model}_plots_{Scaling}_{BalanceClass}_Nov27/"
        model_dir = f"{args.YEAR}/{args.TrainRegion}/Models/Model_{args.Model}_{Scaling}_{BalanceClass}_Nov27/"

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Plot dir:", plot_dir)
    print("Model dir:", model_dir)

    # ===================================================================
    file_list = ['/data/dust/user/wanghaoy/XtoYH4b/Tree_Data_Parking.root']
    # file_list = ['/data/dust/user/wanghaoy/XtoYH4b/split_rootfile/Tree_Data_Parking_0.root']
    feature_names, features, combined_tree, aux_data = processing(file_list, args=args)

    label_name = "signal"

    X_full = np.stack([combined_tree[f] for f in feature_names], axis=1)
    y_full = combined_tree[label_name]

    w_full = aux_data["event_weights"]

    # X_train = np.stack([combined_tree_train[f] for f in feature_names_train], axis=1)
    # y_train = combined_tree_train[label_name]

    # X_test = np.stack([combined_tree_test[f] for f in feature_names_test], axis=1)
    # y_test = combined_tree_test[label_name]    

    test_frac = args.splitfraction if args.splitfraction else 0.2
    print(f"Splitting data into {100*(1-test_frac):.0f}% Train / {100*test_frac:.0f}% Test...")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_full, y_full, w_full, test_size=test_frac, random_state=42
    )

    # ===================================================================
    plt.figure(figsize=(8, 6))
    
    feature_to_plot = X_train[:, 0] # Plotting JetAK4_pt_1
    
    mask_3b_test = (y_train == 1)
    mask_2b_test = (y_train == 0)

    # Plot 1: True 3b (Target)
    plt.hist(feature_to_plot[mask_3b_test], bins=50, density=True, histtype='step', 
             color='red', linewidth=2, label='True 3b')
             
    # Plot 2: Unweighted 2b (What it looked like originally)
    plt.hist(feature_to_plot[mask_2b_test], bins=50, density=True, histtype='step', 
             color='gray', linestyle='dotted', label='Unweighted 2b')
             
    # Plot 3: Weighted 2b (Did the weights move it to match the red line?)
    plt.hist(feature_to_plot[mask_2b_test], bins=50, density=True, histtype='step', 
             color='blue', linewidth=2, weights=w_train[mask_2b_test], label='Weighted 2b')

    plt.title("First Feature (Closure Check)")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "Closure_Check_Input.png"))
    plt.close()
    # ===================================================================

    if args.isScaling == 1:
        Scaling = "Scaling"

        scaler = StandardScaler()
        scaler.fit(X_train)

        # Scale both train and test
        # X_train = scaler.transform(X_train)
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # X_train, y_train = shuffle(X_train, y_train, random_state=42)

        # Save scaler
        save_dir = model_dir
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(scaler, f"{save_dir}/scaler.pkl")
        
        full_scaled = scaler.transform(features)

        # corr_matrix = np.corrcoef(full_scaled, rowvar=False)
        corr_matrix = np.corrcoef(X_train, rowvar=False)

    else:
        Scaling = "NoScaling"
        # corr_matrix = np.corrcoef(features, rowvar=False)
        corr_matrix = np.corrcoef(X_train, rowvar=False)

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
                    # metrics=['accuracy', tf.keras.metrics.AUC()])
                    weighted_metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)  

        # ===================================================================
        sum_w_3b = np.sum(w_train[y_train == 1])
        sum_w_2b = np.sum(w_train[y_train == 0])
        print(f"Total weight for 3b (signal): {sum_w_3b:.2f}")
        print(f"Total weight for 2b (background): {sum_w_2b:.2f}")

        scale_factor_test = sum_w_3b / sum_w_2b
        w_train[y_train == 0] *= scale_factor_test
        w_test[y_test == 0] *= scale_factor_test
        
        print(f"Scaled 2b weights by a factor of {scale_factor_test:.4f} to balance classes.")
        
        # ===================================================================

        history = model.fit(X_train, y_train, 
              sample_weight=w_train,
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

    print("Classification Report (Unweighted):")
    print(classification_report(y_test, y_pred))
    
    # Just print the true, weighted physics score!
    weighted_auc = roc_auc_score(y_test, y_score, sample_weight=w_test)
    print(f"Weighted ROC AUC Score: {weighted_auc:.4f}")

    plt.figure(figsize=(8, 6))
    plot_hist(score_train, y_train == 1, "Signal (train)", "r", weights=w_train)
    plot_hist(score_test, y_test == 1, "Signal (test)", "orange", linestyle="dashed", weights=w_test)

    plot_hist(score_train, y_train == 0, "Background (train)", "b", weights=w_train)
    plot_hist(score_test, y_test == 0, "Background (test)", "green", linestyle="dashed", weights=w_test)

    # plot_hist(score_train, y_train == 1, "Signal (train)", "r")
    # plot_hist(score_test, y_test == 1, "Signal (test)", "orange", linestyle="dashed")
    # # Background
    # plot_hist(score_train, y_train == 0, "Background (train)", "b")
    # plot_hist(score_test, y_test == 0, "Background (test)", "green", linestyle="dashed")

    plt.xlabel(f"{args.Model} Score")
    plt.ylabel("Arbitrary units")
    plt.yscale("log")
    plt.legend(loc="best", fontsize=10)
    plt.title(f"{args.Model} Score Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{args.Model}_Score_Distribution.png"), dpi=300)
    plt.close()

    #Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_test, y_score)
    # roc_auc = auc(fpr, tpr)

    fpr, tpr, _ = roc_curve(y_test, y_score, sample_weight=w_test)
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
    sns.heatmap(corr_matrix, xticklabels=feature_names, yticklabels=feature_names, annot=False, cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.75})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "Feature_Correlation_Matrix.png"), dpi=300)
    plt.close()

    # plot also the train roc
    fpr_train, tpr_train, _ = roc_curve(y_train, score_train, sample_weight=w_train)
    roc_auc = auc(fpr_train, tpr_train)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Train Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "Train_ROC_Curve.png"), dpi=300)
    plt.close()


elif args.runType == "test-only":
    pass

