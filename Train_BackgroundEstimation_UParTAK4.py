import uproot
import numpy as np
import vector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import os

parser = argparse.ArgumentParser(description="")

parser.add_argument('--YEAR', default="2024", type=str, help="Which era?")
#parser.add_argument('--UseWeights', action='store', default=False, type=bool, help = "Use event weights?")
parser.add_argument('--isScaling', default=1, type=int, help = "Standard Scaling")
# parser.add_argument('--TrainingScheme', default=0, type=int, help = "Integer for choosing training samples, 0: sum of all samples, 1: high-priority samples, 2: low mass-split samples")
parser.add_argument('--isBalanceClass', default=1, type=int, help = "Balance class?")
parser.add_argument('--splitfraction', default=0.2, type=float, help = "Fraction of test data")
# parser.add_argument('--isExcludedJetMass', default=1, type=int, help = "Balance class?")
parser.add_argument('--Model', default="DNN", type=str, help = "Model for training")
parser.add_argument('--region', default="4b", type=str, help = "Region of training data? Select from: '4b', '3b'")

args = parser.parse_args()

isHcand_index_available = False

def plot_hist(scores, mask, label, color, linestyle="solid"):
    """Utility to draw one DNN histogram."""
    scores = np.asarray(scores).ravel()
    mask = np.asarray(mask).ravel() 
    plt.hist(
        scores[mask],
        bins=50,
        range=(0, 1),
        histtype='step',
        linewidth=1.5,
        label=label,
        color=color,
        linestyle=linestyle,
        density=True
    )

input_file = uproot.open(f"/data/dust/user/chokepra/XtoYH4b/For_Haoyu/Tree_Data_parking.root")
# input_file = uproot.open(f"/data/dust/user/chatterj/XToYHTo4b/SmallNtuples/Histograms/{args.YEAR}/Tree_Data_Parking.root")

tree = input_file["Tree_JetInfo"]
n_events = tree.num_entries
# n_events = 1000
njets = 4

### 2024
columns = ['JetAK4_btag_UParTAK4B_WP_1', 'JetAK4_btag_UParTAK4B_WP_2', 'JetAK4_btag_UParTAK4B_WP_3', 'JetAK4_btag_UParTAK4B_WP_4', 
           'JetAK4_pt_1', 'JetAK4_pt_2', 'JetAK4_pt_3', 'JetAK4_pt_4', 
           'JetAK4_eta_1', 'JetAK4_eta_2', 'JetAK4_eta_3', 'JetAK4_eta_4', 
           'JetAK4_phi_1', 'JetAK4_phi_2', 'JetAK4_phi_3', 'JetAK4_phi_4', 
           'JetAK4_mass_1', 'JetAK4_mass_2', 'JetAK4_mass_3', 'JetAK4_mass_4', 
        #    "JetAK4_Hcand_index_1", "JetAK4_Hcand_index_2", "JetAK4_Hcand_index_3", "JetAK4_Hcand_index_4",
        #   'JetAK4_btag_UParTAK4CvB_1', 'JetAK4_btag_UParTAK4CvB_2', 'JetAK4_btag_UParTAK4CvB_4', #'JetAK4_btag_UParTAK4CvB_3'
        #   'JetAK4_btag_UParTAK4CvL_1', 'JetAK4_btag_UParTAK4CvL_2', 'JetAK4_btag_UParTAK4CvL_4', #'JetAK4_btag_UParTAK4CvL_3'
        #   'JetAK4_btag_UParTAK4QG_1', 'JetAK4_btag_UParTAK4QG_2', 'JetAK4_btag_UParTAK4QG_4', #'JetAK4_btag_UParTAK4QG_3'
           'Hcand_mass', 'Ycand_mass']

# columns = ["JetAK4_btag_PNetB_WP_1", "JetAK4_btag_PNetB_WP_2", "JetAK4_btag_PNetB_WP_3", "JetAK4_btag_PNetB_WP_4",
#            "JetAK4_pt_1", "JetAK4_pt_2", "JetAK4_pt_3", "JetAK4_pt_4",
#            "JetAK4_eta_1", "JetAK4_eta_2", "JetAK4_eta_3", "JetAK4_eta_4",
#            "JetAK4_phi_1", "JetAK4_phi_2", "JetAK4_phi_3", "JetAK4_phi_4",
#            "JetAK4_mass_1", "JetAK4_mass_2", "JetAK4_mass_3", "JetAK4_mass_4",
#            "JetAK4_Hcand_index_1", "JetAK4_Hcand_index_2", "JetAK4_Hcand_index_3", "JetAK4_Hcand_index_4",
#            "JetAK4_btag_PNetCvB_1", "JetAK4_btag_PNetCvB_2", "JetAK4_btag_PNetCvB_4", #"JetAK4_btag_PNetCvB_3", 
#            "JetAK4_btag_PNetCvL_1", "JetAK4_btag_PNetCvL_2", "JetAK4_btag_PNetCvL_4", #"JetAK4_btag_PNetCvL_3", 
#            "JetAK4_btag_PNetQG_1", "JetAK4_btag_PNetQG_2", "JetAK4_btag_PNetQG_4", #"JetAK4_btag_PNetQG_3", 
#            "Hcand_mass", "Ycand_mass"]

tree_arr = tree.arrays(columns, library="np", entry_stop=n_events)

wp1 = tree_arr["JetAK4_btag_UParTAK4B_WP_1"]
wp2 = tree_arr["JetAK4_btag_UParTAK4B_WP_2"]
wp3 = tree_arr["JetAK4_btag_UParTAK4B_WP_3"]
wp4 = tree_arr["JetAK4_btag_UParTAK4B_WP_4"]

H_mass = tree_arr["Hcand_mass"]

# CvB4 = tree_arr["JetAK4_btag_UParTAK4CvB_4"]
# CvL4 = tree_arr["JetAK4_btag_UParTAK4CvL_4"]
# QG4 = tree_arr["JetAK4_btag_UParTAK4QG_4"]

if args.region == "3b":
    common_mask = ((H_mass < 90) | (H_mass > 150)) #& (CvB4 >= 0) & (CvL4 >= 0) & (QG4 >= 0)
    sig_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 2) & (wp4 < 2) & common_mask

elif args.region == "4b":
    common_mask = ((H_mass < 90) | (H_mass > 150)) #& (CvB4 >= 0) & (CvL4 >= 0) & (QG4 >= 0)
    sig_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 >= 3) & (wp4 >= 2) & common_mask

bkg_mask = (wp1 >= 3) & (wp2 >= 3) & (wp3 < 2) & (wp4 < 2) & common_mask

sig_idx = np.where(sig_mask)[0]
bkg_idx = np.where(bkg_mask)[0]

if args.isBalanceClass == 1:
    n_min = min(len(sig_idx), len(bkg_idx))

    sig_idx_bal = sig_idx[:n_min]
    bkg_idx_bal = bkg_idx[:n_min]

    all_idx = np.concatenate([sig_idx_bal, bkg_idx_bal])
    signal_flag = np.concatenate([np.ones(n_min), np.zeros(n_min)])

    BalanceClass = "BalanceClass"
else:
    all_idx = np.concatenate([sig_idx, bkg_idx])
    signal_flag = np.concatenate([np.ones(len(sig_idx)), np.zeros(len(bkg_idx))])

    BalanceClass = "NoBalanceClass"

n_events = len(all_idx)

combined_tree = {
    "signal": signal_flag.astype(np.int32),
    **{key: val[all_idx] for key, val in tree_arr.items()}
}

jets = vector.arr({
    "pt":   np.stack([combined_tree[f"JetAK4_pt_{i+1}"]   for i in range(njets)], axis=1),
    "eta":  np.stack([combined_tree[f"JetAK4_eta_{i+1}"]  for i in range(njets)], axis=1),
    "phi":  np.stack([combined_tree[f"JetAK4_phi_{i+1}"]  for i in range(njets)], axis=1),
    "mass": np.stack([combined_tree[f"JetAK4_mass_{i+1}"] for i in range(njets)], axis=1),
})

Hcand_index_cols = [f"JetAK4_Hcand_index_{i+1}" for i in range(njets)]

dR1_arr = np.full(n_events, np.nan)
dR2_arr = np.full(n_events, np.nan)

if isHcand_index_available:

    Hcand_index = np.stack([combined_tree[col] for col in Hcand_index_cols], axis=1)

    for i in range(n_events):
        val = Hcand_index[i]
        pair1 = np.where(val == 1)[0]
        pair2 = np.where(val == 2)[0]

        if pair1.size == 2 and pair2.size == 2:
            dR1_arr[i] = jets[i, pair1[0]].deltaR(jets[i, pair1[1]])
            dR2_arr[i] = jets[i, pair2[0]].deltaR(jets[i, pair2[1]])

else:

    jet_pairs = [[(1, 2), (3, 4)],
                [(1, 3), (2, 4)],
                [(1, 4), (2, 3)]]

    for i in range(n_events):
        best_idx   = None
        best_min   = np.inf
        best_dRs   = (np.nan, np.nan)

        for pair_set in jet_pairs:
            dR1 = jets[i, pair_set[0][0]-1].deltaR(jets[i, pair_set[0][1]-1])
            dR2 = jets[i, pair_set[1][0]-1].deltaR(jets[i, pair_set[1][1]-1])

            if min(dR1, dR2) < best_min:
                best_min = min(dR1, dR2)
                best_dRs = (dR1, dR2)

        dR1_arr[i], dR2_arr[i] = best_dRs

combined_tree["dR_1"] = dR1_arr
combined_tree["dR_2"] = dR2_arr

drop_cols = Hcand_index_cols 
drop_cols += [f"JetAK4_btag_UParTAK4B_WP_{i+1}" for i in range(njets)]
#drop_cols += [f"JetAK4_mass_{i+1}" for i in range(njets)] 
drop_cols += ["Hcand_mass", "Ycand_mass"]
drop_cols += ["dR_1", "dR_2"]

for col in drop_cols:
    if col in combined_tree:
        del combined_tree[col]

label_name = "signal"
feature_names = [col for col in combined_tree.keys() if col != label_name]
features = np.stack([combined_tree[col] for col in feature_names], axis=1)

if args.isScaling == 1:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    combined_tree = {
        "signal": signal_flag.astype(np.int32),
        **{name: features_scaled[:, i] for i, name in enumerate(feature_names)}
    }

    Scaling = "Scaling"

else:
    Scaling = "NoScaling"

n_sig = np.sum(combined_tree["signal"] == 1)
n_bkg = np.sum(combined_tree["signal"] == 0)

print("Signal events:", n_sig)
print("Background events:", n_bkg)

plot_dir =  f"{args.YEAR}/{args.region}/{args.Model}_plots/without_dR/{args.Model}_plots_{Scaling}_{BalanceClass}_HcandSelection_addMoreTagging_remove3rdtagging_UParTAK4_All/"
model_dir = f"{args.YEAR}/{args.region}/Models/without_dR/Model_{args.Model}_{Scaling}_{BalanceClass}_HcandSelection_addMoreTagging_remove3rdtagging_UParTAK4_All/"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

print(plot_dir)
print(model_dir)

X = np.stack([combined_tree[f] for f in feature_names], axis=1)
y = combined_tree[label_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.splitfraction, stratify=y, random_state=42)

if args.Model == "DNN":

    inputs = Input(shape=(len(X_train[0]),))
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=64)

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
plt.ylabel("Events")
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

# Compute correlation matrix
corr_matrix = np.corrcoef(features, rowvar=False)
# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, xticklabels=feature_names, yticklabels=feature_names, annot=False, cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.75})
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Feature_Correlation_Matrix.png"), dpi=300)
plt.close()




