import os
import argparse

argument_parser = argparse.ArgumentParser(description="Check the existence of trained models and scalers for all folds and splits.")
argument_parser.add_argument('--YEAR', default="2025", type=str, help="Which year to check.")
argument_parser.add_argument('--TrainRegion', default="4b", type=str, help = "Which training region to check.")
args = argument_parser.parse_args()

base_dir_template = f"/data/dust/user/wanghaoy/XtoYH4b/Background_{args.YEAR}/{args.YEAR}/{args.TrainRegion}/Models/Model_DNN_Scaling_BalanceClass_Nov27/"
missing_count = 0
found_count = 0

if args.TrainRegion == "3b":
    print(f"Checking for 3b training models and scalers for year {args.YEAR}...")
    expected_models = 10 * 5  # 10 folds * 5 splits
    for split in range(5):
        for fold in range(1, 11):
            fold_dir = os.path.join(base_dir_template, f"MODEL_{fold}_{split}")

            model_path = os.path.join(fold_dir, "model.h5")
            scaler_path = os.path.join(fold_dir, "scaler.pkl")

            has_model = os.path.exists(model_path)
            has_scaler = os.path.exists(scaler_path)

            if has_model and has_scaler:
                found_count += 1
            else:
                missing_count += 1
                print(f"[MISSING] Split {split} / Fold {fold}")
                if not has_model:  print(f"   -> Model missing: {model_path}")
                if not has_scaler: print(f"   -> Scaler missing: {scaler_path}")

elif args.TrainRegion == "4b":
    print(f"Checking for 4b training models and scalers for year {args.YEAR}...")
    expected_models = 10  # 10 folds, no splits
    for fold in range(1, 11):
        fold_dir = os.path.join(base_dir_template, f"MODEL_{fold}")

        model_path = os.path.join(fold_dir, "model.h5")
        scaler_path = os.path.join(fold_dir, "scaler.pkl")

        has_model = os.path.exists(model_path)
        has_scaler = os.path.exists(scaler_path)

        if has_model and has_scaler:
            found_count += 1
        else:
            missing_count += 1
            print(f"[MISSING] Fold {fold}")
            if not has_model:  print(f"   -> Model missing: {model_path}")
            if not has_scaler: print(f"   -> Scaler missing: {scaler_path}")

print("-" * 50)
print(f"Summary:")
print(f"  Total Expected Models: {expected_models}")
print(f"  Successfully Found:    {found_count}")
print(f"  Missing / Incomplete:  {missing_count}")
print("-" * 50)

if missing_count == 0:
    print("✅ All files are present. You are ready to run evaluation!")
else:
    print("❌ Some files are missing. Please fix/retrain before evaluating.")