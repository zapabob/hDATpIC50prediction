import argparse
from pathlib import Path
from dat_predictor import DATPredictor, get_reference_pIC50s


def train(args):
    predictor = DATPredictor()
    df = predictor.fetch_data(target_chembl_id=args.target)
    predictor.prepare_data(df)
    if args.optimize:
        predictor.optimize_hyperparameters(n_trials=20)
    else:
        predictor.train_model(early_stopping=True, patience=10, scheduler=True)
    output = Path(args.output or Path(predictor.config.MODEL_DIR) / "dat_transformer_model.pt")
    predictor.save_model(str(output))
    print(f"Model saved to {output}")
    reference_results = get_reference_pIC50s(predictor, args.target)
    print("=== Reference Compounds pIC50 ===")
    for name, smiles, pred in reference_results:
        print(f"{name}: {pred:.2f} (SMILES: {smiles})")


def predict(args):
    predictor = DATPredictor()
    predictor.load_model(args.model)
    smiles_list = []
    if args.smiles:
        smiles_list.append(args.smiles)
    if args.input:
        with open(args.input, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    smiles_list.append(line)
    if not smiles_list:
        print("No SMILES provided")
        return
    for sm in smiles_list:
        pred, _ = predictor.predict(sm)
        if pred is not None:
            print(f"{sm}\t{pred:.4f}")
        else:
            print(f"{sm}\tPrediction failed")


parser = argparse.ArgumentParser(description="DAT/GPCR/Opioid prediction CLI. Supported targets: DAT, 5HT2A, CB1, CB2, μ-opioid, δ-opioid, κ-opioid.")
subparsers = parser.add_subparsers(dest="command")

train_parser = subparsers.add_parser("train", help="train model")
train_parser.add_argument("--output", help="output model path")
train_parser.add_argument("--optimize", action="store_true", help="use Optuna for hyperparameter optimization")
train_parser.add_argument(
    "--target",
    default="CHEMBL238",
    help="ChEMBL target ID (e.g. CHEMBL238=DAT, CHEMBL224=5HT2A, CHEMBL218=CB1, CHEMBL1861=CB2, CHEMBL233=μ-opioid, CHEMBL236=δ-opioid, CHEMBL237=κ-opioid)"
)
train_parser.set_defaults(func=train)

predict_parser = subparsers.add_parser("predict", help="predict pIC50")
predict_parser.add_argument("--model", required=True, help="path to trained model")
predict_parser.add_argument("--smiles", help="SMILES string")
predict_parser.add_argument("--input", help="path to file with SMILES, one per line")
predict_parser.set_defaults(func=predict)

args = parser.parse_args()
if hasattr(args, "func"):
    args.func(args)
else:
    parser.print_help()
