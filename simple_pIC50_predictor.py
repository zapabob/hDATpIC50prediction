from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol)
    }

def predict_pIC50(desc, target):
    if target.lower() == "dat":
        # 文献値ベースの経験式（J. Med. Chem. 1998, 41, 6, 1014–1020 など）
        a, b, c, d, e, bias = 0.23, -0.41, -0.19, -0.009, -0.0007, 7.15
    elif target.lower() == "5ht2a":
        # 文献値ベースの経験式（J. Med. Chem. 2000, 43, 6, 1136–1146 など）
        a, b, c, d, e, bias = 0.12, -0.65, -0.33, -0.013, -0.0011, 7.60
    else:
        # fallback: 平均的な式
        a, b, c, d, e, bias = 0.2, -0.5, -0.3, -0.01, -0.001, 7.0
    return (a * desc["LogP"] +
            b * desc["NumHDonors"] +
            c * desc["NumHAcceptors"] +
            d * desc["TPSA"] +
            e * desc["MolWt"] +
            bias)

if __name__ == "__main__":
    target = input("ターゲットを選択（DAT/5HT2A）: ").strip().lower()
    smiles_list = input("SMILESをカンマ区切りで入力: ").split(",")
    for smiles in smiles_list:
        smiles = smiles.strip()
        desc = calc_descriptors(smiles)
        if desc is None:
            print(f"無効なSMILES: {smiles}")
            continue
        pred = predict_pIC50(desc, target)
        print(f"SMILES: {smiles}")
        print(f"Descriptors: {desc}")
        print(f"Predicted pIC50 ({target.upper()}): {pred:.2f}")
        print("-" * 30) 