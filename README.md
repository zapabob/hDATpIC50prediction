# hDAT pIC50 Prediction System

---

## 概要
本リポジトリは、7つの分子ターゲット（DAT, 5HT2A, CB1, CB2, μ/δ/κ-opioid）に対応した、分子構造（SMILES）からpIC50値をTransformerベースの深層学習モデルで予測するシステムです。

- **分子特徴量**: RDKit記述子、ECFP4/MACCSフィンガープリント、サイケデリックスSMARTS特徴量、アゴニストスキャフォールド
- **モデル**: PyTorch LightningによるTransformer回帰
- **CLI/GUI**: Typer CLIとPySide6 GUI両対応
- **電源断保護**: 自動保存・キャッシュ・チェックポイント
- **RTX3080等CUDA対応**

---

## 背景・目的

創薬・化学分野では、分子の生物活性（pIC50等）予測は新規化合物設計・スクリーニングの要。特に多ターゲット（DAT, 5HT2A, CB1, CB2, オピオイド）に対し、
- **分子記述子の多様性**
- **深層学習による表現学習**
- **ターゲットごとの代表スキャフォールド考慮**
を組み合わせることで、従来法より高精度な予測を目指します。

---

## 特徴量設計

- **RDKit分子記述子**: MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, RotatableBonds, AromaticRings, FractionCSP3, LabuteASA, BalabanJ, BertzCT, HeavyAtomCount, など
- **フィンガープリント**: ECFP4 (1024bit), MACCS (167bit)
- **サイケデリックスSMARTS特徴量**: インドール環, トリプタミン, フェネチルアミン, メトキシ基数, ハロゲン数, N,N-ジメチルアミン基
- **アゴニストスキャフォールド**: ターゲットごとに代表的なSMARTSパターンを付与
- **相関除去**: 高相関特徴量は自動除去

---

## モデル構造

- **入力層**: 分子特徴量ベクトル
- **埋め込み層**: Linear(input_dim→256)
- **Transformer Encoder**: 2層, 4ヘッド, 256次元（Optunaで最適化可）
- **グローバルプーリング**: 平均
- **出力層**: Linear(256→1)
- **損失関数**: MSELoss
- **最適化**: Adam, 学習率スケジューラ, 早期停止
- **クロスバリデーション/Optuna最適化対応**

---

## ディレクトリ構成
```
├── src/                # メインモジュール（train.py, predict.py など）
├── models/             # 学習済みモデル
├── tests/              # テストコード
├── _docs/              # 実装ログ・要件定義
├── cli.py              # CLIエントリポイント
├── main.py             # GUIエントリポイント
├── dat_predictor.py    # コアロジック
├── requirements.txt    # 依存パッケージ
└── README.md           # 本ファイル
```

---

## 依存パッケージ
- Python 3.10+
- torch==2.3
- pytorch-lightning==2.0
- rdkit==2024.03
- optuna==3.6
- PySide6==6.5
- tqdm
- seaborn, matplotlib, pandas, scikit-learn

`pip install -r requirements.txt` で一括インストール可能。

---

## 使い方

### 1. CLI
#### モデル学習
```sh
py -3 cli.py train --target CHEMBL238 --output models/dat_transformer_model.pt
```
- `--target` : ChEMBLターゲットID（例: CHEMBL238=DAT, CHEMBL224=5HT2A, ...）
- `--optimize` : Optunaによるハイパーパラメータ最適化

#### 予測
```sh
py -3 cli.py predict --model models/dat_transformer_model.pt --smiles "CC(CC1=CC=CC=C1)NC"
```
- `--input` : SMILESファイル（1行1分子）も可

### 2. GUI
```sh
py -3 main.py
```
- 学習・予測・バッチ予測・特徴量重要度グラフ・分布可視化など

---

## FAQ・トラブルシュート

- **Q. CUDAが使われない/遅い**
  - A. torch, pytorch-lightning, CUDA toolkit, GPUドライバのバージョンを確認。
- **Q. RDKit記述子エラー**
  - A. RDKitのバージョンとimport名を確認。
- **Q. ChEMBLからデータが取得できない**
  - A. chembl_webresource_clientのAPI制限やネットワークを確認。
- **Q. GUIが起動しない**
  - A. PySide6のバージョン、PyQt6との競合を確認。

---

## 開発指針・拡張例
- 新規ターゲット追加：`REFERENCE_COMPOUNDS`/SMARTS/ChEMBL IDを追加
- 特徴量追加：`MolecularDescriptorCalculator`に記述子関数を追加
- モデル改良：アンサンブル/多層化/Attention可視化など
- テスト追加：`tests/`配下にpytestでユニットテスト
- 実装ログ：`_docs/`に日付+機能名で記録

---

## 参考文献・リンク
- [ChEMBL](https://www.ebi.ac.uk/chembl/)
- [RDKit](https://www.rdkit.org/)
- [PyTorch](https://pytorch.org/)
- [Optuna](https://optuna.org/)
- [PySide6](https://doc.qt.io/qtforpython/)

---

## ライセンス
MIT License

---

## 貢献
- Issue/PR歓迎。新規ターゲット・特徴量・モデル改良・バグ修正など大歓迎。
- コーディング規約: PEP8, 型ヒント, 実装ログ必須

---

## 更新履歴
- 2024-06-09: リポジトリ整理・README刷新
- 2024-06-01: GUI機能拡張・Optuna最適化追加
- 2024-05-20: 多ターゲット対応・SMARTS特徴量追加
- ...

