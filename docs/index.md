# hDAT pIC50 Multi-Target Predictor

> **多ターゲット対応pIC50予測AI**  
> DAT, 5HT2A, CB1, CB2, オピオイド受容体など7ターゲットに対応。  
> Transformer回帰・RDKit特徴量・Optuna最適化・電源断復旧・GUI/CLI両対応。

---

## Overview

**hDAT pIC50 Multi-Target Predictor** is an advanced platform for predicting pIC50 values of psychoactive and drug-like compounds.  
Supports DAT, 5HT2A, CB1, CB2, and opioid receptors.  
Features Transformer regression, RDKit descriptors, SMARTS scaffolds, Optuna optimization, robust session recovery, and both GUI/CLI interfaces.

---

## 特徴 / Features

- 7つの分子ターゲット（DAT, 5HT2A, CB1, CB2, μ/δ/κ-opioid）対応
- Transformerベースの回帰モデル（PyTorch Lightning）
- Optunaによるハイパーパラメータ自動最適化
- RDKit記述子・SMARTSスキャフォールド・フィンガープリント
- 特徴量キャッシュ・電源断保護・セッション管理
- GUI（PySide6）/ CLI 両対応
- ROC AUCカーブ・特徴量重要度グラフの可視化
- 実装ログ・理論数式・FAQ・専門家向けドキュメント

---

## 使い方 / Usage

### 1. インストール

```bash
pip install -r requirements.txt
```

### 2. CLI起動

```bash
py -3 cli.py
```

### 3. GUI起動

```bash
py -3 dat_predictor.py
```

---

## 主要技術 / Technologies

- Python, PyTorch, PyTorch Lightning, RDKit, Optuna, scikit-learn, PySide6, tqdm, seaborn, matplotlib
- Transformer回帰・アンサンブル学習
- ChEMBLデータ自動取得・キャッシュ
- 電源断復旧・自動チェックポイント保存

---

## ライセンス / License

MIT License

---

## リンク / Links

- [GitHubリポジトリ](https://github.com/yourname/hDAT-pIC50-multi-target-predictor)
- [実装ログ・理論数式（_docs/）](../_docs/)
- [GitHub Pages公式ドキュメント](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)

---

© 2024 hDAT pIC50 Multi-Target Predictor Project 