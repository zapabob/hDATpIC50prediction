# hDAT/5HT2A pIC50 Prediction

## 概要
本プロジェクトは、ヒトドーパミントランスポーター（DAT, CHEMBL238）および5HT2A受容体（CHEMBL224）に対する化合物のpIC50値を、SMILESから機械学習（Transformerモデル）で予測するPythonアプリケーションです。

- サイケデリックス化合物特有の特徴量抽出に対応
- 標準物質（メタンフェタミン、コカイン、LSD等）のpIC50を自動表示
- CLI/GUI（PyQt6）両対応
- CUDA（RTX3080等）による高速学習
- 電源断・異常終了時の自動リカバリー/バックアップ

---

## 使い方

### 1. 必要なパッケージのインストール
```bash
pip install -r requirements.txt
```

### 2. CLIでの学習・予測
```bash
# DATで学習
py -3 cli.py train --target CHEMBL238
# 5HT2Aで学習
py -3 cli.py train --target CHEMBL224
```

### 3. GUIの起動
```bash
py -3 dat_predictor.py
```

---

## 主な機能
- ChEMBLからの自動データ取得・キャッシュ
- 分子記述子・フィンガープリント・サイケデリックス特徴量の自動計算
- 標準物質のpIC50自動表示
- 学習・予測・可視化（分布・残差・構造図）
- Optunaによるハイパーパラメータ最適化
- 電源断/異常終了時の自動保存・復旧

---

## 標準物質リスト
- DAT: メタンフェタミン, コカイン, メチルフェニデート
- 5HT2A: LSD, DMT, シロシビン

---

## 依存関係
- Python 3.8+
- RDKit
- PyQt6
- torch (CUDA対応推奨)
- tqdm, seaborn, optuna, chembl_webresource_client など

---

## ディレクトリ構成
- `dat_predictor.py` : メインアプリ/GUI
- `cli.py` : CLIインターフェース
- `requirements.txt` : 依存パッケージ
- `_docs/` : 実装ログ・要件定義

---

## ライセンス
MIT License

---

## 開発・貢献
Pull Request・Issue歓迎！

---

## リンク
- [プロジェクトGitHub](https://github.com/zapabob/hDATpIC50prediction)

