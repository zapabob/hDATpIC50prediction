# 2024-06-09 5HT2A/サイケデリックス特徴量対応 実装ログ

## 概要
- DAT（CHEMBL238）だけでなく、5HT2A（CHEMBL224）受容体のpIC50データ取得・学習に対応
- サイケデリックス化合物特有の特徴量（インドール環、トリプタミン骨格、メトキシ基数、ハロゲン数、N,N-ジメチルアミン基など）をSMARTSで自動抽出し、学習・予測に利用
- CLI/GUI両方でターゲット受容体（DAT/5HT2A）を切り替えて学習・予測可能に

## 変更点
- `dat_predictor.py`:
    - `fetch_data(target_chembl_id)`でターゲットIDを指定可能に
    - `MolecularDescriptorCalculator`にサイケデリックス特徴量を追加
    - GUI学習パネルにQComboBoxでターゲット選択を追加
    - 学習スレッド/CLIでターゲットIDを渡す設計に
- `cli.py`:
    - `--target`オプションでターゲットID指定可能

## 使い方
- CLI例: `py -3 cli.py train --target CHEMBL224`
- GUI: 学習パネルの「Target」ドロップダウンでDAT/5HT2Aを選択

## 備考
- サイケデリックス特徴量はSMARTSパターンで自動抽出
- 既存のキャッシュ・前処理・学習・予測・可視化機能と連携
- 実装日: 2024-06-09 