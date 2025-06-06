# hDATpIC50prediction
transformerモデルを構築し、決定係数90　テストデータのR2は70でありまだまだ特徴量エンジニアリングが重要
DAT活性予測器
このプロジェクトは、分子記述子とフィンガープリントに基づいてドーパミントランスポーター（DAT）の活性を予測するためのグラフィカルユーザーインターフェイス（GUI）アプリケーションです。SMILES表記から化合物のpIC50値を予測するために、Transformerベースのニューラルネットワークモデルなどの機械学習技術を活用しています。

特徴
モデル学習: ChEMBLから取得したデータを使用して、Transformerベースのモデルを学習します。
ハイパーパラメータ最適化: Optunaを使用してモデルのハイパーパラメータを最適化します。
単一予測: SMILES文字列を入力して、単一の化合物のpIC50値を予測します。
バッチ予測: 複数の化合物のpIC50値を一度に予測します。
可視化: 分子構造と分子記述子を表示します。
キャッシュ: 計算された分子特徴量をキャッシュして、処理を高速化します。
インストール
必要条件
Python 3.7以上
依存ライブラリのインストール
以下のコマンドを使用して、必要なPythonライブラリをインストールします。

bash
コードをコピーする
pip install -r requirements.txt
注意: 一部のライブラリは追加のシステム依存関係を必要とする場合があります。

RDKit: RDKitが正しくインストールされていることを確認してください。問題が発生した場合は、RDKitのインストールガイドを参照してください。
PyQt5: PyQt5はC++コンパイラを必要とする場合があります。Windowsではpip経由でインストールできます。macOSやLinuxでも同様です。
使用方法
リポジトリのクローンまたはダウンロード

提供されたコードをdat_predictor.pyというファイル名で保存してください。

アプリケーションの実行

bash
コードをコピーする
python dat_predictor.py
GUIの操作

モデル学習

Train Modelボタンをクリックして、デフォルトのパラメータでモデルを学習します。
**Optimize (Optuna)**ボタンをクリックして、ハイパーパラメータの最適化を行います。
単一予測

Single Predictionセクションで、化合物のSMILES文字列を入力します。
Predictボタンをクリックして、予測されたpIC50値と分子記述子を表示します。
バッチ予測

Batch Predictionセクションで、複数のSMILES文字列を1行ずつ入力します。
Predict Batchボタンをクリックして、すべての入力化合物の予測を実行します。
Export Resultsボタンをクリックして、予測結果をCSVファイルに保存します。
可視化

単一予測時に、分子構造と分子記述子が表示されます。
キャッシュ管理

Clear Cacheボタンをクリックして、キャッシュされた分子特徴量を削除します。

## CLIの使用
コマンドラインから予測を実行する簡単なインターフェースを追加しました。

### 単一予測の例
```bash
python cli.py predict --model models/dat_transformer_model.pt --smiles "CCO"
```

### バッチ予測の例
```bash
python cli.py predict --model models/dat_transformer_model.pt --input smiles.txt
```

プロジェクト構成
dat_predictor.py: すべてのクラスとロジックを含むメインのアプリケーションスクリプト。
.cache/: キャッシュされた分子特徴量を保存するディレクトリ。
models/: 学習済みモデルを保存するディレクトリ。
dat_predictor.log: アプリケーションの詳細なログを含むログファイル。
モジュールとクラス
ModelConfig: モデルと学習のパラメータを含む設定用のデータクラス。
FeatureCache: 分子特徴量のキャッシュを管理します。
MolecularDescriptorCalculator: 分子記述子とフィンガープリントを計算します。
TransformerModel: Transformerベースのニューラルネットワークモデルを定義します。
ModelPipeline: 学習、検証、予測のプロセスを管理します。
DATPredictor: データ準備とモデルの処理を統括する高レベルのクラス。
TrainingThread: GUIでのモデル学習を管理するQThread。
BatchPredictionThread: GUIでのバッチ予測を処理するQThread。
DATPredictorGUI: アプリケーションのメインGUIクラス。
ロギング
アプリケーションは詳細な情報とエラーをdat_predictor.logに記録します。これにはデータ取得、前処理ステップ、学習の進行状況、予測の詳細が含まれます。

エラーハンドリング
無効なSMILES入力: 無効なSMILES文字列が入力された場合、エラーメッセージが表示されます。
モデル未学習: モデルの学習前に予測が試みられた場合、警告が表示されます。
キャッシュの問題: キャッシュに関連するエラーはログに記録され、ユーザーに通知されます。
依存関係のバージョン情報
以下のバージョン以上を使用することを推奨します。

Python: 3.7+
numpy: 1.18+
pandas: 1.0+
RDKit: 2020.03.1+
scikit-learn: 0.22+
torch: 1.4+
optuna: 1.3+
PyQt5: 5.14+
matplotlib: 3.1+
seaborn: 0.10+
scipy: 1.4+
トラブルシューティング
RDKitのインポートエラー: RDKitが正しくインストールされていることを確認してください。RDKitのインストールガイドを参照してください。
ModuleNotFoundError: モジュールが見つからない場合、すべての依存関係がインストールされ、最新であることを確認してください。
アプリケーションのクラッシュ: 詳細なエラーメッセージについてはdat_predictor.logを確認してください。
ライセンス
このプロジェクトはMITライセンスの下で提供されています。

謝辞
RDKit: オープンソースのケモインフォマティクスツールキット。
ChEMBLデータベース: 創薬のためのオープンデータリソース。
Optuna: ハイパーパラメータ最適化フレームワーク。
お問い合わせ
ご質問や問題がありましたら、[r.minegishi1987@gmail.com]までご連絡ください。

