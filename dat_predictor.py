import sys
import os
import io
import pickle
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from functools import partial

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, MACCSkeys, Crippen
from chembl_webresource_client.new_client import new_client
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import optuna

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox,
    QGroupBox, QProgressBar, QFileDialog, QPlainTextEdit,
    QComboBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp  # Kolmogorov-Smirnov test


@dataclass
class ModelConfig:
    TEST_SIZE: float = 0.2
    RANDOM_SEED: int = 42
    N_EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    CACHE_DIR: str = '.cache'
    MODEL_DIR: str = 'models'
    LOG_FILE: str = 'dat_predictor.log'
    EARLY_STOPPING: bool = True
    PATIENCE: int = 10
    SCHEDULER: bool = True


class FeatureCache:
    """分子特徴量のキャッシュシステム"""
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, smiles: str) -> Path:
        return self.cache_dir / f"{hashlib.md5(smiles.encode()).hexdigest()}.npz"

    def get(self, smiles: str) -> Optional[np.ndarray]:
        cache_path = self._get_cache_path(smiles)
        if cache_path.exists():
            return np.load(cache_path)['features']
        return None

    def save(self, smiles: str, features: np.ndarray) -> None:
        cache_path = self._get_cache_path(smiles)
        np.savez_compressed(cache_path, features=features)


class MolecularDescriptorCalculator:
    """分子記述子計算クラス + サイケデリックス特徴量"""
    def __init__(self) -> None:
        from rdkit.Chem import Descriptors, Crippen, AllChem, MACCSkeys
        
        self.descriptor_functions = {
            'MolWt': Descriptors.MolWt,
            'MolLogP': Crippen.MolLogP,
            'NumHDonors': Descriptors.NumHDonors,
            'NumHAcceptors': Descriptors.NumHAcceptors,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumAromaticRings': Descriptors.NumAromaticRings,
            'TPSA': Descriptors.TPSA,
            'FractionCSP3': Descriptors.FractionCSP3,
            'LabuteASA': Descriptors.LabuteASA,
            'BalabanJ': Descriptors.BalabanJ,
            'BertzCT': Descriptors.BertzCT,
            # 必要に応じて追加
        }

        self.fingerprint_functions = {
            'ECFP4': partial(AllChem.GetMorganFingerprintAsBitVect, radius=2, nBits=1024),
            'MACCS': MACCSkeys.GenMACCSKeys
        }
        # サイケデリックス特徴量SMARTSパターン
        self.psychedelic_patterns = {
            'HasIndole': Chem.MolFromSmarts('c1cc2c(cc1)[nH]c2'),
            'HasTryptamine': Chem.MolFromSmarts('CCN(CC)CCC1=CNC2=CC=CC=C12'),
            'HasPhenethylamine': Chem.MolFromSmarts('NCCc1ccc(O)cc1'),
            'MethoxyCount': Chem.MolFromSmarts('CO'),
            'HalogenCount': Chem.MolFromSmarts('[F,Cl,Br,I]'),
            'HasNNDimethyl': Chem.MolFromSmarts('N(C)C'),
        }

    def calculate(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """分子記述子とフィンガープリントを計算"""
        if mol is None:
            return None

        try:
            # 分子記述子の計算
            descriptors = [func(mol) for func in self.descriptor_functions.values()]

            # フィンガープリントの計算
            fingerprints = []
            for name, func in self.fingerprint_functions.items():
                fp = func(mol)
                if hasattr(fp, 'ToBitString'):
                    fingerprints.extend([int(b) for b in fp.ToBitString()])
                else:
                    fingerprints.extend(fp)

            # サイケデリックス特徴量
            psychedelic_features = []
            psychedelic_features.append(int(mol.HasSubstructMatch(self.psychedelic_patterns['HasIndole'])))
            psychedelic_features.append(int(mol.HasSubstructMatch(self.psychedelic_patterns['HasTryptamine'])))
            psychedelic_features.append(int(mol.HasSubstructMatch(self.psychedelic_patterns['HasPhenethylamine'])))
            # メトキシ基数
            methoxy_count = len(mol.GetSubstructMatches(self.psychedelic_patterns['MethoxyCount']))
            psychedelic_features.append(methoxy_count)
            # ハロゲン数
            halogen_count = len(mol.GetSubstructMatches(self.psychedelic_patterns['HalogenCount']))
            psychedelic_features.append(halogen_count)
            # N,N-ジメチルアミン基
            psychedelic_features.append(int(mol.HasSubstructMatch(self.psychedelic_patterns['HasNNDimethyl'])))

            return np.array(descriptors + fingerprints + psychedelic_features)

        except Exception as e:
            logging.error(f"特徴量計算エラー: {e}", exc_info=True)
            return None

    def get_feature_names(self) -> List[str]:
        """全特徴量名を取得"""
        descriptor_names = list(self.descriptor_functions.keys())
        fingerprint_names = []
        for name in self.fingerprint_functions.keys():
            if name == 'ECFP4':
                n_bits = 1024
            elif name == 'MACCS':
                n_bits = 167
            else:
                n_bits = 0
            fingerprint_names.extend([f"{name}_{i}" for i in range(n_bits)])
        psychedelic_names = [
            'HasIndole', 'HasTryptamine', 'HasPhenethylamine',
            'MethoxyCount', 'HalogenCount', 'HasNNDimethyl'
        ]
        return descriptor_names + fingerprint_names + psychedelic_names


class TransformerModel(nn.Module):
    """Transformerベースのモデル"""
    def __init__(self, input_dim: int, num_layers: int = 2, num_heads: int = 4, dim_feedforward: int = 256, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(dim_feedforward, 1)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)  # [batch_size, dim] -> [1, batch_size, dim]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # プーリング
        x = self.fc_out(x)
        return x


class ModelPipeline:
    """モデルパイプライン"""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.scaler = RobustScaler()
        self.y_scaler = StandardScaler()
        self.model = None  # TransformerModel
        self.model_type = 'transformer'

    def fit(self, X: np.ndarray, y: np.ndarray, config: ModelConfig,
            num_layers: int = 2, num_heads: int = 4, dim_feedforward: int = 256, dropout: float = 0.1,
            weight_decay: float = 1e-5,
            early_stopping: bool = False, patience: int = 10, scheduler: bool = False):
        """モデルの学習"""
        X = self.scaler.fit_transform(X)
        y = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        input_dim = X.shape[1]

        # モデルの初期化
        self.model = TransformerModel(input_dim, num_layers=num_layers, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout).to(self.device)

        # データセットの準備
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.BATCH_SIZE, shuffle=True
        )

        # 損失関数と最適化手法
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=weight_decay)

        # 学習率スケジューラの設定
        if scheduler:
            scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # 学習ループ
        self.model.train()
        best_loss = float('inf')
        epochs_no_improve = 0

        train_losses = []  # 学習曲線用
        for epoch in range(config.N_EPOCHS):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
            avg_loss = epoch_loss / len(dataset)
            train_losses.append(avg_loss)
            logging.info(f"Epoch [{epoch+1}/{config.N_EPOCHS}], Loss: {avg_loss:.4f}")

            # 学習率スケジューラのステップ
            if scheduler:
                scheduler_step.step()

            # 早期停止のチェック
            if early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                    # ベストモデルを保存
                    torch.save(self.model.state_dict(), 'best_model.pt')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logging.info("早期停止を実行しました")
                        break

        # 早期停止後、ベストモデルをロード
        if early_stopping and os.path.exists('best_model.pt'):
            self.model.load_state_dict(torch.load('best_model.pt'))
            os.remove('best_model.pt')

        # 学習曲線をプロット
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('learning_curve.png')
        plt.close()
        logging.info("学習曲線を保存しました: learning_curve.png")

    def cross_validate(self, X: np.ndarray, y: np.ndarray, config: ModelConfig, n_splits: int = 3,
                       num_layers: int = 2, num_heads: int = 4, dim_feedforward: int = 256, dropout: float = 0.1,
                       weight_decay: float = 1e-5) -> float:
        """K-Foldクロスバリデーション"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = []

        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            logging.info(f"Fold {fold+1}/{n_splits}")
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # 各フォールドで独立したスケーラーを使用
            scaler = RobustScaler()
            y_scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            X_val_scaled = scaler.transform(X_val)
            y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

            input_dim = X_train_scaled.shape[1]

            # モデルの初期化
            model = TransformerModel(input_dim, num_layers=num_layers, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout).to(self.device)

            # データセットの準備
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train_scaled, dtype=torch.float32),
                torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.BATCH_SIZE, shuffle=True
            )

            # 損失関数と最適化手法
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=weight_decay)

            # 学習ループ
            model.train()
            best_loss = float('inf')
            epochs_no_improve = 0
            patience = config.PATIENCE  # 早期停止のパラメータ

            for epoch in range(config.N_EPOCHS):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * batch_X.size(0)
                avg_loss = epoch_loss / len(dataset)
                logging.info(f"Fold {fold+1}, Epoch [{epoch+1}/{config.N_EPOCHS}], Loss: {avg_loss:.4f}")

                # 早期停止のチェック
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                    # ベストモデルを保存
                    torch.save(model.state_dict(), 'fold_best_model.pt')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logging.info("早期停止を実行しました")
                        break

            # ベストモデルのロード
            if os.path.exists('fold_best_model.pt'):
                model.load_state_dict(torch.load('fold_best_model.pt'))
                os.remove('fold_best_model.pt')

            # 検証セットでの評価
            model.eval()
            with torch.no_grad():
                inputs = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
                outputs = model(inputs).cpu().numpy().flatten()
                score = r2_score(y_val_scaled, outputs)
                logging.info(f"Fold {fold+1} R2 Score: {score:.4f}")
                scores.append(score)

        mean_score = np.mean(scores)
        logging.info(f"Cross-Validation Mean R2 Score: {mean_score:.4f}")
        return mean_score

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測の実行"""
        self.model.eval()
        X = self.scaler.transform(X)
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            predictions = outputs.cpu().numpy().flatten()
        # 逆変換
        predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions


class DATPredictor:
    """DAT活性予測モデル"""
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()
        self.descriptor_calculator = MolecularDescriptorCalculator()
        self.pipeline = ModelPipeline(random_state=self.config.RANDOM_SEED)
        self.cache = FeatureCache(self.config.CACHE_DIR)
        self.is_trained = False
        self._setup_logging()
        self.importances = None  # 特徴量重要度
        self.model_type = 'transformer'  # 'transformer' をデフォルトに設定
        self.removed_features = []  # 削除された特徴量名を保存
        self.feature_names = []  # 使用する特徴量名を保存
        self.full_feature_names = []  # 全特徴量名を保存
        self.feature_indices = []  # 使用する特徴量のインデックスを保存

    def _setup_logging(self) -> None:
        """ロギング設定"""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # FileHandler with utf-8 encoding
        file_handler = logging.FileHandler(self.config.LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # StreamHandler with utf-8 encoding
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

    def fetch_data(self, target_chembl_id: str = 'CHEMBL238') -> pd.DataFrame:
        """ChEMBLからのデータ取得（ターゲットID指定可）"""
        cache_path = Path(self.config.CACHE_DIR) / f'chembl_data_{target_chembl_id}.pkl'
        try:
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                logging.info("キャッシュからデータを読み込みました")
                return df
            target = new_client.target
            activity = new_client.activity
            dat = target.filter(target_chembl_id=target_chembl_id)[0]
            activities = activity.filter(
                target_chembl_id=dat['target_chembl_id'],
                standard_type="IC50",
                standard_units="nM"
            )
            df = pd.DataFrame(activities)
            if 'standard_value' in df.columns:
                df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
            if df.empty:
                raise ValueError("データが取得できませんでした")
            result_df = df[['molecule_chembl_id', 'canonical_smiles', 'standard_value']].dropna()
            result_df = result_df[result_df['standard_value'] < 1_000_000]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(result_df, f)
            logging.info(f"ChEMBL({target_chembl_id})から{len(result_df)}件のデータを取得しました")
            return result_df
        except Exception as e:
            logging.error(f"データ取得エラー: {e}", exc_info=True)
            raise

    def prepare_data(self, df: pd.DataFrame) -> None:
        """データ前処理と分布確認、リサンプリング"""
        try:
            if df.empty:
                raise ValueError("入力データが空です")

            # pIC50の計算
            df['pIC50'] = -np.log10(df['standard_value'].values * 1e-9)

            # 全てのSMILESを取得
            smiles_list = df['canonical_smiles'].tolist()

            # キャッシュされていないSMILESを特定
            uncached_smiles = [smiles for smiles in smiles_list if self.cache.get(smiles) is None]

            # 特徴量を計算（シーケンシャルに処理）
            for smiles in tqdm(uncached_smiles, desc="特徴量計算中"):
                mol = Chem.MolFromSmiles(smiles)
                features = self.descriptor_calculator.calculate(mol)
                if features is not None:
                    self.cache.save(smiles, features)
                else:
                    logging.warning(f"SMILESの特徴量計算に失敗しました: {smiles}")

            # すべての特徴量を収集
            descriptors = []
            valid_indices = []
            for i, smiles in enumerate(smiles_list):
                features = self.cache.get(smiles)
                if features is not None:
                    descriptors.append(features)
                    valid_indices.append(i)
                else:
                    logging.warning(f"キャッシュに特徴量が見つかりませんでした: {smiles}")

            if not valid_indices:
                raise ValueError("有効なデータがありません")

            self.X = np.vstack(descriptors)
            self.y = df['pIC50'].values[valid_indices]

            # 特徴量名を設定
            self.feature_names = self.descriptor_calculator.get_feature_names()
            # 全特徴量名を保存
            self.full_feature_names = self.feature_names.copy()

            # 特徴量の前処理（相関の高い特徴量の削除）
            self.remove_highly_correlated_features()

            # データ分割
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_SEED
            )

            logging.info(f"データ前処理完了: {len(self.X)}件の有効データ")

            # データの分布確認
            self._check_distribution()

            # 特徴量の重要度分析
            self.analyze_feature_importance()

        except Exception as e:
            logging.error(f"データ前処理エラー: {e}", exc_info=True)
            raise

    def _check_distribution(self):
        """学習データとテストデータの分布を確認し、必要に応じてリサンプリングを行う"""
        try:
            # 分布を可視化（オプション：GUI外で保存するなど）
            plt.figure(figsize=(10, 6))
            sns.kdeplot(self.y_train, label='Train')
            sns.kdeplot(self.y_test, label='Test')
            plt.title('Distribution of pIC50 in Train and Test Sets')
            plt.xlabel('pIC50')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig('distribution_comparison.png')
            plt.close()
            logging.info("分布比較プロットを保存しました: distribution_comparison.png")

            # 統計的に分布が異なるか検定（Kolmogorov-Smirnov test）
            ks_stat, p_value = ks_2samp(self.y_train, self.y_test)
            logging.info(f"Kolmogorov-Smirnov test statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")

            if p_value < 0.05:
                logging.warning("学習データとテストデータの分布が統計的に有意に異なります。リサンプリングを行います。")
                self._resample_data()

        except Exception as e:
            logging.error(f"分布確認エラー: {e}", exc_info=True)

    def _resample_data(self):
        """学習データをリサンプリングして分布のバランスを取る"""
        try:
            # ターゲット変数をビニングしてカテゴリカル変数に変換
            num_bins = 10
            y_train_binned = pd.cut(self.y_train, bins=num_bins, labels=False)
            y_test_binned = pd.cut(self.y_test, bins=num_bins, labels=False)

            # リサンプリング（アンダーサンプリング）
            df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
            df_train['y'] = self.y_train
            df_train['y_bin'] = y_train_binned

            # 各ビンの最小サンプル数を決定
            bin_counts = df_train['y_bin'].value_counts()
            min_count = bin_counts.min()

            # 各ビンからランダムにサンプルを抽出
            df_resampled = pd.DataFrame()
            for bin_label in bin_counts.index:
                bin_data = df_train[df_train['y_bin'] == bin_label]
                bin_resampled = resample(bin_data, replace=False, n_samples=min_count, random_state=self.config.RANDOM_SEED)
                df_resampled = pd.concat([df_resampled, bin_resampled], axis=0)

            self.X_train = df_resampled.drop(['y', 'y_bin'], axis=1).values
            self.y_train = df_resampled['y'].values

            logging.info(f"リサンプリング後の学習データサイズ: {self.X_train.shape[0]}")

            # 再度分布を確認
            plt.figure(figsize=(10, 6))
            sns.kdeplot(self.y_train, label='Resampled Train')
            sns.kdeplot(self.y_test, label='Test')
            plt.title('Distribution of pIC50 after Resampling')
            plt.xlabel('pIC50')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig('distribution_comparison_resampled.png')
            plt.close()
            logging.info("リサンプリング後の分布比較プロットを保存しました: distribution_comparison_resampled.png")

        except Exception as e:
            logging.error(f"リサンプリングエラー: {e}", exc_info=True)

    def remove_highly_correlated_features(self, threshold=0.9):
        """相関の高い特徴量を削除"""
        try:
            df = pd.DataFrame(self.X, columns=self.feature_names)
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            logging.info(f"Removing {len(to_drop)} highly correlated features")

            # 削除された特徴量名を保存
            self.removed_features = to_drop.copy()

            df_reduced = df.drop(columns=to_drop)
            self.X = df_reduced.values

            # 特徴量名を更新
            self.feature_names = df_reduced.columns.tolist()

            # 特徴量のインデックスを保存
            self.feature_indices = df_reduced.columns.map(lambda x: self.full_feature_names.index(x)).tolist()

        except Exception as e:
            logging.error(f"相関の高い特徴量の削除エラー: {e}", exc_info=True)

    def analyze_feature_importance(self):
        """特徴量の重要度を分析"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(random_state=self.config.RANDOM_SEED)
            rf.fit(self.X_train, self.y_train)
            self.importances = rf.feature_importances_
            indices = np.argsort(self.importances)[::-1]
            feature_names = self.feature_names

            # 重要度の高い特徴量トップ20を表示
            top_n = 20
            logging.info("Feature importances (top 20):")
            for f in range(top_n):
                if f < len(indices):
                    logging.info(f"{f + 1}. {feature_names[indices[f]]} ({self.importances[indices[f]]:.4f})")
        except Exception as e:
            logging.error(f"特徴量重要度の分析エラー: {e}", exc_info=True)

    def train_model(self, early_stopping: bool = False, patience: int = 10, scheduler: bool = False) -> None:
        """モデル学習"""
        try:
            self.pipeline.fit(
                self.X_train, self.y_train, self.config,
                num_layers=2,
                num_heads=4,
                dim_feedforward=256,
                dropout=0.1,
                weight_decay=1e-4,
                early_stopping=early_stopping,
                patience=patience,
                scheduler=scheduler
            )
            self.is_trained = True
            logging.info(f"モデル学習完了（{self.model_type}）")

        except Exception as e:
            logging.error(f"モデル学習エラー: {e}", exc_info=True)
            raise

    def cross_validate_model(self) -> float:
        """モデルのクロスバリデーション"""
        return self.pipeline.cross_validate(
            self.X_train, self.y_train, self.config
        )

    def optimize_hyperparameters(self, n_trials: int = 20) -> None:
        """ハイパーパラメータ最適化（Optuna）"""
        try:
            def objective(trial):
                # ハイパーパラメータの提案
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
                batch_size = trial.suggest_categorical('batch_size', [32, 64])
                dropout = trial.suggest_uniform('dropout', 0.1, 0.3)
                weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
                num_layers = trial.suggest_int('num_layers', 1, 3)
                num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
                dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])

                # 一部のハイパーパラメータを更新
                config = ModelConfig(
                    LEARNING_RATE=learning_rate,
                    BATCH_SIZE=batch_size,
                    N_EPOCHS=50,  # 最適化時はエポック数を減らす
                    PATIENCE=self.config.PATIENCE
                )

                # クロスバリデーションで評価
                score = self.pipeline.cross_validate(
                    self.X_train, self.y_train, config, n_splits=3,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    weight_decay=weight_decay
                )

                return score  # R2スコアを最大化

            # Optunaのプルーナーを設定
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            study = optuna.create_study(direction='maximize', pruner=pruner)
            study.optimize(objective, n_trials=n_trials)

            # 最適なハイパーパラメータで再学習
            best_params = study.best_params
            config = ModelConfig(
                LEARNING_RATE=best_params['learning_rate'],
                BATCH_SIZE=best_params['batch_size'],
                N_EPOCHS=self.config.N_EPOCHS,
                PATIENCE=self.config.PATIENCE
            )

            self.pipeline = ModelPipeline(random_state=config.RANDOM_SEED)
            self.pipeline.fit(
                self.X_train, self.y_train, config,
                num_layers=best_params['num_layers'],
                num_heads=best_params['num_heads'],
                dim_feedforward=best_params['dim_feedforward'],
                dropout=best_params['dropout'],
                weight_decay=best_params['weight_decay'],
                early_stopping=True,
                patience=config.PATIENCE,
                scheduler=True
            )
            self.is_trained = True
            logging.info(f"Optuna最適化完了: {best_params}")

        except Exception as e:
            logging.error(f"ハイパーパラメータ最適化エラー: {e}", exc_info=True)
            raise

    def predict(self, smiles: str) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """予測実行"""
        try:
            if not self.is_trained:
                raise RuntimeError("モデルが学習されていません")

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("無効なSMILES文字列です")

            # 特徴量を計算
            features = self.descriptor_calculator.calculate(mol)
            if features is None:
                raise ValueError("特徴量の計算に失敗しました")
            X = features.reshape(1, -1)

            # 特徴量を選択
            if hasattr(self, 'feature_indices'):
                X = X[:, self.feature_indices]

            prediction = float(self.pipeline.predict(X)[0])

            # モデルの不確実性を推定（ここでは簡易的に標準偏差を0とします）
            confidence = {
                'mean': prediction,
                'std': 0.0,
                'min': prediction,
                'max': prediction
            }

            return prediction, confidence

        except Exception as e:
            logging.error(f"予測エラー: {e}", exc_info=True)
            return None, None

    def save_model(self, path: str) -> None:
        """モデルの保存"""
        try:
            model_dir = Path(path).parent
            model_dir.mkdir(parents=True, exist_ok=True)

            temp_path = f"{path}.tmp"

            torch.save({
                'version': 1.0,  # バージョン情報の追加
                'model_state_dict': self.pipeline.model.state_dict(),
                'scaler': self.pipeline.scaler,
                'y_scaler': self.pipeline.y_scaler,  # y_scaler を含める
                'is_trained': self.is_trained,
                'input_dim': self.pipeline.model.embedding.in_features,
                'num_layers': self.pipeline.model.transformer_encoder.num_layers,
                'num_heads': self.pipeline.model.transformer_encoder.layers[0].self_attn.num_heads,
                'dim_feedforward': self.pipeline.model.transformer_encoder.layers[0].linear1.in_features,
                'dropout': self.pipeline.model.transformer_encoder.layers[0].dropout.p,
                'model_type': self.model_type,
                'timestamp': datetime.now().isoformat(),
                'removed_features': self.removed_features,
                'feature_names': self.feature_names,
                'full_feature_names': self.full_feature_names,
                'feature_indices': self.feature_indices,  # 追加
            }, temp_path)

            os.replace(temp_path, path)
            logging.info(f"モデルを保存しました: {path}")

        except Exception as e:
            logging.error(f"モデル保存エラー: {e}", exc_info=True)
            raise

    def load_model(self, path: str) -> None:
        """モデルの読み込み"""
        try:
            checkpoint = torch.load(path, map_location=self.pipeline.device)

            # 必要なキーがすべて存在するか確認
            required_keys = ['version', 'model_state_dict', 'scaler', 'y_scaler', 'is_trained', 'input_dim', 'dropout', 'model_type', 'timestamp']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise KeyError(f"チェックポイントに必要なキーが不足しています: {missing_keys}")

            self.pipeline.scaler = checkpoint['scaler']
            self.pipeline.y_scaler = checkpoint['y_scaler']
            self.is_trained = checkpoint['is_trained']
            self.model_type = checkpoint['model_type']
            self.removed_features = checkpoint.get('removed_features', [])
            self.feature_names = checkpoint.get('feature_names', [])
            self.full_feature_names = checkpoint.get('full_feature_names', [])
            self.feature_indices = checkpoint.get('feature_indices', [])  # 追加

            # モデルの初期化
            input_dim = checkpoint['input_dim']
            dropout = checkpoint['dropout']
            num_layers = checkpoint.get('num_layers', 2)
            num_heads = checkpoint.get('num_heads', 4)
            dim_feedforward = checkpoint.get('dim_feedforward', 256)

            self.pipeline.model = TransformerModel(
                input_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ).to(self.pipeline.device)

            self.pipeline.model.load_state_dict(checkpoint['model_state_dict'])
            self.pipeline.model.eval()

            logging.info(f"モデルを読み込みました: {path}")

        except KeyError as e:
            logging.error(f"モデル読み込みエラー: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"モデル読み込みエラー: {e}", exc_info=True)
            raise


class TrainingThread(QThread):
    """学習進捗管理スレッド"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, predictor: DATPredictor, method: str = 'optuna', target_chembl_id: str = 'CHEMBL238') -> None:
        super().__init__()
        self.predictor = predictor
        self.method = method
        self.target_chembl_id = target_chembl_id

    def run(self) -> None:
        try:
            self.status.emit("データ取得中...")
            df = self.predictor.fetch_data(target_chembl_id=self.target_chembl_id)
            self.progress.emit(10)

            self.status.emit("データ前処理中...")
            self.predictor.prepare_data(df)
            self.progress.emit(30)

            if self.method == 'optuna':
                self.status.emit("ハイパーパラメータ最適化中（Optuna）...")
                self.predictor.optimize_hyperparameters(n_trials=20)
            else:
                self.status.emit("モデル学習中...")
                self.predictor.train_model(early_stopping=True, patience=10, scheduler=True)

            self.progress.emit(80)

            metrics = self._calculate_metrics()
            self.progress.emit(100)
            self.finished.emit(metrics)

        except Exception as e:
            self.error.emit(str(e))
            logging.error(f"学習エラー: {e}", exc_info=True)

    def _calculate_metrics(self) -> Dict[str, float]:
        """評価指標の計算"""
        y_train_pred = self.predictor.pipeline.predict(self.predictor.X_train)
        y_test_pred = self.predictor.pipeline.predict(self.predictor.X_test)

        # R2スコアの計算
        train_score = r2_score(self.predictor.y_train, y_train_pred)
        test_score = r2_score(self.predictor.y_test, y_test_pred)

        # 残差プロットを作成
        self._plot_residuals(self.predictor.y_test, y_test_pred)

        return {
            'R2 Score (Train)': train_score,
            'R2 Score (Test)': test_score,
            'Training Samples': len(self.predictor.X_train),
            'Test Samples': len(self.predictor.X_test),
            'Total Features': self.predictor.X_train.shape[1]
        }

    def _r2_score(self, y_true, y_pred):
        """R2スコアの計算"""
        return r2_score(y_true, y_pred)

    def _plot_residuals(self, y_true, y_pred):
        """残差プロットの作成"""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residuals Plot')
        plt.xlabel('Predicted pIC50')
        plt.ylabel('Residuals')
        plt.savefig('residuals_plot.png')
        plt.close()
        logging.info("残差プロットを保存しました: residuals_plot.png")


class BatchPredictionThread(QThread):
    """バッチ予測管理スレッド"""
    progress = pyqtSignal(int)
    result = pyqtSignal(tuple)
    error = pyqtSignal(str)

    def __init__(self, predictor: DATPredictor, smiles_list: List[str]) -> None:
        super().__init__()
        self.predictor = predictor
        self.smiles_list = smiles_list

    def run(self) -> None:
        try:
            results = []
            for i, smiles in enumerate(self.smiles_list):
                prediction, confidence = self.predictor.predict(smiles)
                results.append((smiles, prediction, confidence))
                self.progress.emit(int((i + 1) / len(self.smiles_list) * 100))
            self.result.emit((True, results))
        except Exception as e:
            self.error.emit(str(e))
            logging.error(f"バッチ予測エラー: {e}", exc_info=True)


class DATPredictorGUI(QMainWindow):
    """DAT活性予測モデルのGUI"""
    def __init__(self, predictor: DATPredictor) -> None:
        super().__init__()
        self.predictor = predictor
        self.training_thread = None
        self.batch_thread = None
        self._init_ui()

    def _init_ui(self) -> None:
        """UIの初期化"""
        self.setWindowTitle('DAT Activity Predictor')
        self.setGeometry(100, 100, 1500, 900)

        # メインウィジェットとレイアウト
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        # 左パネル（学習セクション）
        left_panel = self._create_training_panel()
        layout.addWidget(left_panel)

        # 中央パネル（予測セクション）
        center_panel = self._create_prediction_panel()
        layout.addWidget(center_panel)

        # 右パネル（可視化セクション）
        right_panel = self._create_visualization_panel()
        layout.addWidget(right_panel)

    def _create_training_panel(self) -> QGroupBox:
        """学習パネルの作成"""
        group = QGroupBox("Model Training")
        layout = QVBoxLayout()

        # ターゲット選択
        target_layout = QHBoxLayout()
        target_label = QLabel('Target:')
        self.target_combo = QComboBox()
        self.target_combo.addItem('DAT (CHEMBL238)', 'CHEMBL238')
        self.target_combo.addItem('5HT2A (CHEMBL224)', 'CHEMBL224')
        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)

        # 学習コントロール
        control_layout = QHBoxLayout()
        self.train_btn = QPushButton('Train Model')
        self.train_btn.clicked.connect(self.handle_training)
        control_layout.addWidget(self.train_btn)

        self.optimize_optuna_btn = QPushButton('Optimize (Optuna)')
        self.optimize_optuna_btn.clicked.connect(self.handle_optuna_training)
        control_layout.addWidget(self.optimize_optuna_btn)

        layout.addLayout(control_layout)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # ステータス表示
        self.status_label = QLabel('Status: Not trained')
        layout.addWidget(self.status_label)

        # メトリクステーブル
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.metrics_table)

        # キャッシュクリアボタンの追加
        self.clear_cache_btn = QPushButton('Clear Cache')
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        layout.addWidget(self.clear_cache_btn)

        group.setLayout(layout)
        return group

    def _create_prediction_panel(self) -> QGroupBox:
        """予測パネルの作成"""
        group = QGroupBox("Prediction")
        layout = QVBoxLayout()

        # 単一予測セクション
        single_group = QGroupBox("Single Prediction")
        single_layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        self.smiles_input = QLineEdit()
        self.smiles_input.setPlaceholderText('Enter SMILES')
        predict_btn = QPushButton('Predict')
        predict_btn.clicked.connect(self.handle_single_prediction)
        input_layout.addWidget(QLabel('SMILES:'))
        input_layout.addWidget(self.smiles_input)
        input_layout.addWidget(predict_btn)
        single_layout.addLayout(input_layout)

        self.prediction_label = QLabel('Predicted pIC50: ')
        single_layout.addWidget(self.prediction_label)

        # 信頼性指標
        confidence_layout = QHBoxLayout()
        self.confidence_labels = {
            'mean': QLabel('Mean: '),
            'std': QLabel('Std: '),
            'min': QLabel('Min: '),
            'max': QLabel('Max: ')
        }
        for label in self.confidence_labels.values():
            confidence_layout.addWidget(label)
        single_layout.addLayout(confidence_layout)

        single_group.setLayout(single_layout)
        layout.addWidget(single_group)

        # バッチ予測セクション
        batch_group = QGroupBox("Batch Prediction")
        batch_layout = QVBoxLayout()

        self.batch_input = QPlainTextEdit()
        self.batch_input.setPlaceholderText("Enter SMILES (one per line)")
        batch_layout.addWidget(self.batch_input)

        batch_control_layout = QHBoxLayout()
        batch_predict_btn = QPushButton("Predict Batch")
        batch_predict_btn.clicked.connect(self.handle_batch_prediction)
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_batch_results)
        batch_control_layout.addWidget(batch_predict_btn)
        batch_control_layout.addWidget(export_btn)
        batch_layout.addLayout(batch_control_layout)

        self.batch_progress = QProgressBar()
        batch_layout.addWidget(self.batch_progress)

        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(4)
        self.batch_table.setHorizontalHeaderLabels([
            "SMILES", "Predicted pIC50", "Confidence Std", "Status"
        ])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        batch_layout.addWidget(self.batch_table)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        group.setLayout(layout)
        return group

    def _create_visualization_panel(self) -> QGroupBox:
        """可視化パネルの作成"""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()

        # 構造図表示
        self.structure_view = QLabel()
        self.structure_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.structure_view)

        # 分子記述子テーブル
        self.descriptor_table = QTableWidget()
        self.descriptor_table.setColumnCount(2)
        self.descriptor_table.setHorizontalHeaderLabels(['Descriptor', 'Value'])
        self.descriptor_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.descriptor_table)

        group.setLayout(layout)
        return group

    def handle_training(self) -> None:
        """学習処理の開始"""
        self._start_training(method='default')

    def handle_optuna_training(self) -> None:
        """Optunaによるハイパーパラメータ最適化の開始"""
        self._start_training(method='optuna')

    def _start_training(self, method: str) -> None:
        try:
            if not self.predictor:
                raise ValueError("Predictor is not initialized")

            self.train_btn.setEnabled(False)
            self.optimize_optuna_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.status_label.setText("Status: Starting training...")

            # ターゲットID取得
            target_chembl_id = self.target_combo.currentData()

            self.training_thread = TrainingThread(self.predictor, method=method, target_chembl_id=target_chembl_id)
            self.training_thread.progress.connect(self.progress_bar.setValue)
            self.training_thread.status.connect(self._update_status)
            self.training_thread.error.connect(self._handle_training_error)
            self.training_thread.finished.connect(self._handle_training_complete)
            self.training_thread.start()

        except Exception as e:
            self.status_label.setText("Status: Training failed")
            self.train_btn.setEnabled(True)
            self.optimize_optuna_btn.setEnabled(True)
            QMessageBox.critical(self, 'Error', f"Failed to start training: {str(e)}")
            logging.error(f"Training start error: {e}", exc_info=True)

    def _update_status(self, status: str) -> None:
        """ステータス表示の更新"""
        self.status_label.setText(f'Status: {status}')

    def _handle_training_error(self, error_message: str) -> None:
        """学習エラーの処理"""
        self._update_status("Training failed")
        self.train_btn.setEnabled(True)
        self.optimize_optuna_btn.setEnabled(True)
        self.training_thread = None
        QMessageBox.critical(self, 'Error', f"Training error: {error_message}")
        logging.error(f"Training error: {error_message}", exc_info=True)

    def _handle_training_complete(self, metrics: Dict[str, float]) -> None:
        """学習完了の処理"""
        try:
            self._update_metrics_table(metrics)
            model_path = Path(self.predictor.config.MODEL_DIR) / 'dat_transformer_model.pt'
            self.predictor.save_model(str(model_path))

            self._update_status("Training completed")
            QMessageBox.information(
                self,
                'Success',
                f'Model trained successfully!\nSaved to {self.predictor.config.MODEL_DIR}'
            )
            # 標準物質のpIC50を表示
            self._display_reference_pIC50s()

        except Exception as e:
            self._handle_training_error(str(e))
        finally:
            self.train_btn.setEnabled(True)
            self.optimize_optuna_btn.setEnabled(True)
            self.training_thread = None

    def _update_metrics_table(self, metrics: Dict[str, float]) -> None:
        """メトリクステーブルの更新"""
        self.metrics_table.setRowCount(0)
        for name, value in metrics.items():
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            if isinstance(value, float):
                self.metrics_table.setItem(row, 1, QTableWidgetItem(f'{value:.4f}'))
            else:
                self.metrics_table.setItem(row, 1, QTableWidgetItem(str(value)))

    def handle_single_prediction(self) -> None:
        """単一予測の実行"""
        try:
            if not self.predictor.is_trained:
                QMessageBox.warning(self, 'Warning', 'モデルが学習されていません。先にモデルを学習してください。')
                return

            smiles = self.smiles_input.text().strip()
            if not smiles:
                raise ValueError("SMILESを入力してください。")

            prediction, confidence = self.predictor.predict(smiles)
            if prediction is None or confidence is None:
                raise ValueError("予測に失敗しました。")

            self._update_prediction_display(prediction, confidence)
            self._update_molecular_display(smiles)

        except Exception as e:
            QMessageBox.warning(self, 'Error', str(e))
            logging.error(f"単一予測エラー: {e}", exc_info=True)

    def handle_batch_prediction(self) -> None:
        """バッチ予測の実行"""
        try:
            if not self.predictor.is_trained:
                QMessageBox.warning(self, 'Warning', 'モデルが学習されていません。先にモデルを学習してください。')
                return

            smiles_list = [s.strip() for s in self.batch_input.toPlainText().split('\n') if s.strip()]
            if not smiles_list:
                raise ValueError("SMILES文字列を入力してください。")

            self.batch_thread = BatchPredictionThread(self.predictor, smiles_list)
            self.batch_thread.progress.connect(self.batch_progress.setValue)
            self.batch_thread.result.connect(self._handle_batch_results)
            self.batch_thread.error.connect(self._handle_batch_error)
            self.batch_thread.start()

        except Exception as e:
            QMessageBox.warning(self, 'Error', str(e))
            logging.error(f"バッチ予測エラー: {e}", exc_info=True)

    def _handle_batch_results(self, result_tuple: tuple) -> None:
        """バッチ予測結果の処理"""
        success, results = result_tuple
        if not success:
            return

        self.batch_table.setRowCount(len(results))
        for i, (smiles, prediction, confidence) in enumerate(results):
            self.batch_table.setItem(i, 0, QTableWidgetItem(smiles))
            if prediction is not None:
                self.batch_table.setItem(i, 1, QTableWidgetItem(f"{prediction:.2f}"))
                self.batch_table.setItem(i, 2, QTableWidgetItem(f"{confidence['std']:.2f}"))
                self.batch_table.setItem(i, 3, QTableWidgetItem("Success"))
            else:
                self.batch_table.setItem(i, 1, QTableWidgetItem("N/A"))
                self.batch_table.setItem(i, 2, QTableWidgetItem("N/A"))
                self.batch_table.setItem(i, 3, QTableWidgetItem("Failed"))

    def _handle_batch_error(self, error_message: str) -> None:
        """バッチ予測エラーの処理"""
        QMessageBox.warning(self, 'Error', f"Batch prediction error: {error_message}")
        logging.error(f"バッチ予測エラー: {error_message}", exc_info=True)

    def export_batch_results(self) -> None:
        """バッチ予測結果のエクスポート"""
        try:
            if self.batch_table.rowCount() == 0:
                raise ValueError("エクスポートする結果がありません。")

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "", "CSV Files (*.csv);;All Files (*)"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    # ヘッダー書き込み
                    headers = [
                        self.batch_table.horizontalHeaderItem(i).text()
                        for i in range(self.batch_table.columnCount())
                    ]
                    f.write(','.join(headers) + '\n')

                    # データ書き込み
                    for row in range(self.batch_table.rowCount()):
                        row_data = [
                            self.batch_table.item(row, col).text()
                            for col in range(self.batch_table.columnCount())
                        ]
                        f.write(','.join(row_data) + '\n')

                QMessageBox.information(self, 'Success', 'Results exported successfully!')

        except Exception as e:
            QMessageBox.warning(self, 'Error', f"Export failed: {str(e)}")
            logging.error(f"エクスポートエラー: {e}", exc_info=True)

    def _update_prediction_display(self, prediction: float, confidence: Dict[str, float]) -> None:
        """予測結果の表示更新"""
        self.prediction_label.setText(f'Predicted pIC50: {prediction:.2f}')

        for key, value in confidence.items():
            if key in self.confidence_labels:
                self.confidence_labels[key].setText(f'{key.capitalize()}: {value:.2f}')

    def _update_molecular_display(self, smiles: str) -> None:
        """分子情報の表示更新"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("無効なSMILES文字列です。")

            # 構造図の更新
            img = Draw.MolToImage(mol)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            qimg = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                400, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.structure_view.setPixmap(scaled_pixmap)

            # 分子記述子の更新
            features = self.predictor.descriptor_calculator.calculate(mol)
            if features is None:
                raise ValueError("分子記述子の計算に失敗しました。")

            feature_names = self.predictor.descriptor_calculator.get_feature_names()

            # 基本的な分子記述子のみを表示（フィンガープリントは除外）
            n_descriptors = len(self.predictor.descriptor_calculator.descriptor_functions)

            self.descriptor_table.setRowCount(0)
            for i in range(n_descriptors):
                row = self.descriptor_table.rowCount()
                self.descriptor_table.insertRow(row)
                self.descriptor_table.setItem(row, 0, QTableWidgetItem(feature_names[i]))
                self.descriptor_table.setItem(row, 1, QTableWidgetItem(f'{features[i]:.2f}'))

        except Exception as e:
            QMessageBox.warning(self, 'Error', f"Display update failed: {str(e)}")
            logging.error(f"分子表示エラー: {e}", exc_info=True)

    def clear_cache(self) -> None:
        """キャッシュをクリアする"""
        try:
            cache_dir = Path(self.predictor.config.CACHE_DIR)
            if cache_dir.exists():
                for item in cache_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                QMessageBox.information(self, 'Success', 'Cache cleared successfully!')
                logging.info("キャッシュをクリアしました")
            else:
                QMessageBox.information(self, 'Info', 'Cache directory does not exist.')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f"Failed to clear cache: {str(e)}")
            logging.error(f"キャッシュクリアエラー: {e}", exc_info=True)

    def closeEvent(self, event) -> None:
        """アプリケーション終了時の処理"""
        try:
            # トレーニングスレッドの終了
            if self.training_thread and self.training_thread.isRunning():
                self.training_thread.terminate()
                self.training_thread.wait()

            # バッチ予測スレッドの終了
            if self.batch_thread and self.batch_thread.isRunning():
                self.batch_thread.terminate()
                self.batch_thread.wait()

            event.accept()
        except Exception as e:
            logging.error(f"終了処理エラー: {e}", exc_info=True)
            event.accept()

    def _display_reference_pIC50s(self) -> None:
        """学習完了時に標準物質のpIC50を表示する"""
        target_chembl_id = self.target_combo.currentData()
        reference_results = get_reference_pIC50s(self.predictor, target_chembl_id)

        # 既存のメトリクステーブルの下に新しいテーブルを追加
        new_row_count = self.metrics_table.rowCount()
        self.metrics_table.setRowCount(new_row_count + 1)
        self.metrics_table.insertRow(new_row_count)
        self.metrics_table.setItem(new_row_count, 0, QTableWidgetItem("Reference Compounds pIC50"))
        self.metrics_table.setItem(new_row_count, 1, QTableWidgetItem("")) # 空のセルを作成

        # 新しいテーブルを作成
        reference_table = QTableWidget()
        reference_table.setColumnCount(3)
        reference_table.setHorizontalHeaderLabels(["Name", "SMILES", "Predicted pIC50"])
        reference_table.horizontalHeader().setStretchLastSection(True)

        for i, (name, smiles, pred) in enumerate(reference_results):
            row = reference_table.rowCount()
            reference_table.insertRow(row)
            reference_table.setItem(row, 0, QTableWidgetItem(name))
            reference_table.setItem(row, 1, QTableWidgetItem(smiles))
            reference_table.setItem(row, 2, QTableWidgetItem(f"{pred:.2f}"))

        # 新しいテーブルを中央パネルに追加
        center_layout = self.centralWidget().layout()
        if center_layout:
            # 既存の中央パネルのレイアウトを取得
            current_center_layout = center_layout.itemAt(1).layout() # 予測パネルのレイアウト
            if current_center_layout:
                # 予測パネルの下に新しいテーブルを追加
                new_layout = QVBoxLayout()
                new_layout.addWidget(reference_table)
                current_center_layout.addLayout(new_layout)
            else:
                # 予測パネルがない場合は、新しいテーブルを中央パネルに直接追加
                new_layout = QVBoxLayout()
                new_layout.addWidget(reference_table)
                self.centralWidget().setLayout(new_layout)
        else:
            # 中央パネルがない場合は、新しいテーブルをウィンドウに直接追加
            new_layout = QVBoxLayout()
            new_layout.addWidget(reference_table)
            self.setCentralWidget(QWidget()) # 既存の中央パネルをクリア
            self.setCentralWidget(QWidget()) # 新しい中央パネルを設定
            self.centralWidget().setLayout(new_layout)


REFERENCE_COMPOUNDS = {
    'CHEMBL238': {  # DAT
        'Methamphetamine': 'CC(CC1=CC=CC=C1)NC',
        'Cocaine': 'CN1C2CCC1C(C2)OC(=O)C3=CC=CC=C3C(=O)OC',
        'Methylphenidate': 'COC(=O)C1=CC=CC=C1C(C)N',
    },
    'CHEMBL224': {  # 5HT2A
        'LSD': 'CN(C)C1CCC2=C1C3C(C2)C4=CC=CC=C4N3C',
        'DMT': 'CN(C)CCC1=CNC2=CC=CC=C12',
        'Psilocybin': 'COP(=O)(O)OCC1C2=CC=CC=C2NC1',
    }
}

def get_reference_pIC50s(predictor, target_chembl_id):
    refs = REFERENCE_COMPOUNDS.get(target_chembl_id, {})
    results = []
    for name, smiles in refs.items():
        pred, _ = predictor.predict(smiles)
        results.append((name, smiles, pred))
    return results


def main():
    """メイン関数"""
    try:
        app = QApplication(sys.argv)
        predictor = DATPredictor()

        # 保存済みモデルの読み込み
        model_path = Path(predictor.config.MODEL_DIR) / 'dat_transformer_model.pt'
        if model_path.exists():
            predictor.load_model(str(model_path))

        gui = DATPredictorGUI(predictor)
        gui.show()
        sys.exit(app.exec())  # PyQt6ではexec_()ではなくexec()

    except Exception as e:
        logging.error(f"アプリケーション実行エラー: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
