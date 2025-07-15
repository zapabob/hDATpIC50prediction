# 理論的背景と主要数式

本システムで用いる理論的背景および主要な数式をまとめる。

---

## 1. pIC50値の変換

生化学的活性値IC50 (nM) からpIC50への変換は以下の式で行う：

$$
pIC_{50} = -\log_{10}(IC_{50} \times 10^{-9})
$$

---

## 2. 回帰モデルの目的関数

教師あり学習による回帰モデルの損失関数（平均二乗誤差, MSE）は：

$$
\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

ここで $y_i$ は真のpIC50、$\hat{y}_i$ はモデル予測値。

---

## 3. Transformerモデルの基本構造

本システムの回帰モデルはTransformer Encoderを基盤とする。

### 入力埋め込み

$$
\mathbf{h}_0 = \mathrm{ReLU}(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)
$$

### Multi-Head Self-Attention

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
\mathrm{MultiHead}(X) = [\mathrm{head}_1; \ldots; \mathrm{head}_h]W^O
$$

各headは異なる重みで自己注意を計算。

### Feedforward層

$$
\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

---

## 4. 分子記述子の例

- 分子量：
  $$
  \mathrm{MolWt} = \sum_{i=1}^{N_{atoms}} m_i
  $$
- LogP（疎水性）：
  $$
  \mathrm{MolLogP} = \sum_{i} c_i
  $$
- トポロジカル極表面積（TPSA）：
  $$
  \mathrm{TPSA} = \sum_{i \in \text{polar atoms}} a_i
  $$

---

## 5. フィンガープリント（ECFP4）

分子の局所構造を半径$r=2$で符号化し、bitベクトル化：

$$
\mathrm{ECFP4}(\text{mol}) = \mathrm{Hash}(\text{substructures up to 2 bonds})
$$

---

## 6. クロスバリデーション

$K$分割交差検証：

$$
\mathrm{CV\ Score} = \frac{1}{K} \sum_{k=1}^K R^2_k
$$

---

## 7. 参考文献
- Vaswani et al., "Attention is All You Need", NeurIPS 2017
- RDKit Documentation
- ChEMBL Database 