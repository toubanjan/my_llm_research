# 技術提案書: 実数事前学習モデルを活用した複素RWKV-7（C-RWKV-7）位相共鳴拡張アーキテクチャ

## 1. 概要（Abstract）

本提案書は、RWKV-7の内部隠れ状態および時間混合（Time-Mixing）機構を複素数空間 $\mathbb{C}^d$ へ拡張し、蔵本モデル（Kuramoto Model）に基づく位相共鳴ダイナミクスを組み込んだ次世代アーキテクチャ「C-RWKV-7（Complex-Valued RWKV-7）」およびその段階的導入手法を定義するものです。

---

## 2. 背景と解決する課題（Motivation & Problem Statement）

### 2.1 課題背景

RNNやLinear Attention構造を持つモデル（RWKV等）は、計算量 $\mathcal{O}(1)$ の推論効率を誇る一方、超長文コンテキストにおいて古い記憶が新しい情報によって上書き・汚染される「情報干渉（Interference）」が生じやすい限界があります。

### 2.2 本提案によるアプローチと改訂ポイント

情報を「振幅（エネルギー・確信度）」**と**「位相（文脈的役割・因果関係）」へ分離拡張（複素数化）し、波動力学の概念を適用します。

* **高密度記憶と干渉減衰**: 同相共鳴による構造化記憶の保持（Constructive Interference）と、逆相干渉によるノイズ減衰（Destructive Interference）。
* **過度同期（Phase Locking）の防止**: 固定結合係数による完全位相同期（表現力の潰滅）を回避するため、入力依存の動的結合制御を導入。
* **Zero-Shot & C-LoRA 移行**: 実数事前学習モデルの知識を100%保持したまま無次元初期化し、転移学習コストを低減。

また、本提案は、単なるコンテキスト長の延伸にとどまらず、「1チャネルあたりの表現容量を極限まで高める（高密度化）」 こと、および 「脳の神経同期に類似した引き込み現象で複雑な論理を固定・操作する（思考の深化）」 ことも主目的として掲げます。

---

## 3. 改訂数理モデルとコア・メカニズム（Core Mathematical Formulation）

### 3.1 状態ベクトルの複素極座標表示

隠れ状態 $h_t^{(c)}$ を直交形式および極座標形式で表現します。

$$h_t^{(c)} = h_t^{\text{real}} + i \, h_t^{\text{imag}} = r_t \odot e^{i \theta_t}$$

* **振幅 ($r_t = \vert{}h_t^{(c)}\vert{}$)**: 情報の強度・概念の確信度（旧来の実数出力の基盤）。
* **位相 ($\theta_t = \arg(h_t^{(c)})$)**: 概念間の文脈的役割・相対位置・依存関係。

### 3.2 動的結合項を伴う $\mathcal{O}(d)$ 蔵本位相共鳴

チャネル間の平均場展開（Mean-Field Approximation）により、全チャネル相互作用計算を $\mathcal{O}(d)$ に削減します。さらに、単一のスカラー値ではなく入力 $x_t$ に依存する動的結合ベクトル $K_t \in (0, 1)^d$ を導入します。

$$K_t = \sigma(W_K x_t + b_K)$$

$$\text{coupling}(\theta_t) = \langle \sin \theta \rangle \cos \theta_t - \langle \cos \theta \rangle \sin \theta_t$$

ここで $\langle \sin \theta \rangle = \frac{1}{d}\sum_{m=1}^d \sin \theta_m$, $\langle \cos \theta \rangle = \frac{1}{d}\sum_{m=1}^d \cos \theta_m$ です。

### 3.3 完全統合型 C-RWKV-7 ステップ更新式

Time-Mixing の Key/Value 射影項 $k_t v_t^{(c)}$ を完全統合した状態更新式は以下の通りです。

$$h_{t+1}^{(c)} = \underbrace{\left( r_t \odot \text{decay}_t \right) \cdot \exp \left( i \left( \theta_t + \omega_t + K_t \odot \text{coupling}(\theta_t) \right) \right)}_{\text{位相共鳴減衰・引き込み項}} \;+\; \underbrace{k_t \odot v_t^{(c)}}_{\text{新規入力項}}$$

* $\omega_t = W_\omega x_t + \omega_{\text{base}}$: 入力依存の周波数シフト
* $\text{decay}_t = \sigma(W_d x_t)$: 振幅減衰率
* $k_t = \sigma(W_k x_t) \in \mathbb{R}^d$, $v_t^{(c)} = v_t^{\text{real}} + i v_t^{\text{imag}} \in \mathbb{C}^d$: 時間混合書き込み項

---

## 4. 事前学習済みモデルの適応構造（Real-to-Complex Adaptation）

```
                     ┌────────────────────────────────┐
                     │   Frozen Real Weight W_real    │ ──▶ 振幅基盤 (Real Output)
                     └────────────────────────────────┘
                                     │
入力 x_real ─────────────────────────┼──────────────────────────┐
                                     │                          │
                                     ▼                          ▼
                     ┌────────────────────────────────┐  ┌──────────────────┐
                     │   Trainable Complex C-LoRA     │  │ Dynamic Kuramoto │
                     │   ΔW_c = A_c × B_c (Rank r)    │  │  Phase Engine    │
                     └────────────────────────────────┘  └──────────────────┘
                                     │                          │
                                     └───────────────┬──────────┘
                                                     ▼
                                         複素状態 h_c = r · e^(iθ)

```

### 4.1 ゼロショット等価初期化（Zero-Shot Equivalence）

事前学習重み $W_{\text{pretrained}}$ を用いて初期化します。初期状態（Step 0）において虚数成分を $0$、位相変化量を $0$ と設計することで、転移学習開始時点での推論結果を実数モデルと完全一致させます。

$$W^{(c)} = W_{\text{pretrained}} + i \cdot \mathbf{0}, \quad B^{(c)} = \mathbf{0}$$

---

## 5. 堅牢化・完全化された PyTorch 参照実装

原点付近（$r \to 0$）の極座標特異点回避、動的結合係数 $K_t$、KV結合処理、および位相同期モニタリング機能を完全統合したセル実装です。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustComplexKuramotoRWKV7Cell(nn.Module):
    """
    動的結合係数(K_t)およびKV結合処理を完全統合し、
    数値的安定性と位相ロック防止の評価機能を備えたC-RWKV-7セル
    """
    def __init__(self, dim: int, eps: float = 1e-7):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # 1. 動的パラメータ射影層
        self.to_omega = nn.Linear(dim, dim)
        self.to_decay = nn.Linear(dim, dim)
        self.to_k_coupling = nn.Linear(dim, dim)  # 入力依存の動的結合係数 K_t
        self.omega_base = nn.Parameter(torch.randn(dim) * 0.01)

        # 2. Time-Mixing 射影層 (複素Value)
        self.to_key = nn.Linear(dim, dim)
        self.to_value_r = nn.Linear(dim, dim)
        self.to_value_i = nn.Linear(dim, dim)

    def forward(self, x_real: torch.Tensor, state_c: torch.Tensor):
        """
        x_real:  [batch, dim] (実数入力)
        state_c: [batch, dim] (複素隠れ状態: torch.cfloat)
        """
        # --- A. 状態分解と極座標抽出 (特異点防止) ---
        r = torch.abs(state_c) + self.eps
        theta = torch.angle(state_c)

        # --- B. 入力依存パラメータの生成 ---
        omega = self.to_omega(x_real) + self.omega_base
        decay = torch.sigmoid(self.to_decay(x_real))
        k_coupling = torch.sigmoid(self.to_k_coupling(x_real)) # K_t in (0, 1)

        # --- C. 平均場近似による蔵本位相引き込み (O(d) 計算) ---
        mean_sin = torch.mean(torch.sin(theta), dim=-1, keepdim=True)
        mean_cos = torch.mean(torch.cos(theta), dim=-1, keepdim=True)
        coupling = mean_sin * torch.cos(theta) - mean_cos * torch.sin(theta)

        # 秩序パラメータ R (秩序度のリアルタイム計測)
        order_parameter = torch.abs(torch.mean(torch.exp(1j * theta), dim=-1))

        # --- D. 隠れ状態の位相・振幅更新 ---
        d_theta = omega + k_coupling * coupling
        new_theta = theta + d_theta
        new_r = r * decay

        # 直交形式への復元（polar演算子の特異点回避）
        state_decayed_r = new_r * torch.cos(new_theta)
        state_decayed_i = new_r * torch.sin(new_theta)

        # --- E. Time-Mixing KV 項の統合 (k_t * v_t) ---
        k_t = torch.sigmoid(self.to_key(x_real))
        v_r = self.to_value_r(x_real)
        v_i = self.to_value_i(x_real)

        # 複素状態の完全更新
        new_state_r = state_decayed_r + k_t * v_r
        new_state_i = state_decayed_i + k_t * v_i
        new_state_c = torch.complex(new_state_r, new_state_i)

        return new_state_c, new_theta, order_parameter

```

---

## 6. 位相ロック防止と学習安定化（Loss Design）

全チャネルの位相が同一値に収束して表現力が崩壊する「過度同期（Phase Locking）」を防ぐため、秩序パラメータ $R$ に対する正則化損失を主損失関数に導入します。

### 6.1 秩序パラメータ（Order Parameter）

$$R_t = \left\vert \frac{1}{d} \sum_{j=1}^d e^{i \theta_{t, j}} \right\vert \in [0, 1]$$

* $R_t \to 0$: 無秩序状態（位相が完全にバラバラ）
* $R_t \to 1$: 完全同期状態（位相ロック・表現力の退化）

### 6.2 位相ターゲット損失 (Phase Target Loss)

目的とする同期度 $R_{\text{target}} \in [0.3, 0.6]$（部分同期状態）を維持させるためのペナルティ項を適用します。

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Task}} + \lambda_{\text{phase}} \cdot \frac{1}{T} \sum_{t=1}^T \left( R_t - R_{\text{target}} \right)^2$$

---

## 7. ハードウェア最適化・Tritonカーネル設計

複素数演算（`torch.cfloat`）に伴うメモリ帯域圧迫および Tensor Core 非活用問題を解決するため、専用 Triton カーネルによるパッキング処理を定義します。

1. **BF16 パッキング（Split-Complex Layout）**:
メモリ転送時は複素数を個別の配列ではなく、[Batch, Dim, 2]（実部, 虚部）の連続領域として BF16 精度で配置し、L1/L2 キャッシュのヒット率を最大化。
2. **Fused Phase-Update Kernel**:
三角関数（$\sin, \cos$）および平均場集約（`mean_sin`, `mean_cos`）を単一の SRAM ブロック内で完了させ、VRAM への書き戻しレイテンシを削減。

---

## 8. 実験・検証ロードマップと Go/No-Go ゲート

```
[ Phase 0: ゼロショット検証 ] ──▶ [ Phase 1: アダプター適応 ] ──▶ [ Phase 2: 全系微調整 & カーネル最適化 ]
 (実数モデルとの出力100%一致)    (C-LoRA & K_t & L_phase)       (基幹重み解禁 & Triton化)

```

---
