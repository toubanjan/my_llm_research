# my_llm_research
The ideas presented here were developed in collaboration with AI.

## Infinite-RoSA (v2.0)
> **Dense Every-Token Retrieval for RWKV-8 via Dynamic Context-Adapted Rank-$K$ Attractors and Speculative Prefetching**

#### Abstract (English)

This proposal specifies **Infinite-RoSA v2.0**, an architecture that deepens the mathematical foundation of RWKV-8's **RoSA (Rank-One State Architecture)** to seamlessly integrate a 10GB+ parametric external memory corpus into local LLMs with strictly $O(1)$ time complexity and zero additional VRAM consumption.

Infinite-RoSA v2.0 addresses two critical bottlenecks of previous state-space memory models: **multi-memory crosstalk noise** during synthesis and **geometric misalignment** between precomputed static bases and dynamic input contexts.

The architecture introduces three mathematical and systems-level innovations:
1. **Crosstalk-Free Rank-$K$ State Addition**: Eliminates cross-term noise ($\sum_{i \neq j} w_i w_j k_i v_j^T$) by applying direct-sum Rank-$K$ outer-product additions onto the state matrix:
   $$S_{\text{att}} = \sum_{i \in B_q} w_i \left( k_i' {v_i'}^T \right)$$
   $$S_t = \text{diag}(e^{-\mathbf{w}}) \cdot S_{t-1} + k_t v_t^T + \alpha \cdot S_{\text{att}}$$
2. **Dynamic Context-Adapted Adapter**: A lightweight runtime layer conditioned on query $q_t$ that aligns precomputed static memory bases $(k_i, v_i)$ with the current hidden space geometry:
   $$k_i' = k_i + \text{MLP}_k(k_i \odot q_t) \quad , \quad v_i' = v_i + \text{MLP}_v(v_i \odot q_t)$$
3. **IVF Speculative Cluster Prefetching**: Uses probabilistic centroid transition paths $P(C_m \mid q_t, \Delta q_t)$ to asynchronously stream candidate memory blocks from CPU/NVMe storage into VRAM L1 cache via CUDA Streams, bypassing I/O stalls during decoding.

By harmonizing continuous **Hopfield Attractor Dynamics** ( $\beta_t = \beta_0 \cdot g(q_t)$ ) with RWKV's inherent **Time Decay** ( $e^{-\mathbf{w}}$ ), Infinite-RoSA v2.0 achieves dense every-token retrieval with high factual accuracy, zero crosstalk noise, and robust noise tolerance.

[Full Document](./idea4_2.md)

## H-RWKV (Hamiltonian RWKV)
> **Constant-Memory $\mathcal{O}(1)$ Iterative Reasoning via Phase-Space Attractor Dynamics and Implicit Function Theorem**

#### Abstract (English)

This proposal specifies **H-RWKV (Hamiltonian RWKV)**, an architecture that integrates **Hamiltonian phase-space dynamics** and **Deep Equilibrium Models (DEQ / Implicit Function Theorem)** into the hidden state space of pre-trained RWKV (RWKV-6/7/8) backbones.

Traditional test-time compute and iterative reasoning models suffer from a **BPTT (Backpropagation Through Time) memory wall**, where training memory scales linearly with thinking steps $K$ ($\mathcal{O}(K)$). H-RWKV resolves this bottleneck by treating iterative thinking loops as a search for equilibrium states (attractors) in a phase space, enabling **strictly constant memory $\mathcal{O}(1)$ training** independent of the number of reasoning steps $K$.

The architecture introduces three mathematical and systems-level innovations:
1. **Hamiltonian Phase-Space Projection**: Maps hidden state $h_t \in \mathbb{R}^d$ into position $q$ and momentum $p$ coordinates under a potential field $V(q; \Theta_{\text{adapter}})$ representing conceptual inconsistency:
   $$H(q, p) = \frac{1}{2} \Vert p \Vert^2 + V(q; \Theta_{\text{adapter}})$$
   $$\frac{dp}{d\tau} = -\left( \frac{\partial V(q)}{\partial q} + \gamma p \right), \quad \frac{dq}{d\tau} = p$$
2. **Implicit Function Theorem (DEQ) Backpropagation**: Computes analytical gradients directly from the converged attractor point $z^{\ast} = [q^{\ast}, p^{\ast}]^T$ via Vector-Jacobian Products (VJP) without saving intermediate computational graphs ($\tau = 0, \dots, K$):
   $$\frac{\partial \mathcal{L}}{\partial \theta} = -\frac{\partial \mathcal{L}}{\partial z^{\ast}} \left( J_f(z^{\ast}) \right)^{-1} \frac{\partial f(z^{\ast}; \theta)}{\partial \theta}$$
3. **100% Parameter Reuse & Zero-Distortion Adapter**: Fully freezes the pre-trained RWKV backbone and attaches a lightweight **Phase Engine Adapter** (~0.5%–2.0% total parameters), allowing small models (1B–3B) to achieve high "intelligence density" on complex reasoning, mathematics, and code generation tasks.

By harmonizing symplectic phase-space integration with implicit differentiation, H-RWKV enables deep iterative reasoning with minimal memory consumption, zero context window degradation, and stable gradient flow.

[Full Document](./idea_3.md)
