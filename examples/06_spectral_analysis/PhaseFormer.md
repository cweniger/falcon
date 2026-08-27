# PhaseFormer

A dual-stream hierarchical transformer for coherent phase accumulation
in spectral token sequences. Designed for precision parameter estimation
of chirping signals — gravitational waves, radar, sonar, spectroscopy.

```
                        ╔═══════════════════════════════╗
                        ║          PhaseFormer          ║
                        ║                               ║
                        ║  "Coherence is all you need"  ║
                        ╚═══════════════════════════════╝
```

---

## 1. Motivation

The Cramér-Rao bound (CRB) on frequency estimation scales as ~1/(N·SNR),
where N is the total number of samples. Achieving this precision requires
**coherent integration** of phase across the full observation — exactly
what a matched filter does.

A standard transformer operating on spectral tokens must learn to:
1. Identify which tokens belong to the same spectral track (frequency matching)
2. Compute phase differences between adjacent tokens (complex arithmetic)
3. Accumulate these differences over the full signal (sequential chaining)

Step 3 is the bottleneck. Standard self-attention computes weighted sums,
not products. Phase accumulation requires chaining complex multiplications
— a fundamentally different operation.

**PhaseFormer bakes coherent phase accumulation into the architecture.**

---

## 2. Token Representation

Each token represents one spectral peak in one time window, carrying
a **feature vector** (real-valued) and a **phase vector** (complex-valued):

```
    Token = (f, p)

    f ∈ R^d_f    feature vector: frequency, amplitude, time
    p ∈ C²       phase vectors:  p_start, p_end (unit complex)
```

From the raw ToneTokenizer output `(f_start, f_end, amp, φ_start, φ_end)`:

```
    ┌─────────────────────────────────────────────────────────┐
    │  Feature stream (real-valued, for attention routing):   │
    │                                                         │
    │    f_start, f_end     raw frequency in [-1, 1]          │
    │    fourier(f_start)   multi-scale sin/cos encoding      │
    │    fourier(f_end)     multi-scale sin/cos encoding      │
    │    fourier(log1p(A))  multi-scale amplitude encoding    │
    │    t                  window time position in [-1, 1]   │
    │                                                         │
    ├─────────────────────────────────────────────────────────┤
    │  Phase stream (complex-valued, for accumulation):       │
    │                                                         │
    │    p_start = (cos φ_start, sin φ_start)                 │
    │    p_end   = (cos φ_end,   sin φ_end)                   │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

The feature stream determines **which** tokens to link (Q·K attention).
The phase stream carries the **payload** that gets coherently accumulated.

### Spectral embedding of amplitude

Amplitude is embedded with the same multi-scale Fourier features as
frequency: `fourier(log1p(A))` → `sin/cos` at geometrically spaced scales.

This ensures:
- **Amplitude matching via dot products**: tokens with similar brightness
  score high in attention, same mechanism as frequency matching
- **Dynamic range handling**: log scale compresses the range, spectral
  embedding distributes it across multiple features — very bright and
  very faint signals get equal representation
- **Balanced feature space**: amplitude gets comparable dimensionality
  to frequency, preventing frequency features from dominating attention

### Higher harmonics

Higher harmonic structure (integer multiples, or more complex
relationships like f = f₀ + m·f₁ + n·f₂ for EMRIs) is not handled
by the PhaseFormer architecture itself. Instead, harmonic detection
is delegated to an upstream peak detector that identifies harmonic
relationships and encodes them as additional token features. This
keeps the transformer focused on what it does best: frequency matching
and phase accumulation.

---

## 3. Architecture Overview

```
    Input: (B, W, K, 5) raw tokens
           W = time windows, K = peaks per window


    ┌──────────────────────────────────────────────┐
    │            SpectralTokenEmbed                 │
    │  raw tokens → feature vectors + phase vectors │
    └──────────────────┬───────────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   W=50, K tokens    │  ← full resolution
            └─────────┬───────────┘
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
    ┌─────────────┐    ┌──────────────────┐
    │ Intra-window │    │  Inter-window    │
    │ self-attn    │    │  cross-attn      │
    │ (K × K)      │    │  (w ↔ w±1)      │
    │              │    │                  │
    │ standard     │    │  feature: sum    │
    │ attention    │    │  phase: complex  │
    │              │    │  multiply        │
    └──────┬──────┘    └────────┬─────────┘
           │                    │
           └────────┬───────────┘
                    │
                    ▼  stride 2 (drop odd windows)
            ┌─────────────────────┐
            │   W=25, K tokens    │
            └─────────┬───────────┘
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
    ┌─────────────┐    ┌──────────────────┐
    │ Intra-window │    │  Inter-window    │
    │ self-attn    │    │  cross-attn      │
    └──────┬──────┘    └────────┬─────────┘
           │                    │
           └────────┬───────────┘
                    │
                    ▼  stride 2
            ┌─────────────────────┐
            │   W=12, K tokens    │
            └─────────┬───────────┘
                      │
                     ...  (repeat log₂(W) times)
                      │
                      ▼
            ┌─────────────────────┐
            │   W=1,  K tokens    │  ← full phase accumulated
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │    Pool → (B, d)    │
            │  summary embedding  │
            └─────────────────────┘
```

Each pyramid level can contain multiple (MHA → FF) sublayers for the
intra-window step before the inter-window cross-attention and stride.
The number of sublayers per level (`n_intra`) is a hyperparameter —
more sublayers allow deeper feature extraction per level at the cost
of compute.

---

## 4. The Two Attention Mechanisms

### 4.1 Intra-Window Self-Attention (Standard)

Processes peaks within the same time window. Learns to classify signal
vs noise peaks, extract local spectral features, and build per-token
representations for downstream frequency matching.

```
    Window w:   [peak₀, peak₁, peak₂, ..., peak_K]

                     ┌───────────────────┐
                     │  Self-Attention    │
                     │                   │
    peak_i ────Q─────┤   A_ij · V_j     ├──── peak_i'
    peak_j ────K,V───┤   (standard sum)  │
                     └───────────────────┘

    Q, K, V all derived from feature stream f.
    Phase stream p passes through unchanged.
```

Standard multi-head attention. Each window processed independently.
Complexity: O(K²) per window, O(W·K²) total.

### 4.2 Inter-Window Cross-Attention (Complex Phase)

Links tokens between adjacent windows. **This is where the magic happens.**

The inter-window step has two stages: first, multiple cross-attention
rounds refine the feature representations using phase information;
then, a final step performs the actual coherent phase accumulation.

#### Stage 1: Feature refinement with phase information (N_inter rounds)

Phase angles are spectrally embedded — `(cos φ, sin φ, cos 2φ, sin 2φ,
..., cos Mφ, sin Mφ)` — and concatenated into the feature stream for
attention computation. Higher harmonics of the phase provide finer
angular resolution in the dot product, allowing the attention to
distinguish peaks whose phases are close but not identical. This also
balances the feature space: phase gets comparable dimensionality to
frequency features.

```
    ┌─────────────────────────────────────────────────────┐
    │  Augmented feature vector for attention (stage 1):  │
    │                                                     │
    │    f_start, f_end, fourier(f), fourier(log A), t    │
    │    + fourier(φ_start): cos(φ), sin(φ), cos(2φ),..  │
    │    + fourier(φ_end):   cos(φ), sin(φ), cos(2φ),..  │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    repeat N_inter times:
        cross-attention between w ↔ w+1 (standard, features only)
        features updated via residual + FF

    After N_inter rounds, the feature stream encodes which tokens
    across windows are phase-coherent — noise peaks get suppressed,
    signal tracks get reinforced.
```

#### Stage 2: Phase accumulation (single final step)

```
    Window w:        Window w+1:
    ┌──────────┐     ┌──────────┐
    │ f_w, p_w │     │f_w+1,p_w+1│
    └────┬─────┘     └────┬──────┘
         │                │
         │    ┌───────────┴────────────┐
         └───►│  Cross-Attention       │
              │                        │
              │  A_ij from refined     │
              │  features (well-       │
              │  informed by phase)    │
              │                        │
              │  Phase accumulation:   │
              │    Δp = conj(p_end_w)  │
              │        ⊗ p_start_w+1   │
              │                        │
              │    p'_w+1 = normalize( │
              │      Σ_j A_ij · Δp_ij) │
              │                        │
              └────────────────────────┘
```

The attention weights A_ij are now informed by multiple rounds of
feature exchange *including phase information*. Tokens that showed
phase continuity in stage 1 get high attention; noise peaks that
matched nothing get suppressed.

The phase accumulation uses the raw `(cos φ, sin φ)` vectors —
higher spectral modes of φ are only used for computing attention
weights, not for the complex multiplication itself.

When token i in window w+1 attends strongly to token j in window w
(same spectral track, high frequency, amplitude, and phase match):

```
    Δp = conj(p_end_j) ⊗ p_start_i = (cos Δφ, sin Δφ)
```

This Δφ is the phase jump at the window boundary. For a clean signal,
Δφ ≈ 0 (phase continuity). The magnitude of the aggregated vector
gives **coherence** — a built-in SNR indicator.

---

## 5. Phase Accumulation via Hierarchical Striding

The key insight: after each inter-window layer, surviving tokens carry
phase information spanning their window's time range. Striding by 2
doubles the effective span each layer.

```
    Layer 0:  Phase spans 1 window
              ───┬───┬───┬───┬───┬───┬───┬───
              w0  w1  w2  w3  w4  w5  w6  w7       W = 8

              inter-window cross-attention (w ↔ w+1)
              stride 2: keep even windows

    Layer 1:  Phase spans 2 windows
              ───┬───────┬───────┬───────┬───
              w0          w2          w4          w6      W = 4

              inter-window cross-attention
              stride 2

    Layer 2:  Phase spans 4 windows
              ───┬───────────────┬───────────────
              w0                          w4              W = 2

              inter-window cross-attention
              stride 2

    Layer 3:  Phase spans 8 windows (full signal)
              ───┬───────────────────────────────
              w0                                          W = 1


    Phase evolution across layers for one spectral track:

    Layer 0:   p₀──Δφ₀₁──p₁──Δφ₁₂──p₂──Δφ₂₃──p₃──Δφ₃₄──p₄
                    │                      │
    Layer 1:   p₀───┴──Δφ₀₂──────p₂───────┴──Δφ₂₄──────p₄
                         │                      │
    Layer 2:   p₀────────┴────Δφ₀₄──────────p₄──┘
                                │
    Layer 3:   p₀───────────────┴── = (cos Σ Δφ, sin Σ Δφ)
                                         └── total accumulated phase
```

After log₂(W) layers, each token carries the **total phase integral**
for its spectral track. This is the matched filter output.

---

## 6. Complex Phase Operations

All phase operations use 2D real arithmetic (no complex dtype needed):

```python
def complex_mul(a_cos, a_sin, b_cos, b_sin):
    """Multiply two unit complex numbers: a ⊗ b."""
    return (a_cos * b_cos - a_sin * b_sin,
            a_cos * b_sin + a_sin * b_cos)

def complex_conj_mul(a_cos, a_sin, b_cos, b_sin):
    """Phase difference: conj(a) ⊗ b = (cos(θb-θa), sin(θb-θa))."""
    return (a_cos * b_cos + a_sin * b_sin,
            a_cos * b_sin - a_sin * b_cos)

def complex_normalize(c_cos, c_sin):
    """Project back onto unit circle."""
    mag = (c_cos**2 + c_sin**2).sqrt().clamp(min=1e-8)
    return c_cos / mag, c_sin / mag
```

The magnitude before normalization is the **coherence**:
- |c| ≈ 1 → attended phases agree → real signal
- |c| ≈ 0 → attended phases random → noise

This can be fed back into the feature stream as a confidence signal.

---

## 7. Detailed Layer Pseudocode

```python
class PhaseFormerLayer(nn.Module):
    """One level of the PhaseFormer pyramid.

    Each level:
      1. Intra-window self-attention (n_intra rounds)
      2. Inter-window cross-attention with phase features (n_inter rounds)
      3. Phase accumulation via complex multiply (single step)
      4. Stride 2 (implicit — outputs have W' = W // 2)
    """

    def __init__(self, d_model, n_heads, d_ff, n_intra=1, n_inter=1,
                 n_phase_harmonics=4):
        self.n_phase_harmonics = n_phase_harmonics
        # Phase feature dimension: n_harmonics * 2 (cos,sin) * 2 (start,end)
        d_phase = n_phase_harmonics * 2 * 2

        # Intra-window self-attention (repeated n_intra times)
        self.intra_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': LayerNorm(d_model),
                'attn':  MultiHeadAttention(d_model, n_heads),
                'norm2': LayerNorm(d_model),
                'ff':    FeedForward(d_model, d_ff),
            })
            for _ in range(n_intra)
        ])

        # Inter-window cross-attention with phase features (n_inter rounds)
        # Input is d_model + d_phase (features augmented with phase)
        d_aug = d_model + d_phase
        self.inter_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': LayerNorm(d_aug),
                'q':     nn.Linear(d_aug, d_model),
                'k':     nn.Linear(d_aug, d_model),
                'v':     nn.Linear(d_aug, d_model),
                'norm2': LayerNorm(d_model),
                'ff':    FeedForward(d_model, d_ff),
            })
            for _ in range(n_inter)
        ])

        # Final cross-attention for phase accumulation
        self.phase_q = nn.Linear(d_aug, d_model)
        self.phase_k = nn.Linear(d_aug, d_model)
        self.phase_norm = LayerNorm(d_aug)

    def _phase_features(self, ps_cos, ps_sin, pe_cos, pe_sin):
        """Spectral embedding of phase angles for attention.

        Returns (cos φ, sin φ, cos 2φ, sin 2φ, ...) for both
        start and end phases, concatenated.
        """
        parts = []
        for m in range(1, self.n_phase_harmonics + 1):
            if m == 1:
                parts += [ps_cos, ps_sin, pe_cos, pe_sin]
            else:
                # cos(mφ) and sin(mφ) via Chebyshev recurrence
                # or direct: angle = m * atan2(sin, cos)
                ps_angle = torch.atan2(ps_sin, ps_cos)
                pe_angle = torch.atan2(pe_sin, pe_cos)
                parts += [torch.cos(m * ps_angle), torch.sin(m * ps_angle),
                          torch.cos(m * pe_angle), torch.sin(m * pe_angle)]
        return torch.stack(parts, dim=-1)  # (..., n_harmonics * 4)

    def forward(self, f, ps_cos, ps_sin, pe_cos, pe_sin):
        """
        f:           (B, W, K, d_model)   feature stream
        p_*_cos/sin: (B, W, K)            phase streams (unit complex)

        Returns same shapes with W' = W // 2 (strided).
        """
        B, W, K, d = f.shape

        # ─── Intra-window self-attention ───────────────────────
        f_flat = f.reshape(B * W, K, d)
        for block in self.intra_blocks:
            f_flat = f_flat + block['attn'](
                block['norm1'](f_flat))
            f_flat = f_flat + block['ff'](
                block['norm2'](f_flat))
        f = f_flat.reshape(B, W, K, d)

        # ─── Inter-window cross-attention (feature refinement) ─
        f_even = f[:, 0::2]                          # (B, W//2, K, d)
        f_odd  = f[:, 1::2]                          # (B, W//2, K, d)
        W2 = f_even.shape[1]

        # Build phase features for attention
        pf_even = self._phase_features(
            ps_cos[:, 0::2], ps_sin[:, 0::2],
            pe_cos[:, 0::2], pe_sin[:, 0::2])         # (B, W2, K, d_phase)
        pf_odd = self._phase_features(
            ps_cos[:, 1::2], ps_sin[:, 1::2],
            pe_cos[:, 1::2], pe_sin[:, 1::2])

        for block in self.inter_blocks:
            # Augment features with phase for Q, K computation
            f_even_aug = torch.cat([f_even, pf_even], dim=-1)
            f_odd_aug  = torch.cat([f_odd,  pf_odd],  dim=-1)

            fe = f_even_aug.reshape(B * W2, K, -1)
            fo = f_odd_aug.reshape(B * W2, K, -1)

            q = block['q'](block['norm1'](fe))
            k = block['k'](fo)
            v = block['v'](fo)

            scale = d ** 0.5
            A = (q @ k.transpose(-1, -2) / scale).softmax(dim=-1)

            # Update even-window features (standard residual)
            f_cross = (A @ v).reshape(B, W2, K, d)
            f_even = f_even + f_cross
            f_even = f_even + self.inter_blocks[0]['ff'](
                block['norm2'](f_even.reshape(B * W2, K, d))
            ).reshape(B, W2, K, d)

        # ─── Phase accumulation (single final step) ───────────
        f_even_aug = torch.cat([f_even, pf_even], dim=-1)
        f_odd_aug  = torch.cat([f_odd,  pf_odd],  dim=-1)

        fe = f_even_aug.reshape(B * W2, K, -1)
        fo = f_odd_aug.reshape(B * W2, K, -1)

        q = self.phase_q(self.phase_norm(fe))
        k = self.phase_k(fo)
        A = (q @ k.transpose(-1, -2) / (d ** 0.5)).softmax(dim=-1)

        # Complex conjugate multiply: conj(p_end_odd) ⊗ p_start_even
        pe_cos_o = pe_cos[:, 1::2].reshape(B * W2, K)
        pe_sin_o = pe_sin[:, 1::2].reshape(B * W2, K)
        ps_cos_e = ps_cos[:, 0::2].reshape(B * W2, K)
        ps_sin_e = ps_sin[:, 0::2].reshape(B * W2, K)

        # Phase difference for each (even_i, odd_j) pair
        delta_cos = (ps_cos_e.unsqueeze(2) * pe_cos_o.unsqueeze(1)
                   + ps_sin_e.unsqueeze(2) * pe_sin_o.unsqueeze(1))
        delta_sin = (ps_sin_e.unsqueeze(2) * pe_cos_o.unsqueeze(1)
                   - ps_cos_e.unsqueeze(2) * pe_sin_o.unsqueeze(1))

        # Attention-weighted aggregation
        agg_cos = (A * delta_cos).sum(dim=-1)
        agg_sin = (A * delta_sin).sum(dim=-1)

        # Coherence + normalize
        coherence = (agg_cos**2 + agg_sin**2).sqrt()
        mag = coherence.clamp(min=1e-8)
        ps_cos_out = (agg_cos / mag).reshape(B, W2, K)
        ps_sin_out = (agg_sin / mag).reshape(B, W2, K)
        pe_cos_out = pe_cos[:, 0::2]
        pe_sin_out = pe_sin[:, 0::2]

        return f_even, ps_cos_out, ps_sin_out, \
               pe_cos_out, pe_sin_out, coherence.reshape(B, W2, K)
```

---

## 8. Full Architecture

```python
class PhaseFormer(nn.Module):
    """
    Hierarchical dual-stream transformer for coherent phase accumulation.

    Input:  (B, W, K, 5) raw spectral tokens
    Output: (B, d_out)   summary embedding for downstream estimation
    """

    def __init__(self, d_model=64, n_heads=4, d_ff=256,
                 n_freq=3, amp_scale=10.0, n_intra=1,
                 max_layers=8):
        # Token embedding (SpectralTokenEmbed-like)
        self.embed = TokenEmbedding(n_freq, amp_scale)
        self.input_proj = nn.LazyLinear(d_model)

        # Pyramid layers: one per halving step
        self.layers = nn.ModuleList([
            PhaseFormerLayer(d_model, n_heads, d_ff, n_intra)
            for _ in range(max_layers)
        ])

        # Final pooling
        self.output_proj = nn.Linear(d_model + 2, d_out)
        #                              │       └── accumulated phase (cos, sin)
        #                              └────────── feature stream

    def forward(self, raw_tokens):
        B, W, K, _ = raw_tokens.shape

        # Embed tokens → feature + phase streams
        f, ps_cos, ps_sin, pe_cos, pe_sin = self.embed(raw_tokens)
        f = self.input_proj(f)                     # (B, W, K, d_model)

        # Pad W to next power of 2 if needed
        ...

        # Pyramid: log₂(W) layers
        coherences = []
        layer_idx = 0
        while f.shape[1] > 1:
            f, ps_cos, ps_sin, pe_cos, pe_sin, coh = \
                self.layers[layer_idx](f, ps_cos, ps_sin, pe_cos, pe_sin)
            coherences.append(coh)
            layer_idx += 1

        # f: (B, 1, K, d_model), phase: (B, 1, K)
        # Concatenate accumulated phase into feature stream
        f_final = torch.cat([
            f.squeeze(1),                          # (B, K, d_model)
            ps_cos.squeeze(1).unsqueeze(-1),       # (B, K, 1)
            ps_sin.squeeze(1).unsqueeze(-1),       # (B, K, 1)
        ], dim=-1)                                 # (B, K, d_model+2)

        # Pool over K peaks → (B, d_model+2)
        summary = f_final.mean(dim=1)

        return self.output_proj(summary)           # (B, d_out)
```

---

## 9. Complexity Analysis

```
    Standard Transformer:          O(W² · K²)     per layer
    PhaseFormer (per layer):       O(W_l · K²)    intra + inter
    PhaseFormer (total):           O(W · K² · log₂W)

    ┌────────────┬────────────┬──────────────────────────┐
    │ Component  │ Per layer  │ Example (W=50, K=100)    │
    ├────────────┼────────────┼──────────────────────────┤
    │ Intra-attn │ W_l · K²  │ 50 · 10,000 = 500K      │
    │ Inter-attn │ W_l · K²  │ 50 · 10,000 = 500K      │
    │ Total/layer│ 2W_l · K² │ 1M                       │
    │ All layers │ Σ 2W_l·K² │ ≈ 2M (geometric sum)     │
    ├────────────┼────────────┼──────────────────────────┤
    │ Standard   │ (WK)² = 25M per layer × L layers     │
    └────────────┴────────────┴──────────────────────────┘

    Speedup: ~10-50x for typical configurations.
```

---

## 10. Why This Works for Gravitational Waves

```
    ┌──────────────────────────────────────────────────────┐
    │                   LISA Signal                        │
    │                                                      │
    │   Source 1:  f₁(t) ───── slowly chirping ─────►     │
    │   Source 2:  f₂(t) ──── different chirp rate ──►    │
    │   Source 3:  f₃(t) ─── overlapping in freq ───►    │
    │              + noise                                 │
    │                                                      │
    │   Standard transformer: must learn to separate      │
    │   sources, track them, AND accumulate phase         │
    │   — all through generic attention. Hard.            │
    │                                                      │
    │   PhaseFormer: frequency + amplitude attention      │
    │   separates sources automatically. Phase             │
    │   accumulation is structural. Network only learns   │
    │   spectral matching — the easy part.                │
    │                                                      │
    └──────────────────────────────────────────────────────┘

    Phase precision for frequency estimation:

         δf ~ 1 / (2π · T_obs · SNR)

    Matched filter achieves this by integrating phase over T_obs.
    PhaseFormer does the same through log₂(W) layers of complex
    phase accumulation — structurally guaranteed, not learned.

    ┌───────────────────────────────────────────┐
    │                                           │
    │    "The architecture IS the physics"      │
    │                                           │
    └───────────────────────────────────────────┘
```

---

## 11. Implementation Plan

```
    Phase 1: Single-source prototype (falcon example 06)
    ├── Implement PhaseFormerLayer in examples/06_spectral_analysis/src/
    ├── Test on current chirp signal (N=100k)
    ├── Compare posterior width to CRB and standard TransformerEmbedding
    └── Validate: inspect accumulated phases, check coherence values

    Phase 2: Multi-source (fuge integration)
    ├── Move PhaseFormer to fuge.nn
    ├── Test with 2-3 overlapping chirps
    ├── Verify frequency-based track separation
    └── Benchmark scaling with K (peaks per window)

    Phase 3: LISA scale
    ├── Long observations (W > 1000)
    ├── Hundreds of peaks per window
    ├── Multiple simultaneous sources
    ├── Upstream harmonic peak detector for EMRI-like signals
    └── Integration with LISA data pipeline
```

---

## 12. Open Questions

- **Phase stream gradient flow**: The complex multiply + normalize chain
  may have vanishing/exploding gradients through many layers. May need
  gradient clipping or residual connections in the phase stream.

- **Odd W handling**: When W is odd, the stride-2 merge drops one window.
  Pad to next power of 2, or handle remainder explicitly?

- **Coherence feedback**: Should coherence (|aggregated phase|) be fed
  back into the feature stream? This would let the network learn to
  trust high-coherence tracks and ignore noise.

- **Learnable scales**: Should the Fourier frequency/amplitude scales be
  learnable rather than fixed geometric? May help adapt to
  signal-specific distributions.

- **Phase stream initialization**: Currently p_start and p_end come
  directly from the tokenizer. Should they be projected/rotated by a
  learned transformation first?

- **Number of intra-window sublayers**: How many (MHA → FF) blocks per
  pyramid level? More depth helps feature extraction but costs compute.
  May need more at early levels (many peaks to sort) than later levels
  (fewer, better-refined tokens).

- **Uncertainty-damped spectral modes (Debye-Waller damping)**: The peak
  detector can estimate measurement uncertainty σ_f for each frequency.
  Higher Fourier modes in the spectral embedding should then be damped:

  ```
      fourier_i(f) = sin(s_i · f) · exp(-s_i² · σ_f² / 2)
  ```

  This is the exact result of marginalizing `cos(s·(f + σ·ε))` over
  Gaussian noise ε — the expected value picks up a damping factor
  `exp(-s²σ²/2)`. Physically: don't encode fine frequency structure
  that the measurement can't resolve.

  Analogous to the **Debye-Waller factor** in crystallography, where
  thermal vibrations damp high-order Bragg peaks by the same factor.
  Here, measurement noise plays the role of thermal motion, and the
  spectral embedding modes play the role of diffraction orders.

  This provides natural regularization: noisy peaks only contribute
  through low-frequency modes (coarse matching), while precise peaks
  retain their full multi-scale resolution. Acts as a physics-informed
  alternative to dropout — the damping is set by actual measurement
  precision, not a tuning parameter.

```
    ┌────────────────────────────────────────────────────┐
    │                                                    │
    │         ╱╲    Signal goes in                       │
    │        ╱  ╲                                        │
    │       ╱    ╲   Phases accumulate                   │
    │      ╱  ╱╲  ╲                                      │
    │     ╱  ╱  ╲  ╲  Sources separate                   │
    │    ╱  ╱    ╲  ╲                                    │
    │   ╱  ╱  ╱╲  ╲  ╲  Noise falls away                │
    │  ╱  ╱  ╱  ╲  ╲  ╲                                 │
    │ ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔                               │
    │  K tokens, total accumulated phase                 │
    │  CRB-level precision on f₀                         │
    │                                                    │
    │         Physics in, posteriors out.                 │
    │                                                    │
    └────────────────────────────────────────────────────┘
```
