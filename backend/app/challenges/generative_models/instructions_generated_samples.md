Upload **`vae_samples.npz`** or **`diffusion_samples.npz`**, exported by the corresponding
notebook. It contains `samples` (256×28×28 grayscale values in `[0, 1]`), `model_family`
(`vae` or `diffusion`), the benchmark digest and a short experiment description.

The server validates the sample count/range and recomputes quality metrics. It accepts
data arrays only; no model weights or executable code are needed. The family stored in
the NPZ determines which leaderboard receives the score. Keep your notebook as the
record of architecture, training and sampling settings.
