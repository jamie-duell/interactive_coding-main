The notebook exports **`submission.zip`** containing `model.py` (your architecture),
`weights.pt` (the validation-selected model weights), and `submission.py` (inference).
Upload the ZIP; keep your `.ipynb` as the record of your work.

The server loads your model on **CPU**, runs it against the official test images, and
computes accuracy itself. It does not use an accuracy value reported by the notebook.
Inference runs in batches without network access; the runner has PyTorch and NumPy,
but does not have torchvision. Put custom layers and any helper functions in `model.py`.
After changing the architecture, rerun model creation and training before exporting.

If writing your own inference code, `predict(X)` receives `(N, 3072)` raw pixels in
`0..255`: the red channel's 1024 pixels, then green, then blue. Reshape to `(N, 3, 32, 32)`.
Return an `(N, 10)` probability matrix in the class order above (apply softmax).
Keep inference compact enough for the configured CPU time and memory limits.
