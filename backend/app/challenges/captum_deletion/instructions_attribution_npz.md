Upload **`attributions.npz`**, exported by the notebook. It contains:

* `ids`: all 64 scoring image ids, exactly once;
* `attributions`: `(64, 784)` finite pixel-importance values, one row per id;
* `model_sha256`: the digest of the supplied frozen model;
* `method`: a short description of your Captum method and settings.

The server aligns rows by id and re-evaluates deletion with its fixed classifier.
It accepts data arrays only; do not upload model weights, Python code, test labels,
predictions, or notebook-reported scores. Signed and constant maps are valid, and
ties are resolved by row-major pixel order. Keep your `.ipynb` as your experiment record.
