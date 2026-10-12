## Your task: classify the PneumoniaMNIST chest X-rays

Open the **Starter Colab notebook**, save a copy to Drive, and select a GPU runtime
if available. The notebook imports **PneumoniaMNIST from MedMNIST**, using its standard
28×28 grayscale images and official splits: **4,708 training**, **524 validation**,
and **624 test** images. The server uses the test split for scoring.

1. Develop the CNN architecture in the `model.py` cell. Keep `CNN()` callable without
   arguments; accept `(batch, 1, 28, 28)` images and return `(batch, 2)` logits.
2. Train from scratch on the training split. Compare architectures using the validation
   split; the notebook saves the checkpoint with the best validation accuracy.
3. Try different convolutional blocks, pooling, dropout, optimiser settings, or class
   weighting. Record what changed, your validation metrics, and what you learned.
4. Export `submission.zip` from the notebook and upload it here.

The leaderboard ranks your best successful submission by **test accuracy (%)** on all
624 official test images. It also shows **AUROC**, computed from pneumonia probabilities,
and **balanced accuracy**, the average recall across the two classes. Discuss how these
metrics differ when the classes have different numbers of examples.

Class order: **0 = normal**, **1 = pneumonia**. The notebook scales pixels to `[0, 1]`
and then normalises with mean 0.5 and standard deviation 0.5. Keep exported inference
preprocessing consistent with validation preprocessing. Use only training labels for
training and validation labels for model selection; do not use test labels or pretrained
weights. Test labels are public in MedMNIST, so this exercise relies on classroom trust.

Dataset attribution: PneumoniaMNIST is distributed as part of
[MedMNIST v2 (Yang et al., Scientific Data, 2023)](https://www.nature.com/articles/s41597-022-01721-8),
derived from [Kermany et al., Cell, 2018](https://doi.org/10.1016/j.cell.2018.02.010).
The [MedMNIST metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
lists this dataset under **CC BY 4.0**. Images are resized by MedMNIST; this classroom
challenge uses those published benchmark images and splits.
