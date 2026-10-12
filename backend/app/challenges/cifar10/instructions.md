## Your task: design a convolutional neural network

Open the **Starter Colab notebook** and choose **Runtime → Change runtime type → T4 GPU**
when available. The notebook supplies data loading, training, validation and export.
Your main task is to develop the architecture in the `model.py` cell.

1. Start with the working CNN, then change its convolutional blocks, channel counts,
   kernel sizes, pooling and classifier. Keep `CNN()` callable without arguments and
   output **10 logits** for each `(3, 32, 32)` RGB image.
2. Train from scratch on the official CIFAR-10 training split. The notebook reserves
   5,000 training images for validation; use validation accuracy to compare architectures.
3. Experiment with batch normalisation, dropout, augmentation and training settings.
   Briefly record what you changed and why in the notebook.
4. Run the export cell, download `submission.zip`, and upload it here.

The leaderboard shows **test accuracy (%) on all 10,000 official CIFAR-10 test images**.
Higher is better; your best successful submission determines your rank. Test labels are
kept out of the inference container. Use the validation set for architecture selection.

Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
The notebook uses RGB tensors scaled to `[0, 1]`, then normalised with mean and standard
deviation `(0.5, 0.5, 0.5)`. Keep training and exported inference preprocessing consistent.

Train only on CIFAR-10 training images; do not use pretrained weights, test labels or
label lookup tables. The official test labels are publicly available, so this is a
classroom exercise that relies on these rules.

[Dataset and attribution: Krizhevsky, Learning Multiple Layers of Features from Tiny Images (2009)](https://www.cs.toronto.edu/~kriz/cifar.html).
