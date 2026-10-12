Run the notebook's export cell and upload **`submission.zip`**. It contains
`model.py` (your architecture), `weights.pt` (your validation-selected weights),
and `submission.py` (batched inference). Keep the `.ipynb` as the record of your work.

The server loads your model on **CPU** and computes the test metrics itself. Its
isolated inference container has PyTorch and NumPy, without network access, MedMNIST,
or torchvision. Put all custom model layers and helpers in `model.py`.

`predict(X)` receives `(N, 784)` raw grayscale pixels in `0..255`, in row-major order.
Reshape to `(N, 1, 28, 28)` and apply the same normalisation used during training.
Return `(N, 2)` class probabilities: column 0 normal, column 1 pneumonia. Use softmax
on the two logits. Rerun model creation and training after changing the architecture.
