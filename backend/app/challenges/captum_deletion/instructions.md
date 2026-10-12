## Find the Captum method with the lowest deletion score

Use the **Starter Colab notebook** to explain the supplied, frozen PneumoniaMNIST
classifier. Your job is to choose a **Captum attribution method** and tune its settings,
such as Integrated Gradients integration steps, Occlusion window/stride, Feature Ablation
groups, or NoiseTunnel smoothing. This task compares explanations of the same model.

Start by comparing methods on the **32 practice images**, then generate maps for all
**64 scoring images** and export `attributions.npz`. Keep an experiment table explaining
your choice. The notebook loads the fixed model and data and supplies local scoring,
plots, and export. Do not retrain or replace the model.

### What the deletion score means

For each image, fix the target to the classifier's **original predicted class**. Sort
its 784 pixels by attribution value, largest first. Replace the first 28 pixels with
**black (raw pixel value 0)**, then the next 28, until all pixels are deleted.
Evaluate the probability of that same target at **29 points**, including the untouched
and fully deleted image. Ties use row-major pixel order.

The score is the **trapezoidal area under the target-probability curve** against the
fraction deleted, averaged over all 64 images. It lies in `[0, 1]`; **lower is better**.
It is not ROC AUC, accuracy, or an initial-confidence-normalised score. The server
computes this score from your maps; it does not accept a self-reported score.

Captum returns signed attributions for some methods. Investigate using signed,
positive-only, or absolute values before exporting: the server sorts the numbers you
submit. Keep the target class fixed when inputs are perturbed. The classifier scales
pixels by 255 and normalises using mean/std 0.5, so a black deletion baseline is **-1**
in normalised input space.

Use Captum-generated explanations and record the method/settings honestly. The maps
alone cannot prove which library generated them. A low deletion score measures this
specific perturbation benchmark; discuss what it does and does not demonstrate about
an explanation. See [RISE's deletion metric](https://github.com/eclique/RISE/blob/master/evaluation.py)
and [Captum attribution methods](https://captum.ai/api/index.html).

Images are from [MedMNIST's PneumoniaMNIST](https://medmnist.com/), distributed under
CC BY 4.0. Attribution: Yang et al., MedMNIST v2, Scientific Data (2023), derived from
Kermany et al., Cell (2018). The frozen classifier is a small ReLU MLP trained on the
official training split; practice images use validation and scoring images use test.
