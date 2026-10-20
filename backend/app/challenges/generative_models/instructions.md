## Two model families, two leaderboards

Choose the **VAE starter notebook** or **Diffusion starter notebook** in Resources.
Both use the official PneumoniaMNIST training images: 28×28 grayscale chest X-rays.
Develop the model architecture and training settings in Colab, then generate **256 new,
unconditional samples** and upload the notebook's NPZ export. You may try both families.

Your best VAE score and best diffusion score are saved **independently**. The student,
teacher and projector screens show both leaderboards together; a submission in one
family cannot replace your entry in the other. Lower scores are better within each board.

### The generation-quality benchmark

The server computes **RBF feature MMD²** against a fixed set of held-out PneumoniaMNIST
features, using the same frozen domain classifier, feature normalisation and kernel
bandwidth for everyone. It includes the diagonal terms (the biased V-statistic), uses
exactly 256 generated samples and clips tiny numerical negatives to zero. This is a
classroom distribution-matching score, **not Inception-based FID or KID**. The notebook
supplies the identical scorer to help you compare experiments.

The boards also show feature diversity; inspect the generated-image gallery when
discussing quality and mode collapse. A single score does not establish clinical realism.
VAE reconstruction/KL loss and diffusion noise-prediction loss are training diagnostics;
they are not compared across the two boards.

### Your work

* **VAE:** change the encoder/decoder, latent dimension, KL weight, optimiser or schedule.
  Generate by drawing fresh latent vectors from the prior, not by reconstructing inputs.
* **Diffusion:** develop the time-conditioned noise predictor, noise schedule, number of
  steps, optimiser or schedule. Generate by denoising independent Gaussian noise.
* Train only on training images, from scratch. Do not upload copied training/test images,
  reconstructions, external images, pretrained-model outputs or hand-selected best samples.
  Export the first 256 outputs from the fixed sampling seed in your notebook.
* Keep an experiment table and update the model family and description honestly. Sample
  arrays alone cannot verify which architecture generated them; this is a trusted classroom.

Attribution: [MedMNIST v2, Yang et al. (2023)](https://www.nature.com/articles/s41597-022-01721-8),
derived from [Kermany et al. (2018)](https://doi.org/10.1016/j.cell.2018.02.010).
PneumoniaMNIST is distributed under CC BY 4.0. The feature encoder was trained only on
the official training split; reference-distribution features come from test images.
