# Project 3 — PathMNIST Classification and Robustness

## Question
How does a small image classifier perform on PathMNIST, and how much do controlled image corruptions change its performance?

## Dataset
PathMNIST is part of MedMNIST. Use the official package and the supplied train/validation/test splits. See https://github.com/MedMNIST/MedMNIST and cite MedMNIST plus the source dataset as required.

## Experiment
- Start with a small CNN; compare with a ResNet only if time permits.
- Report accuracy, macro-F1, and confusion matrix.
- Select checkpoints using validation macro-F1.
- Evaluate the selected checkpoint once on the official test split.
- Run separate test-time stress tests: Gaussian noise and brightness shift. These are synthetic corruptions, not a substitute for external validation.

## Run
Open `notebooks/02_pathmnist_robustness.ipynb` in Colab. The notebook downloads PathMNIST through MedMNIST, uses fixed official splits, logs metrics, and plots results.

## Limitations
PathMNIST images are low-resolution benchmark images. High benchmark performance does not imply clinical utility or generalization to clinical workflows.
