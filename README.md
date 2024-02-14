# CausalScore 
## a model-based metric for dialogue response evaluation.

## Setup
We use conda to manage python packages.
```bash
conda env create -n dialog_rel --file environment.yml
conda activate dialog_rel
```

## Usage
`run_text_classification.py` is the training script for unconditional independence classifier.

`run_condtional_classifier_EM.py` is the training script for conditional independence classifier.

`interact_models.py` is the script to interact with the trained models.

CGDIALOG+ is in `cgdialog+` folder.