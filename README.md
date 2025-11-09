# Mice Representation  
**Final Project — Deep Learning Course, University of Trieste**

## Objective

This project investigates the **similarity between visual representations** in the **mouse visual cortex** and **artificial neural networks (ANNs)**.  
Specifically, we compare ANN activations on two variations of the same image dataset:

- **Mouse-like preprocessed images**
- **Raw (non-preprocessed) images**

The goal is to understand how biologically inspired preprocessing affects ANN representations and their correspondence with neural data.


> For a complete analysis and results, see the [final report](./report/main.pdf).


## Project Structure

```text
🐭 mice-representation/
├── 🧹 allen-data-clean/          # preprocessing code and preprocessed data
│   ├── preproc.py
│   ├── preprocessing.ipynb
│   ├── stimuli_images/
│   └── PreprocData/
├── 🗄️ allen-data-raw/            # original Allen resources
│   ├── neural_data/
│   └── neuropixels.zarr/
├── 🤖 models/                    # training code and configs
│   ├── configs/
│   ├── scripts/
│   ├── src/
│   └── weights/
├── 🧠 neural-mapping/            # map ANN activations to neural data (RSA/CKA/PLS)
│   ├── activation/
│   ├── model_to_neural.ipynb
│   ├── neural_to_neural.ipynb
│   ├── plot.ipynb
│   └── src/
├── 🛠️ pipeline/                  # mouse-like vision preprocessing
│   ├── mouse_pipeline.ipynb
│   ├── artifacts/
│   └── src/
├── 📚 doc/                       # references
├── 📝 report/                    # LaTeX report
├── pyproject.toml
├── uv.lock
├── README.md
└── LICENSE
````


## Quickstart (with `uv`)

You can use [`uv`](https://github.com/astral-sh/uv) to manage your Python environment efficiently.

```bash
# Create the environment
uv venv

# Activate it
source .venv/bin/activate        # Linux/MacOS
# .venv\Scripts\Activate.ps1     # Windows (PowerShell)

# Sync dependencies
uv sync
```


## Manual Environment Setup

If you prefer to use standard `venv` + `pip`:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # Linux/MacOS
# .venv\Scripts\Activate.ps1     # Windows (PowerShell)

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```


## Training and Testing

Training and testing use **layered YAML configurations** for reproducibility and flexibility.
Each run automatically creates a **timestamped checkpoint directory** containing logs, configs, and artifacts.

### Train

Run a basic training experiment:

```bash
python scripts/train.py \
  --project-config configs/project.yaml \
  --config configs/train/supervised_no-diet.yaml
```

Override any configuration value using `--set` (values are parsed as JSON when possible):

```bash
python scripts/train.py \
  --project-config configs/project.yaml \
  --config configs/train/supervised_no-diet.yaml \
  --set train.num_epochs=120 train.optimizer.learning_rate=1e-4 device.device=cpu
```

Augmentations and "diet" toggles are defined under the `diet` section in the YAML file.


### Test

Evaluate a trained model using the same experiment YAML:

```bash
python scripts/test.py \
  --project-config configs/project.yaml \
  --config configs/train/supervised_no-diet.yaml \
  --checkpoint best_model.pth
```

If `--checkpoint` is a relative path, the script first searches the current run directory,
then the most recent run of the same experiment. Absolute paths are used as-is.

Example with overrides:

```bash
python scripts/test.py \
  --project-config configs/project.yaml \
  --config configs/train/supervised_no-diet.yaml \
  --checkpoint checkpoint_epoch_90.pth \
  --set device.device=cuda
```

## Checkpoint Directory Structure

Each experiment produces a dedicated directory with all logs and results:

```
checkpoints/
  <experiment-subdir>/
    <YYYYMMDD_HHMMSS>/
      train.log
      resolved_config.yaml
      best_model.pth
      checkpoint_epoch_<N>.pth
      artifacts/
        training_history.png
        training_history.csv
        training_history_scaled.png
```

* `<experiment-subdir>` is specified in the YAML (`train.checkpoint_sub_dir`).
* Each run is **self-contained** for full reproducibility.

## Utilities

Standardize plots across multiple runs or experiments for easier comparison:

```bash
python scripts/scale_history_plots.py \
  --experiments supervised_no-diet supervised_diet \
  --latest-only --max-loss 4.0 --max-acc 100
```

## Authors

* Giacomo Amerio, Giovanni Lucarelli, Andrea Spinelli
