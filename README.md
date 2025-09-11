# Mice Representation
Final Project for the Deep Learning course at University of Trieste

## Objective

This project investigates the similarity between the visual representations of a mouse's visual cortex and artificial neural networks (ANNs). Specifically, we compare ANN activations on two variations of the same image dataset:

- **Mouse-like preprocessed images**  
- **Raw (non-preprocessed) images**

## Environment Setup

To create and activate a Python virtual environment with the required dependencies:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the environment
# On Linux/MacOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Training and Testing

We provide CLI scripts that load layered YAML configs and save each run into a timestamped checkpoint directory.

### Train

Basic training run with an experiment YAML:

```bash
python scripts/train.py \
  --project-config configs/project.yaml \
  --config configs/train/supervised_no-diet.yaml
```

Override any config value via `--set` using dot notation (the value is parsed as JSON when possible):

```bash
python scripts/train.py \
  --project-config configs/project.yaml \
  --config configs/train/supervised_no-diet.yaml \
  --set train.num_epochs=120 train.optimizer.learning_rate=1e-4 device.device=cpu
```

Augmentations and “diet” toggles are controlled in the experiment YAML under `diet`.

### Test

Test a trained model using the same experiment YAML:

```bash
python scripts/test.py \
  --project-config configs/project.yaml \
  --config configs/train/supervised_no-diet.yaml \
  --checkpoint best_model.pth
```

If `--checkpoint` is a relative filename (e.g., `best_model.pth`), the script first looks in the current run directory; otherwise it searches the most recent run under the same experiment. Absolute paths are used as-is.

You can switch device or other settings with `--set`, exactly like in training:

```bash
python scripts/test.py \
  --project-config configs/project.yaml \
  --config configs/train/supervised_no-diet.yaml \
  --checkpoint checkpoint_epoch_90.pth \
  --set device.device=cuda
```

## Checkpoint Directory Structure

Each run is saved under a timestamped directory within the experiment’s subdirectory:

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
        training_history_scaled.png  # optional, from scripts/scale_history_plots.py
```

- `<experiment-subdir>` comes from `train.checkpoint_sub_dir` in the experiment YAML (e.g., `supervised_no-diet`).
- Each run gets a unique timestamp; logs and artifacts are colocated with checkpoints for reproducibility.
- `resolved_config.yaml` contains the final merged configuration for the run (project + experiment + CLI overrides).

## Utilities

- Scale plots across runs/experiments with unified axes:

```bash
python scripts/scale_history_plots.py --experiments supervised_no-diet supervised_diet --latest-only --max-loss 4.0 --max-acc 100
```


## References

- [Unraveling the complexity of rat object vision requires a full convolutional network and beyond](https://www.sciencedirect.com/science/article/pii/S2666389924003210)

- [BrainScore](https://www.brain-score.org/)

- [A large-scale examination of inductive biases shaping high-level visual representation in brains and machines](https://www.nature.com/articles/s41467-024-53147-y)

- [Mouse visual cortex as a limited resource system that self-learns an ecologically-general representation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011506)