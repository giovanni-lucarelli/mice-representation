import os
from pathlib import Path

"""Project-wide configuration constants.

All file-system locations are defined with `pathlib.Path` and resolved relative
to the repository layout to ensure portability and correctness.
"""

# discover important locations
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

#? ------------- data constants -------------
NEUROPIXELS_PKL_URL: str = "https://mouse-vision-neuraldata.s3.amazonaws.com/mouse_neuropixels_visual_data_with_reliabilities.pkl"
CALCIUM_PKL_URL: str = "https://mouse-vision-neuraldata.s3.amazonaws.com/mouse_calcium_visual_data_with_reliabilities.pkl"
DATA_PATH = Path(os.path.expanduser("~/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1"))
LABELS_PATH = SRC_DIR / "data" / "labels.txt"
BATCH_SIZE = 512
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
SPLIT_SEED = 42
NUM_WORKERS = 12

#? ------------- checkpoint path -------------
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pth"

#? ------------- model constants -------------
TRAIN = False
LOAD_MODEL = True

#? ------------- training constants -------------
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 3e-4
DROPOUT_RATE = 0.3
PATIENCE = 15
LABEL_SMOOTHING = 0.1
