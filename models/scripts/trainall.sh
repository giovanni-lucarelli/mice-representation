#!/bin/bash

# activate virtual environment
source .mouse_vision/bin/activate

# Train supervised no-diet
python scripts/train.py --project-config configs/project.yaml --config configs/train/supervised_no-diet.yaml

# Train supervised diet
python scripts/train.py --project-config configs/project.yaml --config configs/train/supervised_diet.yaml