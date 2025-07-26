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

## References

- [Unraveling the complexity of rat object vision requires a full convolutional network and beyond](https://www.sciencedirect.com/science/article/pii/S2666389924003210)

- [BrainScore](https://www.brain-score.org/)

- [A large-scale examination of inductive biases shaping high-level visual representation in brains and machines](https://www.nature.com/articles/s41467-024-53147-y)

- [Mouse visual cortex as a limited resource system that self-learns an ecologically-general representation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011506)