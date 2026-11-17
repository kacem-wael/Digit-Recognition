# Handwritten Digit Recognition

Draw a digit with the mouse and a neural network recognizes it in real time. While you draw, the
interface shows the probability of each digit 0–9, the shape descriptors extracted from your drawing,
and the network's neurons activating layer by layer.

The neural network is written from scratch in **NumPy** (forward pass, backpropagation and the Adam
optimizer), with no PyTorch or TensorFlow. It reaches **99.17 % accuracy** on the 10 000 MNIST test images.

## Quick start

```bash
pip install numpy opencv-python scikit-image matplotlib scikit-learn
python train.py     # downloads MNIST and trains the network (the trained model is already in model/)
python app.py       # opens http://localhost:8000
```

The web server uses only Python's standard library, so Flask isn't needed. `scikit-learn` is used
only to download MNIST and for the baseline in `main.py`.

## Results

Evaluated on the MNIST test set: 10 000 images the network never saw during training.

| Method | Accuracy |
|---|---|
| Baseline: 7 Hu moments → PCA → KNN (`main.py`) | 49.45 % |
| **Shape descriptors (1314 values) → neural network** | **99.17 %** |

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy (%) | 99.5 | 99.7 | 99.2 | 99.4 | 99.3 | 99.3 | 98.9 | 99.2 | 98.9 | 98.2 |

Running `train.py` also saves the confusion matrix to `figures/nn_confusion.png`. One prediction takes about 30 ms on a CPU.

## How it works

```
drawing or MNIST image
   │  normalization: crop → fit in a 20×20 box → center of mass at the center of a 28×28 frame
   ▼
1314 descriptors
   Pixels 784 · HOG 441 · Hu moments 7 · Zoning 16 · Projections 56 · Topology 10
   ▼
standardization → Dense 128 (ReLU) → Dense 64 (ReLU) → Dense 10 (softmax)
   ▼
probability for each digit 0–9
```

### 1. Normalization

MNIST digits were made by fitting each digit in a 20×20 box and centering it by its center of mass
inside a 28×28 image. Mouse drawings go through exactly the same step, so they look like the training
data whatever their size or position on the canvas.

### 2. Shape descriptors

Instead of giving the network raw pixels only, each image is described by six groups of features:

| Group | Size | What it captures |
|---|---|---|
| Pixels | 784 | the normalized 28×28 image |
| HOG | 441 | Histogram of Oriented Gradients: stroke directions in 7×7 cells, 9 orientations each |
| Hu moments | 7 | invariant moments describing global shape (robust to translation, scale and rotation) |
| Zoning | 16 | ink density in a 4×4 grid |
| Projections | 56 | ink profile of each row and each column |
| Topology | 10 | number of holes (0, 6, 8, 9 have loops), stroke end-points and junctions from the skeleton, aspect ratio |

### 3. The neural network

The network is a **multilayer perceptron (MLP)**, a stack of fully connected layers. Each neuron
computes a weighted sum of its inputs plus a bias, then applies an activation function:

- **ReLU** (`max(0, x)`) in the two hidden layers, which lets the network learn non-linear combinations of descriptors.
- **Softmax** on the 10 output neurons, which turns the raw scores into probabilities that sum to 1.

Training minimizes the **cross-entropy loss** between the predicted probabilities and the true digit.
**Backpropagation** computes the gradient of the loss with respect to every weight, layer by layer
from the output back to the input. The **Adam** optimizer then updates the weights using running
averages of these gradients.

Techniques used to train the network and keep it from overfitting:

| Technique | Role |
|---|---|
| He initialization | starts weights at a scale suited to ReLU so the signal doesn't vanish or explode |
| Input standardization | every descriptor rescaled to mean 0 and variance 1 |
| Dropout (10 % / 30 % / 20 %) | randomly switches off neurons during training so the network can't rely on a few of them |
| L2 weight decay | keeps weights small |
| Label smoothing | the target is 95 % on the true digit instead of 100 %, which avoids overconfident predictions |
| Cosine learning-rate schedule | big steps early, fine adjustments at the end |
| Data augmentation | each training image is copied twice with random rotation (±15°), shear, scale and stroke thickness, giving 180 000 samples |

Augmentation is what makes the model work on mouse drawings, which are rarely as clean as MNIST
handwriting. Training runs for 30 epochs with mini-batches of 128.

## The interface

| Panel | What it shows |
|---|---|
| Draw a digit | drawing pad (mouse, touch or pen), brush size, undo, clear |
| Network input / HOG / Zoning / Topology | what the network actually sees, and the descriptors computed live |
| Neural network, live | descriptor groups → 128 → 64 → 10 neurons. Brighter neurons are more active, blue and red curves are the strongest excitatory and inhibitory connections, and moving pulses show the signal flowing |
| Live prediction | predicted digit, confidence and a probability bar for each digit |
| Accuracy per digit | the model's accuracy on the MNIST test set next to your own session accuracy (press Enter if the prediction is correct, or the real digit 0–9 if it is wrong) |

Shortcuts: `F` fullscreen · `C` / `Esc` clear · `Ctrl+Z` undo · `Enter` correct · `0`–`9` correct the prediction

## Project structure

| File | Role |
|---|---|
| `features.py` | normalization and the six descriptor groups |
| `nn.py` | the NumPy neural network: inference, training, save/load |
| `train.py` | data augmentation, training, evaluation, confusion matrix |
| `app.py` | local web server and the `/api/predict` endpoint |
| `web/index.html` | the interface |
| `main.py` | the baseline study: Hu moments + PCA + KNN |
| `model/` | trained weights and metrics |

## Tech stack

Python · NumPy · OpenCV · scikit-image · Matplotlib · HTML/CSS/JavaScript (Canvas)
