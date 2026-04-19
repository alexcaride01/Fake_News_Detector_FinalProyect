# Fake News Detector

A multimodal system for detecting fake news in images. It combines a CNN visual classifier (MobileNetV3-Small) with a text verification pipeline based on OCR, Wikipedia retrieval and an LLM (Mistral), and a rule-based decision engine that fuses both signals into a final explainable verdict.

---

## How it works

When an image is submitted, the system runs two analyses in parallel.

The **visual module** passes the image through a MobileNetV3-Small convolutional neural network that was fine-tuned on a dataset of fake and real news images. The network outputs a probability that the image is fake, based purely on visual features. The model was trained in two phases: first only the classifier head was trained while the backbone remained frozen, then the last three backbone blocks were unfrozen and fine-tuned at a smaller learning rate. This two-phase strategy stabilises training and avoids destroying the ImageNet representations learned during pre-training.

The **text module** tries to extract any text visible in the image using Tesseract OCR. Because social media images vary widely in font, colour and quality, the OCR runs on over 40 differently preprocessed versions of the image and keeps the result with the most recognised words. If enough text is found, the module identifies named entities and keywords using NLTK, builds a search query, retrieves the most relevant Wikipedia article, and sends the extracted claim together with the Wikipedia excerpt to Mistral (served locally via Ollama). Mistral reasons over the evidence and returns one of three signals: support, refute, or unknown. If Ollama is not available, the module falls back to TF-IDF cosine similarity between the claim and the Wikipedia text.

The **decision engine** takes the CNN fake probability and the text verdict and applies a set of rules with three thresholds to produce a final verdict: **FAKE**, **REAL**, **DOUBTFUL**, or **UNKNOWN**. Every verdict is accompanied by a list of human-readable reasons so the user can understand why the system reached that conclusion.

The system is accessible through a **FastAPI web interface** where images can be uploaded and results are returned in real time as a JSON response.

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd "Proyecto Final"
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract

Tesseract must be installed as a system binary — it is not available via pip.

- **Windows**: download the installer from https://github.com/UB-Mannheim/tesseract/wiki and install to `C:\Program Files\Tesseract-OCR\`. The path is already set in `src/text/ocr.py`.
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

### 5. Install Ollama and pull Mistral (optional but recommended)

If Ollama is not available the system automatically falls back to TF-IDF similarity, so this step is optional.

```bash
# Install Ollama from https://ollama.com
ollama pull mistral
```

---

## Usage

### Analyse a single image (CLI)

```bash
python main.py path/to/image.jpg
```

The script prints a full structured report showing the CNN probability, the text module output and the final verdict with its explanation.

### Start the web interface

```bash
cd app
uvicorn server:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser. You can upload any image and get the full analysis in real time.

---

## Training

To train the model from scratch you need the dataset organised as:

```
dataset/
  train/fake/  train/real/
  valid/fake/  valid/real/
  test/fake/   test/real/
```

Then run:

```bash
python src/train.py
```

This runs both training phases automatically. Checkpoints are saved to `checkpoints/` and training curves are saved to `results/`.

### Training configuration

All hyperparameters are defined at the top of `src/train.py`:

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| Epochs (max) | 10 | 10 |
| Learning rate | 1e-3 | 1e-4 |
| Backbone | Frozen | Last 3 blocks unfrozen |
| Early stopping patience | 5 | 5 |
| Batch size | 32 | 32 |

---

## Evaluation

```bash
python src/evaluation.py
```

This evaluates the Phase 2 checkpoint on the test set and saves the following plots to `results/`:

- `training_curves.png` — loss and accuracy for both phases
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `probability_distribution.png`
- `per_class_metrics.png`
- `metrics.txt` — accuracy, precision, recall, F1-score

### Results on the test set (81 images)

| Metric | Fake | Real | Weighted avg. |
|---|---|---|---|
| Precision | 0.93 | 0.93 | 0.93 |
| Recall | 0.93 | 0.93 | 0.93 |
| F1-score | 0.93 | 0.93 | 0.93 |
| **Accuracy** | | | **0.9259** |

---

## Dataset

The model was trained on the [Roboflow Fake News Image Classifier](https://universe.roboflow.com) dataset (CC BY 4.0), which contains approximately 2,050 images split across train, validation and test sets.

The dataset is **not included** in this repository. Download it from Roboflow and place it in the `dataset/` folder following the structure shown above.

---

## Requirements

- Python 3.9+
- Tesseract OCR (system binary, see Installation)
- Ollama + Mistral (optional, for LLM-based text verification)