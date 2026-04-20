# Fake News Detector

A multimodal system that detects fake news in images by combining a CNN classifier with a text verification pipeline.

## How it works

When an image is submitted, the system runs two analyses in parallel and combines them into a single verdict.

The **visual module** passes the image through a MobileNetV3-Small CNN fine-tuned on a dataset of ~2,050 fake and real news images. The model was trained in two phases: first the backbone was frozen and only the classifier head was trained, then the last three backbone blocks were unfrozen and fine-tuned at a lower learning rate. The output is a probability that the image is fake.

The **text module** tries to extract any visible text using Tesseract OCR. Since social media images vary a lot in font, colour and quality, the OCR runs on over 40 differently preprocessed versions of the image and picks the one with the most recognised words. If text is found, the module identifies named entities and keywords using NLTK, searches Wikipedia for related information, and sends the claim together with the Wikipedia excerpt to Mistral (via Ollama) to decide whether the evidence supports or refutes it. If Ollama is not available, TF-IDF cosine similarity is used as a fallback.

The **decision engine** combines the CNN probability and the text verdict using a set of rules and returns one of four verdicts — **FAKE**, **REAL**, **DOUBTFUL** or **UNKNOWN** — along with a human-readable explanation of why.

Results are available through a command-line script (`main.py`) or a FastAPI web interface (`app/server.py`).

## Setup

```bash
pip install -r requirements.txt
```

Tesseract also needs to be installed as a system binary (not via pip). On Windows, download it from https://github.com/UB-Mannheim/tesseract/wiki. On Linux: `sudo apt install tesseract-ocr`. Ollama with Mistral is optional — the system falls back to TF-IDF if it is not available.

## Results

Evaluated on a test set of 81 images: **92.59% accuracy**, with precision, recall and F1-score of 0.93 for both classes.