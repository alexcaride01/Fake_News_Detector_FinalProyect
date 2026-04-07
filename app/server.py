import os
import sys
import shutil
import uuid

# We add the src and src/text directories to the Python path so that
# all our custom modules can be imported from anywhere in the project.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "text"))

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model import build_model, get_device
from dataset import get_transforms
from text.pipeline import run_text_pipeline
from decision_engine import decide

# We define all directory paths relative to this file's location
# so the server works correctly regardless of where it is launched from.
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
CHECKPOINT     = "phase2_best.pth"
# We store uploaded images in a dedicated folder so we can serve them
# back to the frontend for display alongside the analysis results.
UPLOAD_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
STATIC_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
CLASS_NAMES = ["fake", "real"]

# We create the uploads directory at startup if it does not already exist.
os.makedirs(UPLOAD_DIR, exist_ok=True)

# We load the model once at server startup rather than on every request.
# This avoids reloading the weights for each image analysis, which would
# make the response time much slower.
device = get_device()
model  = build_model(num_classes=2, freeze_backbone=False)
model.load_state_dict(torch.load(
    os.path.join(CHECKPOINT_DIR, CHECKPOINT), map_location=device
))
model.to(device)
# We set eval mode to disable dropout and use running batch norm statistics.
model.eval()
print("Model loaded and ready.")

# We create the FastAPI application instance.
# FastAPI provides automatic request validation, serialization and API documentation.
app = FastAPI(title="Fake News Detector")

# We add CORS middleware to allow the frontend to make requests to our API
# from the browser without being blocked by the browser's same-origin policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# We serve the static HTML, CSS and JS files from the static folder.
# We also serve the uploaded images so the frontend can display them.
app.mount("/static",  StaticFiles(directory=STATIC_DIR),  name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR),  name="uploads")

def run_visual(image_path):
    # We apply the same preprocessing pipeline used during training
    # to ensure the model receives inputs in the expected format.
    transform = get_transforms("test")
    image     = Image.open(image_path).convert("RGB")
    tensor    = transform(image).unsqueeze(0).to(device)

    # We disable gradient computation since we only need the forward pass.
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).squeeze()

    # We round both probabilities to 4 decimal places for cleaner JSON output.
    # index 0 = fake, index 1 = real (alphabetical order from ImageFolder).
    return {
        "p_fake": round(probs[0].item(), 4),
        "p_real": round(probs[1].item(), 4),
    }

@app.get("/", response_class=HTMLResponse)
def index():
    # We serve the main HTML page directly from the static folder.
    # This allows the user to open the demo by navigating to http://localhost:8000.
    html_path = os.path.join(STATIC_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # We generate a unique filename for each uploaded image using UUID
    # to avoid overwriting previous uploads and to prevent naming conflicts.
    ext       = os.path.splitext(file.filename)[1]
    filename  = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    # We save the uploaded file to disk so we can pass the path to our modules.
    # All our analysis functions expect a file path rather than a file object.
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # We run the visual module to get the CNN fake probability.
    visual = run_visual(save_path)

    # We run the text module which performs OCR, entity extraction,
    # Wikipedia retrieval and TF-IDF similarity in a single call.
    text = run_text_pipeline(save_path)

    # We pass both module outputs to the decision engine to get
    # the final verdict, confidence score and explanation.
    # We also forward the LLM explanation so it can be included in the reasons.
    verdict, confidence, reasons = decide(
        p_fake          = visual["p_fake"],
        text_verdict    = text["verdict"],
        text_confidence = text["confidence"],
        text_found      = text["text_found"],
        llm_explanation = text.get("llm_explanation", ""),
    )

    # We return a structured JSON response with all the information
    # the frontend needs to render the full analysis report.
    # We limit the extracted text to 200 characters to keep the response compact.
    return {
        "image_url" : f"/uploads/{filename}",
        "visual"    : visual,
        "text"      : {
            "text_found"      : text["text_found"],
            "extracted_text"  : text["extracted_text"][:200] if text["text_found"] else "",
            "verdict"         : text["verdict"],
            "confidence"      : round(text["confidence"], 4),
            "source_title"    : text["source_title"],
            "source_url"      : text["source_url"],
            "similarity"      : round(text["similarity"], 4),
            "llm_used"        : text.get("llm_used", False),
            "llm_explanation" : text.get("llm_explanation", ""),
        },
        "decision"  : {
            "verdict"    : verdict,
            "confidence" : round(confidence, 4),
            "reasons"    : reasons,
        },
    }