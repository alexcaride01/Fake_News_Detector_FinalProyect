import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "text"))

import torch
from model import build_model, get_device
from dataset import get_transforms
from text.pipeline import run_text_pipeline
from decision_engine import decide, print_decision
from PIL import Image


# We point to the Phase 2 checkpoint because it achieved the best validation accuracy.
# CLASS_NAMES must match the alphabetical order used by ImageFolder during training.
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT     = "phase2_best.pth"
CLASS_NAMES    = ["fake", "real"]


def load_model(device):
    # We load the trained CNN model from the saved checkpoint.
    # We build the model with freeze_backbone=False because during inference
    # we want all layers to contribute to the prediction.
    # We use map_location=device so the checkpoint loads correctly
    # whether it was originally saved on GPU or CPU.
    model = build_model(num_classes=2, freeze_backbone=False)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    # We set eval mode to disable dropout and use running batch norm statistics.
    model.eval()
    return model


def run_visual_module(image_path, model, device):
    # We apply the same preprocessing used during validation and testing.
    # It is important to use exactly the same transforms as during training
    # so the model receives inputs in the format it was trained on.
    transform = get_transforms("test")
    image     = Image.open(image_path).convert("RGB")
    # We add a batch dimension with unsqueeze(0) because the model expects
    # a tensor of shape [batch_size, 3, 224, 224] even for a single image.
    tensor    = transform(image).unsqueeze(0).to(device)

    # We disable gradient computation since we only need the forward pass.
    with torch.no_grad():
        outputs = model(tensor)
        # We apply softmax to convert raw logits into probabilities.
        # squeeze() removes the batch dimension so we get a tensor of shape [2].
        probs   = torch.softmax(outputs, dim=1).squeeze()

    # index 0 = fake, index 1 = real (alphabetical order from ImageFolder)
    p_fake = probs[0].item()
    p_real = probs[1].item()

    return {"p_fake": p_fake, "p_real": p_real}


def print_full_result(image_path, visual, text, verdict, confidence, reasons):
    # We print a structured report showing the output of each module
    # and the final decision so the user can understand how the verdict was reached.
    print("\n" + "="*60)
    print("  FAKE NEWS DETECTOR - Full Analysis")
    print("="*60)
    print(f"  Image             : {image_path}")
    print(f"\n  [ Visual Module ]")
    print(f"  p(fake)           : {visual['p_fake']:.4f}")
    print(f"  p(real)           : {visual['p_real']:.4f}")
    print(f"\n  [ Text Module ]")
    print(f"  Text found        : {text['text_found']}")
    if text["text_found"]:
        print(f"  Extracted text    : {text['extracted_text'][:80]}...")
        print(f"  Wikipedia source  : {text['source_title']}")
        print(f"  Text verdict      : {text['verdict'].upper()}")
        print(f"  Text confidence   : {text['confidence']:.4f}")
    print(f"\n  [ Final Decision ]")
    print(f"  VERDICT           : {verdict.upper()}")
    print(f"  Confidence        : {confidence:.4f}")
    print(f"\n  Explanation:")
    for reason in reasons:
        print(f"    - {reason}")
    print("="*60)


def analyze(image_path):
    # We validate the image path before doing anything else
    # to give the user a clear error message if the file does not exist.
    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        sys.exit(1)

    device = get_device()

    # We load the model once at the start and reuse it for both modules.
    model = load_model(device)

    # We run the visual module first since it is faster and does not
    # require any external API calls.
    print(f"\nAnalyzing image: {image_path}")
    print("Running visual module...")
    visual = run_visual_module(image_path, model, device)

    # We run the text module which includes OCR, entity extraction,
    # Wikipedia retrieval and TF-IDF similarity computation.
    print("Running text module...")
    text = run_text_pipeline(image_path)

    # We pass both module outputs to the decision engine which applies
    # our rule set to produce the final explainable verdict.
    verdict, confidence, reasons = decide(
        p_fake          = visual["p_fake"],
        text_verdict    = text["verdict"],
        text_confidence = text["confidence"],
        text_found      = text["text_found"],
    )

    # We print the full structured report showing all intermediate results
    # and the final decision with its explanation.
    print_full_result(image_path, visual, text, verdict, confidence, reasons)

    # We return all results as a dictionary so the function can also be
    # called programmatically from other scripts such as the web demo server.
    return {
        "verdict"    : verdict,
        "confidence" : confidence,
        "visual"     : visual,
        "text"       : text,
        "reasons"    : reasons,
    }


if __name__ == "__main__":
    # We require exactly one command line argument: the path to the image to analyze.
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_image>")
        print("Example: python main.py dataset/test/fake/image.jpg")
        sys.exit(1)

    analyze(sys.argv[1])