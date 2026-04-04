import os
import sys
import torch
from PIL import Image

from model import build_model, get_device
from dataset import get_transforms


# We point to the Phase 2 checkpoint because it is the best model we trained.
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT     = "phase2_best.pth"

# We define the class names here to match the alphabetical order that
# ImageFolder used when loading the dataset during training.
# This means index 0 always corresponds to "fake" and index 1 to "real".
CLASS_NAMES = ["fake", "real"]


def load_model(device):
    # We build the model with freeze_backbone=False because during inference
    # we want all layers to participate in the forward pass.
    model = build_model(num_classes=2, freeze_backbone=False)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT)
    # We use map_location=device so the checkpoint loads correctly
    # regardless of whether it was saved on GPU or CPU.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    # We set the model to evaluation mode to disable dropout and ensure
    # batch normalization uses its running statistics.
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def preprocess_image(image_path):
    # We apply the same transforms used during validation and testing,
    # which means only resizing and normalization without any augmentation.
    # It is important to use exactly the same preprocessing as during training
    # so the model receives inputs in the format it expects.
    transform = get_transforms("test")
    image     = Image.open(image_path).convert("RGB")
    # We add a batch dimension with unsqueeze(0) because the model expects
    # inputs of shape [batch_size, 3, 224, 224] even for a single image.
    tensor = transform(image).unsqueeze(0)
    return tensor


def predict(image_path, model, device):
    tensor = preprocess_image(image_path)
    tensor = tensor.to(device)

    # We disable gradient computation during inference to save memory and speed things up.
    with torch.no_grad():
        outputs = model(tensor)
        # We apply softmax to convert the raw logits into probabilities.
        # squeeze() removes the batch dimension so we get a tensor of shape [2].
        probs = torch.softmax(outputs, dim=1).squeeze()

    # index 0 corresponds to "fake" and index 1 to "real".
    fake_prob = probs[0].item()
    real_prob = probs[1].item()

    # We select the class with the highest probability as our prediction.
    predicted_class = CLASS_NAMES[probs.argmax().item()]
    confidence      = probs.max().item()

    # We return all relevant values so the decision engine can use p_fake directly.
    return {
        "predicted_class" : predicted_class,
        "confidence"      : confidence,
        "p_fake"          : fake_prob,
        "p_real"          : real_prob,
    }


def print_result(image_path, result):
    print("\n" + "="*50)
    print(f"  Image     : {image_path}")
    print(f"  Prediction: {result['predicted_class'].upper()}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  p(fake)   : {result['p_fake']:.4f}")
    print(f"  p(real)   : {result['p_real']:.4f}")
    print("="*50)


if __name__ == "__main__":
    # We require the user to provide an image path as a command line argument.
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
        print("Example: python src/predict.py dataset/test/fake/img001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # We check that the file exists before trying to load it
    # to give the user a clear error message if the path is wrong.
    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        sys.exit(1)

    device = get_device()
    model  = load_model(device)

    result = predict(image_path, model, device)
    print_result(image_path, result)