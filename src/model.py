import torch
import torch.nn as nn
from torchvision import models


# We follow a two-phase training strategy to take full advantage of transfer learning.
# In Phase 1, we freeze the backbone and train only the new classifier head.
# This allows the head to learn the fake/real task without disturbing the
# pretrained visual features that the backbone already has.
# In Phase 2, we unfreeze the last few blocks of the backbone and fine-tune
# them together with the head using a much smaller learning rate.
# This lets the network adapt its visual representations to our specific domain.


def build_model(num_classes=2, freeze_backbone=True):
    # We load MobileNetV3-Small with weights pretrained on ImageNet.
    # We chose this architecture because it is lightweight, runs well on CPU,
    # and provides strong visual features even with small datasets like ours.
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model   = models.mobilenet_v3_small(weights=weights)

    # During Phase 1, we freeze all parameters in the feature extractor.
    # This means only the classifier head will be updated during backpropagation.
    # Freezing the backbone prevents us from losing the pretrained features
    # before the head has had a chance to learn the classification task.
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # We replace the original final linear layer, which outputs 1000 classes for ImageNet,
    # with a new one that outputs num_classes values (2 in our case: fake and real).
    # All other layers of the classifier remain unchanged.
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def unfreeze_last_blocks(model, num_blocks=3):
    # We unfreeze the last N convolutional blocks of the backbone for Phase 2.
    # We only unfreeze the last few blocks rather than the entire backbone
    # because the early layers learn very general features (edges, textures)
    # that are useful for any vision task and do not need to be retrained.
    # The later blocks learn more task-specific features that benefit from fine-tuning.
    total_blocks  = len(model.features)
    unfreeze_from = total_blocks - num_blocks

    for i, block in enumerate(model.features):
        if i >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True

    # We print the number of trainable parameters so we can verify
    # that the unfreezing worked as expected.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2: {num_blocks} blocks unfrozen -> {trainable:,} trainable parameters")


def get_device():
    # We select the best available device automatically.
    # We prioritize CUDA (NVIDIA GPU) for the fastest training,
    # then MPS (Apple Silicon GPU), and fall back to CPU if neither is available.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device selected: {device}")
    return device


def get_probabilities(outputs):
    # We apply softmax to the raw logits to convert them into probabilities.
    # The output is a tensor of shape [B, 2] where each row sums to 1.
    # Since ImageFolder sorts classes alphabetically, index 0 = fake, index 1 = real.
    # We use the p_fake value (index 0) as our main manipulation confidence score.
    return torch.softmax(outputs, dim=1)


if __name__ == "__main__":
    device = get_device()

    # We build the model in Phase 1 configuration to verify the architecture.
    model = build_model(num_classes=2, freeze_backbone=True)
    model.to(device)

    # We count trainable and total parameters to confirm the backbone is frozen.
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params  = sum(p.numel() for p in model.parameters())
    print(f"\nPhase 1:")
    print(f"  Trainable parameters : {trainable_p1:,}")
    print(f"  Total parameters     : {total_params:,}")

    # We run a forward pass with a dummy batch to verify the output shape.
    # We expect shape [4, 2] since we have 4 images and 2 classes.
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f"\nForward pass OK -> output shape: {out.shape}")

    # We unfreeze the last 3 blocks to simulate the Phase 2 configuration.
    unfreeze_last_blocks(model, num_blocks=3)

    print("\nmodel.py loaded successfully")