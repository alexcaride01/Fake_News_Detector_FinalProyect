import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import build_model, unfreeze_last_blocks, get_device
from dataset import get_dataloaders


# We define all configuration values at the top so they are easy to find and modify.
# DATA_DIR points to the folder containing our train, valid and test splits.
DATA_DIR        = "dataset"
# We save our model checkpoints in a dedicated folder to keep things organized.
CHECKPOINT_DIR  = "checkpoints"
# We save our training plots and metrics in the results folder.
RESULTS_DIR     = "results"

# Phase 1 configuration: we train only the classifier head with a relatively high
# learning rate since the head starts with random weights and needs to learn quickly.
PHASE1_EPOCHS   = 10
PHASE1_LR       = 1e-3

# Phase 2 configuration: we fine-tune the last backbone blocks with a much smaller
# learning rate to avoid destroying the pretrained features we want to preserve.
PHASE2_EPOCHS   = 10
PHASE2_LR       = 1e-4

BATCH_SIZE               = 32
# We stop training early if the validation loss does not improve for this many epochs.
# This prevents overfitting and saves computation time on CPU.
EARLY_STOPPING_PATIENCE  = 5


class EarlyStopping:
    # We implement early stopping as a class so we can easily reset its state
    # between Phase 1 and Phase 2 without creating a new instance each time.
    def __init__(self, patience=5):
        self.patience    = patience
        self.counter     = 0
        self.best_loss   = float("inf")
        self.should_stop = False

    def step(self, val_loss):
        # We check whether the validation loss has improved compared to the best
        # value we have seen so far. If it has, we reset the counter.
        # If it has not improved for patience epochs in a row, we set the stop flag.
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True


def run_epoch(model, loader, criterion, optimizer, device, is_train):
    # We set the model to training or evaluation mode depending on the split.
    # In eval mode, dropout and batch normalization behave differently,
    # which is important for getting correct validation metrics.
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss    = 0.0
    running_correct = 0

    for images, labels in loader:
        # We move both images and labels to the selected device (CPU or GPU).
        images = images.to(device)
        labels = labels.to(device)

        # We zero the gradients before each batch to avoid accumulation
        # from the previous iteration.
        optimizer.zero_grad()

        # We use torch.set_grad_enabled to avoid computing gradients during
        # validation, which saves memory and speeds up the evaluation.
        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss    = criterion(outputs, labels)
            # We take the class with the highest logit as our prediction.
            preds   = outputs.argmax(dim=1)

            if is_train:
                # We compute gradients and update the weights only during training.
                loss.backward()
                optimizer.step()

        # We accumulate the loss weighted by the batch size so we can compute
        # the correct average loss at the end of the epoch.
        running_loss    += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()

    # We divide by the total number of samples to get the average epoch loss and accuracy.
    epoch_loss = running_loss    / len(loader.dataset)
    epoch_acc  = running_correct / len(loader.dataset)

    return epoch_loss, epoch_acc


def train(model, dataloaders, dataset_sizes, num_epochs, lr, device, phase_name):
    # We use CrossEntropyLoss because this is a multi-class classification problem.
    # It combines log-softmax and negative log-likelihood loss in one step.
    criterion = nn.CrossEntropyLoss()

    # We use Adam optimizer because it adapts the learning rate for each parameter
    # and generally converges faster than plain SGD for fine-tuning tasks.
    # We filter parameters to only optimize those that have requires_grad=True,
    # which means only the unfrozen layers will be updated.
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    # We keep a copy of the best model weights found during training.
    # At the end we restore these weights so the returned model is the best one,
    # not necessarily the one from the last epoch.
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_acc       = 0.0

    # We track loss and accuracy for both train and validation splits
    # so we can plot the training curves at the end.
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*50}")
    print(f"  {phase_name}")
    print(f"{'='*50}")

    for epoch in range(num_epochs):
        start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # We run both the training and validation pass within the same epoch loop
        # so we can monitor overfitting in real time.
        for split in ["train", "valid"]:
            is_train = (split == "train")
            loss, acc = run_epoch(
                model, dataloaders[split], criterion, optimizer, device, is_train
            )

            tag = "train" if is_train else "val"
            history[f"{tag}_loss"].append(loss)
            history[f"{tag}_acc"].append(acc)

            print(f"  {split:6s} -> loss: {loss:.4f}  acc: {acc:.4f}")

            # We save the model weights whenever we see a new best validation accuracy.
            # This ensures we always keep the best model seen during training.
            if not is_train and acc > best_val_acc:
                best_val_acc       = acc
                best_model_weights = copy.deepcopy(model.state_dict())

        # We check the early stopping condition after each full epoch.
        val_loss = history["val_loss"][-1]
        early_stopper.step(val_loss)

        elapsed = time.time() - start
        print(f"  Epoch time: {elapsed:.1f}s")

        if early_stopper.should_stop:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # We restore the best weights before returning the model.
    # This is important because the last epoch is not necessarily the best one.
    model.load_state_dict(best_model_weights)

    return model, history


def save_checkpoint(model, filename):
    # We create the checkpoints directory if it does not exist yet.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    # We save only the model's state dict (weights), not the full model object.
    # This is the recommended approach in PyTorch as it is more portable.
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved -> {path}")


def plot_history(history_p1, history_p2):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # We concatenate the histories from both phases to get a continuous plot
    # that shows the full training process from start to finish.
    train_loss = history_p1["train_loss"] + history_p2["train_loss"]
    val_loss   = history_p1["val_loss"]   + history_p2["val_loss"]
    train_acc  = history_p1["train_acc"]  + history_p2["train_acc"]
    val_acc    = history_p1["val_acc"]    + history_p2["val_acc"]

    epochs     = range(1, len(train_loss) + 1)
    # We mark the boundary between Phase 1 and Phase 2 with a vertical dashed line
    # so it is easy to see how the fine-tuning step affected the metrics.
    phase1_end = len(history_p1["train_loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # We plot training and validation loss side by side with the accuracy plot
    # so we can quickly spot overfitting by comparing the two curves.
    ax1.plot(epochs, train_loss, label="Train loss")
    ax1.plot(epochs, val_loss,   label="Val loss")
    ax1.axvline(x=phase1_end, color="gray", linestyle="--", label="Phase 1 -> 2")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, train_acc, label="Train acc")
    ax2.plot(epochs, val_acc,   label="Val acc")
    ax2.axvline(x=phase1_end, color="gray", linestyle="--", label="Phase 1 -> 2")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"Training curves saved -> {path}")


if __name__ == "__main__":
    device = get_device()

    dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # We build the model in Phase 1 configuration with the backbone frozen.
    # Only the new classifier head will be trained in this first phase.
    model = build_model(num_classes=2, freeze_backbone=True)
    model.to(device)

    model, history_p1 = train(
        model, dataloaders, dataset_sizes,
        num_epochs=PHASE1_EPOCHS, lr=PHASE1_LR,
        device=device, phase_name="Phase 1 - Classifier head only"
    )
    # We save the best model from Phase 1 before moving on to Phase 2.
    save_checkpoint(model, "phase1_best.pth")

    # We unfreeze the last backbone blocks and start Phase 2 fine-tuning.
    # We use a much smaller learning rate to avoid destroying the pretrained features.
    unfreeze_last_blocks(model, num_blocks=3)

    model, history_p2 = train(
        model, dataloaders, dataset_sizes,
        num_epochs=PHASE2_EPOCHS, lr=PHASE2_LR,
        device=device, phase_name="Phase 2 - Fine-tuning"
    )
    # We save the best model from Phase 2, which is the final model we will use.
    save_checkpoint(model, "phase2_best.pth")

    # We generate and save the training curves combining both phases.
    plot_history(history_p1, history_p2)

    print("\nTraining complete.")