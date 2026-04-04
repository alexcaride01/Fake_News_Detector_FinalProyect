import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from model import build_model, get_device
from dataset import get_dataloaders


# We always evaluate on the test split, which was never seen during training
# or used to make any decisions such as early stopping or checkpoint saving.
# This guarantees that our reported metrics are unbiased estimates of
# how the model will perform on real unseen data.
DATA_DIR       = "dataset"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
# We load the best model from Phase 2 since it achieved the highest validation accuracy.
CHECKPOINT     = "phase2_best.pth"
BATCH_SIZE     = 32


def get_predictions(model, loader, device):
    # We set the model to evaluation mode to disable dropout and ensure
    # batch normalization uses its running statistics instead of batch statistics.
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    # We disable gradient computation during inference because we do not need
    # to compute gradients and disabling them saves memory and speeds up the process.
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # We apply softmax to get proper probabilities that sum to 1 for each image.
            probs   = torch.softmax(outputs, dim=1)
            # We take the class with the highest probability as our final prediction.
            preds   = outputs.argmax(dim=1)

            # We move the results back to CPU and convert to numpy arrays
            # so we can use scikit-learn metrics functions on them.
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # We concatenate all batches into single arrays covering the full test set.
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    return all_preds, all_labels, all_probs


def compute_metrics(preds, labels, class_names):
    # We compute four standard classification metrics to evaluate our model thoroughly.
    # Accuracy tells us the overall percentage of correct predictions.
    acc  = accuracy_score(labels, preds)
    # Precision tells us how many of the images we predicted as fake were actually fake.
    prec = precision_score(labels, preds, average="weighted")
    # Recall tells us how many of the actually fake images we correctly identified.
    rec  = recall_score(labels, preds, average="weighted")
    # F1-score is the harmonic mean of precision and recall, useful for imbalanced datasets.
    f1   = f1_score(labels, preds, average="weighted")

    print("\n" + "="*50)
    print("  Evaluation on Test Set")
    print("="*50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    # We also print a per-class report to see if the model performs
    # differently on fake vs real images.
    print("\nPer-class report:")
    print(classification_report(labels, preds, target_names=class_names))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(preds, labels, class_names):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # We compute the confusion matrix which shows us exactly how many images
    # were correctly classified and how many were confused with the other class.
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - Test Set")

    # We write the count of images inside each cell of the matrix.
    # We choose white or black text depending on the background color
    # so the numbers are always readable.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix saved -> {path}")


def save_metrics(metrics, class_names, preds, labels):
    # We save the metrics to a text file so we can include them in our report
    # without having to run the evaluation script again.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "metrics.txt")

    with open(path, "w") as f:
        f.write("Evaluation on Test Set\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy  : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision : {metrics['precision']:.4f}\n")
        f.write(f"Recall    : {metrics['recall']:.4f}\n")
        f.write(f"F1-score  : {metrics['f1']:.4f}\n\n")
        f.write("Per-class report:\n")
        f.write(classification_report(labels, preds, target_names=class_names))

    print(f"Metrics saved -> {path}")


if __name__ == "__main__":
    device = get_device()

    dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print(f"Classes: {class_names}")
    print(f"Test set size: {dataset_sizes['test']} images")

    # We load the Phase 2 checkpoint and build the model with freeze_backbone=False
    # because during inference we do not need to freeze anything and we want
    # all layers to contribute to the final prediction.
    model = build_model(num_classes=2, freeze_backbone=False)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {checkpoint_path}")

    # We run inference on the full test set and collect all predictions.
    preds, labels, probs = get_predictions(model, dataloaders["test"], device)

    # We compute all metrics and print them to the console.
    metrics = compute_metrics(preds, labels, class_names)

    # We save both the confusion matrix image and the metrics text file
    # to the results folder so they are ready to include in our report.
    plot_confusion_matrix(preds, labels, class_names)
    save_metrics(metrics, class_names, preds, labels)