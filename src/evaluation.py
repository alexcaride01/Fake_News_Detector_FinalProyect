import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
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
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {path}")

    return cm


def plot_roc_curve(labels, probs):
    # We plot the ROC curve which shows the trade-off between true positive rate
    # and false positive rate at different classification thresholds.
    # We use class 0 (fake) as the positive class.
    fpr, tpr, _ = roc_curve(labels, probs[:, 0], pos_label=0)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Test Set")
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC curve saved -> {path}")


def plot_precision_recall_curve(labels, probs):
    # We plot the precision-recall curve which is particularly informative
    # when the dataset has class imbalance. A good model keeps high precision
    # even at high recall values.
    precision_curve, recall_curve, _ = precision_recall_curve(
        (labels == 0).astype(int), probs[:, 0]
    )
    ap = average_precision_score((labels == 0).astype(int), probs[:, 0])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_curve, precision_curve, color="darkorange", lw=2,
            label=f"PR curve (AP = {ap:.3f})")
    ax.fill_between(recall_curve, precision_curve, alpha=0.1, color="darkorange")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (fake class)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "precision_recall_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Precision-recall curve saved -> {path}")


def plot_probability_distribution(labels, probs):
    # We plot the distribution of predicted fake probabilities separately for
    # true fake and true real images. A well-trained model should show two clearly
    # separated peaks near 0 and 1 with little overlap in the middle.
    fake_probs_on_fakes = probs[labels == 0, 0]
    fake_probs_on_reals = probs[labels == 1, 0]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(fake_probs_on_fakes, bins=20, alpha=0.6, color="red",
            label="True fake images", density=True)
    ax.hist(fake_probs_on_reals, bins=20, alpha=0.6, color="green",
            label="True real images", density=True)
    ax.axvline(x=0.5, color="black", linestyle="--", lw=1.5, label="Decision boundary (0.5)")
    ax.set_xlabel("p(fake)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Probability Distribution - Test Set")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "probability_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Probability distribution saved -> {path}")


def plot_per_class_metrics(preds, labels, class_names):
    # We plot a bar chart comparing precision, recall and F1 for each class side by side.
    # This makes it easy to spot if the model is performing differently on fake vs real images.
    prec_per_class = precision_score(labels, preds, average=None)
    rec_per_class  = recall_score(labels, preds, average=None)
    f1_per_class   = f1_score(labels, preds, average=None)

    x     = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_p = ax.bar(x - width, prec_per_class, width, label="Precision", color="steelblue")
    bars_r = ax.bar(x,         rec_per_class,  width, label="Recall",    color="darkorange")
    bars_f = ax.bar(x + width, f1_per_class,   width, label="F1-score",  color="green")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Score")
    ax.set_title("Per-class Metrics - Test Set")
    ax.legend()

    # We annotate each bar with its value for easy reading.
    for bar, val in zip(list(bars_p) + list(bars_r) + list(bars_f),
                        list(prec_per_class) + list(rec_per_class) + list(f1_per_class)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "per_class_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Per-class metrics saved -> {path}")


def save_metrics(metrics, class_names, preds, labels):
    # We save the metrics to a text file so we can include them in our report
    # without having to run the evaluation script again.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "metrics.txt")

    with open(path, "w") as f:
        f.write("Evaluation on Test Set\n")
        f.write("=" * 50 + "\n")
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

    # We run inference on the full test set and collect all predictions and probabilities.
    preds, labels, probs = get_predictions(model, dataloaders["test"], device)

    # We compute and print the standard metrics as before.
    metrics = compute_metrics(preds, labels, class_names)

    # We generate all plots individually so each one can be included separately in the paper.
    plot_confusion_matrix(preds, labels, class_names)
    plot_roc_curve(labels, probs)
    plot_precision_recall_curve(labels, probs)
    plot_probability_distribution(labels, probs)
    plot_per_class_metrics(preds, labels, class_names)

    # We save the metrics text file as before.
    save_metrics(metrics, class_names, preds, labels)