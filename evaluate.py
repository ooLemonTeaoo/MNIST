import torch

# Metrics Plot
import matplotlib.pyplot as plt
# Confusion Matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Classification Report
from sklearn.metrics import classification_report


# Metrics Plot
def plot_metrics(train_acc, val_acc, train_loss):

    epochs = range(1, len(train_acc) + 1)

    # ---------------- Accuracy Plot ----------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_acc,
        marker="o",
        linewidth=2,
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_acc,
        marker="o",
        linewidth=2,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.title("Training vs Validation Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("results/accuracy_plot.png")

    plt.close()

    print("Saved results → results/accuracy_plot.png")

    # ---------------- Loss Plot ----------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("results/loss_plot.png")

    plt.close()

    print("Saved results → results/loss_plot.png")

# Confusion Matrix
def plot_confusion_matrix(model, test_loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("results/confusion_matrix.png")

    plt.close()

    print(f"Saved results → results/confusion_matrix.png")

# Digit Not Classified (9 Representative Mistakes)
def show_misclassified(model, test_loader, device):

    model.eval()

    misclassified = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            wrong = predicted != labels

            for img, pred, label in zip(
                images[wrong],
                predicted[wrong],
                labels[wrong]
            ):

                misclassified.append((img.cpu(), pred.cpu(), label.cpu()))

                if len(misclassified) >= 9:
                    break

            if len(misclassified) >= 9:
                break

    plt.figure(figsize=(8, 8))

    for i, (img, pred, label) in enumerate(misclassified):

        plt.subplot(3, 3, i + 1)

        plt.imshow(img.squeeze(), cmap="gray")

        plt.title(
            f"Wrong\nPred: {pred.item()} | True: {label.item()}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig("results/misclassified_digits.png")

    plt.close()

    print(f"Saved results → results/misclassified_digits.png")

# Classification Report
def save_classification_report(model, test_loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds)

    with open("results/classification_report.txt", "w") as f:
        f.write(report)

    print(f"Saved results → results/classification_report.txt")

    print(report)

def show_sample_predictions(model, test_loader, device):

    model.eval()

    images_batch, labels_batch = next(iter(test_loader))

    images = images_batch.to(device)
    labels = labels_batch.to(device)

    with torch.no_grad():

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(10, 10))

    for i in range(9):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].cpu().squeeze(), cmap="gray")

        plt.title(
            f"Pred: {predicted[i].item()} | True: {labels[i].item()}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig("results/sample_predictions.png")

    plt.close()

    print("Saved results/sample_predictions.png")