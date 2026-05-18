import torch
import torch.nn as nn
import torch.optim as optim
import os

from evaluate import (
    plot_metrics,
    plot_confusion_matrix,
    save_classification_report,
    show_misclassified
)

from save_results import save_results

# MLP model
# from preprocess import get_mnist_loaders, flatten_for_mlp
# CNN model
from preprocess import get_mnist_loaders

# MLP model
# from models.baseline_mlp import BaselineMLP
# CNN model
from models.cnn import CNN


def train(model, train_loader, val_loader, epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Logit Roundoff Error
    criterion = nn.CrossEntropyLoss()
    # Faster Learning Rate Approach
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Metric Tracking
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []

    for epoch in range(epochs):

        # ---------------- TRAIN ----------------
        model.train()

        train_correct = 0
        train_total = 0

        # Initiate Loss Record
        running_loss = 0.0

        for images, labels in train_loader:
            # MLP Model
            # images = flatten_for_mlp(images).to(device)
            # CNN Model
            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Record Loss
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # ---------------- VALIDATION ----------------
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                # MLP Model
                # images = flatten_for_mlp(images).to(device)
                # CNN Model
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        # Calculate Epoch Loaa
        epoch_loss = running_loss / len(train_loader)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(epoch_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

    return model, train_acc_history, val_acc_history, train_loss_history


def test(model, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            # MLP Model
            # images = flatten_for_mlp(images).to(device)
            # CNN Model
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_mnist_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MLP model
    # model = BaselineMLP().to(device)
    # CNN model
    model = CNN().to(device)

    model, train_acc, val_acc, train_loss = train(
        model,
        train_loader,
        val_loader,
        epochs=5
    )

    test_acc = test(model, test_loader)

    # Output Current Results
    # MLP model
    # save_results(
    #     filename="baseline_mlp_results.txt",
    #     epochs=5,
    #     train_acc=train_acc[-1],
    #     val_acc=val_acc[-1],
    #     test_acc=test_acc,
    #     model_name="Baseline MLP (784 → 256 → 128 → 10)",
    #     optimizer="Adam (lr=0.001)",
    #     loss="CrossEntropyLoss"
    # )
    # CNN model
    save_results(
        filename="cnn_results.txt",
        epochs=5,
        train_acc=train_acc[-1],
        val_acc=val_acc[-1],
        test_acc=test_acc,
        model_name="CNN (Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Flatten → 128 → 10)",
        optimizer="Adam (lr=0.001)",
        loss="CrossEntropyLoss"
    )

    # Run Evaluation Functions
    plot_metrics(train_acc, val_acc, train_loss)
    plot_confusion_matrix(model, test_loader, device)
    save_classification_report(model, test_loader, device)
    show_misclassified(model, test_loader, device)

    os.makedirs("results", exist_ok=True)
    # MLP model
    # torch.save(model.state_dict(), "results/baseline_mlp.pth")
    # print("\nSaved: results\baseline_mlp.pth")
    # CNN model
    torch.save(model.state_dict(), "results/cnn_model.pth")
    print("\nSaved: results\cnn_model.pth")