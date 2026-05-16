import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import get_mnist_loaders, flatten_for_mlp
from models.mlp import BaselineMLP


def train(model, train_loader, val_loader, epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Logit Roundoff Error
    criterion = nn.CrossEntropyLoss()
    # Faster Learning Rate Approach
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):

        # ---------------- TRAIN ----------------
        model.train()

        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = flatten_for_mlp(images).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

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

                images = flatten_for_mlp(images).to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

    return model


def test(model, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = flatten_for_mlp(images).to(device)
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

    model = BaselineMLP()

    model = train(model, train_loader, val_loader, epochs=5)

    test_acc = test(model, test_loader)

    torch.save(model.state_dict(), "baseline_mlp.pth")

    print("\nSaved: baseline_mlp.pth")