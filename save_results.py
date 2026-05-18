import os
import datetime

def save_results(
    filename,
    epochs,
    train_acc,
    val_acc,
    test_acc,
    model_name,
    optimizer,
    loss
):

    os.makedirs("results", exist_ok=True)

    filepath = os.path.join("results", filename)

    now = datetime.datetime.now()
    timestamp = now.strftime("%d.%m.%Y %H:%M:%S")

    content = f"""\
{timestamp}

Epochs: {epochs}
Train Accuracy: {train_acc:.2f}%
Validation Accuracy: {val_acc:.2f}%
Test Accuracy: {test_acc:.2f}%
Model: {model_name}
Optimizer: {optimizer}
Loss: {loss}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved results → results/{filename}")