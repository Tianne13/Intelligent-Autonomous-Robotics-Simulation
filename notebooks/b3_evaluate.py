import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

from b3_train_model import NavNet, generate_synthetic_dataset

# Folders setup
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)


def discretize_action(steering_val):
    """Map continuous steering output to discrete action classes: 0: Left, 1: Straight, 2: Right"""
    if steering_val < -0.5:
        return 0  # Turn Left
    elif steering_val > 0.5:
        return 2  # Turn Right
    else:
        return 1  # Go Straight


def main():
    print("Executing Step 21: Evaluating Neural Network Navigation Model...")

    # 1. Load test dataset and trained model weights
    X, y_cont = generate_synthetic_dataset(500)
    
    model = NavNet()
    weights_path = os.path.join(data_folder, "nav_model_weights.pth")
    if not os.path.exists(weights_path):
        print("Error: Trained model weights not found. Run b3_train_model.py first.")
        return

    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # 2. Run Inference
    with torch.no_grad():
        preds_cont = model(X).numpy()

    # Convert continuous targets and predictions into discrete classes for confusion matrix
    y_true = np.array([discretize_action(val) for val in y_cont.numpy().flatten()])
    y_pred = np.array([discretize_action(val) for val in preds_cont.flatten()])

    # 3. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    mse = np.mean((y_cont.numpy() - preds_cont) ** 2)

    print(f"\n--- Model Evaluation Results ---")
    print(f"Test Accuracy (Discrete Actions): {acc * 100:.2f}%")
    print(f"Mean Squared Error (Continuous Steering): {mse:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Turn Left", "Go Straight", "Turn Right"]))

    # 4. Generate Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["Turn Left", "Go Straight", "Turn Right"]

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title("B3: Navigation Model Confusion Matrix")
    plt.xlabel("Predicted Action")
    plt.ylabel("True Expert Action")
    plt.tight_layout()

    plt.savefig(os.path.join(figures_folder, "B3_confusion_matrix.png"), dpi=150)
    plt.show()
    print("Confusion matrix plot saved to figures/B3_confusion_matrix.png")


if __name__ == "__main__":
    main()