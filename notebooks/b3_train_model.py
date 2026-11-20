import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Fixed random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Folders setup
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)


# 1. Define PyTorch Neural Network Architecture
class NavNet(nn.Module):
    def __init__(self):
        super(NavNet, self).__init__()
        # Input: 3 sensor readings (left distance, front distance, right distance)
        # Output: 1 steering command (angular velocity w)
        self.fc1 = nn.Linear(3, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_synthetic_dataset(num_samples=1000):
    """Generate synthetic reactive navigation telemetry data."""
    # Inputs: [left_dist, front_dist, right_dist] ranging from 0.05m to 2.0m
    X = np.random.uniform(0.05, 2.0, size=(num_samples, 3))
    
    # Reactive steering logic (expert policy):
    # If front obstacle is close, turn hard right or left. If clear, go straight (steering ~ 0).
    y = np.zeros((num_samples, 1))
    for i in range(num_samples):
        left, front, right = X[i]
        if front < 0.4:
            # Obstacle ahead: turn away from the tighter side
            y[i, 0] = -1.5 if left > right else 1.5
        elif left < 0.2:
            y[i, 0] = 1.0  # steer right
        elif right < 0.2:
            y[i, 0] = -1.0 # steer left
        else:
            y[i, 0] = 0.0  # path clear, go straight

        # Add minor noise for realism
        y[i, 0] += np.random.normal(0, 0.05)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def main():
    print("Executing Step 20: Training Neural Network for Navigation...")

    # 2. Generate Dataset and Split into Train/Test
    X, y = generate_synthetic_dataset(1200)
    train_size = 1000
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # 3. Model, Loss, and Optimizer Setup
    model = NavNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. Training Loop
    epochs = 150
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 30 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Save model weights
    torch.save(model.state_dict(), os.path.join(data_folder, "nav_model_weights.pth"))
    print("Model weights saved to data/nav_model_weights.pth")

    # 5. Plot Training Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(epochs), loss_history, color="tab:red", linewidth=2)
    plt.title("B3: Neural Network Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE) Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_folder, "B3_training_loss.png"), dpi=150)
    plt.show()
    print("Training loss plot saved to figures/B3_training_loss.png")


if __name__ == "__main__":
    main()