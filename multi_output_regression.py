import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# =========================================================
# 1. Load California Housing Dataset
# =========================================================
california_data = fetch_california_housing(as_frame=True)
df = california_data.frame

# Define primary target: Median House Value
df['MedianHouseValue'] = df['MedHouseVal'] * 100_000  # scale to realistic dollars

# Define secondary target: value per median income
df['ValuePerIncome'] = df['MedianHouseValue'] / df['MedInc']

# Features and targets
X = df.drop(columns=['MedHouseVal', 'MedianHouseValue', 'ValuePerIncome'])
Y = df[['MedianHouseValue', 'ValuePerIncome']]

# =========================================================
# 2. Split Data into Training and Test Sets
# =========================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X.values, Y.values, test_size=0.2, random_state=42
)

# =========================================================
# 3. Scale Features
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 4. Generate Polynomial & Interaction Features
# =========================================================
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_transformer.fit_transform(X_train_scaled)
X_test_poly = poly_transformer.transform(X_test_scaled)

# =========================================================
# 5. Initialize Gradient Descent Parameters
# =========================================================
num_samples = X_train_poly.shape[0]
num_features = X_train_poly.shape[1]
num_targets = Y_train.shape[1]

weights = np.zeros((num_features, num_targets))
bias = np.zeros(num_targets)

learning_rate = 0.005
num_epochs = 3000

cost_history = []

# =========================================================
# 6. Gradient Descent Training
# =========================================================
for epoch in range(num_epochs):
    # Predictions
    Y_pred = X_train_poly @ weights + bias

    # Compute cost (MSE for multi-output)
    cost = (1 / (2 * num_samples)) * np.sum((Y_pred - Y_train) ** 2)
    cost_history.append(cost)

    # Compute gradients
    grad_w = (1 / num_samples) * X_train_poly.T @ (Y_pred - Y_train)
    grad_b = (1 / num_samples) * np.sum(Y_pred - Y_train, axis=0)

    # Update parameters
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b

    # Print progress every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Cost: {cost:.4f}")

print("\nTraining Complete!")
print("Final weights and bias learned successfully.\n")

# =========================================================
# 7. Evaluate Model
# =========================================================
Y_pred_train = X_train_poly @ weights + bias
Y_pred_test = X_test_poly @ weights + bias

# RMSE for each target
rmse_train = np.sqrt(np.mean((Y_pred_train - Y_train) ** 2, axis=0))
rmse_test = np.sqrt(np.mean((Y_pred_test - Y_test) ** 2, axis=0))

print(f"RMSE on Training Set (MedianHouseValue, ValuePerIncome): {rmse_train}")
print(f"RMSE on Test Set     (MedianHouseValue, ValuePerIncome): {rmse_test}\n")

# =========================================================
# 8. Plot Cost Reduction Over Epochs
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(cost_history, color='teal')
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.show()

# =========================================================
# 9. Visualize Predictions vs Actual
# =========================================================
target_names = ['MedianHouseValue', 'ValuePerIncome']
for i, name in enumerate(target_names):
    plt.figure(figsize=(6, 4))
    plt.scatter(Y_test[:, i], Y_pred_test[:, i], alpha=0.5, color='orange')
    plt.plot([Y_test[:, i].min(), Y_test[:, i].max()],
             [Y_test[:, i].min(), Y_test[:, i].max()], 'r--', lw=2)
    plt.xlabel("Actual " + name)
    plt.ylabel("Predicted " + name)
    plt.title(f"{name} | Actual vs Predicted")
    plt.grid(True)
    plt.show()

# =========================================================
# 10. Feature Importance
# =========================================================
importance = np.abs(weights).sum(axis=1)
feature_names = poly_transformer.get_feature_names_out(california_data.feature_names)

# Sort features by importance
feature_importance = sorted(
    zip(feature_names, importance),
    key=lambda x: x[1],
    reverse=True
)

print("Top 15 Features by Absolute Weight Importance:\n")
for feat, imp in feature_importance[:15]:
    print(f"{feat}: {imp:.4f}")