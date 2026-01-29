import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Reproducibility
np.random.seed(42)

# Create random dataset
X = np.random.rand(1000, 5)        # 1000 samples, 5 features
y = (X.sum(axis=1) > 2.5).astype(int)  # Binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Complete!")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model saved to models/model.pkl")
