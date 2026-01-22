import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset (UCI Heart Disease)
url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
df = pd.read_csv(url)

# Use only 5 features
features = ["age", "sex", "trestbps", "chol", "fbs"]
X = df[features]
y = df["target"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "heart_disease_model.pkl")

print("Model trained and saved as heart_disease_model.pkl")
