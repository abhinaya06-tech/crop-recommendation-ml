import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("data/crop_data.csv")

# Features & target
X = df.drop("label", axis=1)
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
rf = RandomForestClassifier(n_estimators=100)
dt = DecisionTreeClassifier()

# Train
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Predict
rf_pred = rf.predict(X_test)
dt_pred = dt.predict(X_test)

# Accuracy
rf_acc = accuracy_score(y_test, rf_pred)
dt_acc = accuracy_score(y_test, dt_pred)

print("Random Forest Accuracy:", rf_acc)
print("Decision Tree Accuracy:", dt_acc)

# Save best model
joblib.dump(rf, "model/model.pkl")

# Feature Importance
importances = rf.feature_importances_

plt.bar(X.columns, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()