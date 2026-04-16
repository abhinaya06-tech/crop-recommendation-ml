import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/crop_data.csv")

print(df.describe())
print(df["label"].value_counts())

# Plot crop distribution
df["label"].value_counts().plot(kind="bar")
plt.title("Crop Distribution")
plt.show()

# Feature relationships
df.boxplot(column="N", by="label", figsize=(10,6))
plt.xticks(rotation=90)
plt.show()