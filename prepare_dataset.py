import pandas as pd

# Read Excel files instead of CSV
true = pd.read_excel("True.xlsx")
fake = pd.read_excel("Fake.xlsx")

# Add labels
true["label"] = 0   # real news
fake["label"] = 1   # fake news

# Combine
data = pd.concat([true, fake])
data = data[["text", "label"]]  # keep only these two columns
data.to_csv("dataset/combined.csv", index=False)

print("✅ Combined dataset saved as dataset/combined.csv")
