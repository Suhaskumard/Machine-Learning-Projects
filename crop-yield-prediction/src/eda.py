import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/crop_production.csv")
df = df.dropna()

# Top crops
top_crops = df.groupby("Crop")["Production"].sum().sort_values(ascending=False).head(10)

plt.figure()
top_crops.plot(kind="bar")
plt.title("Top 10 Crops by Production")
plt.xlabel("Crop")
plt.ylabel("Production")
plt.show()

# Yield distribution
df["Yield"] = df["Production"] / df["Area"]

plt.figure()
sns.histplot(df["Yield"], bins=50)
plt.title("Yield Distribution")
plt.show()
