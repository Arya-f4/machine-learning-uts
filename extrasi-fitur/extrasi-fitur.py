import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat dataset
df = pd.read_csv("train.csv")


# --- Preprocessing ---
# Encode kolom kategorikal
label_cols = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
for col in label_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # ubah ke string dulu
    df[col] = le.fit_transform(df[col])

# Hilangkan baris dengan NaN
df = df.dropna()

# Memisahkan fitur dan target
X = df.drop(columns=["Survived"])  # Semua fitur
y = df["Survived"]

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA dengan 2 komponen
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Buat DataFrame hasil PCA
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Survived"] = y.values

# Variance explained
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca.explained_variance_ratio_))

# Plot hasil PCA
plt.figure(figsize=(8, 6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["Survived"], cmap="coolwarm", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Titanic Dataset")
plt.colorbar(label="Survived")
plt.show()

print(df_pca.head())
