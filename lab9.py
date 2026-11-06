# Lab 9: K-Means and Hierarchical Clustering on Wholesale Customers data
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

df = pd.read_csv("Wholesale customers data.csv")
spend = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
print(df.shape, "| nulls:", df.isnull().sum().sum())
print(df[spend].describe().round(0).astype(int).to_string())

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.boxplot(data=df[spend], ax=ax[0]); ax[0].set_title("Spending boxplots"); ax[0].tick_params(axis="x", rotation=45)
sns.countplot(data=df, x="Channel", ax=ax[1]); ax[1].set_title("Customers per Channel")
plt.tight_layout(); plt.show()

X_scaled = MinMaxScaler().fit_transform(df[spend])
ks = range(2, 7)
sil = {k: silhouette_score(X_scaled, KMeans(n_clusters=k, random_state=77, n_init=10).fit_predict(X_scaled))
       for k in ks}
for k, s in sil.items():
    print(f"k = {k}  silhouette = {s:.4f}")
plt.figure(figsize=(6, 3.5))
plt.bar([str(k) for k in ks], list(sil.values()), color="#dd8452")
plt.title("Silhouette score for each k"); plt.xlabel("k"); plt.ylabel("score")
plt.tight_layout(); plt.show()

best_k = max(sil, key=sil.get)
km = KMeans(n_clusters=best_k, random_state=77, n_init=10).fit(X_scaled)
km_labels = km.labels_
print("Chosen k:", best_k)
print(pd.Series(km_labels).value_counts().sort_index().to_string())
mix = pd.crosstab(df["Channel"], km_labels, normalize="columns") * 100
mix.columns = [f"Cluster {c}" for c in mix.columns]
print("\nChannel share inside each cluster (%):")
print(mix.round(1).to_string())

Z = linkage(X_scaled, method="complete")
plt.figure(figsize=(9, 4))
dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=8)
plt.axhline(y=Z[-(best_k - 1), 2] - 0.01, color="red", linestyle="--", label="cut for chosen k")
plt.title("Dendrogram - complete linkage"); plt.ylabel("Distance"); plt.legend()
plt.tight_layout(); plt.show()

hc = AgglomerativeClustering(n_clusters=best_k, linkage="complete").fit(X_scaled)
rows = []
for name, lab in [("K-Means", km_labels), ("Hierarchical (complete)", hc.labels_)]:
    rows.append([name, silhouette_score(X_scaled, lab), davies_bouldin_score(X_scaled, lab),
                 calinski_harabasz_score(X_scaled, lab)])
print(pd.DataFrame(rows, columns=["Algorithm", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"])
      .round(4).to_string(index=False))
prof = df[spend].groupby(km_labels).mean()
plt.figure(figsize=(7.5, 3.4))
sns.heatmap(prof, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Average spending per K-Means cluster"); plt.ylabel("Cluster")
plt.tight_layout(); plt.show()
