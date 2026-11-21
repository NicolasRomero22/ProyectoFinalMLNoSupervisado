# -*- coding: utf-8 -*-
"""
Analisis complementario (ejecutar despues de segmentacion_clientes.py, en la misma carpeta):
 - Estabilidad de K-Means con distintas semillas (ARI promedio)
 - Perfil de segmentos para k = 2, 3 y 4
 - Barrido de hiperparametros de DBSCAN
 - Efecto de la transformacion logaritmica sobre la silueta
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score

CARPETA = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
SALIDA = os.path.join(CARPETA, "resultados")
os.makedirs(SALIDA, exist_ok=True)
PROD = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


class Tee:
    def __init__(self, *f): self.f = f
    def write(self, t): [x.write(t) for x in self.f]
    def flush(self): [x.flush() for x in self.f]


_log = open(os.path.join(SALIDA, "log_complementario.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

df = pd.read_csv(os.path.join(CARPETA, "Wholesale customers data.csv"))
X = StandardScaler().fit_transform(np.log1p(df[PROD]))

# 1. Estabilidad frente a la semilla
print("== 1. Estabilidad (ARI entre 20 semillas distintas, n_init=1) ==")
est = []
for k in (2, 3, 4):
    et = [KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(X) for s in range(20)]
    aris = [adjusted_rand_score(et[i], et[j]) for i in range(20) for j in range(i + 1, 20)]
    est.append({"k": k, "ARI_medio": np.mean(aris), "ARI_minimo": np.min(aris)})
print(pd.DataFrame(est).round(3).to_string(index=False))

# 2. Perfiles para k = 2, 3, 4
print("\n== 2. Perfil (gasto medio, m.u.) y tamano por segmento ==")
fig, ejes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, k in zip(ejes, (2, 3, 4)):
    et = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42).fit_predict(X)
    d = df.copy(); d["seg"] = et
    perf = d.groupby("seg")[PROD].mean().round(0)
    perf["n"] = d["seg"].value_counts().sort_index()
    perf["%Retail"] = (d.groupby("seg")["Channel"].apply(lambda s: (s == 2).mean()) * 100).round(0)
    print(f"\nk = {k}  (silueta = {silhouette_score(X, et):.3f})")
    print(perf.to_string())
    rel = perf[PROD] / df[PROD].mean()
    sns.heatmap(rel, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar=False)
    ax.set_title(f"k = {k}: gasto del segmento / gasto global")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(SALIDA, "06_perfiles_k2_k3_k4.png"), dpi=150)

# 3. DBSCAN
print("\n== 3. Barrido de DBSCAN ==")
filas = []
for eps in (0.8, 1.0, 1.2, 1.5, 1.8, 2.0):
    for ms in (5, 8, 12):
        et = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        ncl = len(set(et)) - (1 if -1 in et else 0)
        m = et != -1
        sil = silhouette_score(X[m], et[m]) if ncl >= 2 else np.nan
        filas.append({"eps": eps, "min_samples": ms, "clusters": ncl,
                      "ruido": int((~m).sum()), "silhouette": sil})
print(pd.DataFrame(filas).round(3).to_string(index=False))

# 4. Efecto del log
print("\n== 4. Silueta de K-Means (k=2) con y sin log1p ==")
X_sin = StandardScaler().fit_transform(df[PROD])
for nombre, M in (("sin log1p", X_sin), ("con log1p", X)):
    et = KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(M)
    print(f"{nombre}: silueta = {silhouette_score(M, et):.3f}, tamanos = {np.bincount(et).tolist()}")
print("\nListo.")
