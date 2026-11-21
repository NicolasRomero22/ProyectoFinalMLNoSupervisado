# -*- coding: utf-8 -*-
"""
Segmentación de clientes de un distribuidor mayorista con ML No Supervisado
Dataset: Wholesale customers (UCI Machine Learning Repository, id=292)
Entorno: PyScripter + Python 3.x
Librerías: pandas, numpy, scikit-learn, matplotlib, seaborn

Uso en PyScripter:
  1) Descargar "Wholesale customers data.csv" desde
     https://archive.ics.uci.edu/dataset/292/wholesale+customers
  2) Guardarlo en la misma carpeta que este script.
  3) Ejecutar con Run (Ctrl+F9).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

SEED = 42
CARPETA = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
ARCHIVO = os.path.join(CARPETA, "Wholesale customers data.csv")
SALIDA = os.path.join(CARPETA, "resultados")
os.makedirs(SALIDA, exist_ok=True)

PRODUCTOS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------
def cargar_datos():
    if os.path.exists(ARCHIVO):
        return pd.read_csv(ARCHIVO)
    # Alternativa: descarga directa desde UCI (requiere internet y ucimlrepo)
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=292)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)


df = cargar_datos()
print("Dimensiones:", df.shape)
print(df.head())
print("\nValores nulos por columna:\n", df.isnull().sum())
print("\nDuplicados:", df.duplicated().sum())
print("\nEstadísticos descriptivos:\n", df[PRODUCTOS].describe().round(1))


# ---------------------------------------------------------------------------
# 2. ANÁLISIS EXPLORATORIO (EDA)
# ---------------------------------------------------------------------------
fig, ejes = plt.subplots(2, 3, figsize=(12, 6))
for ax, col in zip(ejes.ravel(), PRODUCTOS):
    sns.histplot(df[col], bins=40, ax=ax)
    ax.set_title(col)
plt.suptitle("Distribución del gasto anual por categoría (sesgo a la derecha)")
plt.tight_layout()
plt.savefig(os.path.join(SALIDA, "01_histogramas.png"), dpi=150)

plt.figure(figsize=(7, 5))
sns.heatmap(df[PRODUCTOS].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlación entre categorías de producto")
plt.tight_layout()
plt.savefig(os.path.join(SALIDA, "02_correlacion.png"), dpi=150)


# ---------------------------------------------------------------------------
# 3. PREPROCESAMIENTO
#    - log1p para corregir asimetría y reducir el peso de valores extremos
#    - StandardScaler: K-Means y PCA son sensibles a la escala
#    - Channel y Region NO entran al modelo: se usan solo para interpretar
# ---------------------------------------------------------------------------
X_log = np.log1p(df[PRODUCTOS])
X = StandardScaler().fit_transform(X_log)


# ---------------------------------------------------------------------------
# 4. REDUCCIÓN DE DIMENSIONALIDAD (PCA) - visualización y diagnóstico
# ---------------------------------------------------------------------------
pca = PCA(random_state=SEED).fit(X)
print("\nVarianza explicada por componente:", pca.explained_variance_ratio_.round(3))
print("Varianza acumulada:", pca.explained_variance_ratio_.cumsum().round(3))
X_2d = PCA(n_components=2, random_state=SEED).fit_transform(X)


# ---------------------------------------------------------------------------
# 5. SELECCIÓN DEL NÚMERO DE CLÚSTERES (k)
# ---------------------------------------------------------------------------
rango_k = range(2, 11)
metricas = []
for k in rango_k:
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=SEED)
    etiquetas = km.fit_predict(X)
    metricas.append({
        "k": k,
        "inercia": km.inertia_,
        "silhouette": silhouette_score(X, etiquetas),
        "davies_bouldin": davies_bouldin_score(X, etiquetas),
        "calinski_harabasz": calinski_harabasz_score(X, etiquetas),
    })
metricas = pd.DataFrame(metricas)
print("\nMétricas por k:\n", metricas.round(3).to_string(index=False))

fig, ejes = plt.subplots(1, 3, figsize=(14, 4))
ejes[0].plot(metricas["k"], metricas["inercia"], "o-")
ejes[0].set_title("Método del codo (inercia)")
ejes[1].plot(metricas["k"], metricas["silhouette"], "o-")
ejes[1].set_title("Coeficiente de silueta (mayor es mejor)")
ejes[2].plot(metricas["k"], metricas["davies_bouldin"], "o-")
ejes[2].set_title("Davies-Bouldin (menor es mejor)")
for ax in ejes:
    ax.set_xlabel("k")
plt.tight_layout()
plt.savefig(os.path.join(SALIDA, "03_seleccion_k.png"), dpi=150)

K_ELEGIDO = int(metricas.loc[metricas["silhouette"].idxmax(), "k"])
print("\nk con mayor silueta:", K_ELEGIDO,
      "(contrastar con el codo y con la interpretabilidad antes de decidir)")


# ---------------------------------------------------------------------------
# 6. MODELO FINAL Y ALTERNATIVAS PARA COMPARAR
# ---------------------------------------------------------------------------
modelos = {
    "K-Means": KMeans(n_clusters=K_ELEGIDO, init="k-means++", n_init=20, random_state=SEED),
    "Jerárquico (Ward)": AgglomerativeClustering(n_clusters=K_ELEGIDO, linkage="ward"),
    "Mezcla gaussiana": GaussianMixture(n_components=K_ELEGIDO, random_state=SEED),
    "DBSCAN": DBSCAN(eps=1.2, min_samples=8),
}
resultados, etiquetas_modelo = [], {}
for nombre, modelo in modelos.items():
    et = modelo.fit_predict(X)
    etiquetas_modelo[nombre] = et
    n_cl = len(set(et)) - (1 if -1 in et else 0)
    if n_cl >= 2:
        mascara = et != -1  # DBSCAN marca ruido con -1
        resultados.append({
            "modelo": nombre, "clusters": n_cl, "ruido": int((~mascara).sum()),
            "silhouette": silhouette_score(X[mascara], et[mascara]),
            "davies_bouldin": davies_bouldin_score(X[mascara], et[mascara]),
            "calinski_harabasz": calinski_harabasz_score(X[mascara], et[mascara]),
        })
    else:
        resultados.append({"modelo": nombre, "clusters": n_cl, "ruido": int((et == -1).sum()),
                           "silhouette": np.nan, "davies_bouldin": np.nan,
                           "calinski_harabasz": np.nan})
comparacion = pd.DataFrame(resultados)
print("\nComparación de algoritmos:\n", comparacion.round(3).to_string(index=False))

df["cluster"] = etiquetas_modelo["K-Means"]


# ---------------------------------------------------------------------------
# 7. VISUALIZACIÓN E INTERPRETACIÓN DE LOS SEGMENTOS
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=df["cluster"], palette="deep", s=35)
plt.title("Segmentos K-Means proyectados en 2 componentes principales")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(SALIDA, "04_clusters_pca.png"), dpi=150)

perfil = df.groupby("cluster")[PRODUCTOS].mean().round(0)
perfil["n_clientes"] = df["cluster"].value_counts().sort_index()
print("\nPerfil promedio de gasto por segmento (m.u.):\n", perfil)
perfil.to_csv(os.path.join(SALIDA, "perfil_segmentos.csv"))

plt.figure(figsize=(9, 4.5))
sns.heatmap(perfil[PRODUCTOS] / df[PRODUCTOS].mean(), annot=True, fmt=".2f",
            cmap="YlGnBu")
plt.title("Gasto promedio del segmento / gasto promedio global")
plt.tight_layout()
plt.savefig(os.path.join(SALIDA, "05_perfil_segmentos.png"), dpi=150)

# Validación externa (solo si existen las columnas Channel y Region)
if "Channel" in df.columns:
    print("\nSegmentos vs. canal (1=Horeca, 2=Retail):")
    print(pd.crosstab(df["cluster"], df["Channel"]))
    print("ARI K-Means vs Channel:",
          round(adjusted_rand_score(df["Channel"], df["cluster"]), 3))
if "Region" in df.columns:
    print("\nSegmentos vs. región (1=Lisboa, 2=Oporto, 3=Otra):")
    print(pd.crosstab(df["cluster"], df["Region"]))

df.to_csv(os.path.join(SALIDA, "clientes_con_segmento.csv"), index=False)
print("\nListo. Resultados guardados en:", SALIDA)
plt.show()
