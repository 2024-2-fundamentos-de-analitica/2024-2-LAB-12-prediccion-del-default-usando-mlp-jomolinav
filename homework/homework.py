
import pandas as pd
import gzip
import pickle
import json
import os
from glob import glob
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Función para cargar los datos
def cargar_datos():
    df_train = pd.read_csv("./files/input/train_data.csv.zip", compression="zip")
    df_test = pd.read_csv("./files/input/test_data.csv.zip", compression="zip")
    return df_train, df_test

# Función para limpiar los datos
def limpiar_datos(df):
    df = df.rename(columns={'default payment next month': 'default'}).drop(columns=["ID"])
    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    return df.dropna()

# Separar variables predictoras y objetivo
def dividir_datos(df):
    return df.drop(columns=["default"]), df["default"]

# Definir pipeline de preprocesamiento y modelo
def construir_pipeline(x_train):
    categorias = ["SEX", "EDUCATION", "MARRIAGE"]
    numericas = [col for col in x_train.columns if col not in categorias]

    preprocesador = ColumnTransformer([
        ('cat', OneHotEncoder(), categorias),
        ('scaler', StandardScaler(), numericas),
    ])

    return Pipeline([
        ("preprocesador", preprocesador),
        ("seleccion_caracteristicas", SelectKBest(score_func=f_classif)),
        ("pca", PCA()),
        ("clasificador", MLPClassifier(max_iter=15000, random_state=21))
    ])

# Configurar optimización de hiperparámetros
def configurar_estimador(pipeline):
    parametros = {
        "pca__n_components": [None],
        "seleccion_caracteristicas__k": [20],
        "clasificador__hidden_layer_sizes": [(50, 30, 40, 60)],
        "clasificador__alpha": [0.26],
        "clasificador__learning_rate_init": [0.001],
    }
    return GridSearchCV(
        estimator=pipeline, param_grid=parametros, cv=10,
        scoring='balanced_accuracy', n_jobs=-1, refit=True
    )

# Guardar el modelo
def guardar_modelo(ruta, modelo):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open(ruta, "wb") as f:
        pickle.dump(modelo, f)

# Calcular métricas de evaluación
def calcular_metricas(tipo, y_real, y_predicho):
    return {
        "type": "metrics", "dataset": tipo,
        "precision": precision_score(y_real, y_predicho, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_predicho),
        "recall": recall_score(y_real, y_predicho, zero_division=0),
        "f1_score": f1_score(y_real, y_predicho, zero_division=0)
    }

# Calcular matriz de confusión
def calcular_matriz_confusion(tipo, y_real, y_predicho):
    cm = confusion_matrix(y_real, y_predicho)
    return {
        "type": "cm_matrix", "dataset": tipo,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])}
    }

# Ejecutar el flujo de trabajo
def ejecutar_proceso():
    df_train, df_test = cargar_datos()
    df_train, df_test = limpiar_datos(df_train), limpiar_datos(df_test)
    x_train, y_train = dividir_datos(df_train)
    x_test, y_test = dividir_datos(df_test)
    
    modelo_pipeline = construir_pipeline(x_train)
    modelo = configurar_estimador(modelo_pipeline)
    modelo.fit(x_train, y_train)
    
    guardar_modelo("files/models/model.pkl.gz", modelo)
    
    y_pred_train = modelo.predict(x_train)
    y_pred_test = modelo.predict(x_test)
    
    metricas_train = calcular_metricas("train", y_train, y_pred_train)
    metricas_test = calcular_metricas("test", y_test, y_pred_test)
    matriz_train = calcular_matriz_confusion("train", y_train, y_pred_train)
    matriz_test = calcular_matriz_confusion("test", y_test, y_pred_test)
    
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        for entrada in [metricas_train, metricas_test, matriz_train, matriz_test]:
            file.write(json.dumps(entrada) + "\n")

if __name__ == "__main__":
    ejecutar_proceso()