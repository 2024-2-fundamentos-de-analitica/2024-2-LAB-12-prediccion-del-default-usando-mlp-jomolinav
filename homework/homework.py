# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import pandas as pd
import zipfile
import pickle
import gzip
import os
import json
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


def clean_dataset(path):
    """Carga y limpia los datasets."""
    with zipfile.ZipFile(path, "r") as z:
        csv_file = z.namelist()[0]
        with z.open(csv_file) as f:
            df = pd.read_csv(f)

    # Renombrar la columna "default payment next month" a "default"
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    # Remover la columna "ID"
    df.drop(columns=["ID"], inplace=True)
    # Eliminar registros con información no disponible
    df.dropna(inplace=True)
    # Agrupar valores de EDUCATION > 4 como "others" (se reemplaza por 4)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)

    return df


# Cargar datasets
df_test = clean_dataset("./files/input/test_data.csv.zip")
df_train = clean_dataset("./files/input/train_data.csv.zip")

# Separar variables
x_train = df_train.drop(columns=["default"])
y_train = df_train["default"]

x_test = df_test.drop(columns=["default"])
y_test = df_test["default"]


def build_pipeline():

    # Definir columnas categóricas y numéricas
    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]
    numeric_features = [
        col for col in x_train.columns if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    # Ajustamos el preprocesador para conocer el número de características resultantes
    preprocessor.fit(x_train)
    x_train_transformed = preprocessor.transform(x_train)
    num_features_after_preprocessing = x_train_transformed.shape[1]


    # Configuración inicial de SelectKBest (el valor k se sobreescribirá en GridSearchCV)
    k_best = SelectKBest(f_classif, k=min(10, num_features_after_preprocessing))

    # Modelo MLP con random_state para reproducibilidad, early stopping y más iteraciones
    model = MLPClassifier(random_state=21, max_iter=1500, early_stopping=True)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("k_best", k_best),
            ("pca", PCA()),  # Se parametriza en GridSearchCV
            ("classifier", model),
        ]
    )

    return pipeline


def optimize_pipeline(pipeline, x_train, y_train):
  
    param_grid = {
        "pca__n_components": [None],
        "k_best__k": [20],
        "classifier__hidden_layer_sizes": [(50, 30, 40, 60), (100,)],
        "classifier__alpha": [0.26],
        "classifier__learning_rate_init": [0.001],
        "classifier__activation": ["relu"],
        "classifier__solver": ["adam"],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

    grid_search.fit(x_train, y_train)


    return grid_search


def save_model(model, file_path="files/models/model.pkl.gz"):
  
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with gzip.open(file_path, "wb") as f:
        pickle.dump(model, f)



def evaluate_model(
    model, x_train, y_train, x_test, y_test, file_path="files/output/metrics.json"
):
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        for dataset, (x, y) in zip(
            ["train", "test"], [(x_train, y_train), (x_test, y_test)]
        ):
            y_pred = model.predict(x)
            metrics = {
                "type": "metrics",
                "dataset": dataset,
                "precision": precision_score(y, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1_score": f1_score(y, y_pred, zero_division=0),
            }
            f.write(json.dumps(metrics) + "\n")

            cm = confusion_matrix(y, y_pred)
            cm_data = {
                "type": "cm_matrix",
                "dataset": dataset,
                "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
                "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
            }
            f.write(json.dumps(cm_data) + "\n")




pipeline = build_pipeline()

best_pipeline = optimize_pipeline(pipeline, x_train, y_train)
save_model(best_pipeline)

evaluate_model(best_pipeline, x_train, y_train, x_test, y_test)
