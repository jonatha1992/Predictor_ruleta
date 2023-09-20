import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from ultimo_programaV2 import Predictor  # Importa la clase Predictor desde tu archivo predictor.py

# Función para realizar validación cruzada
def cross_validation():
    # Carga los datos desde el archivo Excel o la fuente de datos que desees
    data = pd.read_excel("datos.xlsx", sheet_name="Salidos")

    # Número de divisiones para la validación cruzada
    num_splits = 5
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Inicializa el resultado de la validación cruzada
    cross_val_scores = []

    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Crea una instancia del predictor y entrena el modelo
        predictor = Predictor("datos.xlsx")
        predictor.df = train_data  # Establece el conjunto de datos de entrenamiento
        predictor.model = predictor._crear_modelo()  # Entrena un nuevo modelo

        # Realiza predicciones en el conjunto de prueba
        predictions = []
        for _, row in test_data.iterrows():
            numero = row["Salidos"]
            predictor.verificar_numero(numero)
            predictor.predecir()
            predictions.extend(predictor.resultados)

        # Evalúa el rendimiento del modelo en el conjunto de prueba
        true_labels = test_data["Salidos"].tolist()
        accuracy = accuracy_score(true_labels, predictions)
        cross_val_scores.append(accuracy)

    # Calcula la puntuación media de la validación cruzada
    mean_accuracy = np.mean(cross_val_scores)
    print(f"Puntuación media de validación cruzada: {mean_accuracy}")

if __name__ == "__main__":
    cross_validation()
