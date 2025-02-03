import optuna
import sqlite3
import os
from Entity.parametro import HiperParametros
from Entity.modelo import Modelo


def get_best_combination():
    # Conectar a la base de datos SQLite
    conn = sqlite3.connect("optuna_study.db")
    cursor = conn.cursor()

    # Verificar las tablas existentes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tablas en la base de datos:", tables)

    # Realizar una consulta SQL para enlazar varias tablas
    query = """
    SELECT
        trial_params.param_name,
        trial_params.param_value
    FROM
        trial_values
    JOIN
        trial_params ON trial_values.trial_id = trial_params.trial_id
    WHERE
        trial_values.value = (SELECT MAX(value) FROM trial_values)
    """

    cursor.execute(query)
    results = cursor.fetchall()

    # Cerrar la conexión
    conn.close()

    best_params = {}
    for param_name, param_value in results:
        best_params[param_name] = param_value
        print(param_name, param_value)
    return best_params


def create_model_with_best_params(best_params):
    hiperparametros = HiperParametros(
        numerosAnteriores=int(best_params['numerosAnteriores']),
        capa1=int(best_params['capa1']),
        capa2=int(best_params['capa2']),
        capa3=int(best_params['capa3']),
        dropout_rate=float(best_params['dropout_rate']),
        l2_lambda=float(best_params['l2_lambda']),
        learning_rate=float(best_params['learning_rate']),
        batchSize=int(best_params['batchSize']),
        epochs=int(best_params['epochs'])
    )

    modelo = Modelo("Data/Electromecanica.xlsx", hiperparametros)
    modelo.crear_y_guardar_modelos()
    print("Modelo óptimo guardado!")


# Save best model
print("\nCreando modelo con mejores parámetros encontrados...")
best_params = get_best_combination()

if best_params:
    # Crear el modelo con los mejores parámetros
    create_model_with_best_params(best_params)
else:
    print("No se encontraron parámetros óptimos en la base de datos.")
