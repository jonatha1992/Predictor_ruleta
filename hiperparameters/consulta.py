import optuna
import sqlite3
import os


def get_combined_data():
    # Conectar a la base de datos SQLite
    db_path = os.path.join('hiperparameters', 'optuna_study.db')  # Asegúrate de que la ruta es correcta
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Verificar las tablas existentes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tablas en la base de datos:", tables)

    # Verificar si las tablas 'trials' y 'trial_params' existen
    if ('trials',) in tables and ('trial_params',) in tables:
        # Realizar una consulta SQL para enlazar varias tablas
        query = """
        SELECT
            trial_values.trial_id,
            trial_values.value,
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
    else:
        results = []
        print("Las tablas 'trials' y/o 'trial_params' no existen en la base de datos.")

    # Cerrar la conexión
    conn.close()

    return results


def get_max_value():
    # Conectar a la base de datos SQLite
    db_path = os.path.join('hiperparameters', 'optuna_study.db')  # Asegúrate de que la ruta es correcta
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Obtener el mayor valor de la columna 'value' en la tabla 'trials'
    cursor.execute("SELECT MAX(value) FROM trial_values")
    max_value = cursor.fetchone()[0]

    # Cerrar la conexión
    conn.close()

    return max_value


if __name__ == "__main__":
    try:
        # # Obtener el mejor trial
        # best_trial = get_best_trial()

        # Obtener datos combinados
        combined_data = get_combined_data()
        for row in combined_data:
            print(row)

        # Obtener el mayor valor de la prueba
        max_value = get_max_value()
        print(f"El mayor valor de la prueba es: {max_value}")
    except Exception as e:
        print(f"Error accessing the database: {e}")
