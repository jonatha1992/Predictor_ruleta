#!/usr/bin/env python3
"""
Script para crear y guardar el modelo de predicción de la ruleta.
"""
import os
from Entity.Predictor import Predictor
from Entity.Parametro import HiperParametros, Parametro_Juego
from Entity.vecinos import colores_ruleta
from Entity.ModeloPredictor import Modelo_Predictor
from config import get_excel_file, get_ruleta_types , get_relative_path


def main():
    print("Iniciando creación de modelo...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_name = "Electromecanica.xlsx"

    # Intenta primero en Entity, luego en Data
    entity_path = os.path.join(script_dir, "Entity", data_file_name)
    data_path = os.path.join(script_dir, "Data", data_file_name)

    if os.path.exists(entity_path):
        data_file_path = entity_path
        print(f"Usando archivo en Entity: {data_file_path}")
    elif os.path.exists(data_path):
        data_file_path = data_path
        print(f"Usando archivo en Data: {data_file_path}")
    else:
        print(f"No se encontró el archivo {data_file_name} en Entity ni Data")
        print(f"Rutas buscadas: {entity_path}, {data_path}")
        return

    # Configurar hiperparámetros con valores mínimos para evitar problemas
    hiperparametros = HiperParametros(
        numerosAnteriores=9,
        capa1=356,
        capa2=256,
        capa3=128,
        batchSize=256,
        epochs=100
    )

    # Crear instancia del modelo
    modelo = Modelo_Predictor(filename=data_file_path, hiperparametro=hiperparametros)

    # Generar nombre y ruta de guardado
    basename = os.path.splitext(data_file_name)[0]  # Use the plain file name for the model name
    modelo_nombre = f"Model_{basename}_N{hiperparametros.numerosAnteriores}.keras"
    ruta_modelo = get_relative_path(os.path.join("Models", modelo_nombre))

    # Asegurarse de que exista el directorio
    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)

    # Crear y guardar el modelo
    print(f"Creando y guardando el modelo en: {ruta_modelo}")
    modelo.crear_y_guardar_modelos()
    print("Proceso completado.")


if __name__ == "__main__":
    main()
