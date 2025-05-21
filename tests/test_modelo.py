import numpy as np
import pytest
import os
from Entity.ModeloPredictor import Modelo_Predictor
from Entity.Parametro import HiperParametros

def get_modelo():
    data_file = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Electromecanica.xlsx')
    hiperparametros = HiperParametros(
        numerosAnteriores=10,
        capa1=356,
        capa2=256,
        capa3=128,
        batchSize=256,
        epochs=5  # Menos epochs para test rápido
    )
    return Modelo_Predictor(filename=data_file, hiperparametro=hiperparametros)

def test_analisis_completo_predicciones():
    """
    Análisis detallado de predicciones para diferentes entradas,
    mostrando probabilidades y comportamiento del modelo.
    """
    # Ruta del modelo guardado
    modelo_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'Model_Electromecanica_N10.keras')
    assert os.path.exists(modelo_path), f"No se encontró el modelo guardado en {modelo_path}"

    # Instanciar el predictor y cargar el modelo
    hiperparametros = HiperParametros(
        numerosAnteriores=10,  # Debe coincidir con el modelo guardado
        capa1=356,
        capa2=256,
        capa3=128,
        batchSize=256,
        epochs=5
    )
    modelo = Modelo_Predictor(filename=None, hiperparametro=hiperparametros)
    modelo.cargar_modelo(modelo_path)
    
    # Test 1: Predicción con entrada de ceros
    print("\n\n---- PREDICCIÓN CON ENTRADA DE CEROS ----")
    entrada_ceros = np.zeros((1, hiperparametros.numerosAnteriores * 7))
    pred_ceros = modelo.modelo.predict(entrada_ceros)[0]
    
    # Mostrar top 5 números más probables
    indices_top = np.argsort(pred_ceros)[-5:][::-1]  # Top 5 índices en orden descendente
    print(f"Top 5 números con mayor probabilidad (entrada ceros):")
    for i, idx in enumerate(indices_top):
        print(f"  {i+1}. Número {idx}: {pred_ceros[idx]:.4f} ({pred_ceros[idx]*100:.2f}%)")
    
    # Estadísticas básicas
    print(f"Desviación estándar: {np.std(pred_ceros):.4f}")
    print(f"Media: {np.mean(pred_ceros):.4f}")
    print(f"Suma de probabilidades: {np.sum(pred_ceros):.4f}")
    
    # Test 2: Predicción con secuencia específica
    print("\n\n---- PREDICCIÓN CON SECUENCIA DE NÚMEROS ----")
    # Creando una entrada con números específicos (por ejemplo: 8, 17, 29, 0, 32, 15, 7, 22)
    numeros_secuencia = [8, 17, 29, 0, 32, 15, 7, 22]
    entrada_secuencia = np.zeros((1, hiperparametros.numerosAnteriores * 7))
    
    # Rellenar solo los valores de los números, el resto con ceros (como características)
    for i, num in enumerate(numeros_secuencia):
        entrada_secuencia[0, i * 7] = num  # Asumiendo que la primera posición de cada grupo es el número
    
    pred_secuencia = modelo.modelo.predict(entrada_secuencia)[0]
    
    # Mostrar top 5 números más probables
    indices_top = np.argsort(pred_secuencia)[-5:][::-1]
    print(f"Top 5 números con mayor probabilidad (después de {numeros_secuencia}):")
    for i, idx in enumerate(indices_top):
        print(f"  {i+1}. Número {idx}: {pred_secuencia[idx]:.4f} ({pred_secuencia[idx]*100:.2f}%)")
    
    # Estadísticas básicas
    print(f"Desviación estándar: {np.std(pred_secuencia):.4f}")
    print(f"Media: {np.mean(pred_secuencia):.4f}")
    
    # Test 3: Comparación entre diferentes secuencias
    print("\n\n---- COMPARACIÓN ENTRE DIFERENTES SECUENCIAS ----")
    # Otra secuencia alternativa
    numeros_secuencia2 = [0, 1, 2, 3, 4, 5, 6, 7]  # Secuencia ordenada
    entrada_secuencia2 = np.zeros((1, hiperparametros.numerosAnteriores * 7))
    
    for i, num in enumerate(numeros_secuencia2):
        entrada_secuencia2[0, i * 7] = num
    
    pred_secuencia2 = modelo.modelo.predict(entrada_secuencia2)[0]
    
    # Mostrar top 5 números más probables
    indices_top = np.argsort(pred_secuencia2)[-5:][::-1]
    print(f"Top 5 números con mayor probabilidad (después de {numeros_secuencia2}):")
    for i, idx in enumerate(indices_top):
        print(f"  {i+1}. Número {idx}: {pred_secuencia2[idx]:.4f} ({pred_secuencia2[idx]*100:.2f}%)")
    
    # Diferencia entre predicciones
    diferencia = np.abs(pred_secuencia - pred_secuencia2)
    max_diff_idx = np.argmax(diferencia)
    print(f"\nMayor diferencia en el número {max_diff_idx}: {diferencia[max_diff_idx]:.4f}")
    
    # Verificar que ambas predicciones son diferentes (el modelo responde a entradas diferentes)
    assert not np.allclose(pred_secuencia, pred_secuencia2), "El modelo predice igual para entradas diferentes"
    
    # Verificar balance de probabilidades
    # Si el modelo está balanceado, ningún número debería tener una probabilidad extremadamente alta
    assert np.max(pred_secuencia) < 0.5, f"Probabilidad demasiado alta para un solo número: {np.max(pred_secuencia):.4f}"
    assert np.max(pred_secuencia2) < 0.5, f"Probabilidad demasiado alta para un solo número: {np.max(pred_secuencia2):.4f}"

# Configuración de hiperparámetros de ejemplo

def test_predicciones_no_uniformes():
    """
    Verifica que el modelo guardado no prediga la misma probabilidad para todos los números (0-36).
    Si todas las probabilidades son iguales, el modelo no aprendió nada útil.
    """
    # Ruta del modelo guardado
    modelo_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'Model_Electromecanica_N8.keras')
    assert os.path.exists(modelo_path), f"No se encontró el modelo guardado en {modelo_path}"

    # Instanciar el predictor y cargar el modelo
    hiperparametros = HiperParametros(
        numerosAnteriores=8,  # Debe coincidir con el modelo guardado
        capa1=356,
        capa2=256,
        capa3=128,
        batchSize=256,
        epochs=5
    )
    modelo = Modelo_Predictor(filename=None, hiperparametro=hiperparametros)
    modelo.cargar_modelo(modelo_path)

    # Crear una entrada de prueba (puedes ajustar la forma si es necesario)
    # La forma debe ser (1, numerosAnteriores * 7) según tu modelo
    entrada = np.zeros((1, hiperparametros.numerosAnteriores * 7))
    pred = modelo.modelo.predict(entrada)[0]
    print(f"Predicciones: {pred}")
    std = np.std(pred)
    mean = np.mean(pred)
    # Las probabilidades deben variar
    assert std > 0.01, f"Las probabilidades parecen uniformes: std={std}, mean={mean}, pred={pred}"
    # Además, la suma debe ser 1 (softmax)
    assert np.isclose(np.sum(pred), 1.0), f"La suma de probabilidades no es 1: {np.sum(pred)}"

if __name__ == "__main__":
    pytest.main([__file__])
