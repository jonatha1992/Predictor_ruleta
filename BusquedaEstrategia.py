import pandas as pd
import os

# Asumiendo que ya tienes la ruta al archivo Excel definida y los datos cargados en `df`
ruta_al_archivo = "Reportes_simulacion_GRU.xlsx"
excel_salida = "Estrategias_Optimas_GRU.xlsx"
# ruta_al_archivo = "Reportes_simulacion_LSTM.xlsx"
# excel_salida = "Estrategias_Optimas_LSTM.xlsx"
df = pd.read_excel(ruta_al_archivo)

# Filtrar las estrategias donde la ganancia ha sido positiva
estrategias_positivas = df[df["Ganancia"] > 0]

# Agrupar por las columnas relevantes para definir una estrategia
estrategias_agrupadas = (
    estrategias_positivas.groupby(
        [
            "Cant. Vecinos",
            "Limite_juego",
            "Limite_pretendiente",
            "Probabilidad",
        ]
    )
    .agg(
        Veces_Positive=("Ganancia", "count"),  # Contar cuántas veces fue positiva
        Ganancia_Total=("Ganancia", "sum"),  # Sumar la ganancia total
        Efectividad_Media=("Efectividad", "mean"),  # Calcular la efectividad media
    )
    .reset_index()
)

# Ordenar por efectividad media de forma descendente para encontrar la que tuvo mayor efectividad
estrategias_ordenadas_efectividad = estrategias_agrupadas.sort_values(
    by="Efectividad_Media", ascending=False
)

# Tomar la estrategia que tuviera mayor efectividad
estrategia_con_mayor_efectividad = estrategias_ordenadas_efectividad.head(10)

# Ordenar por 'Cant. Vecinos' y 'Limite_juego' para priorizar estrategias con menos vecinos y menor límite de juego
estrategias_ordenadas = estrategia_con_mayor_efectividad.sort_values(
    by=["Cant. Vecinos", "Limite_juego", "Efectividad_Media"],
    ascending=[True, True, False],
)

# Tomar las estrategias que cumplen con los criterios
estrategias_seleccionadas = estrategias_ordenadas.head(
    20
)  # Ajustar según cuántas quieras seleccionar

# Guardar las estrategias seleccionadas en un nuevo archivo Excel
estrategias_seleccionadas.to_excel(excel_salida, index=False)

print(f"Se imprimió el archivo {excel_salida}.xlsx'")
