import pandas as pd

# Asumiendo que ya tienes la ruta al archivo Excel definida y los datos cargados en `df`
ruta_al_archivo = "Reportes_simulacion copy.xlsx"
df = pd.read_excel(ruta_al_archivo)

# Filtrar las estrategias donde la ganancia ha sido positiva
estrategias_positivas = df[df["Ganancia"] > 0]

# Agrupar por las columnas relevantes para definir una estrategia
estrategias_agrupadas = (
    estrategias_positivas.groupby(
        ["Cant. Vecinos", "Limite_juego", "Limite_pretendiente", "Probabilidad"]
    )
    .agg(
        Veces_Positive=("Ganancia", "count"),  # Contar cuántas veces fue positiva
        Ganancia_Total=("Ganancia", "sum"),  # Sumar la ganancia total
        Efectividad_Media=("Efectividad", "mean"),  # Calcular la efectividad media
    )
    .reset_index()
)

# Filtrar aquellas estrategias que han sido positivas en todas las instancias
# Esto asume que tienes una columna o manera de identificar todas las instancias de cada estrategia
# Y quieres aquellas que siempre fueron positivas, no solo más de una vez
estrategias_100_positivas = estrategias_agrupadas[
    estrategias_agrupadas["Veces_Positive"]
    == estrategias_agrupadas["Veces_Positive"].max()
]

# Ordenar por 'Cant. Vecinos' y 'Limite_juego' para priorizar estrategias con menos vecinos y menor límite de juego
estrategias_ordenadas = estrategias_100_positivas.sort_values(
    by=["Cant. Vecinos", "Limite_juego", "Efectividad_Media"],
    ascending=[True, True, False],
)

# Tomar las estrategias que cumplen con los criterios
estrategias_seleccionadas = estrategias_ordenadas.head(
    10
)  # Ajustar según cuántas quieras seleccionar

# Guardar las estrategias seleccionadas en un nuevo archivo Excel
estrategias_seleccionadas.to_excel("Estrategias_Optimas.xlsx", index=False)

print("Se imprimió el archivo 'Estrategias_Optimas.xlsx'")
