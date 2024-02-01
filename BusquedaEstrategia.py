import pandas as pd

# Reemplaza 'ruta_al_archivo.xlsx' con la ruta de tu archivo Excel real
ruta_al_archivo = "Reportes_simulacion copy.xlsx"

# Cargar los datos en un DataFrame de pandas
df = pd.read_excel(ruta_al_archivo)

# Filtrar las estrategias donde la ganancia ha sido positiva
estrategias_positivas = df[df["Ganancia"] > 0]

# Agrupar por 'numeros jugados', 'plenos', 'valor ficha' para definir una estrategia
estrategias_agrupadas = estrategias_positivas.groupby(
    ["Cant. Vecinos", "Limite_juego", "Limite_pretendiente", "Probabilidad"]
)

# Calcular la cantidad de veces que la estrategia fue positiva y la efectividad media
estrategias_stats = estrategias_agrupadas.agg(
    Veces_Positive=("Ganancia", "count"), Efectividad_Media=("Efectividad", "mean")
).reset_index()

# Filtrar aquellas estrategias que han sido positivas más de una vez y tienen una efectividad alta
# Supongamos que consideramos una efectividad alta como aquella que está en el cuartil superior
umbral_efectividad = estrategias_stats["Efectividad_Media"].quantile(0.75)
estrategias_recomendadas = estrategias_stats[
    (estrategias_stats["Veces_Positive"] > 1)
    & (estrategias_stats["Efectividad_Media"] > umbral_efectividad)
]

# Ordenar las estrategias recomendadas por efectividad media de manera descendente
estrategias_recomendadas = estrategias_recomendadas.sort_values(
    by="Efectividad_Media", ascending=False
)

# Tomar las primeras 10 estrategias recomendadas
top_estrategias = estrategias_recomendadas.head(10)
top_estrategias.to_excel("Mejores_estrategias.xlsx", index=False)

print("Se imprimio el archivo 'Mejores_estrategias.xlsx'")
