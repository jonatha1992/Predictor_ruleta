# Importando librerías adicionales para el análisis
import re
from collections import defaultdict


# Función para extraer datos de tardanza de las columnas relevantes
def extraer_tardanza(columna):
    tardanzas = []
    if isinstance(columna, str):
        # Extrae todos los elementos que coinciden con el patrón "(N:#,P:#,T:#)"
        matches = re.findall(r"\(N:\d+,P:\d+,T:(\d+)\)", columna)
        tardanzas = [int(t) for t in matches]
    return tardanzas


# Aplicando la función a las columnas relevantes
tardanzas_acertados = updated_data["Acertados"].apply(extraer_tardanza)
tardanzas_no_salidos = updated_data["No salidos"].apply(extraer_tardanza)
tardanzas_resultados = updated_data["Resultados"].apply(extraer_tardanza)

# Contando la frecuencia de cada tardanza
frecuencia_tardanzas_acertados = defaultdict(int)
frecuencia_tardanzas_no_salidos = defaultdict(int)
frecuencia_tardanzas_resultados = defaultdict(int)

for lista_tardanzas in tardanzas_acertados:
    for t in lista_tardanzas:
        frecuencia_tardanzas_acertados[t] += 1

for lista_tardanzas in tardanzas_no_salidos:
    for t in lista_tardanzas:
        frecuencia_tardanzas_no_salidos[t] += 1

for lista_tardanzas in tardanzas_resultados:
    for t in lista_tardanzas:
        frecuencia_tardanzas_resultados[t] += 1

# Convirtiendo a DataFrame para mejor visualización
df_frecuencia_tardanzas_acertados = pd.DataFrame(
    list(frecuencia_tardanzas_acertados.items()),
    columns=["Tardanza", "Frecuencia_Acertados"],
)
df_frecuencia_tardanzas_no_salidos = pd.DataFrame(
    list(frecuencia_tardanzas_no_salidos.items()),
    columns=["Tardanza", "Frecuencia_No_Salidos"],
)
df_frecuencia_tardanzas_resultados = pd.DataFrame(
    list(frecuencia_tardanzas_resultados.items()),
    columns=["Tardanza", "Frecuencia_Resultados"],
)

df_frecuencia_tardanzas_acertados.sort_values(
    by="Tardanza"
), df_frecuencia_tardanzas_no_salidos.sort_values(
    by="Tardanza"
), df_frecuencia_tardanzas_resultados.sort_values(
    by="Tardanza"
)
