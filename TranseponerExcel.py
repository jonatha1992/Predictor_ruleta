import pandas as pd

# Suponiendo que tienes tus datos en un archivo de Excel
file_path = "Datos Quini6.xlsx"
df = pd.read_excel(file_path, header=None, engine='openpyxl')

# Una lista para guardar las filas procesadas
output_data = []

# Un contador temporal para guardar números entre separadores
temp_nums = []

for index, row in df.iterrows():
    # Si el valor en la celda es un número, lo añadimos a temp_nums
    if isinstance(row[0], int):
        temp_nums.append(row[0])
    # Si temp_nums tiene exactamente 6 números, lo añadimos a output_data y limpiamos temp_nums
    if len(temp_nums) == 6:
        output_data.append(temp_nums)
        temp_nums = []

# Creamos un nuevo DataFrame con los datos procesados
output_df = pd.DataFrame(output_data)

# Guardamos el nuevo DataFrame a un archivo Excel
output_df.to_excel("Datos Quini6 nuevo.xlsx", header=False, index=False)
