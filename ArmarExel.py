import os
from openpyxl import Workbook

# Crear un nuevo archivo Excel y la hoja de trabajo
wb = Workbook()
ws = wb.active

# Especificar la ubicación de la carpeta 'Reportes'
carpeta_reportes = "Reportes"

# Agregar las cabeceras a la hoja de trabajo
cabeceras = ["Juego fecha y hora", "Numeros jugados", "Aciertos de resultados", "Aciertos de vecinos cercanos", 
             "Aciertos de vecinos lejanos", "Aciertos Totales", "Maximo valor que tardo sin salir acierto", 
             "Maximo valor que tardo sin salir vecinos cercanos", "Maximo valor que tardo sin salir vecinos lejanos", 
             "l2", "dropout rate", "learning rate", "epoca", "batch_size"]
ws.append(cabeceras)

# Recorrer cada archivo TXT en la carpeta 'Reportes'
for archivo in os.listdir(carpeta_reportes):
    if archivo.endswith(".txt"):
        # Leer el archivo TXT
        with open(os.path.join(carpeta_reportes, archivo), 'r') as f:
            lines = f.readlines()

            # Escribir la información en la hoja de trabajo
            
            fecha_hora = lines[0].strip().split(' ')[-1]
            numeros_jugados = lines[1].strip().split(': ')[-1]
            aciertos_resultados = lines[4].strip().split(': ')[-1]
            aciertos_cercanos = lines[5].strip().split(': ')[-1]
            aciertos_lejanos = lines[6].strip().split(': ')[-1]
            aciertos_totales = lines[7].strip().split(': ')[-1]
            max_acierto = lines[8].strip().split(': ')[-1]
            max_cercanos = lines[9].strip().split(': ')[-1]
            max_lejanos = lines[10].strip().split(': ')[-1]
            l2 = lines[11].strip().split(',')[0].split(': ')[-1]
            dropout_rate = lines[11].strip().split(',')[1].split(': ')[-1]
            learning_rate = lines[11].strip().split(',')[2].split(': ')[-1]
            epoca = lines[12].strip().split(',')[0].split(': ')[-1]
            batch_size = lines[12].strip().split(',')[1].split(': ')[-1]
            # Escribir la información en la hoja de trabajo
            ws.append([fecha_hora, numeros_jugados, aciertos_resultados, aciertos_cercanos, aciertos_lejanos,
                       aciertos_totales, max_acierto, max_cercanos, max_lejanos, l2, dropout_rate, learning_rate,
                       epoca, batch_size])
# Guardar el archivo Excel
wb.save("reportes.xlsx")
