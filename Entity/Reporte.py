from datetime import datetime
import pandas as pd
import os


class Reporte:
    def __init__(self) -> None:
        pass

    def generar_reporte(
        self,
        contador,
        Parametro_Juego,
        filename,
        filename_reporte,
        numeros_anteriores,
    ):
        fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        juego = getattr(Parametro_Juego, "juego", "No es Simulación")

        # juego = Parametro_Juego.juego if Parametro_Juego.juego else "No es Simulación"
        # Crear un diccionario con los datos
        datos = {
            "Juego": juego,
            "Juego fecha y hora": fecha_hora_actual,
            "Predecidos": contador.jugados,
            "Aciertos Totales": contador.aciertos_totales,
            "Aciertos de Predecidos": contador.acierto_predecidos,
            "V1L": contador.acierto_vecinos_1lugar,
            "V2L": contador.acierto_vecinos_2lugar,
            "V3L": contador.acierto_vecinos_3lugar,
            "V4L": contador.acierto_vecinos_4lugar,
            "Nros a Predecir": Parametro_Juego.numeros_a_predecir,
            "Nros Anteriores": numeros_anteriores,
            "Cant. Vecinos": Parametro_Juego.lugares_vecinos,
            "Limite_tardancia": Parametro_Juego.limite_tardancia,
            "Probabilidad": Parametro_Juego.umbral_probilidad,
            "Efectividad": contador.sacarEfectividad(),
            "Ruleta": filename,
        }

        # Convertir el diccionario en un DataFrame de Pandas
        df = pd.DataFrame([datos])

        archivo_excel = filename_reporte

        # Comprobar si el archivo de Excel ya existe
        if os.path.exists(archivo_excel):
            # Si existe, leerlo y agregar la nueva fila
            df_existente = pd.read_excel(archivo_excel)
            df_final = pd.concat([df_existente, df], ignore_index=True)
        else:
            # Si no existe, usar el DataFrame creado
            df_final = df

        # Guardar el DataFrame en el archivo de Excel
        df_final.to_excel(archivo_excel, index=False)
        return datos
