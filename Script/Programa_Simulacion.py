import os
import itertools
from Entity.Contador import Contador
from Entity.Parametro import Parametro_Juego
from Entity.Predictor import Predictor
from Script.Numeros_Simulacion import Simulador

import pandas as pd


class Parametro_Juego_simulacion:
    def __init__(self):
        self.valores_ficha = [200]
        self.cantidad_vecinos = [0, 1, 2, 3]
        self.limites_juego = [1, 2, 3, 4, 5, 6, 7]
        self.limites_pretendiente = [0, 1]
        self.umbrales_probabilidad = [50, 70, 90, 100]

    def obtener_todas_combinaciones(self):
        return itertools.product(
            self.valores_ficha,
            self.cantidad_vecinos,
            self.limites_juego,
            self.limites_pretendiente,
            self.umbrales_probabilidad,
        )


def leer_combinaciones_procesadas(archivo):
    try:
        df = pd.read_excel(archivo)
        # Asumiendo que tienes columnas para cada parámetro de combinación en tu Excel
        combinaciones_procesadas = df[
            [
                "Valor_ficha",
                "Cant. Vecinos",
                "Limite_juego",
                "Limite_pretendiente",
                "Probabilidad",
            ]
        ].values.tolist()
        return set(tuple(combinacion) for combinacion in combinaciones_procesadas)
    except FileNotFoundError:
        print("Archivo no encontrado. Se procesarán todas las combinaciones.")
        return set()


def main():
    simulador = Simulador()
    nombre_archivo_reporte = "Reportes_simulacion_GRU.xlsx"
    parametros_simulacion = Parametro_Juego_simulacion()
    combinaciones_procesadas = leer_combinaciones_procesadas(nombre_archivo_reporte)

    for combinacion in parametros_simulacion.obtener_todas_combinaciones():
        if combinacion in combinaciones_procesadas:
            print(f"Saltando la combinación ya procesada: {combinacion}")
            continue

        (
            valor_ficha,
            cantidad_vecinos,
            limite_juego,
            limite_pretendiente,
            umbral_probabilidad,
        ) = combinacion
        carpeta = "Data/bombay1.xlsx"

        if not os.path.exists(carpeta):
            print("El archivo Excel no existe. No se puede instanciar el Predictor.")
            continue

        for nombre_juego, lista_numeros in simulador.arrays.items():
            parametro_juego = Parametro_Juego(
                valor_ficha,
                cantidad_vecinos,
                limite_juego,
                limite_pretendiente,
                umbral_probabilidad,
                juego=nombre_juego,
            )
            predictor = Predictor(carpeta, parametro_juego)
            predictor.contador = Contador()
            predictor.Parametro_juego = parametro_juego

            for numero in lista_numeros:
                if numero == "salir":
                    predictor.guardar_excel(False, nombre_archivo_reporte)
                    break
                predictor.verificar_resultados(numero)
                predictor.predecir()
            print(f"ganancia_neta_final: {predictor.contador.ganancia_neta}")
            # Guardar resultados después de cada combinación
            print(
                f"Simulación completada con: J: {parametro_juego.juego}, VF:{parametro_juego.valor_ficha}, CV:{parametro_juego.lugares_vecinos}, LJ:{parametro_juego.limite_juego}, LP:{parametro_juego.limite_pretendiente}, P:{parametro_juego.umbral_probilidad} , G: {predictor.contador.ganancia_neta}"
            )

    print("Finalizo la simulacion de todas las combinaciones")


if __name__ == "__main__":
    main()
