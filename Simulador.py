import pandas as pd
from Config import get_excel_file, get_ruleta_types
from Entity.Predictor import Predictor
from Entity.Modelo import Modelo
from Entity.Parametro import Parametro_Juego, HiperParametros


def crear_modelos():
    tipos_ruleta = get_ruleta_types()
    for tipo_ruleta in tipos_ruleta:
        filename = get_excel_file(tipo_ruleta)
        hiperparametros = HiperParametros()
        modelo = Modelo(filename, hiperparametros)
        modelo.crear_y_guardar_modelos()


def simular_juego(predictor, datos_simulacion):
    for numero in datos_simulacion:
        predictor.verificar_resultados(numero)
        predictor.predecir()
    return predictor.contador


def ejecutar_simulaciones(datos_simulacion):
    tipos_ruleta = get_ruleta_types()
    resultados = []

    for tipo_ruleta in tipos_ruleta:
        filename = get_excel_file(tipo_ruleta)
        print(f"\nSimulando para {tipo_ruleta}:")

        for numeros_anteriores in [4]:
            print(f"  Configuración con {numeros_anteriores} números anteriores")

            for cantidad_vecinos in range(0, 5):
                for limite_juego in range(1, 6):
                    for umbral_probabilidad in range(10, 101, 10):
                        parametros_juego = Parametro_Juego(
                            cantidad_vecinos=cantidad_vecinos,
                            limite_juego=limite_juego,
                            umbral_probabilidad=umbral_probabilidad
                        )
                        hiperParametros = HiperParametros(numerosAnteriores=numeros_anteriores)
                        predictor = Predictor(filename, parametros_juego, hiperParametros)

                        contador = simular_juego(predictor, datos_simulacion[tipo_ruleta])
                        efectividad = contador.sacarEfectividad()
                        resultado = {
                            "Ruleta": tipo_ruleta,
                            "Números Anteriores": numeros_anteriores,
                            "Cantidad Vecinos": cantidad_vecinos,
                            "Limite Juego": limite_juego,
                            "Umbral Probabilidad": umbral_probabilidad,
                            "Aciertos Totales": contador.aciertos_totales,
                            "Jugados": contador.jugados,
                            "Efectividad": efectividad,
                            "Aciertos Predecidos": contador.acierto_predecidos,
                            "Aciertos Vecinos 1L": contador.acierto_vecinos_1lugar,
                            "Aciertos Vecinos 2L": contador.acierto_vecinos_2lugar,
                            "Aciertos Vecinos 3L": contador.acierto_vecinos_3lugar,
                            "Aciertos Vecinos 4L": contador.acierto_vecinos_4lugar,
                            "Sin Salir Nada": contador.Sin_salir_nada
                        }
                        resultados.append(resultado)
                        # print(
                        #     f" Números Anteriores: {numeros_anteriores}, Predicidos: {contador.jugados}, Aciertos: {contador.acierto_predecidos}, Efectividad: {efectividad:.2f}. "
                        #     f"Vecinos: {cantidad_vecinos}, Límite: {limite_juego}, Umbral: {umbral_probabilidad}")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel("Resultados_simulacion_optimizada.xlsx", index=False)

    mejores_configuraciones = df_resultados.nlargest(5, 'Efectividad')
    print("\nMejores configuraciones:")
    print(mejores_configuraciones)


if __name__ == "__main__":
    datos_simulacion = {
        "Electromecanica": [
            6, 6, 36, 32, 7, 31, 6, 8, 31, 21, 16, 23, 21, 26, 6, 32, 11, 35, 16, 21,
            24, 4, 17, 4, 5, 30, 2, 33, 29, 24, 24, 15, 3, 13, 8, 24, 36, 26, 22, 4,
            5, 28, 25, 14, 2, 14, 16, 36, 19, 22, 31, 33, 36, 20, 25, 19, 12, 8, 31, 19,
            29, 1, 10, 32, 22, 28, 34, 23, 35, 14, 20, 30, 36, 8, 5, 14, 18, 13, 29, 21,
            18, 1, 35, 13, 12, 14, 34, 0, 8, 12, 13, 9, 0
        ],
        "Crupier": [
            36, 18, 33, 20, 24, 34, 36, 15, 17, 25, 26, 9, 36, 8, 19, 23, 1, 36, 17, 1,
            34, 31, 29, 11, 9, 23, 14, 1, 10, 6, 35, 18, 33, 32, 26, 23, 2, 19, 1, 6,
            17, 16, 30, 27, 18, 10, 9, 7, 10, 35, 25, 31, 33, 17, 3, 14, 9, 31, 6, 0,
            32, 34, 0, 18, 22, 0, 20, 8, 36, 12, 35, 8, 13, 20, 7, 23, 5, 17, 5, 22,
            17, 35, 26, 20, 27, 13, 17, 15, 22, 26, 30, 32, 4, 33, 32, 0, 31, 26, 21, 2,
            10, 15, 24, 18, 18, 23, 18
        ]
    }

    crear_modelos()
    ejecutar_simulaciones(datos_simulacion)
