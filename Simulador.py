import pandas as pd
from Config import get_excel_file, get_ruleta_types
from Entity.Predictor import Predictor
from Entity.Parametro import Parametro_Juego, HiperParametros
from Entity.Modelo import Modelo


def simular_juego(predictor, datos_simulacion):
    aciertos = 0
    total_predicciones = 0

    for numero in datos_simulacion:
        predictor.verificar_resultados(numero)
        prediccion = predictor.predecir()
        predictor.actualizar_dataframe(numero)

        if prediccion:
            total_predicciones += 1
            if numero in prediccion:
                aciertos += 1

        print(f"Número ingresado: {numero}")
        print(predictor.mostrar_resultados())
        print(f"Números a jugar: {[n.numero for n in predictor.numeros_a_jugar]}")
        print("---")

    efectividad = aciertos / total_predicciones if total_predicciones > 0 else 0
    return efectividad, predictor.contador


def ejecutar_simulaciones(datos_simulacion):
    tipos_ruleta = get_ruleta_types()
    resultados = []

    for tipo_ruleta in tipos_ruleta:
        filename = get_excel_file(tipo_ruleta)
        print(f"\nSimulando para {tipo_ruleta}:")

        for numeros_anteriores in [3, 4, 5]:
            print(f"  Configuración con {numeros_anteriores} números anteriores")

            for cantidad_vecinos in range(0, 5):
                for limite_juego in range(1, 6):
                    for umbral_probabilidad in range(10, 101, 10):
                        parametros_juego = Parametro_Juego(
                            cantidad_vecinos=cantidad_vecinos,
                            limite_juego=limite_juego,
                            umbral_probabilidad=umbral_probabilidad,
                            num_Anteriores=numeros_anteriores
                        )
                        hiperParametros = HiperParametros(numerosAnteriores=numeros_anteriores)
                        predictor = Predictor(filename, parametros_juego, hiperParametros)

                        efectividad, contador = simular_juego(predictor, datos_simulacion)

                        resultado = {
                            "Ruleta": tipo_ruleta,
                            "Números Anteriores": numeros_anteriores,
                            "Cantidad Vecinos": cantidad_vecinos,
                            "Limite Juego": limite_juego,
                            "Umbral Probabilidad": umbral_probabilidad,
                            "Efectividad": efectividad,
                            "Aciertos Totales": contador.aciertos_totales,
                            "Jugados": contador.jugados,
                            "Aciertos Predecidos": contador.acierto_predecidos,
                            "Aciertos Vecinos 1L": contador.acierto_vecinos_1lugar,
                            "Aciertos Vecinos 2L": contador.acierto_vecinos_2lugar,
                            "Aciertos Vecinos 3L": contador.acierto_vecinos_3lugar,
                            "Aciertos Vecinos 4L": contador.acierto_vecinos_4lugar,
                            "Sin Salir Nada": contador.Sin_salir_nada
                        }
                        resultados.append(resultado)

                        print(f"    Vecinos: {cantidad_vecinos}, Límite: {limite_juego}, "
                              f"Umbral: {umbral_probabilidad}, Efectividad: {efectividad:.2f}")

        predictor.guardar_reporte()
        predictor.guardar_excel()

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel("Resultados_simulacion_optimizada.xlsx", index=False)

    mejores_configuraciones = df_resultados.nlargest(5, 'Efectividad')
    print("\nMejores configuraciones:")
    print(mejores_configuraciones)


if __name__ == "__main__":
    # Array fijo de números para simulación
    datos_simulacion = [
        6, 6, 36, 32, 7, 31, 6, 8, 31, 21, 16, 23, 21, 26, 6, 32, 11, 35, 16, 21,
        24, 4, 17, 4, 5, 30, 2, 33, 29, 24, 24, 15, 3, 13, 8, 24, 36, 26, 22, 4,
        5, 28, 25, 14, 2, 14, 16, 36, 19, 22, 31, 33, 36, 20, 25, 19, 12, 8, 31, 19,
        29, 1, 10, 32, 22, 28, 34, 23, 35, 14, 20, 30, 36, 8, 5, 14, 18, 13, 29, 21,
        18, 1, 35, 13, 12, 14, 34, 0, 8, 12, 13, 9, 0
    ]

    ejecutar_simulaciones(datos_simulacion)
