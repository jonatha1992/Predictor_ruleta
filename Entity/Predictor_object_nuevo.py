import numpy as np
import pandas as pd
import tensorflow as tf
import os
from Entity.Contador import Contador
from Entity.Modelo import Modelo
from Entity.Numero import Numero_pretendiente, Numero_jugar
from datetime import datetime
from Entity.Parametro import HiperParametros, Parametro_Juego
from Entity.Vecinos import vecino1lugar, vecino2lugar, vecinos3lugar, Vecino4lugar
from tensorflow.keras.models import load_model
from Entity.Reporte import Reporte


class Predictor:
    # Inicializa el objeto de la clase con un nombre de archivo y crea el modelo.
    def __init__(self, filename, valor_ficha, cantidad_vecinos, Limite):
        self.Parametro = Parametro_Juego(
            valor_ficha, cantidad_vecinos=cantidad_vecinos, limite=Limite
        )
        self.filename = filename
        self.contador = Contador()
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        self.contador.numeros = self.df["Salidos"].values.tolist()
        self.hiperparametros = HiperParametros(len(self.contador.numeros))
        
        Modelo(self.filename, self.hiperparametros)
        self.reporte = Reporte()
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]
        self.model = load_model("./Models/Model_" + self.filebasename)



        self.numeros_a_jugar = list()
        self.numeros_pretendientes = list()
        self.numeros_predecidos = list()
        self.no_salidos = list()

        self.df_nuevo = self.df.copy()

    def predecir(self):
        if self.contador.ingresados > self.Parametro.numerosAnteriores:
            secuencia_entrada = np.array(
                self.contador.numeros[-self.Parametro.numerosAnteriores :]
            ).reshape(1, self.Parametro.numerosAnteriores, 1)
            predicciones = self.model.predict(secuencia_entrada, verbose=0)

            # Filtrar predicciones basadas en el umbral de probabilidad

            predicciones_filtradas = [
                i
                for i, probabilidad in enumerate(predicciones[0])
                if probabilidad > 0.10
            ]
            # Ordenar las predicciones filtradas por probabilidad descendente
            predecidos = sorted(
                predicciones_filtradas, key=lambda i: predicciones[0][i], reverse=True
            )

            for numero_jugado in self.numeros_a_jugar:
                numero_jugado.Jugar()
            for Numero_pretendiente in self.numeros_pretendientes:
                Numero_pretendiente.Jugar()

            self.verificar_predecidos(predicciones, predecidos)

            self.verificar_pretendientes()

    def verificar_pretendientes(self):
        for pretendiente in self.numeros_pretendientes:
            if pretendiente.probabilidad >= self.Parametro.umbral_probilidad:
                new_numero = Numero_jugar(
                    pretendiente.numero,
                    pretendiente.probabilidad,
                    self.Parametro.valor_ficha,
                    self.Parametro.lugares_vecinos,
                )
                self.numeros_a_jugar.append(new_numero)
                self.numeros_pretendientes.remove(pretendiente)
                self.contador.incrementar_jugados()

            # Ordena los numeros_a_jugar2 por probabilidad descendente
        self.numeros_a_jugar = sorted(
            self.numeros_a_jugar, key=lambda x: x.probabilidad, reverse=True
        )
        self.numeros_pretendientes = sorted(
            self.numeros_pretendientes, key=lambda x: x.probabilidad, reverse=True
        )

    def verificar_predecidos(self, predicciones, predecidos):
        for num in predecidos:
            probabilidad_redondeada = int(round(predicciones[0][num], 2) * 100)
            if any(n.numero == num for n in self.numeros_a_jugar):
                for pretendiente in self.numeros_a_jugar:
                    if pretendiente.numero == num:
                        pretendiente.Aumentar_probailidad(probabilidad_redondeada)
            else:
                nuevo_pretendiente = Numero_pretendiente(num, probabilidad_redondeada)
                if any(n.numero == num for n in self.numeros_pretendientes):
                    for pretendiente in self.numeros_pretendientes:
                        if pretendiente.numero == num:
                            pretendiente.Aumentar_probailidad(probabilidad_redondeada)
                else:
                    self.numeros_pretendientes.append(nuevo_pretendiente)

    def verificar_resultados(self, numero):
        acierto = False
        es_vecino1lugar = False
        es_vecino2lugar = False
        es_vecino3lugar = False
        es_vecino4lugar = False

        self.numeros_predecidos = list()
        self.no_salidos = []
        self.contador.incrementar_ingresados(numero)

        if len(self.numeros_a_jugar) > 0:
            if any(
                n.numero == numero and n.probabilidad > 0 for n in self.numeros_a_jugar
            ):
                # Acertó un número en numeros_a_jugar2
                numero_acertado = next(
                    n for n in self.numeros_a_jugar if n.numero == numero
                )
                self.numeros_predecidos.append(numero_acertado)
                self.contador.incrementar_predecidos()
                self.df_nuevo.at[len(self.df_nuevo), "Acierto"] = "P"
                acierto = True

            for vecino in self.numeros_a_jugar:
                if (
                    numero in vecino1lugar[vecino.numero]
                    and self.Parametro.lugares_vecinos >= 1
                    and vecino.probabilidad > 0
                ):
                    if vecino not in self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_1lugar()
                        es_vecino1lugar = True

                if (
                    numero in vecino2lugar[vecino.numero]
                    and self.Parametro.lugares_vecinos >= 2
                    and vecino.probabilidad > 0
                ):
                    if vecino not in self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_2lugar()
                        es_vecino2lugar = True

                if (
                    numero in vecinos3lugar[vecino.numero]
                    and self.Parametro.lugares_vecinos >= 3
                    and vecino.probabilidad > 0
                ):
                    if vecino not in self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_3lugar()
                        es_vecino3lugar = True

                if (
                    numero in Vecino4lugar[vecino.numero]
                    and self.Parametro.lugares_vecinos >= 4
                    and vecino.probabilidad > 0
                ):
                    if vecino not in self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_4lugar()
                        es_vecino4lugar = True

            # Filtra los números predecidos de la lista numeros_a_jugar2
            self.numeros_a_jugar = [
                n for n in self.numeros_a_jugar if n not in self.numeros_predecidos
            ]

            for x in self.numeros_predecidos:
                x.Pego()
                self.contador.incrementar_ganancias_totales(x.ganancia_neta)

            self.Verificar_limites_numeros()

            if es_vecino1lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V1L"] = "V1L"

            if es_vecino2lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V2L"] = "V2L"

            if es_vecino3lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V3L"] = "V3L"

            if es_vecino4lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V4L"] = "V4L"

            if len(self.numeros_predecidos) > 0:
                self.contador.incrementar_aciertos_totales(len(self.numeros_predecidos))

    def Verificar_limites_numeros(self):
        for obj in self.numeros_a_jugar[:]:
            if obj.tardancia == self.Parametro.limite:
                self.no_salidos.append(obj.numero)
                self.numeros_a_jugar.remove(obj)
                self.contador.incrementar_supero_limite(obj.jugado)

        for x in self.numeros_pretendientes[:]:
            if x.tardancia == self.Parametro.limite:
                self.numeros_pretendientes.remove(x)

    # Actualiza el DataFrame con el número ingresado y los resultados de las predicciones.
    def actualizar_dataframe(self, numero_ingresado):
        self.df_nuevo.loc[len(self.df_nuevo) + 1, "Salidos"] = numero_ingresado

        # Imprime para asegurarte de que self.numeros_a_jugar contiene instancias de Numero_pretendiente
        # Convierte la lista de instancias a una cadena única
        resultados_str = ",".join([str(obj) for obj in self.numeros_a_jugar])
        predecidos_str = ",".join([str(obj) for obj in self.numeros_predecidos])
        self.df_nuevo.loc[len(self.df_nuevo), "Resultados"] = resultados_str
        self.df_nuevo.loc[len(self.df_nuevo), "Acertados"] = predecidos_str
        self.df_nuevo.loc[len(self.df_nuevo), "No salidos"] = str(self.no_salidos)
        self.df_nuevo.loc[len(self.df_nuevo), "Ganancia_neta"] = str(
            self.contador.Calcular_ganancia()
        )
        self.df_nuevo.loc[len(self.df_nuevo), "Orden"] = self.contador.ingresados

    # Guarda el DataFrame en un archivo de Excel.
    def guardar_excel(self):
        self.reporte.generar_reporte(self.contador, self.hiperparametros, self.Parametro, self.filename)
        self.df_nuevo.to_excel(self.filename, sheet_name="Salidos", index=False)

    # Muestra los resultados y las estadísticas.
    def mostrar_resultados(self):
        print("\nTabla de resultados:")
        print(self.df_nuevo.tail(3))
        print(f"Numeros Jugados: {self.contador.jugados}")
        print(f"Aciertos Totales: {self.contador.aciertos_totales}")
        print(f"Sin salir: {self.contador.Sin_salir_nada}")
        print(f"Ganancia_neta: {self.contador.ganancia_neta}\n")

        for e in self.numeros_predecidos:
            print(
                f"El Número {e.numero} fue acertado de la lista de predecidos ganancia neta {e.ganancia_neta}."
            )

        if len(self.no_salidos) > 0:
            print(f"Eliminados por superar el limite: {self.no_salidos}")

        if len(self.numeros_a_jugar) > 0:
            print("\nLas posibles predicciones para el próximo número son:")
            for x in self.numeros_a_jugar:
                print(x)

    # Borra el último número ingresado y actualiza el contador.
    def borrar(self):
        if self.contador.numeros:
            self.contador.borrar_ultimo_numero()
            ultimo = self.contador.numeros[-1]
            self.df_nuevo = self.df_nuevo[
                :-1
            ]  # Eliminar la última fila del DataFrame nuevo

            if len(self.numeros_a_jugar) > 0:
                for key in self.numeros_a_jugar:
                    self.numeros_a_jugar[key] -= 1

        print(f"Último número borrado {ultimo}")