import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from Entity.Contador import Contador
from Entity.Modelo import Modelo
from Entity.Numero import NumeroJugar, NumeroHistorial  # Asegúrate de que las clases estén en módulos adecuados
from datetime import datetime
from Entity.Parametro import HiperParametros, Parametro_Juego
from Entity.Vecinos import vecino1lugar, vecino2lugar, vecinos3lugar, Vecino4lugar, colores_ruleta
from Entity.Reporte import Reporte
from Config import get_relative_path
import pprint
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Predictor:
    def __init__(self, filename: str, parametro_juego: Parametro_Juego, hiperparametros: HiperParametros, **kwargs):
        self.filename = filename
        self.parametro_juego = parametro_juego
        self.contador = Contador()
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        self.contador.numeros = self.df["Salidos"].tolist()
        self.hiperparametros = hiperparametros
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]

        modelo_nombre = f"Model_{self.filebasename}_N{self.hiperparametros.numerosAnteriores}"
        modelo_path = get_relative_path(f"./Models/{modelo_nombre}.keras")

        if not os.path.exists(modelo_path):
            modelo = Modelo(filename, hiperparametros)
            modelo.crear_y_guardar_modelos()

        self.model = tf.keras.models.load_model(modelo_path)
        self.numeros_a_jugar = {}  # Usar diccionario para eficiencia
        self.numeros_predecidos = []
        self.historial_predecidos = {}
        self.no_salidos = {}
        self.df_nuevo = self.df.copy()

    def verificar_resultados(self, numero: int):
        """
        Verifica si el número ingresado es un acierto o un vecino.
        """
        acierto = False
        es_vecino1lugar = False
        es_vecino2lugar = False
        es_vecino3lugar = False
        es_vecino4lugar = False

        self.numeros_predecidos = []
        self.contador.incrementar_ingresados(numero)
        self.no_salidos = {}

        if self.numeros_a_jugar:
            # Verificar aciertos directos
            if numero in self.numeros_a_jugar and self.numeros_a_jugar[numero].probabilidad > 0:
                numero_acertado = self.numeros_a_jugar.pop(numero)
                self.numeros_predecidos.append(numero_acertado)
                self.contador.incrementar_predecidos()
                self.df_nuevo.at[len(self.df_nuevo), "Acierto"] = "P"
                acierto = True

                # Eliminar también del historial ya que el número salió
                if numero in self.historial_predecidos:
                    del self.historial_predecidos[numero]

            # Verificar vecinos
            for vecino_numero, vecino_obj in list(self.numeros_a_jugar.items()):
                if (numero in vecino1lugar.get(vecino_numero, []) and self.parametro_juego.lugares_vecinos >= 1 and vecino_obj.probabilidad > 0):
                    self.numeros_predecidos.append(vecino_obj)
                    self.contador.incrementar_aciertos_vecinos_1lugar()
                    es_vecino1lugar = True

                if (numero in vecino2lugar.get(vecino_numero, []) and self.parametro_juego.lugares_vecinos >= 2 and vecino_obj.probabilidad > 0):
                    self.numeros_predecidos.append(vecino_obj)
                    self.contador.incrementar_aciertos_vecinos_2lugar()
                    es_vecino2lugar = True

                if (numero in vecinos3lugar.get(vecino_numero, []) and self.parametro_juego.lugares_vecinos >= 3 and vecino_obj.probabilidad > 0):
                    self.numeros_predecidos.append(vecino_obj)
                    self.contador.incrementar_aciertos_vecinos_3lugar()
                    es_vecino3lugar = True

                if (numero in Vecino4lugar.get(vecino_numero, []) and self.parametro_juego.lugares_vecinos >= 4 and vecino_obj.probabilidad > 0):
                    self.numeros_predecidos.append(vecino_obj)
                    self.contador.incrementar_aciertos_vecinos_4lugar()
                    es_vecino4lugar = True

                # Eliminar el vecino si fue predecido
                if vecino_obj in self.numeros_predecidos:
                    del self.numeros_a_jugar[vecino_numero]

                    # Si este vecino también estaba en el historial, eliminarlo
                    if vecino_numero in self.historial_predecidos:
                        del self.historial_predecidos[vecino_numero]

            # Actualizar columnas de vecinos en el DataFrame
            if es_vecino1lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V1L"] = "V1L"
            if es_vecino2lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V2L"] = "V2L"
            if es_vecino3lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V3L"] = "V3L"
            if es_vecino4lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V4L"] = "V4L"

        # self.verificar_limites_numeros()

        if self.numeros_predecidos:
            self.contador.incrementar_aciertos_totales(len(self.numeros_predecidos))

    def verificar_limites_numeros(self):
        """
        Verifica y elimina números que han alcanzado el límite de tardancia.
        """
        objetos_a_eliminar = [num for num, obj in self.numeros_a_jugar.items() if obj.tardancia >= self.parametro_juego.limite_juego]

        for num in objetos_a_eliminar:
            obj = self.numeros_a_jugar.pop(num)
            self.no_salidos[num] = obj
            self.contador.incrementar_supero_limite()

    def actualizar_dataframe(self, numero_ingresado: int):
        """
        Actualiza el DataFrame con el número ingresado y los resultados actuales.
        """
        nueva_fila = {
            "Salidos": numero_ingresado,
            "Resultados": ",".join([str(obj) for obj in self.numeros_a_jugar.values()]),
            "Acertados": ",".join([str(obj) for obj in self.numeros_predecidos]),
            "No salidos": ",".join([str(obj) for obj in self.no_salidos.values()]),
            "Orden": self.contador.ingresados
        }
        # self.df_nuevo = self.df_nuevo.append(nueva_fila, ignore_index=True)  # type: ignore
        self.df_nuevo = pd.concat([self.df_nuevo, pd.DataFrame([nueva_fila])], ignore_index=True)

    def guardar_reporte(self):
        """
        Guarda el reporte en un archivo Excel.
        """
        reporte = Reporte()
        reporte.generar_reporte(
            self.contador,
            self.parametro_juego,
            self.filename,
            filename_reporte="Data/Reporte_juego.xlsx",
            numeros_anteriores=self.hiperparametros.numerosAnteriores,
        )

    def guardar_excel(self):
        """
        Guarda el DataFrame actualizado en el archivo Excel original.
        """
        self.df_nuevo.to_excel(self.filename, sheet_name="Salidos", index=False)

    def mostrar_resultados(self) -> str:
        """
        Muestra los resultados de los números predecidos y no salidos.
        """
        resultados = []
        for e in self.numeros_predecidos:
            resultados.append(
                f"Núm {e.numero} fue ACERTADO, Probabilidad: {e.probabilidad}, Tardancia: {e.tardancia}, Repetidos: {e.repetido}"
            )

        if self.no_salidos:
            for e in self.no_salidos.values():
                resultados.append(
                    f"Núm {e.numero} NO SALIÓ, Probabilidad: {e.probabilidad}, Tardancia: {e.tardancia}, Repetidos: {e.repetido}"
                )

        return "\n".join(resultados)

    def borrar(self):
        """
        Borra el último número ingresado y actualiza las listas y el DataFrame.
        """
        if self.contador.numeros:
            self.contador.borrar_ultimo_numero()
            ultimo = self.contador.numeros[-1]
            self.df_nuevo = self.df_nuevo[:-1]  # Eliminar la última fila del DataFrame nuevo

            if self.numeros_a_jugar:
                self.numeros_a_jugar.clear()

            print(f"Último número borrado {ultimo}")

    def predecir(self):
        """
        Genera predicciones a partir del modelo, actualiza el historial y los números a jugar.
        """
        if self.contador.ingresados > self.hiperparametros.numerosAnteriores:
            # Crear la secuencia de entrada para el modelo
            secuencia_entrada = np.array(
                self.contador.numeros[-self.hiperparametros.numerosAnteriores:]
            ).reshape(1, self.hiperparametros.numerosAnteriores, 1)
            predicciones = self.model.predict(secuencia_entrada, verbose=0)

            # Convertir las predicciones en un formato manejable
            predecidos = [
                {
                    "numero": i,
                    "probabilidad": prob
                }
                for i, prob in enumerate(predicciones[0])
            ]

            # Redondear probabilidades y procesar los números
            for pred in predecidos:
                pred["probabilidad"] = int(round(pred["probabilidad"], 2) * 100)

            # Marcar los números actuales como jugados
            for numero_jugado in self.numeros_a_jugar.values():
                numero_jugado.jugar()

            # Ordenar historial_predecidos por probabilidad descendente
            self.historial_predecidos = dict(sorted(
                self.historial_predecidos.items(),
                key=lambda item: item[1].numero,
                reverse=True
            ))

            if self.historial_predecidos:
                print("Historial antes:")
                for num in self.historial_predecidos.values():
                    print(f"numero {num.numero}, probabilidad {num.probabilidad}")
                    # Actualizar el historial con las predicciones actuales

            self.actualizar_historial(predecidos)

            # Verificar y actualizar números a jugar según las predicciones y el historial
            self.verificar_historial()
            self.verificar_probabilidad_cero(predecidos)

            print("Predicciones:")
            pprint.pprint(predecidos)

            self.historial_predecidos = dict(sorted(
                self.historial_predecidos.items(),
                key=lambda item: item[1].numero,
                reverse=True
            ))

            # Ordenar números a jugar por probabilidad descendente
            self.numeros_a_jugar = dict(sorted(
                self.numeros_a_jugar.items(),
                key=lambda item: item[1].numero,
                reverse=True
            ))

            print("Historial posterior:")
            for num in self.historial_predecidos.values():
                print(f"numero {num.numero}, probabilidad {num.probabilidad}")

    def actualizar_historial(self, predecidos: list):
        """
        Actualiza el historial con las predicciones.
        Si ya existen en el historial, suma la probabilidad.
        Si son nuevos y probabilidad > 0, los agrega.
        """
        for pred in predecidos:
            num = pred["numero"]
            prob = pred["probabilidad"]

            if num in self.historial_predecidos:
                # Si ya está en el historial, sumar la probabilidad
                self.historial_predecidos[num].aumentar_probabilidad(prob)
            else:
                # Agregar al historial si probabilidad > 0
                if prob > 0:
                    self.historial_predecidos[num] = NumeroHistorial(
                        numero=num,
                        probabilidad=prob
                    )

    def verificar_historial(self):
        """
        Verifica los números en el historial y agrega los que superan el umbral a numeros_a_jugar.
        """
        umbral = self.parametro_juego.umbral_probilidad

        for num, historial_numero in self.historial_predecidos.items():
            prob = historial_numero.probabilidad

            if prob >= umbral:
                if num not in self.numeros_a_jugar:
                    # Agregar al numeros_a_jugar si no está presente
                    self.numeros_a_jugar[num] = NumeroJugar(
                        numero=num,
                        probabilidad=prob,
                        vecinos=self.parametro_juego.lugares_vecinos
                    )
                    self.contador.incrementar_jugados()
                else:
                    # Actualizar la probabilidad si ya está en numeros_a_jugar
                    self.numeros_a_jugar[num].probabilidad = prob

        logging.debug(f"Estado final de numeros_a_jugar: {[f'{num}: {obj.probabilidad}' for num, obj in self.numeros_a_jugar.items()]}")

    def verificar_probabilidad_cero(self, predecidos: list):
        """
        Verifica los números que ya están en numeros_a_jugar o historial.
        Si aparecen con probabilidad 0, los elimina de ambas listas.
        """
        logging.debug("Iniciando verificación de probabilidad cero.")
        numeros_con_prob_0 = {p["numero"] for p in predecidos if p["probabilidad"] == 0}
        logging.debug(f"Números con probabilidad 0 identificados: {numeros_con_prob_0}")

        for num in list(self.numeros_a_jugar.keys()):
            if num in numeros_con_prob_0:
                obj = self.numeros_a_jugar.pop(num)
                self.no_salidos[num] = obj
                self.contador.incrementar_supero_limite()
                logging.debug(f"Número {num} movido a no_salidos.")

        for num in list(self.historial_predecidos.keys()):
            if num in numeros_con_prob_0:
                del self.historial_predecidos[num]
                logging.debug(f"Número {num} eliminado del historial.")

        logging.debug("Finalizada verificación de probabilidad cero.")
