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
        self.numeros_acertados = []
        self.historial_predecidos = []
        self.numeros_a_jugar = []

        self.no_salidos = {}
        self.df_nuevo = self.df.copy()

    def verificar_limites_numeros(self):
        """
        Verifica y elimina números que han alcanzado el límite de tardancia.
        """
        numeros_a_eliminar = [num for num in self.numeros_a_jugar if num.tardancia >= self.parametro_juego.limite_tardancia]
        print(f"Números a eliminar por tardancia: {[num.numero for num in numeros_a_eliminar]}")

        for num in numeros_a_eliminar:
            self.numeros_a_jugar.remove(num)
            self.no_salidos[num.numero] = num
            self.contador.incrementar_supero_limite()
            print(f"Número {num.numero} eliminado y movido a no_salidos.")

        print(f"Números a jugar después de verificar límites: {[num.numero for num in self.numeros_a_jugar]}")

    def actualizar_dataframe(self, numero_ingresado: int):
        """
        Actualiza el DataFrame con el número ingresado y los resultados actuales.
        """
        nueva_fila = {
            "Orden": self.contador.ingresados,
            "Salidos": numero_ingresado,
            "Resultados": ",".join([str(obj) for obj in self.numeros_a_jugar]),
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
        for e in self.numeros_acertados:
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

                # Filtrar y mostrar solo predicciones con probabilidad mayor a 2
            predicciones_filtradas = [pred for pred in predecidos if pred["probabilidad"] > 0]
            print("Predicciones con probabilidad mayor a 0:")
            pprint.pprint(predicciones_filtradas)

            # Marcar los números actuales como jugados
            for numero_jugado in self.numeros_a_jugar:
                numero_jugado.jugar()

            self.historial_predecidos.sort(key=lambda x: x.numero, reverse=True)

            if self.historial_predecidos:
                print("Historial antes:")
                for num in self.historial_predecidos:
                    print(f"numero {num.numero}, probabilidad {num.probabilidad}")
                    # Actualizar el historial con las predicciones actuales

            self.actualizar_historial(predecidos)

            # Verificar y actualizar números a jugar según las predicciones y el historial
            self.verificar_historial()
            self.verificar_probabilidad_cero(predecidos)

            self.historial_predecidos.sort(key=lambda x: x.numero, reverse=True)

            # Ordenar números a jugar por probabilidad descendente
            self.numeros_a_jugar.sort(key=lambda x: x.numero, reverse=True)

            print("Historial posterior:")
            for num in self.historial_predecidos:
                print(f"numero {num.numero}, probabilidad {num.probabilidad}")

    def actualizar_historial(self, predecidos: list):
        for pred in predecidos:
            num = pred["numero"]
            prob = pred["probabilidad"]

            # Buscar en ambas listas
            num_existente = None
            for nh in self.historial_predecidos:
                if nh.numero == num:
                    num_existente = nh
                    break

            if not num_existente:
                for nj in self.numeros_a_jugar:
                    if nj.numero == num:
                        num_existente = nj
                        break

            if num_existente:
                num_existente.actualizar_probabilidad(prob)
            else:
                if prob > 0:
                    self.historial_predecidos.append(
                        NumeroHistorial(numero=num, probabilidad=prob)
                    )

    def verificar_historial(self):
        umbral = self.parametro_juego.umbral_probilidad
        for numero in self.historial_predecidos[:]:
            if numero.probabilidad >= umbral:
                self.historial_predecidos.remove(numero)
                nuevo_numero = NumeroJugar(numero=numero.numero, probabilidad=numero.probabilidad, repetido=numero.repetido)
                self.numeros_a_jugar.append(nuevo_numero)
                self.contador.incrementar_jugados()

    def verificar_probabilidad_cero(self, predecidos: list):
        """
        Verifica los números que ya están en numeros_a_jugar o historial.
        Si aparecen con probabilidad 0, los elimina de ambas listas.
        """
        logging.debug("Iniciando verificación de probabilidad cero.")
        numeros_con_prob_0 = {p["numero"] for p in predecidos if p["probabilidad"] <= 1}
        logging.debug(f"Números con probabilidad menor e igual a 1 identificados: {numeros_con_prob_0}")

        for item in self.numeros_a_jugar[:]:
            if item.numero in numeros_con_prob_0:
                self.numeros_a_jugar.remove(item)
                self.no_salidos[item.numero] = item
                self.contador.incrementar_supero_limite()
                logging.debug(f"Número {item.numero} movido a no_salidos.")

        for item in self.historial_predecidos[:]:
            if item.numero in numeros_con_prob_0:
                self.historial_predecidos.remove(item)

        logging.debug(f"Número {item.numero} eliminado del historial.")
        logging.debug("Finalizada verificación de probabilidad cero.")

    def verificar_resultados(self, numero_salido: int):
        # Obtener el índice de la última fila del DataFrame
        indice_actual = len(self.df_nuevo) - 1

        # Variables para marcar vecinos acertados
        es_vecino1lugar = False
        es_vecino2lugar = False
        es_vecino3lugar = False
        es_vecino4lugar = False

        self.numeros_acertados = []
        self.contador.incrementar_ingresados(numero_salido)
        self.no_salidos = {}

        numeros_a_eliminar = set()

        if self.numeros_a_jugar:
            # Buscar el índice del número en la lista si probabilidad > 0
            indice_a_jugar = None
            for i, n in enumerate(self.numeros_a_jugar):
                if n.numero == numero_salido and n.probabilidad > 0:
                    indice_a_jugar = i
                    break

            if indice_a_jugar is not None:
                numero_acertado = self.numeros_a_jugar.pop(indice_a_jugar)
                self.numeros_acertados.append(numero_acertado)
                self.contador.incrementar_predecidos()
                acierto = True

                # Quitar también de historial_predecidos si está
                indice_historial = None
                for j, h in enumerate(self.historial_predecidos):
                    if h.numero == numero_salido:
                        indice_historial = j
                        break
                if indice_historial is not None:
                    self.historial_predecidos.pop(indice_historial)

            # Recolectar vecinos a eliminar
            vecinos = []
            if self.parametro_juego.lugares_vecinos >= 1:
                vecinos.extend(vecino1lugar.get(numero_salido, []))
                if vecinos:
                    es_vecino1lugar = True
            if self.parametro_juego.lugares_vecinos >= 2:
                vecinos.extend(vecino2lugar.get(numero_salido, []))
                if vecinos:
                    es_vecino2lugar = True
            if self.parametro_juego.lugares_vecinos >= 3:
                vecinos.extend(vecinos3lugar.get(numero_salido, []))
                if vecinos:
                    es_vecino3lugar = True
            if self.parametro_juego.lugares_vecinos >= 4:
                vecinos.extend(Vecino4lugar.get(numero_salido, []))
                if vecinos:
                    es_vecino4lugar = True

            # Agregar vecinos al conjunto de eliminación
            numeros_a_eliminar.update(vecinos)

            # Eliminar números de `numeros_a_jugar` una sola vez
            for vecino_numero in numeros_a_eliminar:
                vecino_obj = next((n for n in self.numeros_a_jugar if n.numero == vecino_numero), None)
                if vecino_obj:
                    self.numeros_a_jugar.remove(vecino_obj)
                    self.numeros_acertados.append(vecino_obj)
                    # Incrementar contadores según el nivel de vecino
                    if vecino_numero in vecino1lugar.get(numero_salido, []):
                        self.contador.incrementar_aciertos_vecinos_1lugar()
                    if vecino_numero in vecino2lugar.get(numero_salido, []):
                        self.contador.incrementar_aciertos_vecinos_2lugar()
                    if vecino_numero in vecinos3lugar.get(numero_salido, []):
                        self.contador.incrementar_aciertos_vecinos_3lugar()
                    if vecino_numero in Vecino4lugar.get(numero_salido, []):
                        self.contador.incrementar_aciertos_vecinos_4lugar()

            self.verificar_limites_numeros()

        # Actualizar la fila existente
        self.df_nuevo.at[indice_actual, "Acierto"] = "P" if 'acierto' in locals() and acierto else ""
        self.df_nuevo.at[indice_actual, "V1L"] = "V1L" if es_vecino1lugar else ""
        self.df_nuevo.at[indice_actual, "V2L"] = "V2L" if es_vecino2lugar else ""
        self.df_nuevo.at[indice_actual, "V3L"] = "V3L" if es_vecino3lugar else ""
        self.df_nuevo.at[indice_actual, "V4L"] = "V4L" if es_vecino4lugar else ""
        self.df_nuevo.at[indice_actual, "Resultados"] = ",".join([str(obj.numero) for obj in self.numeros_acertados])
        self.df_nuevo.at[indice_actual, "Acertados"] = ",".join([str(obj.numero) for obj in self.numeros_acertados])
        self.df_nuevo.at[indice_actual, "No salidos"] = ",".join([str(obj.numero) for obj in self.no_salidos.values()])

        if self.numeros_acertados:
            self.contador.incrementar_aciertos_totales(len(self.numeros_acertados))
