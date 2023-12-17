import numpy as np
import pandas as pd
import tensorflow as tf
import os
from Entity.Contador import Contador
from Entity.Numeros_Simulacion import Simulador
from datetime import datetime
from Entity.Vecinos import vecino1lugar, vecino2lugar, vecinos3lugar, Vecino4lugar
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Nadam, Adadelta
from tensorflow.keras.callbacks import EarlyStopping


class Predictor:
    # Inicializa el objeto de la clase con un nombre de archivo y crea el modelo.
    def __init__(self, filename):
        self.filename = filename
        self.foldername = os.path.dirname(filename)
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]

        self.nombreModelo = "Model_" + self.filebasename
        self.df = pd.read_excel(filename, sheet_name="Salidos")

        self.contador = Contador()
        self.contador.numeros = self.df["Salidos"].values.tolist()

        self.resultados = dict()
        self.numeros_predecidos = list()
        self.no_salidos = list()

        # Parametros
        self.numerosAnteriores = 7
        self.numeros_a_predecir = 10
        self.lsmt = 352
        self.gru = 256
        self.lsmt2 = 128
        self.l2_lambda = 0.001
        self.dropout_rate = 0.01
        self.learning_rate = 0.003  # Tasa de aprendizaje inicial
        self.epoc = 100
        self.batchSize = 512
        self.umbral_probilidad = 0.7

        # Ruta relativa a la carpeta "modelo" en el mismo directorio que tu archivo de código
        modelo_path = "./Models/" + self.nombreModelo

        if os.path.exists(modelo_path):  # Verifica si ya hay un modelo guardado
            self.model = load_model(modelo_path)  # Carga el modelo guardado si existe
        else:
            self.model = self._crear_modelo()
            self.guardar_modelo()  # Guarda el modelo después de entrenarlo

        self.df_nuevo = self.df.copy()

    # Crea el modelo de red neuronal LSTM.
    def _crear_modelo(self):
        secuencias, siguientes_numeros = self._crear_secuencias()
        model = Sequential()

        model.add(
            LSTM(
                self.lsmt,  # Incrementar el número de unidades en la primera capa LSTM
                input_shape=(7, 1),
                return_sequences=True,
                kernel_regularizer=l2(self.l2_lambda),
            )
        )
        # Reducir el número de unidades en la última capa LSTM
        model.add(Dropout(self.dropout_rate))
        model.add(
            LSTM(self.gru, return_sequences=True, kernel_regularizer=l2(self.l2_lambda))
        )
        # Cambiar a capa GRU
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.lsmt2, kernel_regularizer=l2(self.l2_lambda)))

        # Reducir el número de unidades en la última capa LSTM
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(37, activation="softmax"))

        # Compilar modelo
        optimizer = Adam(
            learning_rate=self.learning_rate
        )  # Usar una tasa de aprendizaje personalizada

        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        early_stopping = EarlyStopping(monitor="loss", patience=20)
        # Entrenar modelo
        model.fit(
            secuencias,
            siguientes_numeros,
            epochs=self.epoc,
            batch_size=self.batchSize,
            validation_split=0.2,
            callbacks=[early_stopping],
        )

        return model

        # Crea secuencias de números y los números siguientes para entrenar el modelo.

    def _crear_secuencias(self):
        secuencias = []
        siguientes_numeros = []
        for i in range(len(self.contador.numeros) - 8):
            secuencias.append(self.contador.numeros[i : i + 7])
            siguientes_numeros.append(self.contador.numeros[i + 7])
        secuencias = pad_sequences(np.array(secuencias))
        siguientes_numeros = to_categorical(np.array(siguientes_numeros))
        return secuencias, siguientes_numeros

    # Predice los próximos números.

    def guardar_modelo(self):
        modelo_path = (
            "Models/" + self.nombreModelo
        )  # Ruta relativa a la carpeta "modelo"
        self.model.save(modelo_path)  # Guarda el modelo en la ubicación especificada

    def predecir(self):
        if self.contador.ingresados > 7:
            secuencia_entrada = np.array(self.contador.numeros[-7:]).reshape(1, 7, 1)
            predicciones = self.model.predict(secuencia_entrada, verbose=0)

            # Filtrar predicciones basadas en el umbral de probabilidad
            predicciones_filtradas = [
                i
                for i, probabilidad in enumerate(predicciones[0])
                if probabilidad > self.umbral_probilidad
            ]

            # Ordenar las predicciones filtradas por probabilidad descendente
            predecidos = sorted(
                predicciones_filtradas, key=lambda i: predicciones[0][i], reverse=True
            )

            for num in predecidos:
                if num not in self.resultados:
                    self.resultados[num] = 0
                    self.contador.incrementar_jugados()

            self.resultados = {
                k: self.resultados[k] for k in sorted(self.resultados.keys())
            }

    def verificar_predecidos(self, numero):
        acierto = False
        es_vecino1lugar = False
        es_vecino2lugar = False
        es_vecino3lugar = False
        es_vecino4lugar=False

        self.numeros_predecidos = []
        self.no_salidos = []
        self.contador.incrementar_ingresados(numero)

        if len(self.resultados) > 0:
            if numero in self.resultados:
                self.numeros_predecidos.append(numero)
                self.contador.incrementar_predecidos()
                self.df_nuevo.at[len(self.df_nuevo), "Acierto"] = "P"
                acierto = True

            for vecino in self.resultados:
                if numero in vecino1lugar[vecino]:
                    if vecino not in  self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_1lugar()
                        es_vecino1lugar = True

                if numero in vecino2lugar[vecino]:
                    if vecino not in  self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_2lugar()
                        es_vecino2lugar = True

                if numero in vecinos3lugar[vecino]:
                    if vecino not in  self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_3lugar()
                        es_vecino3lugar = True
                
                if numero in Vecino4lugar[vecino]:
                    if vecino not in  self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_4lugar()
                        es_vecino4lugar = True

            for key in self.resultados:
                self.resultados[key] += 1

            for num in list(self.resultados.keys()):   
                if self.resultados[num] >= 7:
                    del self.resultados[num]
                    self.no_salidos.append(num)
                    self.contador.incrementar_supero_limite()
                    
            for x in list(self.numeros_predecidos):   
                del self.resultados[x]
                  
            
            
            
            if es_vecino1lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V1L"] = "V1L"

            if es_vecino2lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V2L"] = "V2L"

            if es_vecino3lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V3L"] = "V3L"
            
            if es_vecino4lugar:
                self.df_nuevo.at[len(self.df_nuevo), "V4L"] = "V4L"
            if (
                acierto
                or es_vecino1lugar
                or es_vecino2lugar
                or es_vecino3lugar
                or es_vecino4lugar
            ):
                self.contador.incrementar_aciertos()

    # Actualiza el DataFrame con el número ingresado y los resultados de las predicciones.
    def actualizar_dataframe(self, numero_ingresado):
        self.df_nuevo.loc[len(self.df_nuevo) + 1, "Salidos"] = (numero_ingresado,)
        # self.df_nuevo.at[len(self.df_nuevo), "Resultados"] = str(self.resultados)
        self.df_nuevo.loc[len(self.df_nuevo), "Resultados"] = str(self.resultados)
        self.df_nuevo.loc[len(self.df_nuevo), "Orden"] = self.contador.ingresados
        self.df_nuevo.loc[len(self.df_nuevo), "Acertados"] = str(self.numeros_predecidos)
        self.df_nuevo.loc[len(self.df_nuevo), "No salidos"] = str(self.no_salidos)

    # Guarda el DataFrame en un archivo de Excel.
    def guardar_excel(self):
        self.generar_reporte()
        self.df_nuevo.to_excel(self.filename, sheet_name="Salidos", index=False)

    # Muestra los resultados y las estadísticas.
    def mostrar_resultados(self):
        print(self.df_nuevo.tail(7))
        print(f"Numeros Jugados: {self.contador.jugados}")
        print(f"Aciertos Totales: {self.contador.aciertos_totales}")
        print(f"Sin salir: {self.contador.Sin_salir_nada}\n")
        print(f"Aciertos Predecidos: {self.contador.acierto_predecidos}")
        print(f"Aciertos v1 lugar : {self.contador.acierto_vecinos_1lugar}")
        print(f"Aciertos v2 lugar: {self.contador.acierto_vecinos_2lugar}")
        print(f"Aciertos v3 lugar: {self.contador.acierto_vecinos_3lugar}")
        print(f"Aciertos v4 lugar : {self.contador.acierto_vecinos_4lugar}\n")

        for e in self.numeros_predecidos:
            print(f"El Número {e} fue acertado de la lista de predecidos.")
        
        
        for x in self.no_salidos:
            print(f"El Número {x} eliminado por que supero el limite.")
            
            
        if len(self.resultados) > 0:
            print(f"\nLas posibles predicciones para el próximo número son: {self.resultados}\n")

    # Borra el último número ingresado y actualiza el contador.
    def borrar(self):
        if self.contador.numeros:
            self.contador.borrar_ultimo_numero()
            self.df_nuevo = self.df_nuevo[
                :-1
            ]  # Eliminar la última fila del DataFrame nuevo
            print("Último número borrado")

    def generar_reporte(self):
        fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Crear un diccionario con los datos
        datos = {
            "Juego fecha y hora": fecha_hora_actual,
            "Numeros jugados": self.contador.jugados,
            "Aciertos Totales": self.contador.aciertos_totales,
            "Aciertos de Predecidos": self.contador.acierto_predecidos,
            "Aciertos de VC": self.contador.acierto_vecinos_1lugar,
            "Aciertos de VL": self.contador.acierto_vecinos_2lugar,
            "Aciertos de VLL": self.contador.acierto_vecinos_3lugar,
            "l2": self.l2_lambda,
            "dropout rate": self.dropout_rate,
            "learning rate": self.learning_rate,
            "epoca": self.epoc,
            "batch_size": self.batchSize,
            "Nros a Predecir": self.numeros_a_predecir,
            "Nros Anteriores": self.numerosAnteriores,
            "Efectividad": self.contador.sacarEfectividad(),
        }

        # Convertir el diccionario en un DataFrame de Pandas
        df = pd.DataFrame([datos])

        archivo_excel = "reportes.xlsx"

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
