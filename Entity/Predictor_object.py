import numpy as np
import pandas as pd
import tensorflow as tf
import os
from Entity.Contador import Contador
from Entity.Numero import Numero_pretendiente as Numero
from datetime import datetime
from Entity.Vecinos import vecino1lugar, vecino2lugar, vecinos3lugar, Vecino4lugar
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    GRU,
    Bidirectional,
    BatchNormalization,
)
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
        # self.contador.numeros = self.df["Salidos"].head(3500).tolist()

        self.numeros_a_jugar = list()
        self.numeros_pretendientes = list()
        self.numeros_predecidos = list()
        self.no_salidos = list()

        # Parametros
        self.numerosAnteriores = 5
        self.lugares_vecinos = 3
        self.numeros_a_predecir = 10
        self.umbral_probilidad = 100
        self.limite = 5

        # hiperparamtros
        self.lsmt = 320
        self.gru = 256
        self.lsmt2 = 128
        self.l2_lambda = 0.001
        self.dropout_rate = 0.05
        self.learning_rate = 0.003
        
        self.epoc = 100 if len(self.contador.numeros) > 1000 else 10

        self.batchSize = 500

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
                input_shape=(self.numerosAnteriores, 1),
                return_sequences=True,
                kernel_regularizer=l2(self.l2_lambda),
            )
        )
        model.add(BatchNormalization())
        # Reducir el número de unidades en la última capa LSTM
        model.add(Dropout(self.dropout_rate))
        model.add(
            GRU(
                self.gru,
                return_sequences=True,
                kernel_regularizer=l2(self.l2_lambda),
            )
        )
        # Cambiar a capa GRU
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        model.add(
            LSTM(
                self.lsmt2,
                kernel_regularizer=l2(self.l2_lambda),
            )
        )
        model.add(BatchNormalization())
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
        )

        return model

        # Crea secuencias de números y los números siguientes para entrenar el modelo.

    def _crear_secuencias(self):
        secuencias = []
        siguientes_numeros = []
        for i in range(len(self.contador.numeros) - (self.numerosAnteriores + 1)):
            secuencias.append(self.contador.numeros[i : i + self.numerosAnteriores])
            siguientes_numeros.append(self.contador.numeros[i + self.numerosAnteriores])
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
        if self.contador.ingresados > self.numerosAnteriores:
            secuencia_entrada = np.array(
                self.contador.numeros[-self.numerosAnteriores :]
            ).reshape(1, self.numerosAnteriores, 1)
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
            
            for pretendiente in self.numeros_a_jugar:
                pretendiente.Jugar()

            for num in predecidos:
                probabilidad_redondeada = int(round(predicciones[0][num], 2) * 100)
                if any(n.numero == num for n in self.numeros_a_jugar):
                    for pretendiente in self.numeros_a_jugar:
                        if pretendiente.numero == num:
                            pretendiente.Aumentar_probailidad(probabilidad_redondeada)
                else:
                    nuevo_pretendiente = Numero(num, probabilidad_redondeada)
                    if any(n.numero == num for n in self.numeros_pretendientes):
                        for pretendiente in self.numeros_pretendientes:
                            if pretendiente.numero == num:
                                pretendiente.Aumentar_probailidad(probabilidad_redondeada)
                    else:
                        self.numeros_pretendientes.append(nuevo_pretendiente)

            for pretendiente in self.numeros_pretendientes:
                if pretendiente.probabilidad >= self.umbral_probilidad:
                    self.numeros_a_jugar.append(pretendiente)
                    self.numeros_pretendientes.remove(pretendiente)       
                    self.contador.incrementar_jugados()
                
            

            # Ordena los numeros_a_jugar2 por probabilidad descendente
            self.numeros_a_jugar = sorted(self.numeros_a_jugar, key=lambda x: x.probabilidad , reverse=True)
            self.numeros_pretendientes = sorted(self.numeros_pretendientes, key=lambda x: x.probabilidad, reverse=True)

    def verificar_predecidos(self, numero):
        acierto = False
        es_vecino1lugar = False
        es_vecino2lugar = False
        es_vecino3lugar = False
        es_vecino4lugar = False

        self.numeros_predecidos = list()
        self.no_salidos = []
        self.contador.incrementar_ingresados(numero)

        if len(self.numeros_a_jugar) > 0:
            if any(n.numero == numero and n.probabilidad > 0 for n in self.numeros_a_jugar):
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
                    and self.lugares_vecinos >= 1
                    and vecino.probabilidad > 0
                ):
                    if vecino not in self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_1lugar()
                        es_vecino1lugar = True

                if (
                    numero in vecino2lugar[vecino.numero]
                    and self.lugares_vecinos >= 2
                    and vecino.probabilidad > 0
                ):
                    if vecino not in self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_2lugar()
                        es_vecino2lugar = True

                if (
                    numero in vecinos3lugar[vecino.numero]
                    and self.lugares_vecinos >= 3
                    and vecino.probabilidad > 0
                ):
                    if vecino not in self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_3lugar()
                        es_vecino3lugar = True

                if (
                    numero in Vecino4lugar[vecino.numero]
                    and self.lugares_vecinos >= 4
                    and vecino.probabilidad > 0
                ):
                    if vecino not in self.numeros_predecidos:
                        self.numeros_predecidos.append(vecino)
                        self.contador.incrementar_aciertos_vecinos_4lugar()
                        es_vecino4lugar = True

            # Filtra los números predecidos de la lista numeros_a_jugar2
            self.numeros_a_jugar = [
                n
                for n in self.numeros_a_jugar
                if n not in self.numeros_predecidos
            ]

            for obj in self.numeros_a_jugar[:]:
                if obj.tardancia == self.limite:
                    self.no_salidos.append(obj.numero) 
                    self.numeros_a_jugar.remove(obj)
                    self.contador.incrementar_supero_limite()

            for x in self.numeros_pretendientes[:]:
                if x.tardancia == self.limite:
                    self.numeros_pretendientes.remove(x)



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

    # Actualiza el DataFrame con el número ingresado y los resultados de las predicciones.
    def actualizar_dataframe(self, numero_ingresado):
        self.df_nuevo.loc[len(self.df_nuevo) + 1, "Salidos"] = (numero_ingresado)
        
        # Imprime para asegurarte de que self.numeros_a_jugar contiene instancias de Numero_pretendiente
        # Convierte la lista de instancias a una cadena única
        resultados_str = ",".join([str(obj) for obj in self.numeros_a_jugar])
        predecidos_str = ",".join([str(obj) for obj in self.numeros_predecidos])
        self.df_nuevo.loc[len(self.df_nuevo), "Resultados"] = resultados_str
        self.df_nuevo.loc[len(self.df_nuevo), "Orden"] = self.contador.ingresados
        self.df_nuevo.loc[len(self.df_nuevo), "Acertados"] = predecidos_str
        self.df_nuevo.loc[len(self.df_nuevo), "No salidos"] = str(self.no_salidos)

    # Guarda el DataFrame en un archivo de Excel.
    def guardar_excel(self):
        self.generar_reporte()
        self.df_nuevo.to_excel(self.filename, sheet_name="Salidos", index=False)

    # Muestra los resultados y las estadísticas.
    def mostrar_resultados(self):
        print("\nTabla de resultados:")
        print(self.df_nuevo.tail(3))
        print(f"Numeros Jugados: {self.contador.jugados}")
        print(f"Aciertos Totales: {self.contador.aciertos_totales}")
        print(f"Sin salir: {self.contador.Sin_salir_nada}\n")

        for e in self.numeros_predecidos:
            print(f"El Número {e.numero} fue acertado de la lista de predecidos.")

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

    def generar_reporte(self):
        fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Crear un diccionario con los datos
        datos = {
            "Juego fecha y hora": fecha_hora_actual,
            "Numeros jugados": self.contador.jugados,
            "Aciertos Totales": self.contador.aciertos_totales,
            "Aciertos de Predecidos": self.contador.acierto_predecidos,
            "V1L": self.contador.acierto_vecinos_1lugar,
            "V2L": self.contador.acierto_vecinos_2lugar,
            "V3L": self.contador.acierto_vecinos_3lugar,
            "V4L": self.contador.acierto_vecinos_4lugar,
            "l2": self.l2_lambda,
            "dropout rate": self.dropout_rate,
            "learning rate": self.learning_rate,
            "epoca": self.epoc,
            "batch_size": self.batchSize,
            "Nros a Predecir": self.numeros_a_predecir,
            "Nros Anteriores": self.numerosAnteriores,
            "Efectividad": self.contador.sacarEfectividad(),
            "Ruleta": self.filename,
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
