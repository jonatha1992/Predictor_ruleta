# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from Config import get_relative_path
# from Entity.Vecinos import colores_ruleta, vecino1lugar, vecino2lugar, vecinos3lugar, Vecino4lugar


# class Modelo:
#     def __init__(self, filename, hiperparametro):
#         self.filename = filename
#         self.filebasename = os.path.splitext(os.path.basename(filename))[0]
#         self.hiperparametros = hiperparametro
#         self.df = pd.read_excel(filename, sheet_name="Salidos")

#         self.df['color'] = self.df['Salidos'].apply(lambda numero: 1 if colores_ruleta.get(numero) == 'red' else 0)
#         self.df['par_impar'] = self.df['Salidos'].apply(lambda numero: 1 if numero % 2 == 0 else 0)
#         self.df['vecino1'] = self.df['Salidos'].apply(lambda numero: vecino1lugar.get(numero, []))
#         self.df['vecino2'] = self.df['Salidos'].apply(lambda numero: vecino2lugar.get(numero, []))
#         self.df['vecino3'] = self.df['Salidos'].apply(lambda numero: vecinos3lugar.get(numero, []))
#         self.df['vecino4'] = self.df['Salidos'].apply(lambda numero: Vecino4lugar.get(numero, []))

#         self.numeros = self.df["Salidos"].values.tolist()

#     def crear_y_guardar_modelos(self):
#         # Usar el mejor valor encontrado para num_anteriores
#         modelo_nombre = f"Model_{self.filebasename}_N{self.hiperparametros.numerosAnteriores}"
#         modelo_path = get_relative_path(f"./Models/{modelo_nombre}.keras")

#         if not os.path.exists(modelo_path):
#             print(f"Creando modelo: {modelo_nombre}")
#             model = self._crear_modelo()
#             tf.keras.models.save_model(model, modelo_path)
#             print(f"Modelo guardado en {modelo_path}")
#         else:
#             print(f"El modelo {modelo_nombre} ya existe.")

#     def _crear_modelo(self):
#         secuencias, siguientes_numeros = self._crear_secuencias()
#         X_train, X_val, y_train, y_val = train_test_split(secuencias, siguientes_numeros, test_size=0.2)

#         model = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Embedding(
#                     input_dim=37,
#                     output_dim=48,
#                     input_length=self.hiperparametros.numerosAnteriores),
#                 # Mejor embedding_dim = 48
#                 tf.keras.layers.LSTM(
#                     self.hiperparametros.gru1, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(self.hiperparametros.l2_lambda)),
#                 tf.keras.layers.BatchNormalization(),
#                 tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
#                 tf.keras.layers.LSTM(
#                     self.hiperparametros.gru2, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(self.hiperparametros.l2_lambda)),
#                 tf.keras.layers.BatchNormalization(),
#                 tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
#                 tf.keras.layers.LSTM(
#                     self.hiperparametros.gru3, kernel_regularizer=tf.keras.regularizers.l2(self.hiperparametros.l2_lambda)),
#                 tf.keras.layers.BatchNormalization(),
#                 tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
#                 tf.keras.layers.Dense(37, activation="softmax"),
#             ]
#         )

#         # Usar el mejor optimizador AdamW y tasa de aprendizaje encontrada
#         optimizer = tf.keras.optimizers.AdamW(learning_rate=self.hiperparametros.learning_rate)
#         model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

#         callbacks = [
#             tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20),
#             tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=40, min_lr=1e-6),
#         ]

#         model.fit(X_train, y_train, epochs=100, batch_size=self.hiperparametros.batchSize,
#                   validation_data=(X_val, y_val), callbacks=callbacks)  # Usar 100 épocas

#         return model

#     def _crear_secuencias(self):
#         secuencias = []
#         siguientes_numeros = []
#         for i in range(len(self.numeros) - (self.hiperparametros.numerosAnteriores + 1)):
#             secuencias.append(self.numeros[i: i + self.hiperparametros.numerosAnteriores])
#             siguientes_numeros.append(self.numeros[i + self.hiperparametros.numerosAnteriores])
#         secuencias = tf.keras.preprocessing.sequence.pad_sequences(np.array(secuencias))
#         siguientes_numeros = tf.keras.utils.to_categorical(np.array(siguientes_numeros), num_classes=37)
#         return secuencias, siguientes_numeros


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Config import get_relative_path
from Entity.Numero import Numero_jugar
from Entity.Vecinos import colores_ruleta, vecino1lugar, vecino2lugar, vecinos3lugar, Vecino4lugar


class Modelo:
    def __init__(self, filename, hiperparametro):
        self.filename = filename
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]
        self.hiperparametros = hiperparametro
        self.df = pd.read_excel(filename, sheet_name="Salidos")

        # Añadir características adicionales
        self.df['color'] = self.df['Salidos'].apply(lambda numero: 1 if colores_ruleta.get(numero) == 'red' else 0)
        self.df['par_impar'] = self.df['Salidos'].apply(lambda numero: 1 if numero % 2 == 0 else 0)
        self.df['vecino1'] = self.df['Salidos'].apply(lambda numero: vecino1lugar.get(numero, []))
        self.df['vecino2'] = self.df['Salidos'].apply(lambda numero: vecino2lugar.get(numero, []))
        self.df['vecino3'] = self.df['Salidos'].apply(lambda numero: vecinos3lugar.get(numero, []))
        self.df['vecino4'] = self.df['Salidos'].apply(lambda numero: Vecino4lugar.get(numero, []))

        # Convertir las secuencias de salida en listas
        self.numeros = self.df["Salidos"].values.tolist()
        self.colores = self.df["color"].values.tolist()
        self.paridades = self.df["par_impar"].values.tolist()
        self.vecinos1 = self.df["vecino1"].values.tolist()
        self.vecinos2 = self.df["vecino2"].values.tolist()
        self.vecinos3 = self.df["vecino3"].values.tolist()
        self.vecinos4 = self.df["vecino4"].values.tolist()

    def crear_y_guardar_modelos(self):

        modelo_nombre = f"Model_{self.filebasename}_N{self.hiperparametros.numerosAnteriores}"
        modelo_path = get_relative_path(f"./Models/{modelo_nombre}.keras")

        if not os.path.exists(modelo_path):
            print(f"Creando modelo: {modelo_nombre}")
            model = self._crear_modelo()
            tf.keras.models.save_model(model, modelo_path)
            print(f"Modelo guardado en {modelo_path}")
        else:
            print(f"El modelo {modelo_nombre} ya existe.")

    def _crear_modelo(self):
        secuencias, siguientes_numeros = self._crear_secuencias()
        X_train, X_val, y_train, y_val = train_test_split(secuencias, siguientes_numeros, test_size=0.2)

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=37,
                output_dim=48,
                input_length=self.hiperparametros.numerosAnteriores * 7),  # Multiplica por 7 para todas las features
            tf.keras.layers.LSTM(
                self.hiperparametros.gru1, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(self.hiperparametros.l2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
            tf.keras.layers.LSTM(
                self.hiperparametros.gru2, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(self.hiperparametros.l2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
            tf.keras.layers.LSTM(
                self.hiperparametros.gru3, kernel_regularizer=tf.keras.regularizers.l2(self.hiperparametros.l2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
            tf.keras.layers.Dense(37, activation="softmax"),
        ])

        optimizer = tf.keras.optimizers.AdamW(learning_rate=self.hiperparametros.learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=40, min_lr=1e-6),
        ]

        model.fit(X_train, y_train, epochs=100, batch_size=self.hiperparametros.batchSize,
                  validation_data=(X_val, y_val), callbacks=callbacks)

        return model

    def _crear_secuencias(self):
        secuencias = []
        siguientes_numeros = []

        for i in range(len(self.numeros) - (self.hiperparametros.numerosAnteriores + 1)):
            secuencia = []
            for j in range(self.hiperparametros.numerosAnteriores):
                idx = i + j
                numero_info = [
                    self.numeros[idx],
                    self.colores[idx],
                    self.paridades[idx],
                    int(self.numeros[idx] in self.vecinos1[idx]),
                    int(self.numeros[idx] in self.vecinos2[idx]),
                    int(self.numeros[idx] in self.vecinos3[idx]),
                    int(self.numeros[idx] in self.vecinos4[idx]),
                ]
                secuencia.extend(numero_info)
            secuencias.append(secuencia)
            siguientes_numeros.append(self.numeros[i + self.hiperparametros.numerosAnteriores])

        secuencias = tf.keras.preprocessing.sequence.pad_sequences(np.array(secuencias))
        siguientes_numeros = tf.keras.utils.to_categorical(np.array(siguientes_numeros), num_classes=37)
        return secuencias, siguientes_numeros

    def predecir(self):
        if self.contador.ingresados > self.hiperparametros.numerosAnteriores:
            secuencia_entrada = np.array(self.contador.numeros[-self.hiperparametros.numerosAnteriores:]).reshape(
                1, self.hiperparametros.numerosAnteriores, 1
            )
            predicciones = self.model.predict(secuencia_entrada, verbose=0)

            # Verificar que `predicciones` tiene el formato esperado
            if not isinstance(predicciones, np.ndarray) or predicciones.ndim != 2 or predicciones.shape[1] != 37:
                print("Error: `predicciones` no tiene el formato esperado.")
                return

            # Filtrar predicciones basadas en el umbral de probabilidad
            predicciones_filtradas = [
                {
                    "numero": i,
                    "probabilidad": float(prob),
                    "color": self.obtener_probabilidad_color(i),
                    "paridad": self.obtener_probabilidad_paridad(i),
                    "vecino": self.obtener_probabilidad_vecinos(i),
                }
                for i, prob in enumerate(predicciones[0]) if prob > 0.10
            ]

            # Ponderar la probabilidad combinada
            for pred in predicciones_filtradas:
                pred["probabilidad_combinada"] = (
                    0.5 * pred["probabilidad"] +
                    0.2 * pred["color"] +
                    0.15 * pred["paridad"] +
                    0.15 * pred["vecino"]
                )

            # Ordenar predicciones por probabilidad combinada descendente
            predecidos = sorted(predicciones_filtradas, key=lambda x: x["probabilidad_combinada"], reverse=True)

            # Obtener los números a jugar
            numeros_a_jugar = [pred["numero"] for pred in predecidos[:5]]  # Seleccionar top 5

            for numero in numeros_a_jugar:
                self.jugar_numero(numero)

            self.verificar_predecidos(predicciones, predecidos)

    def jugar_numero(self, numero):
        # Verificar si `numero` tiene una probabilidad combinada antes de intentar usarlo
        probabilidad = next((pred["probabilidad_combinada"] for pred in self.numeros_a_jugar if pred["numero"] == numero), None)
        if probabilidad is None:
            print(f"Advertencia: El número {numero} no tiene una probabilidad combinada calculada.")
            return  # Salir si el número no tiene probabilidad calculada

        nuevo_numero = Numero_jugar(
            numero,
            probabilidad,
            self.Parametro_juego.lugares_vecinos
        )
        self.numeros_a_jugar.append(nuevo_numero)
        self.contador.incrementar_jugados()
