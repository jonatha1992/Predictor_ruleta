import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Config import get_relative_path
from Entity.Vecinos import colores_ruleta, vecino1lugar, vecino2lugar, vecinos3lugar, Vecino4lugar


class Modelo:
    def __init__(self, filename, hiperparametro):
        self.filename = filename
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]
        self.hiperparametros = hiperparametro
        self.df = pd.read_excel(filename, sheet_name="Salidos")

        # Añadir características adicionales
        self.df['vecino1'] = self.df['Salidos'].apply(lambda numero: vecino1lugar.get(numero, []))

        # Convertir las secuencias de salida en listas
        self.numeros = self.df["Salidos"].values.tolist()
        self.vecinos1 = self.df["vecino1"].values.tolist()

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
                    self.numeros[idx],  # El número actual
                    int(self.numeros[idx] in self.vecinos1[idx]),  # Si está en los vecinos más cercanos
                ]
                secuencia.extend(numero_info)  # Agregar la información del número a la secuencia
            secuencias.append(secuencia)
            siguientes_numeros.append(self.numeros[i + self.hiperparametros.numerosAnteriores])

        # Convertir las secuencias a tensores y normalizarlas
        secuencias = tf.keras.preprocessing.sequence.pad_sequences(np.array(secuencias))
        siguientes_numeros = tf.keras.utils.to_categorical(np.array(siguientes_numeros), num_classes=37)
        return secuencias, siguientes_numeros

    # def jugar_numero(self, numero):
    #     # Verificar si `numero` tiene una probabilidad combinada antes de intentar usarlo
    #     probabilidad = next((pred["probabilidad_combinada"] for pred in self.numeros_a_jugar if pred["numero"] == numero), None)
    #     if probabilidad is None:
    #         print(f"Advertencia: El número {numero} no tiene una probabilidad combinada calculada.")
    #         return  # Salir si el número no tiene probabilidad calculada

    #     nuevo_numero = NumeroJugar(
    #         numero,
    #         probabilidad,
    #         self.Parametro_juego.lugares_vecinos
    #     )
    #     self.numeros_a_jugar.append(nuevo_numero)
    #     self.contador.incrementar_jugados()
