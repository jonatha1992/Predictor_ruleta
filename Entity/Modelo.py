import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Config import get_relative_path


class Modelo:
    def __init__(self, filename, hiperparametro):
        self.filename = filename
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]
        self.hiperparametros = hiperparametro
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        self.numeros = self.df["Salidos"].values.tolist()

    def crear_y_guardar_modelos(self):
        for num_anteriores in [10]:
            self.hiperparametros.numerosAnteriores = num_anteriores
            modelo_nombre = f"Model_{self.filebasename}_N{num_anteriores}"
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

        model = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(
                    self.hiperparametros.gru1, input_shape=(
                        self.hiperparametros.numerosAnteriores, 1), return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(
                        self.hiperparametros.l2_lambda)), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(
                    self.hiperparametros.dropout_rate), tf.keras.layers.GRU(
                            self.hiperparametros.gru2, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(
                                self.hiperparametros.l2_lambda)), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(
                                    self.hiperparametros.dropout_rate), tf.keras.layers.GRU(
                                        self.hiperparametros.gru3, kernel_regularizer=tf.keras.regularizers.l2(
                                            self.hiperparametros.l2_lambda)), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(
                                                self.hiperparametros.dropout_rate), tf.keras.layers.Dense(
                                                    37, activation="softmax"), ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.hiperparametros.learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=40, min_lr=1e-6),
        ]

        model.fit(X_train, y_train, epochs=self.hiperparametros.epoc, batch_size=self.hiperparametros.batchSize,
                  validation_data=(X_val, y_val), callbacks=callbacks)

        return model

    def _crear_secuencias(self):
        secuencias = []
        siguientes_numeros = []
        for i in range(len(self.numeros) - (self.hiperparametros.numerosAnteriores + 1)):
            secuencias.append(self.numeros[i: i + self.hiperparametros.numerosAnteriores])
            siguientes_numeros.append(self.numeros[i + self.hiperparametros.numerosAnteriores])
        secuencias = tf.keras.preprocessing.sequence.pad_sequences(np.array(secuencias))
        siguientes_numeros = tf.keras.utils.to_categorical(np.array(siguientes_numeros))
        return secuencias, siguientes_numeros
