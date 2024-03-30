import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class Modelo:
    def __init__(self, filename, hiperparametro):
        self.filename = filename
        self.foldername = os.path.dirname(filename)
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]
        self.nombreModelo = "Model_LSTM" + self.filebasename
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        self.numeros = self.df["Salidos"].values.tolist()
        self.hiperparametros = hiperparametro

        modelo_path = "./Models/" + self.nombreModelo
        if os.path.exists(modelo_path):
            self.model = load_model(modelo_path)
        else:
            self.model = self._crear_modelo()
            self.guardar_modelo()

    def _crear_modelo(self):
        secuencias, siguientes_numeros = self._crear_secuencias()
        model = Sequential()

        # Primera capa LSTM
        model.add(
            LSTM(
                self.hiperparametros.lsmt,  # Número de unidades en la primera capa LSTM
                input_shape=(self.hiperparametros.numerosAnteriores, 1),
                return_sequences=True,  # Importante para apilar LSTM
                kernel_regularizer=l2(self.hiperparametros.l2_lambda),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(self.hiperparametros.dropout_rate))

        # Segunda capa LSTM
        model.add(
            LSTM(
                self.hiperparametros.lsmt2,  # Ajustar el número de unidades en la segunda capa LSTM
                kernel_regularizer=l2(self.hiperparametros.l2_lambda),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(self.hiperparametros.dropout_rate))

        # Capa de salida
        model.add(Dense(37, activation="softmax"))  # Ajustar según el número de clases

        # Compilar modelo
        optimizer = Adam(learning_rate=self.hiperparametros.learning_rate)
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        early_stopping = EarlyStopping(monitor="loss", patience=20)

        # Entrenar modelo
        model.fit(
            secuencias,
            siguientes_numeros,
            epochs=self.hiperparametros.epoc,
            batch_size=self.hiperparametros.batchSize,
            validation_split=0.2,
            callbacks=[early_stopping],
        )

        return model

    def _crear_secuencias(self):
        secuencias = []
        siguientes_numeros = []
        for i in range(
            len(self.numeros) - (self.hiperparametros.numerosAnteriores + 1)
        ):
            secuencias.append(
                self.numeros[i : i + self.hiperparametros.numerosAnteriores]
            )
            siguientes_numeros.append(
                self.numeros[i + self.hiperparametros.numerosAnteriores]
            )
        secuencias = pad_sequences(
            np.array(secuencias), padding="post", dtype="float32"
        )
        siguientes_numeros = to_categorical(
            np.array(siguientes_numeros), num_classes=37
        )
        return secuencias[:, :, np.newaxis], siguientes_numeros

    def guardar_modelo(self):
        modelo_path = "Models/" + self.nombreModelo
        self.model.save(modelo_path)
