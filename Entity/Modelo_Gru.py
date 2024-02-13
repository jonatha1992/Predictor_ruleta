import numpy as np
import pandas as pd
import tensorflow as tf
import os
from Entity.Contador import Contador
from Entity.Parametro import HiperParametros
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GRU,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Nadam, Adadelta
from tensorflow.keras.callbacks import EarlyStopping


class Modelo:
    # Inicializa el objeto de la clase con un nombre de archivo y crea el modelo.
    def __init__(self, filename, hiperparametro):
        self.filename = filename
        self.foldername = os.path.dirname(filename)
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]
        self.nombreModelo = "Model_gru" + self.filebasename
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        self.numeros = self.df["Salidos"].values.tolist()
        self.hiperparametros = hiperparametro

        # Ruta relativa a la carpeta "modelo" en el mismo directorio que tu archivo de código
        modelo_path = "./Models/" + self.nombreModelo

        if os.path.exists(modelo_path):  # Verifica si ya hay un modelo guardado
            self.model = load_model(modelo_path)  # Carga el modelo guardado si existe
        else:
            self.model = self._crear_modelo()
            self.guardar_modelo()  # Guarda el modelo después de entrenarlo

    def _crear_modelo(self):
        secuencias, siguientes_numeros = self._crear_secuencias()
        model = Sequential()

        model.add(
            GRU(
                self.hiperparametros.gru,  # Usar unidades GRU según los hiperparámetros
                input_shape=(self.hiperparametros.numerosAnteriores, 1),
                return_sequences=True,
                kernel_regularizer=l2(self.hiperparametros.l2_lambda),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(self.hiperparametros.dropout_rate))
        model.add(
            GRU(
                self.hiperparametros.gru,  # Ajuste según sea necesario; podría ser el mismo valor o ajustado
                return_sequences=True,
                kernel_regularizer=l2(self.hiperparametros.l2_lambda),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(self.hiperparametros.dropout_rate))
        model.add(
            GRU(
                self.hiperparametros.gru,  # Para la última capa, ajustar según el rendimiento deseado
                kernel_regularizer=l2(self.hiperparametros.l2_lambda),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(self.hiperparametros.dropout_rate))
        model.add(Dense(37, activation="softmax"))

        optimizer = Adam(learning_rate=self.hiperparametros.learning_rate)
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        early_stopping = EarlyStopping(monitor="loss", patience=20)
        model.fit(
            secuencias,
            siguientes_numeros,
            epochs=self.hiperparametros.epoc,
            batch_size=self.hiperparametros.batchSize,
            validation_split=0.2,
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
        secuencias = pad_sequences(np.array(secuencias))
        siguientes_numeros = to_categorical(np.array(siguientes_numeros))
        return secuencias, siguientes_numeros

    # Predice los próximos números.

    def guardar_modelo(self):
        modelo_path = (
            "Models/" + self.nombreModelo
        )  # Ruta relativa a la carpeta "modelo"
        self.model.save(modelo_path)  # Guarda el modelo en la ubicación especificada
