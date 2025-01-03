import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Config import get_relative_path
from Entity.Vecinos import colores_ruleta, vecino1lugar, vecino2lugar,tercio, juego_del_0, huerfanos, vecinos
from sklearn.preprocessing import LabelEncoder

def calcular_frecuencia(df, rango=10):
    """
    Calcula la frecuencia de aparición de un número en las últimas `rango` partidas.
    """
    frecuencias = []
    for i in range(len(df)):
        if i < rango:
            # Si no hay suficientes datos previos
            frecuencias.append(0)
        else:
            # Calcular frecuencia en la ventana
            ventana = df['Salidos'].iloc[i - rango:i]
            frecuencia_actual = ventana.value_counts().to_dict()
            frecuencias.append(frecuencia_actual.get(df['Salidos'].iloc[i], 0))

    df['Frecuencia'] = frecuencias
    # Normalizar la frecuencia
    scaler = MinMaxScaler()
    df['Frecuencia'] = scaler.fit_transform(df[['Frecuencia']])
    return df

def determinar_sector(numero):
    if numero in tercio:
        return 'tercio'
    elif numero in juego_del_0:
        return 'juego_del_0'
    elif numero in huerfanos:
        return 'huerfanos'
    elif numero in vecinos:
        return 'vecinos'
    else:
        return 'desconocido'



class Modelo:
    def __init__(self, filename, hiperparametro):
        self.filename = filename
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]
        self.hiperparametros = hiperparametro
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        le = LabelEncoder()

        # Añadir características adicionales
        self.df['vecino1'] = self.df['Salidos'].apply(lambda numero: vecino1lugar.get(numero, []))
        self.df['vecino2'] = self.df['Salidos'].apply(lambda numero: vecino2lugar.get(numero, []))
        self.df['sector'] = self.df['Salidos'].apply(determinar_sector)
        # Codificar la columna 'sector'
        self.df['sector_encoded'] = le.fit_transform(self.df['sector'])
        
        # Calcular frecuencias y preparar datos
        self.df = calcular_frecuencia(self.df, rango=10)
        self.numeros = self.df["Salidos"].values.tolist()
        self.vecinos1 = self.df["vecino1"].values.tolist()
        self.vecinos2 = self.df["vecino2"].values.tolist()
        self.frecuencias = self.df["Frecuencia"].values.tolist()

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
                input_dim=37,  # Números posibles en la ruleta
                output_dim=48,
                input_length=self.hiperparametros.numerosAnteriores * 4  # Multiplica por 4 para todas las características
            ),
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
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6),
        ]

        model.fit(X_train, y_train, epochs=100, batch_size=self.hiperparametros.batchSize,
                  validation_data=(X_val, y_val), callbacks=callbacks)

        return model

    def _crear_secuencias(self):
        """
        Crea secuencias de entrada para el modelo a partir de los datos históricos.
        """
        secuencias = []
        siguientes_numeros = []

        for i in range(len(self.numeros) - (self.hiperparametros.numerosAnteriores + 1)):
            secuencia = []
            for j in range(self.hiperparametros.numerosAnteriores):
                idx = i + j
                # Construir la información para cada número en la secuencia
                numero_info = [
                    self.numeros[idx],       # Número actual
                    self.frecuencias[idx],   # Frecuencia en rango
                    int(self.numeros[idx] in self.vecinos1[idx]),
                    int(self.numeros[idx] in self.vecinos2[idx]),
                    self.df['sector_encoded'].iloc[idx]  # Sector codificado


                ]
                secuencia.extend(numero_info)

            # Agregar la secuencia y el siguiente número como objetivo
            secuencias.append(secuencia)
            siguientes_numeros.append(self.numeros[i + self.hiperparametros.numerosAnteriores])

        # Convertir las secuencias a tensores y normalizarlas
        secuencias = tf.keras.preprocessing.sequence.pad_sequences(np.array(secuencias))
        siguientes_numeros = tf.keras.utils.to_categorical(np.array(siguientes_numeros), num_classes=37)

        return secuencias, siguientes_numeros
