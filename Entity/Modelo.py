import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import get_relative_path
from Entity.vecinos import colores_ruleta, vecino1lugar, vecino2lugar, sector1, sector2, sector3, sector4, sector5, sector6
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


def calcular_estadisticas_avanzadas(df, rango=10):
    """
    Calcula estadísticas avanzadas para mejorar la predicción
    """
    # Media móvil ponderada (da más peso a números recientes)
    weights = np.array([1.2**i for i in range(rango)])
    weights = weights / weights.sum()
    df['Media_Ponderada'] = df['Salidos'].rolling(window=rango).apply(
        lambda x: np.sum(weights[-len(x):] * x))

    # Velocidad de cambio (derivada)
    df['Velocidad_Cambio'] = df['Salidos'].diff()

    # Aceleración (segunda derivada)
    df['Aceleracion'] = df['Velocidad_Cambio'].diff()

    # Volatilidad (desviación estándar móvil)
    df['Volatilidad'] = df['Salidos'].rolling(window=rango).std()

    # Momento (diferencia entre medias móviles)
    df['MA_Corta'] = df['Salidos'].rolling(window=5).mean()
    df['MA_Larga'] = df['Salidos'].rolling(window=rango).mean()
    df['Momento'] = df['MA_Corta'] - df['MA_Larga']

    # Llenar NaN con 0
    df = df.fillna(0)

    # Normalizar todas las nuevas características
    scaler = MinMaxScaler()
    columnas_normalizar = ['Media_Ponderada', 'Velocidad_Cambio', 'Aceleracion',
                           'Volatilidad', 'Momento']

    for columna in columnas_normalizar:
        df[columna] = scaler.fit_transform(df[[columna]])

    return df


def determinar_sector(numero):
    if numero in sector1:
        return 'sector1'
    elif numero in sector2:
        return 'sector2'
    elif numero in sector3:
        return 'sector3'
    elif numero in sector4:
        return 'sector4'
    elif numero in sector5:
        return 'sector5'
    elif numero in sector6:
        return 'sector6'
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
        self.df['sector'] = self.df['Salidos'].apply(determinar_sector)
        self.df['sector_encoded'] = le.fit_transform(self.df['sector'])

        # Calcular frecuencias y preparar datos
        self.df = calcular_frecuencia(self.df, rango=10)
        self.numeros = self.df["Salidos"].values.tolist()
        self.vecinos1 = self.df["vecino1"].values.tolist()
        self.frecuencias = self.df["Frecuencia"].values.tolist()

        # Añadir estadísticas avanzadas
        self.df = calcular_estadisticas_avanzadas(self.df, rango=10)
        self.media_ponderada = self.df["Media_Ponderada"].values.tolist()
        self.velocidad_cambio = self.df["Velocidad_Cambio"].values.tolist()
        self.aceleracion = self.df["Aceleracion"].values.tolist()
        self.volatilidad = self.df["Volatilidad"].values.tolist()
        self.momento = self.df["Momento"].values.tolist()

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
        # Set complete seed configuration
        seed = 42
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        secuencias, siguientes_numeros = self._crear_secuencias()
        X_train, X_val, y_train, y_val = train_test_split(
            secuencias,
            siguientes_numeros,
            test_size=0.2,
            random_state=seed,
            shuffle=True
        )

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=37,  # Números posibles en la ruleta
                output_dim=48,
                input_length=self.hiperparametros.numerosAnteriores * 9  # Multiplica por 4 para todas las características
            ),
            tf.keras.layers.LSTM(
                self.hiperparametros.capa1, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(self.hiperparametros.l2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
            tf.keras.layers.LSTM(
                self.hiperparametros.capa2, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(self.hiperparametros.l2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
            tf.keras.layers.LSTM(
                self.hiperparametros.capa3, kernel_regularizer=tf.keras.regularizers.l2(self.hiperparametros.l2_lambda)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
            tf.keras.layers.Dense(37, activation="softmax"),
        ])

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.hiperparametros.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        # Remove problematic thread configuration
        # tf.config.threading.set_inter_op_parallelism_threads(1)
        # tf.config.threading.set_intra_op_parallelism_threads(1)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.0001,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=7,
                min_lr=1e-7,
                mode='min'
            )
        ]

        # Add deterministic training
        history = model.fit(
            X_train,
            y_train,
            epochs=self.hiperparametros.epochs,
            batch_size=self.hiperparametros.batchSize,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

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
                    self.df['sector_encoded'].iloc[idx],  # Sector codificado
                    self.media_ponderada[idx],     # Nueva característica
                    self.velocidad_cambio[idx],    # Nueva característica
                    self.aceleracion[idx],         # Nueva característica
                    self.volatilidad[idx],         # Nueva característica
                    self.momento[idx]              # Nueva característica
                ]
                secuencia.extend(numero_info)

            # Agregar la secuencia y el siguiente número como objetivo
            secuencias.append(secuencia)
            siguientes_numeros.append(self.numeros[i + self.hiperparametros.numerosAnteriores])

        # Convertir las secuencias a tensores y normalizarlas
        secuencias = tf.keras.preprocessing.sequence.pad_sequences(np.array(secuencias))
        siguientes_numeros = tf.keras.utils.to_categorical(np.array(siguientes_numeros), num_classes=37)

        return secuencias, siguientes_numeros
