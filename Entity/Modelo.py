from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import os
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class Modelo:
    def __init__(self, filename, hiperparametro):
        self.filename = filename
        self.foldername = os.path.dirname(filename)
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]
        self.nombreModelo = "Model_gru_" + self.filebasename
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        self.numeros = self.df["Salidos"].values.tolist()
        self.hiperparametros = hiperparametro

        modelo_path = "./Models/" + self.nombreModelo

        if os.path.exists(modelo_path):
            self.model = load_model(modelo_path)
        else:
            self.model = self._crear_modelo()
            self.guardar_modelo()

        # Evaluar el modelo después de crearlo
        self.evaluar_modelo()

    def _crear_modelo(self):
        secuencias, siguientes_numeros = self._crear_secuencias()
        X_train, X_val, y_train, y_val = train_test_split(
            secuencias, siguientes_numeros, test_size=0.2
        )

        model = Sequential()
        model.add(
            GRU(
                self.hiperparametros.gru1,
                input_shape=(self.hiperparametros.numerosAnteriores, 1),
                return_sequences=True,
                kernel_regularizer=l2(self.hiperparametros.l2_lambda),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(self.hiperparametros.dropout_rate))
        model.add(
            GRU(
                self.hiperparametros.gru2,
                return_sequences=True,
                kernel_regularizer=l2(self.hiperparametros.l2_lambda),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(self.hiperparametros.dropout_rate))
        model.add(
            GRU(
                self.hiperparametros.gru3,
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

        early_stopping = EarlyStopping(monitor="val_loss", patience=20)
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
        )
        model_checkpoint = ModelCheckpoint(
            filepath="./Models/best_model.h5", save_best_only=True, monitor="val_loss"
        )

        model.fit(
            X_train,
            y_train,
            epochs=self.hiperparametros.epoc,
            batch_size=self.hiperparametros.batchSize,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
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

    def guardar_modelo(self):
        modelo_path = "Models/" + self.nombreModelo
        self.model.save(modelo_path)

    def evaluar_modelo(self):
        secuencias, siguientes_numeros = self._crear_secuencias()
        X_train, X_test, y_train, y_test = train_test_split(
            secuencias, siguientes_numeros, test_size=0.2
        )

        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print("Accuracy:", accuracy_score(y_true, y_pred_classes))
        print("Classification Report:\n", classification_report(y_true, y_pred_classes))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_classes))
