from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import os


class Modelo:
    def __init__(self, filename, hiperparametro):
        self.filename = filename
        self.foldername = os.path.dirname(filename)
        self.filebasename = os.path.splitext(os.path.basename(filename))[0]
        self.nombreModelo = "Model_" + self.filebasename
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        self.numeros = self.df["Salidos"].values.tolist()
        self.hiperparametros = hiperparametro

        modelo_path = "./Models/" + self.nombreModelo + ".keras"

        try:
            self.model = tf.keras.models.load_model(modelo_path)
            print("Modelo cargado exitosamente.")
        except:
            print("No se pudo cargar el modelo existente. Creando uno nuevo.")
            self.model = self._crear_modelo()
            self.guardar_modelo()

        self.evaluar_modelo()

    def _crear_modelo(self):
        secuencias, siguientes_numeros = self._crear_secuencias()

        X_train, X_val, y_train, y_val = train_test_split(
            secuencias, siguientes_numeros, test_size=0.2
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(
                    self.hiperparametros.gru1,
                    input_shape=(self.hiperparametros.numerosAnteriores, 1),
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.hiperparametros.l2_lambda
                    ),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
                tf.keras.layers.GRU(
                    self.hiperparametros.gru2,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.hiperparametros.l2_lambda
                    ),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
                tf.keras.layers.GRU(
                    self.hiperparametros.gru3,
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.hiperparametros.l2_lambda
                    ),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(self.hiperparametros.dropout_rate),
                tf.keras.layers.Dense(37, activation="softmax"),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.hiperparametros.learning_rate
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="./Models/best_model.keras",
            save_best_only=True,
            monitor="val_loss",
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
        secuencias = tf.keras.preprocessing.sequence.pad_sequences(np.array(secuencias))
        siguientes_numeros = tf.keras.utils.to_categorical(np.array(siguientes_numeros))
        return secuencias, siguientes_numeros

    def guardar_modelo(self):
        modelo_path = "Models/" + self.nombreModelo + ".keras"
        tf.keras.models.save_model(self.model, modelo_path)
        print(f"Modelo guardado en {modelo_path}")

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
