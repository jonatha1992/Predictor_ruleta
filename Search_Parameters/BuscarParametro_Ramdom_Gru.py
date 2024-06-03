import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense, BatchNormalization
from keras_tuner import RandomSearch, HyperParameters
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class HyperparameterTuner:
    def __init__(self, filename, epocas_probar, numeros_anteriores):
        self.filename = filename
        try:
            self.df = pd.read_excel(filename, sheet_name="Salidos")
        except Exception as e:
            print(f"Error leyendo el archivo de Excel: {e}")
            return

        if "Salidos" not in self.df.columns:
            print("Error: la columna 'Salidos' no se encuentra en la hoja de Excel.")
            return

        self.numeros = self.df["Salidos"].values.tolist()
        self.epocas_probar = epocas_probar
        self.numeros_anteriores = numeros_anteriores
        self.results_filename = "resultados_parametros_random_search.xlsx"
        self.best_accuracy = 0
        self.best_model_info = {}

    def build_model(self, hp, input_shape):
        model = Sequential()

        l2_lambda_value = hp.Choice("l2_lambda", values=[0.001, 0.002, 0.003])
        dropout_rate_value = hp.Choice("dropout_rate", values=[0.05, 0.1])
        learning_rate_value = hp.Choice("learning_rate", values=[0.001, 0.003])

        model.add(
            GRU(
                units=hp.Int("gru_3", min_value=256, max_value=512, step=32),
                input_shape=input_shape,
                return_sequences=True,
                kernel_regularizer=L2(l2_lambda_value),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate_value))
        model.add(
            GRU(
                units=hp.Int("gru_2", min_value=128, max_value=256, step=32),
                return_sequences=True,
                kernel_regularizer=L2(l2_lambda_value),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate_value))
        model.add(
            GRU(
                units=hp.Int("gru_1", min_value=64, max_value=128, step=32),
                activation="relu",
                kernel_regularizer=L2(l2_lambda_value),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate_value))
        model.add(Dense(37, activation="softmax"))
        model.compile(
            optimizer=Adam(learning_rate=learning_rate_value),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _crear_secuencias(self, numero_anterior):
        secuencias = []
        siguientes_numeros = []
        if len(self.numeros) < numero_anterior + 1:
            print("Error: Insufficient data in self.numeros")
            return None, None
        for i in range(len(self.numeros) - numero_anterior - 1):
            secuencias.append(self.numeros[i : i + numero_anterior])
            siguientes_numeros.append(self.numeros[i + numero_anterior])
        secuencias = pad_sequences(np.array(secuencias, dtype="float32"))
        siguientes_numeros = to_categorical(np.array(siguientes_numeros))
        return secuencias, siguientes_numeros

    def tune_hyperparameters(self):
        for epocas in self.epocas_probar:
            for numero_anterior in self.numeros_anteriores:
                self.secuencias, self.siguientes_numeros = self._crear_secuencias(
                    numero_anterior
                )

                if self.secuencias is None or self.siguientes_numeros is None:
                    print("Error al crear secuencias y siguientes_numeros.")
                    continue

                if (
                    self.secuencias.shape[0] == 0
                    or self.siguientes_numeros.shape[0] == 0
                ):
                    print(
                        "Error: self.secuencias o self.siguientes_numeros están vacíos."
                    )
                    continue
                if self.secuencias.shape[0] != self.siguientes_numeros.shape[0]:
                    print(
                        "Error: self.secuencias y self.siguientes_numeros no tienen el mismo número de filas."
                    )
                    continue

                early_stopping = EarlyStopping(monitor="val_loss", patience=15)
                reduce_lr = ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
                )
                model_checkpoint = ModelCheckpoint(
                    filepath=f"best_model_{epocas}_epochs_{numero_anterior}_anterior.h5",
                    save_best_only=True,
                    monitor="val_loss",
                )

                tuner = RandomSearch(
                    lambda hp: self.build_model(
                        hp, (numero_anterior, 1)
                    ),  # Cambio aquí
                    objective="val_accuracy",
                    max_trials=50,
                    executions_per_trial=3,
                    directory=f"random_search_{epocas}_epochs_{numero_anterior}_anterior",
                    project_name="hyperparameter_tuning",
                )

                tuner.search(
                    x=self.secuencias,
                    y=self.siguientes_numeros,
                    epochs=epocas,
                    batch_size=512,
                    validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr, model_checkpoint],
                )

                self.save_results_to_excel(tuner, epocas, numero_anterior)
                self.evaluate_best_model(tuner, epocas, numero_anterior)

                print(
                    f"Se guardaron los resultados para {epocas} épocas y {numero_anterior} números anteriores en el excel."
                )

    def save_results_to_excel(self, tuner, epocas, numero_anterior):
        if not os.path.exists(self.results_filename):
            pd.DataFrame().to_excel(self.results_filename, index=False)

        existing_df = pd.read_excel(self.results_filename)
        trials = tuner.oracle.get_best_trials(num_trials=10)
        new_results = []

        for trial in trials:
            trial_info = {
                "epocas": epocas,
                "numero_anterior": numero_anterior,
                "trial_id": trial.trial_id,
                "score": trial.score,
            }
            trial_info.update(trial.hyperparameters.values)
            new_results.append(trial_info)

        new_df = pd.DataFrame(new_results)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df.to_excel(self.results_filename, index=False)

    def evaluate_best_model(self, tuner, epocas, numero_anterior):
        best_model = tuner.get_best_models(num_models=1)[0]
        (
            secuencias_train,
            secuencias_test,
            siguientes_numeros_train,
            siguientes_numeros_test,
        ) = train_test_split(
            self.secuencias, self.siguientes_numeros, test_size=0.2, random_state=42
        )

        best_model.fit(
            secuencias_train,
            siguientes_numeros_train,
            epochs=10,
            batch_size=512,
            verbose=0,
        )
        y_pred = best_model.predict(secuencias_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(siguientes_numeros_test, axis=1)

        accuracy = accuracy_score(y_true, y_pred_classes)
        report = classification_report(y_true, y_pred_classes)
        matrix = confusion_matrix(y_true, y_pred_classes)

        print(
            f"Best Model Evaluation for {epocas} epochs and {numero_anterior} previous numbers:"
        )
        print("Accuracy:", accuracy)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", matrix)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model_info = {
                "epocas": epocas,
                "numero_anterior": numero_anterior,
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": matrix,
            }

    def save_best_model_info(self):
        best_model_df = pd.DataFrame([self.best_model_info])
        with pd.ExcelWriter(
            self.results_filename, mode="a", if_sheet_exists="replace"
        ) as writer:
            best_model_df.to_excel(writer, sheet_name="Best_Model_Info", index=False)


if __name__ == "__main__":
    epocas_probar = [100, 150, 200]
    numeros_anteriores = [9, 8, 7, 6, 5, 4]
    tuner = HyperparameterTuner("bombay1.xlsx", epocas_probar, numeros_anteriores)
    tuner.tune_hyperparameters()
    tuner.save_best_model_info()
