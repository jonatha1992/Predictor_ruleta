import numpy as np
import pandas as pd
import os 
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras_tuner import BayesianOptimization


class HyperparameterTuner:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        if "Salidos" not in self.df.columns:
            print("Error: la columna 'Salidos' no se encuentra en la hoja de Excel.")
            return
        self.numeros = self.df["Salidos"].values.tolist()

        self.length_values = [ 5,6,7,8]
        self.l2_lambda_values = [0.001, 0.002, 0.003]
        self.dropout_rate_values = [0.02, 0.01, 0.03, 0.04, 0.05]
        self.learning_rate_values = [0.001, 0.002, 0.003, 0.005]

    

    def build_model(self, hp, length):
        model = Sequential()
        model.add(
            LSTM(
                units=320,
                input_shape=(length, 1),
                return_sequences=True,
                kernel_regularizer=L2(
                    hp.Choice("l2_lambda", values=self.l2_lambda_values)
                ),
            )
        )
        model.add(BatchNormalization())
        model.add(
            Dropout(rate=hp.Choice("dropout_rate", values=self.dropout_rate_values))
        )
        model.add(
            LSTM(
                units=128,
                activation="relu",
                kernel_regularizer=L2(
                    hp.Choice("l2_lambda", values=self.l2_lambda_values)
                ),
            )
        )
        model.add(BatchNormalization())
        model.add(
            Dropout(rate=hp.Choice("dropout_rate", values=self.dropout_rate_values))
        )
        model.add(Dense(37, activation="softmax"))

        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice(
                    "learning_rate", values=self.learning_rate_values
                )
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def crear_secuencias(self, length):
        secuencias = []
        siguientes_numeros = []
        for i in range(len(self.numeros) - (length + 1)):
            secuencias.append(self.numeros[i : i + length])
            siguientes_numeros.append(self.numeros[i + length])
        secuencias = np.array(secuencias).reshape(len(secuencias), length, 1)
        siguientes_numeros = to_categorical(np.array(siguientes_numeros))
        return secuencias, siguientes_numeros

    def tune_hyperparameters(self):
        for length in self.length_values:
            secuencias, siguientes_numeros = self.crear_secuencias(length)

            tuner = BayesianOptimization(
                lambda hp: self.build_model(hp, length),
                objective="val_accuracy",
                max_trials=200,
                directory=f"bayesian_optimization_length_{length}",
                project_name=f"Bayesian_Optimizer_length_{length}",
            )

            tuner.search(
                x=secuencias,
                y=siguientes_numeros,
                epochs=100,
                batch_size=500,
                verbose=1,
                validation_split=0.2,
                callbacks=[EarlyStopping(monitor="val_accuracy", patience=20)],
            )

            self.evaluate_and_save_results(tuner, length)

    def evaluate_and_save_results(self, tuner, length):
        trials = tuner.oracle.get_best_trials(num_trials=10)
        new_results = []

        for trial in trials:
            trial_info = {
                "trial_id": trial.trial_id,
                "score": trial.score,
            }
            trial_info.update(trial.hyperparameters.values)
            new_results.append(trial_info)

        new_df = pd.DataFrame(new_results)

        filename = f"resultados_bayesian_length_{length}.xlsx"
        existing_df = (
            pd.DataFrame() if not os.path.exists(filename) else pd.read_excel(filename)
        )
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df.to_excel(filename, index=False)


if __name__ == "__main__":
    tuner = HyperparameterTuner("./Data/Datos.xlsx")
    tuner.tune_hyperparameters()
