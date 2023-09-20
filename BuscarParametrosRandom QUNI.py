import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dropout, Dense, Reshape , Lambda
from keras_tuner import RandomSearch
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K


class HyperparameterTuner:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        
         # Revisar que todas las columnas necesarias estén presentes
        if not set(["A", "B", "C", "D", "E", "F"]).issubset(self.df.columns):
            print("Error: las columnas A, B, C, D, E y F no se encuentran en la hoja de Excel.")
            return
         # Obtener los números de las columnas y convertirlos en una lista de combinaciones
        self.numeros = [list(row) for row in self.df[["A", "B", "C", "D", "E", "F"]].values]
        self.secuencias, self.siguientes_numeros = self._crear_secuencias()

    def custom_softmax(self, x):
        shape = K.shape(x)
        x = K.reshape(x, (-1, 46))
        x = K.softmax(x)
        return K.reshape(x, shape)


    def build_model(self, hp):
        model = Sequential()
        l2_lambda_value = hp.Choice("l2_lambda", values=[0.001, 0.002, 0.003])
        dropout_rate_value = hp.Choice("dropout_rate", values=[0.05, 0.01])
        learning_rate_value = hp.Choice("learning_rate", values=[0.001, 0.003])

        model.add(LSTM(units=hp.Int("lstm_units", min_value=256, max_value=512, step=32),
                    input_shape=(10, 6),
                    return_sequences=True,
                    kernel_regularizer=L2(l2_lambda_value)))
        model.add(Dropout(rate=dropout_rate_value))
        model.add(GRU(units=hp.Int("gru_units", min_value=128, max_value=256, step=32),
                    kernel_regularizer=L2(l2_lambda_value)))
        model.add(Dropout(rate=dropout_rate_value))
        model.add(Dense(units=hp.Int("lstm2_units", min_value=32, max_value=64, step=32),
                        activation="relu", kernel_regularizer=L2(l2_lambda_value)))

        model.add(Dense(6 * 46))
        model.add(Lambda(self.custom_softmax))
        model.add(Reshape((6, 46)))

        model.compile(optimizer=Adam(learning_rate=learning_rate_value),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])
        return model

    def _crear_secuencias(self):
        secuencias = []
        siguientes_numeros_onehot  = []

        # Para que haya 10 combinaciones y una adicional que actúe como "siguiente número"
        for i in range(len(self.numeros) - 10):
            # Agrega 10 combinaciones a la lista de secuencias
            secuencias.append(self.numeros[i: i + 10])
            
            siguiente = to_categorical(self.numeros[i + 10], num_classes=46)
            siguientes_numeros_onehot.append(siguiente)
        
        # Convierte las listas en arrays de numpy
        secuencias = np.array(secuencias)  # Shape: (número_de_secuencias, 10, 6)
        siguientes_numeros_onehot = np.array(siguientes_numeros_onehot)  

        return secuencias, siguientes_numeros_onehot



    def tune_hyperparameters(self):
        if self.secuencias is None or self.siguientes_numeros is None:
            print("Error al crear secuencias y siguientes_numeros.")
            return

        early_stopping_loss = EarlyStopping(monitor="loss", patience=10)
        early_stopping_accuracy = EarlyStopping(monitor="accuracy", patience=10)

        for epoch in [30,50 ,70]:
            for batch in [512, 1024]:
                tuner = RandomSearch(
                    self.build_model,
                    objective="accuracy",
                    max_trials=1000,
                    executions_per_trial=1,
                    directory="random_search",
                    project_name=f"Busca_Parametros_E{epoch}_B{batch}"
                )

                tuner.search(
                    x=self.secuencias,
                    y=self.siguientes_numeros,
                    epochs=epoch,
                    batch_size=batch,
                    verbose=1,
                    callbacks=[early_stopping_loss, early_stopping_accuracy]
                )

                self.save_results_to_excel(
                    tuner, "resultados_parametros_QUINI_6_random.xlsx", epoch, batch
                )
        print("Se guardó todo en Excel.")

    def save_results_to_excel(self, tuner, filename, epoch, batch_size):
        if not os.path.exists(filename):
            # Si el archivo no existe, crea un DataFrame vacío y guárdalo
            pd.DataFrame().to_excel(filename, index=False)

        # Leer el archivo existente y los resultados actuales en dos DataFrames
        existing_df = pd.read_excel(filename)
        trials = tuner.oracle.get_best_trials(num_trials=10)
        new_results = []

        for trial in trials:
            trial_info = {
                "trial_id": trial.trial_id,
                "score": trial.score,
                "epochs": epoch,
                "batch_size": batch_size,
            }
            trial_info.update(trial.hyperparameters.values)
            new_results.append(trial_info)

        new_df = pd.DataFrame(new_results)

        # Concatenar los DataFrames y guardar en el mismo archivo
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df.to_excel(filename, index=False)


if __name__ == "__main__":
    tuner = HyperparameterTuner("Datos Quini6.xlsx")
    tuner.tune_hyperparameters()
