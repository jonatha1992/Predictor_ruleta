import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dropout, Dense
from keras_tuner import RandomSearch, GridSearch
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras_tuner import HyperParameters
from tensorflow.keras.regularizers import L2
from keras.callbacks import EarlyStopping



class HyperparameterTuner:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        if not 'Salidos' in self.df.columns:
            print("Error: la columna 'Salidos' no se encuentra en la hoja de Excel.")
            return
        self.numeros = self.df["Salidos"].values.tolist()
        self.secuencias, self.siguientes_numeros = self._crear_secuencias()

    def build_model(self, hp):
        model = Sequential()
        
        # Define l2_lambda, dropout_rate, and learning_rate once
        l2_lambda_value = hp.Choice("l2_lambda", values=[0.001, 0.002, 0.003])
        dropout_rate_value = hp.Choice("dropout_rate", values=[0.05, 0.01])
        learning_rate_value = hp.Choice("learning_rate", values=[0.001, 0.003])

        # LSTM layer
        model.add(
            LSTM(
                units=hp.Int("lstm_units", min_value=256, max_value=512, step=32),
                input_shape=(10, 1),
                return_sequences=True,
                kernel_regularizer=L2(l2_lambda_value)
            )
        )
        model.add(Dropout(rate=dropout_rate_value))

        # GRU layer
        model.add(
            GRU(
                units=hp.Int("gru_units", min_value=128, max_value=256, step=32),
                return_sequences=False,
                kernel_regularizer=L2(l2_lambda_value)
            )
        )
        model.add(Dropout(rate=dropout_rate_value))
        # Dense layers
        model.add(
            Dense(
                units=hp.Int("lstm2_units", min_value=64, max_value=128, step=32),
                activation='relu',
                kernel_regularizer=L2(l2_lambda_value)
            )
        )
        model.add(Dropout(rate=dropout_rate_value))
        model.add(Dense(37, activation='softmax'))

        # Compile - Using Adam optimizer
        model.compile(
            optimizer=Adam(learning_rate=learning_rate_value),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _crear_secuencias(self):
        secuencias = []
        siguientes_numeros = []
        if len(self.numeros) < 11:
            print("Error: Insufficient data in self.numeros")
            return None, None
        for i in range(len(self.numeros) - 11):
            secuencias.append(self.numeros[i : i + 10])
            siguientes_numeros.append(self.numeros[i + 10])
        secuencias = np.array(secuencias).reshape(len(secuencias), 10, 1)
        siguientes_numeros = to_categorical(np.array(siguientes_numeros))
        return secuencias, siguientes_numeros

    def tune_hyperparameters(self):
        if self.secuencias is None or self.siguientes_numeros is None:
            print("Error al crear secuencias y siguientes_numeros.")
            return

        if self.secuencias.shape[0] == 0 or self.siguientes_numeros.shape[0] == 0:
            print("Error: self.secuencias o self.siguientes_numeros están vacíos.")
            return
        if self.secuencias.shape[0] != self.siguientes_numeros.shape[0]:
            print("Error: self.secuencias y self.siguientes_numeros no tienen el mismo número de filas.")
            return

        early_stopping_loss = EarlyStopping(monitor="loss", patience=10)
        early_stopping_accuracy = EarlyStopping(monitor="accuracy", patience=10)
        for epoch in [ 50 ,60,70]:
            for batch in [512, 1024]:
        
                tuner = GridSearch(self.build_model,
                        objective='accuracy',
                        directory='grid_search',
                        project_name=f'Busca_Parametros_infinitos_E{epoch}_B{batch}')
                
                tuner.search(x=self.secuencias, y=self.siguientes_numeros,
                                epochs=epoch,
                                batch_size=batch,
                                verbose=1,
                                callbacks=[early_stopping_loss, early_stopping_accuracy])
               
                self.save_results_to_excel(tuner,"resultados_parametros_grid.xlsx", epoch, batch)
        
        print("se guardo todo en excel")

    def save_results_to_excel(self, tuner,filename, epoch, batch_size):
        if not os.path.exists(filename):
            # Si el archivo no existe, crea un DataFrame vacío y guárdalo
            pd.DataFrame().to_excel(filename, index=False)

        # Leer el archivo existente y los resultados actuales en dos DataFrames
        existing_df = pd.read_excel(filename)
        trials = tuner.oracle.get_best_trials(num_trials=10)
        new_results = []

        for trial in trials:
            trial_info = {
                'trial_id': trial.trial_id,
                'score': trial.score,
                'epochs': epoch,
                'batch_size': batch_size,
            }
            trial_info.update(trial.hyperparameters.values)
            new_results.append(trial_info)

        new_df = pd.DataFrame(new_results)

        # Concatenar los DataFrames y guardar en el mismo archivo
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df.to_excel(filename, index=False)

    
            
    
if __name__ == "__main__":
    tuner = HyperparameterTuner("datos.xlsx")
    tuner.tune_hyperparameters()
