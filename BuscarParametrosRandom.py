import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dropout, Dense
from keras_tuner import RandomSearch
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras.regularizers import L2


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
        model.add(LSTM(units=hp.Int("lstm_units", min_value=256, max_value=512, step=32 ), input_shape=(10, 1), return_sequences=True))
        model.add(Dropout(rate=hp.Choice("dropout_rate", values=[0.05,0.1,0.02,0.5])))
        model.add(GRU(units=hp.Int("gru_units", min_value=128, max_value=256,step=32 )))
        model.add(Dense(units=hp.Int("lstm2_units", min_value=32, max_value=64,step=32), activation='relu', kernel_regularizer=L2(hp.Choice("l2_lambda", values=[0.002,0.003,0.001]))))
        model.add(Dense(37, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.1,0.2])), loss='categorical_crossentropy', metrics=['accuracy'])
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

        tuner = RandomSearch(self.build_model,
                             objective='accuracy',
                             max_trials=300,
                             executions_per_trial=1,
                             directory='random_search',
                             project_name='Busca_Parametros_50')
        tuner.search_space_summary()

        # Utilizar todos los datos en self.numeros para entrenamiento
        tuner.search(x=self.secuencias, y=self.siguientes_numeros, epochs=30, batch_size=256, verbose=1)

        tuner.results_summary()
        # Guardar los resultados en un archivo de Excel
        self.save_results_to_excel(tuner, "resultados.xlsx")
        
    def save_results_to_excel(self,tuner, filename):
        # trials = tuner.oracle.get_best_trials(num_trials=tuner.oracle.trials)
        trials = tuner.oracle.get_best_trials(num_trials=10)
        results = []
        for trial in trials:
            trial_info = {
                'trial_id': trial.trial_id,
                'score': trial.score
            }
            trial_info.update(trial.hyperparameters.values)
            results.append(trial_info)
        df = pd.DataFrame(results)
        df.to_excel(filename, index=False)
        
if __name__ == "__main__":
    tuner = HyperparameterTuner("datos.xlsx")
    tuner.tune_hyperparameters()
