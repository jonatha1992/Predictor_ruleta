from scipy.stats import chisquare
import sys
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras_tuner import BayesianOptimization, RandomSearch, HyperModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import logging

# Configuración de logging para un seguimiento más detallado
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.models.Sequential()

        # Definir hiperparámetros
        l2_lambda_value = hp.Float("l2_lambda", min_value=1e-6, max_value=1e-2, sampling="LOG")
        dropout_rate_value = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.05)
        learning_rate_value = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="LOG")
        embedding_dim = hp.Int("embedding_dim", min_value=8, max_value=64, step=8)
        gru1_units = hp.Int("gru1_units", min_value=32, max_value=512, step=32)
        gru2_units = hp.Int("gru2_units", min_value=32, max_value=256, step=32)
        gru3_units = hp.Int("gru3_units", min_value=32, max_value=128, step=32)
        gru4_units = hp.Int("gru4_units", min_value=32, max_value=64, step=32)
        gru5_units = hp.Int("gru5_units", min_value=32, max_value=32, step=32)
        num_rnn_layers = hp.Int("num_rnn_layers", min_value=1, max_value=5, step=1)
        rnn_type_choices = ["GRU", "LSTM"]
        optimizer_type = hp.Choice("optimizer", values=["adam", "rmsprop", "adamw"])

        # Añadir capa de Embedding
        model.add(
            tf.keras.layers.Embedding(
                input_dim=37,
                output_dim=embedding_dim,
                input_length=self.input_shape[0],
            )
        )

        # Añadir capas RNN
        gru_units_list = [gru1_units, gru2_units, gru3_units, gru4_units, gru5_units]
        rnn_types = []
        for i in range(num_rnn_layers):
            return_sequences = True if i < num_rnn_layers - 1 else False
            units = gru_units_list[i] if i < len(gru_units_list) else gru5_units
            rnn_type = hp.Choice(f"rnn_type_layer_{i+1}", values=rnn_type_choices)
            rnn_types.append(rnn_type)
            if rnn_type == "GRU":
                model.add(
                    tf.keras.layers.GRU(
                        units=units,
                        return_sequences=return_sequences,
                        kernel_regularizer=tf.keras.regularizers.L2(l2_lambda_value),
                        activation='tanh',
                    )
                )
            else:
                model.add(
                    tf.keras.layers.LSTM(
                        units=units,
                        return_sequences=return_sequences,
                        kernel_regularizer=tf.keras.regularizers.L2(l2_lambda_value),
                        activation='tanh',
                    )
                )
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rate_value))

        # Capa de salida
        model.add(tf.keras.layers.Dense(37, activation="softmax"))

        # Definir el optimizador
        if optimizer_type == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value)
        elif optimizer_type == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_value)
        else:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate_value)

        # Compilar el modelo
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Almacenar información sobre la configuración utilizada
        self.best_model_info = {
            "l2_lambda": l2_lambda_value,
            "dropout_rate": dropout_rate_value,
            "learning_rate": learning_rate_value,
            "embedding_dim": embedding_dim,
            "gru1_units": gru1_units,
            "gru2_units": gru2_units,
            "gru3_units": gru3_units,
            "gru4_units": gru4_units,
            "gru5_units": gru5_units,
            "num_rnn_layers": num_rnn_layers,
            "rnn_types": rnn_types,
            "optimizer": optimizer_type
        }

        return model

    def fit(self, hp, model, x, y, validation_data, callbacks, **kwargs):
        batch_size = hp.Choice('batch_size', values=[16, 32, 64, 128])
        return model.fit(
            x,
            y,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            **kwargs
        )


class HyperparameterTuner:
    def __init__(self, filename, epocas_probar, numeros_anteriores):
        """
        Inicializa el tuner con el archivo de datos, épocas a probar y números anteriores a considerar.

        Args:
            filename (str): Nombre del archivo Excel con los datos.
            epocas_probar (list): Lista de números de épocas a probar.
            numeros_anteriores (list): Lista de cantidades de números anteriores a considerar.
        """
        # Establecer semilla para reproducibilidad
        tf.random.set_seed(42)
        np.random.seed(42)

        # Verificar y crear el directorio 'random_search' si no existe
        self.current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        self.random_search_dir = os.path.join(self.current_dir, "search_parameters")
        os.makedirs(self.random_search_dir, exist_ok=True)
        logging.info(f"Directorio 'search_parameters' establecido en: {self.random_search_dir}")

        # Construir la ruta absoluta al archivo Excel
        self.filename = os.path.join(self.current_dir, filename)
        self.epocas_probar = epocas_probar
        self.numeros_anteriores = numeros_anteriores

        # Verificar si el archivo existe
        if not os.path.exists(self.filename):
            logging.error(f"El archivo '{self.filename}' no existe.")
            raise FileNotFoundError(f"Error: El archivo '{self.filename}' no existe.")

        try:
            self.df = pd.read_excel(self.filename, sheet_name="Salidos")
            logging.info(f"Archivo '{filename}' leído correctamente desde la hoja 'Salidos'.")
        except Exception as e:
            logging.error(f"Error leyendo el archivo de Excel: {e}")
            raise Exception(f"Error leyendo el archivo de Excel: {e}")

        if "Salidos" not in self.df.columns:
            logging.error("La columna 'Salidos' no se encuentra en la hoja de Excel especificada.")
            raise ValueError("Error: la columna 'Salidos' no se encuentra en la hoja de Excel.")

        self.numeros = self.df["Salidos"].values.tolist()
        self.results_filename = os.path.join("resultados_parametros_optimization.xlsx")
        self.best_accuracy = 0
        self.best_model_info = {}

    def _crear_secuencias(self, numero_anterior):
        """
        Crea secuencias de números anteriores y las etiquetas correspondientes.

        Args:
            numero_anterior (int): Número de secuencias anteriores a considerar.

        Returns:
            tuple: (secuencias, etiquetas_one_hot) o (None, None) si hay un error.
        """
        secuencias = []
        siguientes_numeros = []
        if len(self.numeros) < numero_anterior + 1:
            logging.warning("Datos insuficientes en self.numeros para crear secuencias.")
            return None, None

        for i in range(len(self.numeros) - numero_anterior):
            secuencia = self.numeros[i: i + numero_anterior]
            siguiente_numero = self.numeros[i + numero_anterior]
            secuencias.append(secuencia)
            siguientes_numeros.append(siguiente_numero)

        # Convertir a numpy arrays
        secuencias = np.array(secuencias)
        siguientes_numeros = np.array(siguientes_numeros)

        # Aplicar one-hot encoding a las etiquetas
        siguientes_numeros_encoded = tf.keras.utils.to_categorical(siguientes_numeros, num_classes=37)

        logging.info(f"Secuencias y etiquetas creadas para {numero_anterior} números anteriores.")
        return secuencias, siguientes_numeros_encoded

    def tune_hyperparameters(self):
        """
        Realiza la búsqueda de hiperparámetros utilizando Keras Tuner con Bayesian Optimization y Random Search.
        """
        tuning_methods = [
            {"method": BayesianOptimization, "name": "bayesian_optimization"},
            {"method": RandomSearch, "name": "random_search"}
        ]

        for epocas in self.epocas_probar:
            for numero_anterior in self.numeros_anteriores:
                for tuning_method in tuning_methods:
                    logging.info(
                        f"Entrenando con {epocas} épocas y {numero_anterior} números anteriores utilizando {tuning_method['name']}...")

                    # Verificar si ya se realizó una prueba con estos parámetros
                    trial_dir = os.path.join(
                        self.random_search_dir,
                        f"{tuning_method['name']}_{epocas}_epochs_{numero_anterior}_anterior"
                    )

                    # Crear secuencias y etiquetas
                    self.secuencias, self.siguientes_numeros = self._crear_secuencias(numero_anterior)
                    if self.secuencias is None or self.siguientes_numeros is None:
                        continue

                    # Dividir los datos en entrenamiento y prueba
                    X_train, X_test, y_train, y_test = train_test_split(
                        self.secuencias, self.siguientes_numeros, test_size=0.2, random_state=42
                    )

                    # Balancear las clases en el conjunto de entrenamiento
                    smote = SMOTE(random_state=42)
                    try:
                        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                        logging.info("Clases balanceadas utilizando SMOTE.")
                    except Exception as e:
                        logging.warning(f"No se pudo balancear las clases: {e}")
                        X_train_res, y_train_res = X_train, y_train

                    input_shape = (X_train_res.shape[1],)

                    # Definir callbacks
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True
                    )
                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    )
                    model_checkpoint_path = os.path.join(
                        self.random_search_dir,
                        f"best_model_{tuning_method['name']}_{epocas}_epochs_{numero_anterior}_anterior.keras"
                    )
                    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        filepath=model_checkpoint_path,
                        save_best_only=True,
                        monitor="val_loss",
                    )

                    # Definir la carpeta específica para este trial
                    os.makedirs(trial_dir, exist_ok=True)

                    # Inicializar el tuner
                    tuner = tuning_method["method"](
                        hypermodel=MyHyperModel(input_shape),
                        objective="val_accuracy",
                        max_trials=300,
                        executions_per_trial=1,
                        directory=trial_dir,
                        project_name="hyperparameter_tuning",
                        overwrite=False
                    )

                    # Realizar la búsqueda
                    tuner.search(
                        x=X_train_res,
                        y=y_train_res,
                        epochs=epocas,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr, model_checkpoint],
                        verbose=1
                    )

                    # Guardar los resultados
                    self.save_results_to_excel(tuner, epocas, numero_anterior, tuning_method['name'])

                    # Evaluar el mejor modelo encontrado
                    self.evaluate_best_model(tuner, X_test, y_test, epocas, numero_anterior, tuning_method['name'])

                    logging.info(
                        f"Resultados guardados para {epocas} épocas y {numero_anterior} números anteriores utilizando {tuning_method['name']}.")

    def save_results_to_excel(self, tuner, epocas, numero_anterior, tuning_method_name):
        """
        Guarda los resultados de los mejores trials en un archivo Excel.

        Args:
            tuner (BayesianOptimization/RandomSearch): Instancia de Keras Tuner.
            epocas (int): Número de épocas utilizadas en el trial.
            numero_anterior (int): Número de secuencias anteriores utilizadas en el trial.
            tuning_method_name (str): Nombre del método de tuning utilizado.
        """
        # Crear el archivo Excel si no existe
        if not os.path.exists(self.results_filename):
            pd.DataFrame().to_excel(self.results_filename, index=False)
            logging.info(f"Archivo de resultados '{self.results_filename}' creado.")

        # Leer los datos existentes
        try:
            existing_df = pd.read_excel(self.results_filename)
        except Exception as e:
            logging.error(f"Error al leer el archivo de resultados: {e}")
            existing_df = pd.DataFrame()

        # Obtener los mejores trials
        trials = tuner.oracle.get_best_trials(num_trials=10)
        new_results = []

        for trial in trials:
            trial_info = {
                "tuning_method": tuning_method_name,
                "epocas": epocas,
                "numero_anterior": numero_anterior,
                "trial_id": trial.trial_id,
                "score": trial.score,
                "val_accuracy": trial.metrics.get_best_value('val_accuracy'),
                "val_loss": trial.metrics.get_best_value('val_loss'),
            }
            # Añadir los hiperparámetros utilizados en el trial
            for param, value in trial.hyperparameters.values.items():
                trial_info[param] = value
            new_results.append(trial_info)

        # Crear un DataFrame con los nuevos resultados
        new_df = pd.DataFrame(new_results)

        # Concatenar con los datos existentes
        final_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Guardar de nuevo en el archivo Excel
        try:
            final_df.to_excel(self.results_filename, index=False)
            logging.info(f"Resultados de los trials guardados en '{self.results_filename}'.")
        except Exception as e:
            logging.error(f"Error al guardar los resultados en Excel: {e}")

    def evaluate_best_model(self, tuner, X_test, y_test, epocas, numero_anterior, tuning_method_name):
        """
        Evalúa el mejor modelo encontrado en el conjunto de prueba y guarda la información si es el mejor hasta ahora.

        Args:
            tuner (BayesianOptimization/RandomSearch): Instancia de Keras Tuner.
            X_test (np.array): Datos de prueba.
            y_test (np.array): Etiquetas de prueba.
            epocas (int): Número de épocas utilizadas en el trial.
            numero_anterior (int): Número de secuencias anteriores utilizadas en el trial.
            tuning_method_name (str): Nombre del método de tuning utilizado.
        """
        best_model = tuner.get_best_models(num_models=1)[0]

        # Evaluar el mejor modelo en el conjunto de prueba
        loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
        y_pred = best_model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        report = classification_report(y_true, y_pred_classes, digits=4)
        matrix = confusion_matrix(y_true, y_pred_classes)

        logging.info(
            f"Evaluación del Mejor Modelo para {epocas} épocas y {numero_anterior} números anteriores utilizando {tuning_method_name}:")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Classification Report:\n{report}")
        logging.info(f"Confusion Matrix:\n{matrix}\n")

        # Guardar la información del mejor modelo si supera la mejor precisión anterior
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model_info = {
                "tuning_method": tuning_method_name,
                "epocas": epocas,
                "numero_anterior": numero_anterior,
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
            }
            self.save_best_model_info()
            logging.info("Nuevo mejor modelo encontrado y guardado.\n")

    def save_best_model_info(self):
        """
        Guarda la información del mejor modelo encontrado en una hoja separada del archivo Excel.
        """
        if self.best_model_info:
            best_model_df = pd.DataFrame([self.best_model_info])
            try:
                with pd.ExcelWriter(
                    self.results_filename, mode="a", if_sheet_exists="replace"
                ) as writer:
                    best_model_df.to_excel(writer, sheet_name="Best_Model_Info", index=False)
                logging.info(f"Información del mejor modelo guardada en 'Best_Model_Info' de '{self.results_filename}'.")
            except Exception as e:
                logging.error(f"Error al guardar la información del mejor modelo: {e}")
        else:
            logging.warning("No se ha encontrado ningún modelo para guardar información.")


def main():
    # Definir las épocas y números anteriores a probar
    epocas_probar = [50, 100]  # Puedes ajustar estos valores según tus necesidades
    numeros_anteriores = [6, 10]  # Ajusta los números anteriores que deseas probar

    # Nombre del archivo Excel
    filename = "Data/Electromecanica.xlsx"  # Asegúrate de que el nombre sea correcto

    # Crear una instancia del tuner
    tuner = HyperparameterTuner(filename, epocas_probar, numeros_anteriores)

    # Realizar la búsqueda de hiperparámetros
    tuner.tune_hyperparameters()

    # Guardar la información del mejor modelo encontrado
    tuner.save_best_model_info()


if __name__ == "__main__":
    main()
