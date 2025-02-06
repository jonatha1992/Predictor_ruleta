from Entity.modelo import Modelo
from Entity.parametro import HiperParametros
from optuna.trial import TrialState
import optuna
from sklearn.model_selection import KFold
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import datetime
import os

# Environment variables first
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Standard imports

# ML imports

# Local imports

STUDY_NAME = "optimization_study"
STORAGE_PATH = "sqlite:///optuna_study.db"


def objective(trial):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Hyperparameter suggestions with better ranges
    hiperparametros = HiperParametros(
        numerosAnteriores=trial.suggest_int('numerosAnteriores', 5, 10),
        capa1=trial.suggest_int('capa1', 64, 512, step=64),
        capa2=trial.suggest_int('capa2', 32, 256, step=32),
        capa3=trial.suggest_int('capa3', 16, 128, step=16),
        dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.5),
        l2_lambda=trial.suggest_float('l2_lambda', 1e-6, 1e-3, log=True),
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        batchSize=trial.suggest_int('batchSize', 32, 256, step=32),
        epochs=trial.suggest_int('epochs', 50, 150, step=25)
    )

    try:
        modelo = Modelo("Data/Electromecanica.xlsx", hiperparametros)
        model = modelo._crear_modelo()
        secuencias, siguientes_numeros = modelo._crear_secuencias()

        # K-fold cross validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
        metrics = {
            'accuracy': [], 'loss': [],
            'val_accuracy': [], 'val_loss': []
        }

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

        for fold, (train_idx, val_idx) in enumerate(kfold.split(secuencias)):
            X_train, X_val = secuencias[train_idx], secuencias[val_idx]
            y_train, y_val = siguientes_numeros[train_idx], siguientes_numeros[val_idx]

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=hiperparametros.epochs,
                batch_size=hiperparametros.batchSize,
                callbacks=callbacks,
                verbose=0
            )

            metrics['accuracy'].append(max(history.history['accuracy']))
            metrics['loss'].append(min(history.history['loss']))
            metrics['val_accuracy'].append(max(history.history['val_accuracy']))
            metrics['val_loss'].append(min(history.history['val_loss']))

        # Store mean metrics in trial
        for metric_name, values in metrics.items():
            mean_value = np.mean(values)
            trial.set_user_attr(metric_name, mean_value)

        return np.mean(metrics['val_accuracy'])

    except Exception as e:
        print(f"Trial failed: {e}")
        return None


# Tiempo inicial
tiempo_inicio = datetime.datetime.now()

# Crear o cargar estudio existente
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_PATH,
    direction='maximize',
    load_if_exists=True  # Carga el estudio si existe
)

print(f"Comenzando optimización desde trial #{len(study.trials)}")

# Configurar y ejecutar estudio
study.optimize(
    objective,
    n_trials=100,
    show_progress_bar=True
)

# Calcular tiempo total
tiempo_total = datetime.datetime.now() - tiempo_inicio

# Guardar resultados
mejores_params = {
    'Números Anteriores': [study.best_params['numerosAnteriores']],
    'CAPA1': [study.best_params['capa1']],
    'CAPA2': [study.best_params['capa2']],
    'CAPA3': [study.best_params['capa3']],
    'Dropout Rate': [study.best_params['dropout_rate']],
    'L2 Lambda': [study.best_params['l2_lambda']],
    'Learning Rate': [study.best_params['learning_rate']],
    'Batch Size': [study.best_params['batchSize']],
    'Epochs': [study.best_params['epochs']],
    'Mejor Pérdida': [study.best_value],
    'Tiempo Total (min)': [tiempo_total.total_seconds() / 60],
    'Trials Completados': [len(study.trials)],
    'Accuracy': [study.best_trial.user_attrs['accuracy']],
    'Loss': [study.best_trial.user_attrs['loss']],
    'Val Accuracy': [study.best_trial.user_attrs['val_accuracy']],
    'Val Loss': [study.best_trial.user_attrs['val_loss']]
}
