import optuna
from optuna.trial import TrialState
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from Entity.Modelo import Modelo
from Entity.Parametro import HiperParametros
import datetime


STUDY_NAME = "optimization_study"
STORAGE_PATH = "sqlite:///optuna_study.db"


def objective(trial):
    # Sugerir hiperparámetros
    numerosAnteriores = trial.suggest_int('numerosAnteriores', 3, 10)
    capa1 = trial.suggest_int('capa1', 128, 512, step=64)
    capa2 = trial.suggest_int('capa2', 64, 256, step=64)
    capa3 = trial.suggest_int('capa3', 32, 128, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batchSize = trial.suggest_int('batchSize', 128, 512, step=128)
    epochs = trial.suggest_int('epochs', 50, 200, step=50)

    # Configurar hiperparámetros
    hiperparametros = HiperParametros(
        numerosAnteriores=numerosAnteriores,
        capa1=capa1,
        capa2=capa2,
        capa3=capa3,
        dropout_rate=dropout_rate,
        l2_lambda=l2_lambda,
        learning_rate=learning_rate,
        batchSize=batchSize,
        epochs=epochs
    )

    # Crear modelo y preparar datos
    modelo = Modelo(filename="Data/Electromecanica.xlsx", hiperparametro=hiperparametros)
    model = modelo._crear_modelo()
    secuencias, siguientes_numeros = modelo._crear_secuencias()
    X_train, X_val, y_train, y_val = train_test_split(secuencias, siguientes_numeros, test_size=0.2)

    # Configurar callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Entrenamiento con validación cruzada
    val_losses = []
    metrics = {
        'accuracy': [],
        'loss': [],
        'val_accuracy': [],
        'val_loss': []
    }
    for fold in range(3):
        X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2)
        history = model.fit(
            X_t, y_t,
            epochs=epochs,
            batch_size=batchSize,
            validation_data=(X_v, y_v),
            callbacks=callbacks,
            verbose=0
        )
        val_losses.append(min(history.history['val_loss']))

        # Guardar mejores valores
        metrics['accuracy'].append(max(history.history['accuracy']))
        metrics['loss'].append(min(history.history['loss']))
        metrics['val_accuracy'].append(max(history.history['val_accuracy']))
        metrics['val_loss'].append(min(history.history['val_loss']))

    # Guardar métricas en trial
    for key, values in metrics.items():
        trial.set_user_attr(key, np.mean(values))

    return np.mean(metrics['val_accuracy'])  # Optimizar val_accuracy


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
    n_trials=20,
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

# Crear DataFrame y guardar en Excel
df = pd.DataFrame(mejores_params)
df.to_excel('mejores_hiperparametros.xlsx', index=False)

print(f"\nOptimización completada en {tiempo_total.total_seconds() / 60:.2f} minutos")
print(f"Trials completados: {len(study.trials)}")
print(f"Mejor pérdida: {study.best_value:.4f}")
print(f"\nMejores métricas encontradas:")
print(f"Accuracy: {study.best_trial.user_attrs['accuracy']:.4f}")
print(f"Loss: {study.best_trial.user_attrs['loss']:.4f}")
print(f"Val Accuracy: {study.best_trial.user_attrs['val_accuracy']:.4f}")
print(f"Val Loss: {study.best_trial.user_attrs['val_loss']:.4f}")
