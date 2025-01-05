import optuna
from optuna.trial import TrialState
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from Entity.Modelo import Modelo
from Entity.Parametro import HiperParametros
import datetime


def objective(trial):
    # Sugerir hiperparámetros
    numerosAnteriores = trial.suggest_int('numerosAnteriores', 5, 10)
    gru1 = trial.suggest_int('gru1', 64, 512)
    gru2 = trial.suggest_int('gru2', 64, 512)
    gru3 = trial.suggest_int('gru3', 64, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    batchSize = trial.suggest_int('batchSize', 128, 512)

    # Configurar hiperparámetros
    hiperparametros = HiperParametros(
        numerosAnteriores=numerosAnteriores,
        gru1=gru1,
        gru2=gru2,
        gru3=gru3,
        dropout_rate=dropout_rate,
        l2_lambda=l2_lambda,
        learning_rate=learning_rate,
        batchSize=batchSize
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
    for fold in range(3):
        X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2)
        history = model.fit(
            X_t, y_t,
            epochs=100,
            batch_size=batchSize,
            validation_data=(X_v, y_v),
            callbacks=callbacks,
            verbose=0
        )
        val_losses.append(min(history.history['val_loss']))

    return np.mean(val_losses)


# Tiempo inicial
tiempo_inicio = datetime.datetime.now()

# Configurar y ejecutar estudio
study = optuna.create_study(direction='minimize')
study.optimize(
    objective,
    n_trials=50,
    show_progress_bar=True
)

# Calcular tiempo total
tiempo_total = datetime.datetime.now() - tiempo_inicio

# Guardar resultados
mejores_params = {
    'Números Anteriores': [study.best_params['numerosAnteriores']],
    'GRU1': [study.best_params['gru1']],
    'GRU2': [study.best_params['gru2']],
    'GRU3': [study.best_params['gru3']],
    'Dropout Rate': [study.best_params['dropout_rate']],
    'L2 Lambda': [study.best_params['l2_lambda']],
    'Learning Rate': [study.best_params['learning_rate']],
    'Batch Size': [study.best_params['batchSize']],
    'Mejor Pérdida': [study.best_value],
    'Tiempo Total (min)': [tiempo_total.total_seconds() / 60],
    'Trials Completados': [len(study.trials)]
}

# Crear DataFrame y guardar en Excel
df = pd.DataFrame(mejores_params)
df.to_excel('mejores_hiperparametros.xlsx', index=False)

print(f"\nOptimización completada en {tiempo_total.total_seconds() / 60:.2f} minutos")
print(f"Trials completados: {len(study.trials)}")
print(f"Mejor pérdida: {study.best_value:.4f}")
