import numpy as np
import pandas as pd
import tensorflow as tf
import os
from Entity.Contador import Contador 
from Entity.Numeros_Simulacion import  Simulador
from datetime import datetime
from Vecinos import vecinosCercanos, vecinosLejanos
from tensorflow.keras.layers import LSTM, Dense, Dropout , GRU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



class Predictor:
    # Inicializa el objeto de la clase con un nombre de archivo y crea el modelo.
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_excel(filename, sheet_name="Salidos")
        self.numeros = self.df["Salidos"].values.tolist()

        self.contador = Contador()


        self.resultados = []
        
        #Parametros
        self.l2_lambda = 0.001
        self.dropout_rate = 0.01
        self.learning_rate = 0.003  # Tasa de aprendizaje inicial
        self.epoc = 40
        self.batchSize =  512

        self.model = self._crear_modelo()
        self.guardar_modelo()

        # # Ruta relativa a la carpeta "modelo" en el mismo directorio que tu archivo de código
        # modelo_path = 'Modelo/mi_modelo'

        # if os.path.exists(modelo_path): # Verifica si ya hay un modelo guardado
        #     self.model = load_model(modelo_path) # Carga el modelo guardado si existe
        # else:
        #     self.model = self._crear_modelo()
        #     self.guardar_modelo() # Guarda el modelo después de entrenarlo

        self.df_nuevo = self.df.copy()

    
     # Crea el modelo de red neuronal LSTM.
    def _crear_modelo(self):
        secuencias, siguientes_numeros = self._crear_secuencias()
        model = Sequential()

        model.add(
            LSTM(
                256,  # Incrementar el número de unidades en la primera capa LSTM
                input_shape=(10, 1),
                return_sequences=True,
                kernel_regularizer=l2(self.l2_lambda),
            )
        )
        model.add(Dropout(self.dropout_rate))
        model.add(GRU(128, return_sequences=True, kernel_regularizer=l2(self.l2_lambda)))  # Cambiar a capa GRU
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(64, kernel_regularizer=l2(self.l2_lambda)))  # Reducir el número de unidades en la última capa LSTM
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(37, activation="softmax"))

        # Compilar modelo
        optimizer = Adam(learning_rate=self.learning_rate)  # Usar una tasa de aprendizaje personalizada
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        # Usar la técnica de Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        # Entrenar modelo
        model.fit(
            secuencias,
            siguientes_numeros,
            epochs=self.epoc,
            batch_size= self.batchSize,
            callbacks=[early_stopping]  # Agregar el callback de Early Stopping
        )
       
        return model
    
    
    
        # Crea secuencias de números y los números siguientes para entrenar el modelo.
    def _crear_secuencias(self):
        secuencias = []
        siguientes_numeros = []
        for i in range(len(self.numeros) - 11):
            secuencias.append(self.numeros[i : i + 10])
            siguientes_numeros.append(self.numeros[i + 10])
        secuencias = pad_sequences(np.array(secuencias))
        siguientes_numeros = to_categorical(np.array(siguientes_numeros))
        return secuencias, siguientes_numeros

    # Predice los próximos números.
    def predecir(self):
        secuencia_entrada = np.array(self.numeros[-10:]).reshape(1, 10, 1)
        predicciones = self.model.predict(secuencia_entrada, verbose=0)
        # self.resultados = sorted(np.random.choice(37, size=5, p=predicciones[0]))
        indices_mayores = predicciones[0].argsort()[-3:][::-1] # Esto te dará los índices de los tres números con las mayores probabilidades
        self.resultados = sorted(indices_mayores)

    # Verifica si un número coincide con los resultados predichos y actualiza los contadores.
    def verificar_numero(self, numero):
        acierto= False
        es_vecino_cercano = False
        es_vecino_lejano = False
        self.contador.ingresados += 1

        if self.contador.ingresados > 10:
            self.contador.jugados += 1
            if numero in self.resultados:
                self.contador.aciertos += 1
                self.contador.sin_aciertos = 0
                print(
                    f"¡Acierto! El número {numero} coincide con uno de los resultados."
                )
                self.df_nuevo.at[len(self.df_nuevo), "Acierto"] = "acierto"
                acierto= True
            else:
                self.contador.sin_aciertos += 1

            for vecino in self.resultados:
                if numero in vecinosCercanos[vecino]:
                    self.contador.sin_vecinos_cercanos = 0
                    self.contador.acierto_vecinos_cercanos += 1
                    es_vecino_cercano = True
                    print(f"¡Vecino! El número {numero} es vecino cercano de {vecino}.")

                if numero in vecinosLejanos[vecino]:
                    self.contador.sin_vecinos_lejanos = 0
                    self.contador.acierto_vecinos_lejanos += 1
                    es_vecino_lejano = True
                    print(f"¡Vecino! El número {numero} es vecino lejano de {vecino}.")

            if es_vecino_cercano:
                self.df_nuevo.at[len(self.df_nuevo), "Vecino"] = "VC"
            else:
                self.contador.sin_vecinos_cercanos += 1
            if es_vecino_lejano:
                self.df_nuevo.at[len(self.df_nuevo), "Vecino lejano"] = "VL"
            else:
                self.contador.sin_vecinos_lejanos += 1

            if acierto or  es_vecino_cercano or es_vecino_lejano:
                self.contador.aciertos_totales += 1

            if self.contador.sin_aciertos > self.contador.max_sin_acierto:
                self.contador.max_sin_acierto = self.contador.sin_aciertos

            if self.contador.sin_vecinos_cercanos > self.contador.max_sin_vecinos_cercanos:
                self.contador.max_sin_vecinos_cercanos = self.contador.sin_vecinos_cercanos

            if self.contador.sin_vecinos_lejanos > self.contador.max_sin_vecinos_lejanos:
                self.contador.max_sin_vecinos_lejanos = self.contador.sin_vecinos_lejanos

    # Actualiza el DataFrame con el número ingresado y los resultados de las predicciones.
    def actualizar_dataframe(self, numero_ingresado):
        
        self.df_nuevo.loc[len(self.df_nuevo) + 1, "Salidos"] = (numero_ingresado,)
        self.df_nuevo.at[len(self.df_nuevo), "Resultados"] = str(self.resultados)
        self.df_nuevo.loc[
            len(self.df_nuevo), "Numero jugado"
        ] = self.contador.ingresados

    # Guarda el DataFrame en un archivo de Excel.
    def guardar_excel(self):
        self.generar_reporte()
        self.df_nuevo.to_excel("Datos.xlsx", sheet_name="Salidos", index=False)

    # Muestra los resultados y las estadísticas.
    def mostrar_resultados(self):
        print(self.df_nuevo.tail(10))
        print(f"Numeros Jugados: {self.contador.jugados}")
        print(f"Aciertos Totales: {self.contador.aciertos_totales}\n")
        print(f"Aciertos Resultados: {self.contador.aciertos}")
        print(f"Aciertos de vecinos Cercanos: {self.contador.acierto_vecinos_cercanos}")
        print(f"Aciertos de vecinos Lejanos: {self.contador.acierto_vecinos_lejanos}\n")
        print(f"Sin salir resultados: {self.contador.sin_aciertos}")
        print(f"Sin salir vecinos Cercanos: {self.contador.sin_vecinos_cercanos}")
        print(f"Sin salir vecinos Lejanos: {self.contador.sin_vecinos_lejanos}")
        print(
            f"\nLas posibles predicciones para el próximo número son: {self.resultados}\n"
        )

    # Borra el último número ingresado y actualiza el contador.
    def borrar(self):
        if self.numeros:
            self.numeros.pop()
            self.contador.ingresados -= 1
            self.contador.jugados -= 1
            self.df_nuevo = self.df_nuevo[
                :-1
            ]  # Eliminar la última fila del DataFrame nuevo
            print("Último número borrado.")
        else:
            print("No hay números para borrar.")

    def guardar_modelo(self):
        modelo_path = "Modelo/mi_modelo"  # Ruta relativa a la carpeta "modelo"
        self.model.save(modelo_path)  # Guarda el modelo en la ubicación especificada

    def generar_reporte(self):
        fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Crear un diccionario con los datos
        datos = {
            'Juego fecha y hora': fecha_hora_actual,
            'Numeros jugados': self.contador.jugados,
            'Aciertos Totales': self.contador.aciertos_totales,
            'Aciertos de resultados': self.contador.aciertos,
            'Aciertos de vecinos cercanos': self.contador.acierto_vecinos_cercanos,
            'Aciertos de vecinos lejanos': self.contador.acierto_vecinos_lejanos,
            'Maximo valor que tardo sin salir acierto': self.contador.max_sin_acierto,
            'Maximo valor que tardo sin salir vecinos cercanos': self.contador.max_sin_vecinos_cercanos,
            'Maximo valor que tardo sin salir vecinos lejanos': self.contador.max_sin_vecinos_lejanos,
            'l2': self.l2_lambda,
            'dropout rate': self.dropout_rate,
            'learning rate': self.learning_rate,
            'epoca': self.epoc,
            'batch_size': self.batchSize
        }
        
        # Convertir el diccionario en un DataFrame de Pandas
        df = pd.DataFrame([datos])
        
        archivo_excel = "reportes.xlsx"
        
        # Comprobar si el archivo de Excel ya existe
        if os.path.exists(archivo_excel):
            # Si existe, leerlo y agregar la nueva fila
            df_existente = pd.read_excel(archivo_excel)
            df_final = pd.concat([df_existente, df], ignore_index=True)
        else:
            # Si no existe, usar el DataFrame creado
            df_final = df

        # Guardar el DataFrame en el archivo de Excel
        df_final.to_excel(archivo_excel, index=False)
        return datos    

# Función principal que ejecuta el programa.
def main():
    predictor = Predictor("datos.xlsx")
    while True:
        array= []
        opcion = input(
            "\nIngresa un nuevo número o 'salir' para terminar el programa: "
        )
        if opcion.lower() == "salir":
                # predictor.guardar_excel()
            predictor.generar_reporte()
            break

        if opcion.lower() == "-":
            predictor.borrar()
            continue
        try:
            numero = int(opcion)
            array.append(numero)
            
            if numero < 0 or numero > 36:
                print("El número debe estar entre 0 y 36. Inténtalo nuevamente.")
                continue
            
            predictor.verificar_numero(numero)
            predictor.predecir()
            predictor.actualizar_dataframe(numero)
            predictor.mostrar_resultados()

        except ValueError:
            print("Valor ingresado no válido. Inténtalo nuevamente.")

# Si el script se ejecuta como programa principal, llama a la función main().
if __name__ == "__main__":
    main()
