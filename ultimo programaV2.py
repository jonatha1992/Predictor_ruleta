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

        self.contador = Contador()
        self.contador.numeros = self.df["Salidos"].values.tolist()


        self.resultados = []
        

        # Parametros
        self.numeros_a_predecir = 3
        self.lsmt = 256
        self.gru = 224
        self.lsmt2 = 128
        self.l2_lambda = 0.001
        self.dropout_rate = 0.01
        self.learning_rate = 0.003  # Tasa de aprendizaje inicial
        self.epoc = 50
        self.batchSize = 512
        
        self.model = self._crear_modelo()
        # self.guardar_modelo()

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

        model.add(LSTM(
                self.lsmt,  # Incrementar el número de unidades en la primera capa LSTM
                input_shape=(10, 1),
                return_sequences=True,
                kernel_regularizer=l2(self.l2_lambda),
                    )
                )
        # Reducir el número de unidades en la última capa LSTM
        model.add(Dropout(self.dropout_rate))
        model.add( GRU(self.gru, return_sequences=True,
                        kernel_regularizer=l2(self.l2_lambda)
                     )
                        )  
        # Cambiar a capa GRU
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.lsmt2, kernel_regularizer=l2(self.l2_lambda)))  
        
        # Reducir el número de unidades en la última capa LSTM
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(37, activation="softmax"))

        # Compilar modelo
        optimizer = Adam(
            learning_rate=self.learning_rate
        )  # Usar una tasa de aprendizaje personalizada
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        # Entrenar modelo
        model.fit(
                secuencias, 
                siguientes_numeros, 
                epochs=self.epoc, 
                batch_size=self.batchSize,
                callbacks=[early_stopping] 
        )

        return model
    
    
    
        # Crea secuencias de números y los números siguientes para entrenar el modelo.
    def _crear_secuencias(self):
        secuencias = []
        siguientes_numeros = []
        for i in range(len(self.contador.numeros) - 11):
            secuencias.append(self.contador.numeros[i : i + 10])
            siguientes_numeros.append(self.contador.numeros[i + 10])
        secuencias = pad_sequences(np.array(secuencias))
        siguientes_numeros = to_categorical(np.array(siguientes_numeros))
        return secuencias, siguientes_numeros

    # Predice los próximos números.
    def predecir(self):
        secuencia_entrada = np.array(self.contador.numeros[-10:]).reshape(1, 10, 1)
        predicciones = self.model.predict(secuencia_entrada, verbose=0)
        self.resultados = sorted(predicciones[0].argsort()[-self.numeros_a_predecir:][::-1])

    # Verifica si un número coincide con los resultados predichos y actualiza los contadores.
    def verificar_numero(self, numero):
        acierto= False
        es_vecino_cercano = False
        es_vecino_lejano = False
        self.contador.incrementar_ingresados(numero)
        
        if self.contador.ingresados > 10:
            self.contador.incrementar_jugados()

            if numero in self.resultados:
                self.contador.incrementar_aciertos()
                print(f"¡Acierto! El número {numero} coincide con uno de los resultados.")
                self.df_nuevo.at[len(self.df_nuevo), "Acierto"] = "acierto"
                acierto= True
            else:
                self.contador.actualizar_sin_aciertos()

            for vecino in self.resultados:
                if numero in vecinosCercanos[vecino]:
                    self.contador.incrementar_aciertos_vecinos_cercanos()
                    es_vecino_cercano = True
                    print(f"¡Vecino! El número {numero} es vecino cercano de {vecino}.")

                if numero in vecinosLejanos[vecino]:
                    self.contador.incrementar_aciertos_vecinos_lejanos()
                    es_vecino_lejano = True
                    print(f"¡Vecino! El número {numero} es vecino lejano de {vecino}.")

            if es_vecino_cercano:
                self.df_nuevo.at[len(self.df_nuevo), "Vecino"] = "VC"
            else:
                self.contador.actualizar_sin_vecinos_cercanos()

            if es_vecino_lejano:
                self.df_nuevo.at[len(self.df_nuevo), "Vecino lejano"] = "VL"
            else:
                self.contador.actualizar_sin_vecinos_lejanos()


            if acierto or es_vecino_cercano or es_vecino_lejano:
                self.contador.reiniciar_sin_salir_nada()
            else:
                self.contador.actualizar_sin_salir_nada()

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
        print(f"Aciertos Totales: {self.contador.aciertos_totales}")
        print(f"Sin salir: {self.contador.Sin_salir_nada}\n")
        print(f"Aciertos Resultados: {self.contador.aciertos}")
        print(f"Aciertos de vecinos Cercanos: {self.contador.acierto_vecinos_cercanos}")
        print(f"Aciertos de vecinos Lejanos: {self.contador.acierto_vecinos_lejanos}\n")
        print(f"\nLas posibles predicciones para el próximo número son: {self.resultados}\n")

    # Borra el último número ingresado y actualiza el contador.
    def borrar(self):
        if self.contador.numeros:
            self.contador.borrar_ultimo_numero()
            self.df_nuevo = self.df_nuevo[:-1]  # Eliminar la última fila del DataFrame nuevo
            print("Último número borrado")

    def guardar_modelo(self):
        modelo_path = "Modelo/mi_modelo"  # Ruta relativa a la carpeta "modelo"
        self.model.save(modelo_path)  # Guarda el modelo en la ubicación especificada

    def generar_reporte(self):
        fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Crear un diccionario con los datos
        datos = {
            "Juego fecha y hora": fecha_hora_actual,
            "Numeros jugados": self.contador.jugados,
            "Aciertos Totales": self.contador.aciertos_totales,
            "Maximo sin salir totales": self.contador.Maximo_Sin_salir_nada,
            "Aciertos de resultados": self.contador.aciertos,
            "Aciertos de vecinos cercanos": self.contador.acierto_vecinos_cercanos,
            "Aciertos de vecinos lejanos": self.contador.acierto_vecinos_lejanos,
            "Maximo valor que tardo sin salir acierto": self.contador.max_sin_acierto,
            "Maximo valor que tardo sin salir vecinos cercanos": self.contador.max_sin_vecinos_cercanos,
            "Maximo valor que tardo sin salir vecinos lejanos": self.contador.max_sin_vecinos_lejanos,
            "l2": self.l2_lambda,
            "dropout rate": self.dropout_rate,
            "learning rate": self.learning_rate,
            "epoca": self.epoc,
            "batch_size": self.batchSize,
            "Nros a Predecir": self.numeros_a_predecir,
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
            predictor.guardar_excel()
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
