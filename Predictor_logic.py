from Entity.Predictor import Predictor
from Entity.Parametro import Parametro_Juego
from Config import get_excel_file


class PredictorLogic:
    def __init__(self):
        self.predictor = None

    def iniciar_predictor(self, ruleta_tipo, cantidad_vecinos, limite_juego, umbral_probabilidad):
        params = {
            "cantidad_vecinos": cantidad_vecinos,
            "limite_juego": limite_juego,
            "umbral_probabilidad": umbral_probabilidad
        }
        parametros_juego = Parametro_Juego(**params)
        excel_file = get_excel_file(ruleta_tipo)
        self.predictor = Predictor(excel_file, parametros_juego)

    def predict_number(self, number):
        if not self.predictor:
            raise ValueError("El predictor no ha sido iniciado.")
        self.predictor.verificar_resultados(number)
        self.predictor.predecir()
        self.predictor.actualizar_dataframe(number)
        return self.predictor.mostrar_resultados()

    def delete_last(self):
        if not self.predictor or not self.predictor.contador.numeros_partida:
            raise ValueError("No hay números para borrar.")
        ultimo_numero = self.predictor.contador.numeros_partida[-1]
        self.predictor.borrar()
        return ultimo_numero

    def reset(self):
        if self.predictor:
            self.predictor.guardar_reporte()
        self.predictor = None

    def guardar_numeros(self):
        if not self.predictor or not self.predictor.contador.numeros_partida:
            raise ValueError("No hay números para guardar.")
        self.predictor.guardar_excel()

    def get_estadisticas(self):
        if not self.predictor or not self.predictor.contador:
            return None
        return {
            "ingresados": self.predictor.contador.ingresados,
            "jugados": self.predictor.contador.jugados,
            "aciertos_totales": self.predictor.contador.aciertos_totales,
            "sin_salir_nada": self.predictor.contador.Sin_salir_nada,
            "numeros_partida": self.predictor.contador.numeros_partida,
            "numeros_a_jugar": self.predictor.numeros_a_jugar
        }
