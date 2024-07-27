class HiperParametros:
    # Inicializa el objeto de la clase con un nombre de archivo y crea el modelo.
    def __init__(self, cantidad):
        # hiperparamtros
        self.numerosAnteriores = 9
        self.gru1 = 320
        self.gru2 = 128
        self.gru3 = 128
        self.l2_lambda = 0.001
        self.dropout_rate = 0.05
        self.learning_rate = 0.003
        self.epoc = 100 if cantidad > 1000 else 10
        self.batchSize = 500


class Parametro_Juego:
    # Inicializa el objeto de la clase con un nombre de archivo y crea el modelo.
    def __init__(
        self,
        cantidad_vecinos,
        limite_juego,
        umbral_probabilidad,
        **kwargs
    ):
        # Parametros juegos
        self.limite_juego = limite_juego
        self.lugares_vecinos = cantidad_vecinos
        self.numerosAnteriores = 6
        self.numeros_a_predecir = 10
        self.umbral_probilidad = umbral_probabilidad

        # Usar kwargs para personalizar atributos adicionales
        if "numeros_a_predecir" in kwargs:
            self.numeros_a_predecir = kwargs["numeros_a_predecir"]

        if "cantidad_vecinos" in kwargs:
            self.lugares_vecinos = kwargs["cantidad_vecinos"]

        if "num_Anteriores" in kwargs:
            self.numerosAnteriores = kwargs["num_Anteriores"]
        if "juego" in kwargs:
            self.juego = kwargs["juego"]
