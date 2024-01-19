class HiperParametros:
    # Inicializa el objeto de la clase con un nombre de archivo y crea el modelo.
    def __init__(self, cantidad):
        # hiperparamtros
        self.numerosAnteriores = 5
        self.lsmt = 320
        self.gru = 256
        self.lsmt2 = 128
        self.l2_lambda = 0.001
        self.dropout_rate = 0.05
        self.learning_rate = 0.003
        self.epoc = 100 if cantidad > 1000 else 10
        self.batchSize = 500


class Parametro_Juego:
    # Inicializa el objeto de la clase con un nombre de archivo y crea el modelo.
    def __init__(self, valor_apuesta, **kwargs):
        # Parametros juegos
        self.valor_ficha = valor_apuesta
        self.lugares_vecinos = 3
        self.numerosAnteriores = 5
        self.numeros_a_predecir = 10
        self.umbral_probilidad = 100
        self.limite = 5

        # Usar kwargs para personalizar atributos adicionales
        if "numeros_a_predecir" in kwargs:
            self.numeros_a_predecir = kwargs["numeros_a_predecir"]

        if "cantidad_vecinos" in kwargs:
            self.lugares_vecinos = kwargs["cantidad_vecinos"]

        if "numerosAnteriores" in kwargs:
            self.numerosAnteriores = kwargs["numerosAnteriores"]

        if "umbral_probilidad" in kwargs:
            self.umbral_probilidad = kwargs["umbral_probilidad"]

        if "limite" in kwargs:
            self.limite = kwargs["limite"]
