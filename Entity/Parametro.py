class HiperParametros:
    # Inicializa el objeto de la clase con un nombre de archivo y crea el modelo.
    def __init__(self, **kwargs):
        # hiperparametros
        self.numerosAnteriores = kwargs.get("numerosAnteriores", 10)
        self.capa1 = kwargs.get("capa1", 384)  # Mejor valor encontrado para rnn_units
        self.capa2 = kwargs.get("capa2", 256)
        self.capa3 = kwargs.get("capa3", 128)
        self.l2_lambda = kwargs.get("l2_lambda", 0.008640737182527825)  # Mejor valor encontrado
        self.dropout_rate = kwargs.get("dropout_rate", 0.3875188255460017)  # Mejor valor encontrado
        self.learning_rate = kwargs.get("learning_rate", 0.00016993366812604327)  # Mejor valor encontrado
        self.epochs = kwargs.get("epochs", 200)  # Mejor valor encontrado
        self.batchSize = kwargs.get("batchSize", 256)


class Parametro_Juego:
    # Inicializa el objeto de la clase con un nombre de archivo y crea el modelo.
    def __init__(
        self,
        cantidad_vecinos,
        limite_tardancia,
        umbral_probabilidad,
        **kwargs
    ):
        # Parametros juegos
        self.limite_tardancia = limite_tardancia
        self.lugares_vecinos = cantidad_vecinos
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
