# Define una clase Contador para agrupar tus variables
class Contador:
    def __init__(self):
        self.numeros = []
        self.numeros_partida = []
        self.acierto_predecidos = 0
        self.acierto_vecinos_1lugar = 0
        self.acierto_vecinos_2lugar = 0
        self.acierto_vecinos_3lugar = 0
        self.acierto_vecinos_4lugar = 0
        self.ingresados = 0
        self.jugados = 0
        self.aciertos_totales = 0
        self.Sin_salir_nada = 0

    def incrementar_ingresados(self, numero):
        self.ingresados += 1
        self.numeros.append(numero)
        self.numeros_partida.append(numero)

    def borrar_ultimo_numero(self):
        self.ingresados -= 1
        self.numeros.pop()
        self.numeros_partida.pop()

        # if self.ingresados > 10 or self.jugados > 0:
        #     self.jugados -= 1

    def incrementar_jugados(self):
        self.jugados += 1

    def incrementar_aciertos_totales(self, numero):
        self.aciertos_totales += numero

    def incrementar_predecidos(self):
        self.acierto_predecidos += 1

    def incrementar_aciertos_vecinos_1lugar(self):
        self.acierto_vecinos_1lugar += 1

    def incrementar_aciertos_vecinos_2lugar(self):
        self.acierto_vecinos_2lugar += 1

    def incrementar_aciertos_vecinos_3lugar(self):
        self.acierto_vecinos_3lugar += 1

    def incrementar_aciertos_vecinos_4lugar(self):
        self.acierto_vecinos_4lugar += 1

    def incrementar_supero_limite(self):
        self.Sin_salir_nada += 1

    def sacarEfectividad(self):
        if self.jugados == 0:
            return "0"
        else:
            return f"{(self.aciertos_totales / (self.jugados)) * 100:.0f}%"
