# Define una clase Contador para agrupar tus variables
class Contador:
    def __init__(self):
        self.numeros = []
        self.acierto_predecidos = 0
        self.acierto_vecinos_1lugar = 0
        self.acierto_vecinos_2lugar = 0
        self.acierto_vecinos_3lugar = 0
        self.acierto_vecinos_4lugar = 0
        self.ingresados = 0
        self.jugados = 0
        self.aciertos_totales = 0
        self.Sin_salir_nada = 0
        self.ganancia_totales = 0
        self.perdida_total = 0
        self.ganancia_neta = 0
        self.capital_inicial = 0

    def Calcular_ganancia(self):
        self.ganancia_neta = self.ganancia_totales - self.perdida_total
        return self.ganancia_neta
    def incrementar_ganancias_totales(self, ganancia_neta):
        self.ganancia_totales += ganancia_neta
    def incrementar_ingresados(self, numero):
        self.ingresados += 1
        self.numeros.append(numero)

    def borrar_ultimo_numero(self):
        self.ingresados -= 1
        self.numeros.pop()

        if self.ingresados > 7 or self.jugados > 0:
            self.jugados -= 1

    def incrementar_jugados(self):
        self.jugados += 1

    def incrementar_aciertos_totales(self):
        self.aciertos_totales += 1

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

    def incrementar_supero_limite(self, perdida):
        self.Sin_salir_nada += 1
        self.perdida_total += perdida

    def sacarEfectividad(self):
        if self.jugados == 0:
            return 0
        else:
          return  (self.aciertos_totales / self.jugados)
