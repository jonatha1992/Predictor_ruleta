# Define una clase Contador para agrupar tus variables
class Numero_pretendiente:
    valor_ficha = 100

    def __init__(self, numero, probabilidad_redondeada):
        self.numero = numero
        self.tardancia = 0
        self.probabilidad = probabilidad_redondeada
        self.repetido = 0
        self.ganancia = 0

    def Jugar(self):
        self.tardancia += 1

    def Aumentar_probailidad(self, nueva_probabilidad):
        self.probabilidad += nueva_probabilidad
        self.tardancia = 0
        self.repetido += 1

    def Calcular_ganancia(self):
        self.ganancia = (self.valor_pleno * self.tardancia) * self.valor_ficha

    def __str__(self):
        return f"(N:{self.numero},P:{self.probabilidad},T:{self.tardancia})"
