# Define una clase Contador para agrupar tus variables
class Numero_pretendiente:
    def __init__(self, numero):
        self.numero = numero
        self.tardancia = 0
        self.probabilidad = 0

    def Jugar(self):
        self.tardancia += 1

    def Aumentar_probailidad(self, nueva_probabilidad):
        self.probabilidad += nueva_probabilidad
