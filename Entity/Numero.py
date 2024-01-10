# Define una clase Contador para agrupar tus variables
class Numero:
    def __init__(self, numero):
        self.numero = numero
        self.tardancia = 0
        self.jugado = 0
        self.sin_salir = 0
        self.valor_pleno = 36
        self.ganado = 0

    def Jugar(self):
        self.jugado += 1

    def Gano(self, orden):
        self.ganado = self.pleno * orden
