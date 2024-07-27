
class Numero_jugar:
    def __init__(self, numero, probabilidad_redondeada, vecinos=1):
        self.numero = numero
        self.probabilidad = probabilidad_redondeada
        self.vecinos = (vecinos * 2) + 1
        self.tardancia = 1
        self.repetido = 0

    def Jugar(self):
        self.tardancia += 1

    def Aumentar_probailidad(self, nueva_probabilidad):
        self.probabilidad += nueva_probabilidad
        self.repetido += 1
        self.tardancia -= 1

    def __str__(self):
        return f"(N:{self.numero},P:{self.probabilidad},T:{self.tardancia},R:{self.repetido})"
