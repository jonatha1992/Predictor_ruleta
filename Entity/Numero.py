# Define una clase Contador para agrupar tus variables
class Numero_pretendiente:
    def __init__(self, numero, probabilidad_redondeada):
        self.numero = numero
        self.probabilidad = probabilidad_redondeada
        self.tardancia = 1
        self.repetido = 0

    def Jugar(self):
        self.tardancia += 1

    def Aumentar_probailidad(self, nueva_probabilidad):
        self.probabilidad += nueva_probabilidad
        self.tardancia -= 1
        self.repetido += 1

    def __str__(self):
        return f"(N:{self.numero},P:{self.probabilidad},T:{self.tardancia},R:{self.repetido})"


class Numero_jugar:
    def __init__(self, numero, probabilidad_redondeada, valor_ficha=100, vecinos=1):
        self.numero = numero
        self.probabilidad = probabilidad_redondeada
        self.valor_ficha = valor_ficha
        self.vecinos = (vecinos * 2) + 1
        self.tardancia = 1
        self.repetido = 0
        self.ganancia = 0
        self.ganancia_neta = 0
        self.jugado = self.vecinos * self.valor_ficha
        self.pleno = 1

    def Jugar(self):
        self.tardancia += 1
        self.pleno += 1
        self.Calcular_jugada()

    def Pego(self):
        self.Calcular_ganancia()

    def Aumentar_probailidad(self, nueva_probabilidad):
        self.probabilidad += nueva_probabilidad
        self.repetido += 1
        self.pleno += 1
        self.tardancia -= 1

    def Calcular_ganancia(self):
        self.Calcular_jugada()

        # Calcular el total ganado si se acierta un pleno
        ganado_partida = 36 * self.pleno * self.valor_ficha

        # Calcular la ganancia neta restando el total jugado del total ganado
        self.ganancia_neta = ganado_partida - self.jugado
        return self.ganancia_neta

    def Calcular_jugada(self):
        jugadaAhora = self.valor_ficha * (self.pleno * self.vecinos)
        # Actualizar el total jugado acumulativo
        self.jugado += jugadaAhora

    def __str__(self):
        return f"(N:{self.numero},P:{self.probabilidad},T:{self.tardancia},R:{self.repetido},PL:{self.pleno},J:{self.jugado})"
