# Define una clase Contador para agrupar tus variables
class Numero_pretendiente:
    def __init__(self, numero, probabilidad_redondeada):
        self.numero = numero
        self.probabilidad = probabilidad_redondeada
        self.tardancia = 1

    def Jugar(self):
        self.tardancia += 1
        
    def Aumentar_probailidad(self, nueva_probabilidad):
        self.probabilidad += nueva_probabilidad
        self.tardancia = 1


    def __str__(self):
        return f"(N:{self.numero},P:{self.probabilidad},T:{self.tardancia})"

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

    def Jugar(self):
        self.tardancia += 1
        jugadaAhora = (self.valor_ficha * self.vecinos)* self.tardancia
        self.jugado += jugadaAhora

    def Pego(self):
        self.Calcular_ganancia()

    def Aumentar_probailidad(self, nueva_probabilidad):
        self.probabilidad += nueva_probabilidad
        self.tardancia -= 1
        self.repetido += 1

    def Calcular_ganancia(self):
        plenos = self.tardancia  + self.repetido
        jugado_con_pleno = plenos * self.valor_ficha
        ganado_partida = 36 * jugado_con_pleno
        self.ganancia_neta = ganado_partida - self.jugado
        return self.ganancia_neta

    def Calcular_perdida(self):
        if self.repetido > 0:
            self.jugado = (36 * self.tardancia ) * self.valor_ficha
        else:
            self.jugado = (36 * self.tardancia) * self.valor_ficha
        return self.jugado

    def __str__(self):
        return f"(N:{self.numero},P:{self.probabilidad},T:{self.tardancia})"
