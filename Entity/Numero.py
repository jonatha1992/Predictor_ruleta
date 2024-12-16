
# class Numero_jugar:
#     def __init__(self, numero, probabilidad_redondeada, vecinos=1):
#         self.numero = numero
#         self.probabilidad = probabilidad_redondeada
#         self.vecinos = (vecinos * 2) + 1
#         self.tardancia = 1
#         self.repetido = 0

#     def Jugar(self):
#         self.tardancia += 1

#     def Aumentar_probailidad(self, nueva_probabilidad):
#         self.probabilidad += nueva_probabilidad
#         self.repetido += 1
#         self.tardancia -= 1

#     def __str__(self):
#         return f"(N:{self.numero},P:{self.probabilidad},T:{self.tardancia},R:{self.repetido})"


from dataclasses import dataclass


class Numero_Historial:
    def __init__(self, numero, probabilidad_redondeada, vecinos=1):
        self.numero = numero
        self.probabilidad = probabilidad_redondeada
        self.repetido = 0

    def aumentar_probailidad(self, nueva_probabilidad):
        self.probabilidad += nueva_probabilidad
        self.repetido += 1

    def __str__(self):
        return f"(N:{self.numero},P:{self.probabilidad},R:{self.repetido})"


@dataclass
class NumeroBase:
    numero: int
    probabilidad: int
    repetido: int = 0

    def aumentar_probabilidad(self, nueva_probabilidad: int):
        """
        Aumenta la probabilidad del número y el contador de repeticiones.
        """
        self.probabilidad += nueva_probabilidad
        self.repetido += 1

    def __str__(self):
        return f"(N:{self.numero}, P:{self.probabilidad}, R:{self.repetido})"


@dataclass
class NumeroJugar(NumeroBase):
    vecinos: int = 1
    tardancia: int = 1

    def jugar(self):
        """
        Incrementa la tardancia cuando se juega el número.
        """
        self.tardancia += 1

    def aumentar_probabilidad(self, nueva_probabilidad: int):
        """
        Aumenta la probabilidad, el contador de repeticiones y decrementa la tardancia.
        """
        super().aumentar_probabilidad(nueva_probabilidad)
        self.tardancia = max(self.tardancia - 1, 0)  # Evita tardancia negativa

    def __str__(self):
        return f"(N:{self.numero}, P:{self.probabilidad}, T:{self.tardancia}, R:{self.repetido})"


@dataclass
class NumeroHistorial(NumeroBase):
    def __str__(self):
        return f"(N:{self.numero}, P:{self.probabilidad}, R:{self.repetido})"
