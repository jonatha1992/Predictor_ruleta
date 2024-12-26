from dataclasses import dataclass


@dataclass
class NumeroBase:
    numero: int
    probabilidad: int
    repetido: int = 0

    def actualizar_probabilidad(self, nueva_probabilidad: int):
        """Actualiza la probabilidad y cuenta repeticiones"""
        self.probabilidad += nueva_probabilidad  # sumar probabilidad
        self.repetido += 1

    def __str__(self):
        return f"(N:{self.numero}, P:{self.probabilidad}, R:{self.repetido})"


@dataclass
class NumeroJugar(NumeroBase):
    vecinos: int = 1
    tardancia: int = 1

    def jugar(self):
        """Incrementa la tardancia"""
        self.tardancia += 1

    def actualizar_probabilidad(self, nueva_probabilidad: int):
        """Actualiza probabilidad y ajusta tardancia"""
        super().actualizar_probabilidad(nueva_probabilidad)
        self.tardancia = max(self.tardancia - 1, 0)

    def __str__(self):
        return f"(N:{self.numero}, P:{self.probabilidad}, T:{self.tardancia}, R:{self.repetido})"


@dataclass
class NumeroHistorial(NumeroBase):
    pass
