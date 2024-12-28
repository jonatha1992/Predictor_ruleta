from dataclasses import dataclass


@dataclass
class NumeroBase:
    numero: int
    probabilidad: float
    repetido: int = 0

    def actualizar_probabilidad(self, nueva_probabilidad: float):
        """Actualiza la probabilidad y cuenta repeticiones"""
        self.probabilidad += nueva_probabilidad
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

    def __str__(self):
        return f"(N:{self.numero}, P:{self.probabilidad}, T:{self.tardancia}, R:{self.repetido})"


@dataclass
class NumeroHistorial(NumeroBase):
    pass
