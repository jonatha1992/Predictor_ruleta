# Define una clase Contador para agrupar tus variables
class Contador:
    def __init__(self):
        self.numeros = []
        self.aciertos = 0
        self.sin_aciertos = 0
        self.acierto_vecinos_cercanos = 0
        self.sin_vecinos_cercanos = 0
        self.acierto_vecinos_lejanos = 0
        self.sin_vecinos_lejanos = 0
        self.ingresados = 0
        self.jugados = 0
        self.max_sin_acierto = 0
        self.max_sin_vecinos_cercanos = 0
        self.max_sin_vecinos_lejanos = 0
        self.aciertos_totales = 0