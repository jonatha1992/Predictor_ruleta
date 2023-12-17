# Define una clase Contador para agrupar tus variables
class Contador:
    def __init__(self):
        self.numeros = []
        self.acierto_predecidos = 0
        self.acierto_vecinos_1lugar = 0
        self.acierto_vecinos_2lugar = 0
        self.acierto_vecinos_3lugar = 0
        self.acierto_vecinos_4lugar = 0
        self.ingresados = 0
        self.jugados = 0
        self.aciertos_totales = 0
        self.Sin_salir_nada = 0
         
        
    def incrementar_ingresados(self, numero):
        self.ingresados += 1
        self.numeros.append(numero)

    def borrar_ultimo_numero (self):
        self.ingresados -= 1
        self.numeros.pop()
        
        if self.ingresados > 7 :  
            self.jugados -= 1
        
    def incrementar_jugados(self):
        self.jugados += 1

    def incrementar_aciertos(self):
        self.aciertos_totales += 1

    def incrementar_predecidos(self):
        self.acierto_predecidos += 1
    def incrementar_aciertos_vecinos_1lugar(self):
        self.acierto_vecinos_1lugar += 1

    def incrementar_aciertos_vecinos_2lugar(self):
        self.acierto_vecinos_2lugar += 1

    def incrementar_aciertos_vecinos_3lugar(self):
        self.acierto_vecinos_3lugar += 1
    def incrementar_aciertos_vecinos_4lugar(self):
        self.acierto_vecinos_4lugar += 1
    def incrementar_supero_limite(self):
        self.Sin_salir_nada += 1


    def sacarEfectividad(self):
        return (self.aciertos_totales / self.jugados) 
        