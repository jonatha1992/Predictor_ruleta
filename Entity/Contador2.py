# Define una clase Contador para agrupar tus variables
class Contador2:
    def __init__(self):
        self.numeros = []
        self.acierto_predecidos = 0
        self.acierto_vecinos_cercanos = 0
        self.acierto_vecinos_lejanos = 0
        self.acierto_vecinos_lejanos_lejano = 0
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
    def incrementar_aciertos_vecinos_cercanos(self):
        self.acierto_vecinos_cercanos += 1

    def incrementar_aciertos_vecinos_lejanos(self):
        self.acierto_vecinos_lejanos += 1

    def incrementar_aciertos_vecinos_lejanos_lejano(self):
        self.acierto_vecinos_lejanos_lejano += 1

    def incrementar_supero_limite(self):
        self.Sin_salir_nada += 1


    def sacarEfectividad(self):
        return (self.aciertos_totales / self.jugados) 
        