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
        self.Sin_salir_nada = 0
        self.Maximo_Sin_salir_nada = 0
         
        
    def incrementar_ingresados(self, numero):
        self.ingresados += 1
        self.numeros.append(numero)

    def borrar_ultimo_numero (self):
        self.ingresados -= 1
        self.numeros.pop()
        
        if self.ingresados > 10 :  
            self.jugados -= 1
        
    def incrementar_jugados(self):
        self.jugados += 1

    def incrementar_aciertos(self):
        self.aciertos += 1
        self.sin_aciertos = 0
        # self.aciertos_totales += 1

    def incrementar_aciertos_vecinos_cercanos(self):
        self.acierto_vecinos_cercanos += 1
        self.sin_vecinos_cercanos = 0
        # self.aciertos_totales += 1

    def incrementar_aciertos_vecinos_lejanos(self):
        self.acierto_vecinos_lejanos += 1
        self.sin_vecinos_lejanos = 0
        # self.aciertos_totales += 1

    def actualizar_sin_aciertos(self):
        self.sin_aciertos += 1
        if self.sin_aciertos > self.max_sin_acierto:
            self.max_sin_acierto = self.sin_aciertos

    def actualizar_sin_vecinos_cercanos(self):
        self.sin_vecinos_cercanos += 1
        if self.sin_vecinos_cercanos > self.max_sin_vecinos_cercanos:
            self.max_sin_vecinos_cercanos = self.sin_vecinos_cercanos

    def actualizar_sin_vecinos_lejanos(self):
        self.sin_vecinos_lejanos += 1
        if self.sin_vecinos_lejanos > self.max_sin_vecinos_lejanos:
            self.max_sin_vecinos_lejanos = self.sin_vecinos_lejanos
    
    def actualizar_sin_salir_nada(self):
        self.Sin_salir_nada += 1
        if self.Sin_salir_nada > self.Maximo_Sin_salir_nada:
            self.Maximo_Sin_salir_nada = self.Sin_salir_nada
    
    def reiniciar_sin_salir_nada(self):
        self.aciertos_totales += 1
        self.Sin_salir_nada = 0
        
    def sacarEfectividad(self):
        return (self.aciertos_totales / self.jugados)*100
        