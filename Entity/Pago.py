# Define una clase Contador para agrupar tus variables
class Pago:
    
    def __init__(self, valor_ficha_inicial = 100):
        self.valor_ficha = valor_ficha_inicial
        self.ganancia_total = 0
        self.jugado = 0
        self.valor_pleno = 36

    def verificar_ganado(self, orden  ):
        self.ganancia_total+= (self.valor_ficha* orden)

    

     