import os
from Entity.Parametro import Parametro_Juego
from Entity.Predictor_GRU import Predictor


def main():
    while True:
        excel_datos = input("\nIngresa el excel de datos: ") + ".xlsx"
        valor_ficha_inicial = int(input("Ingresa el valor inicial de la ficha: "))
        cantidad_vecinos = int(input("Ingresa la cantidad de vecinos: "))
        limite_juego = int(input("Ingresa el Limite de Juego: "))
        limite_pretendiente = int(input("Ingresa el Limite de Pretendiente: "))
        umbral_probabilidad = int(input("Ingresa el umbral de Probilidad: "))
        carpeta = "Data/" + excel_datos

        if not os.path.exists(carpeta):
            print("El archivo Excel no existe. No se puede instanciar el Predictor.")
        else:
            predictor = None
            Parametro_juego = Parametro_Juego(
                valor_ficha_inicial,
                cantidad_vecinos,
                limite_juego,
                limite_pretendiente,
                umbral_probabilidad,
            )
            predictor = Predictor(carpeta, Parametro_juego)
            numerosingresado = []

            while True:
                opcion = input(
                    "\nIngresa un nuevo número o 'salir' para terminar el programa: "
                )
                if opcion.lower() == "salir":
                    print(numerosingresado)
                    predictor.guardar_excel()
                    os.system(f"start excel Reporte_juego.xlsx")
                    break

                if opcion.lower() == "-":
                    predictor.borrar()
                    continue
                try:
                    numero = int(opcion)
                    numerosingresado.append(numero)

                    if numero < 0 or numero > 36:
                        print(
                            "El número debe estar entre 0 y 36. Inténtalo nuevamente."
                        )
                        continue

                    predictor.verificar_resultados(numero)
                    predictor.predecir()
                    predictor.actualizar_dataframe(numero)
                    predictor.mostrar_resultados()

                except ValueError as e:
                    print(f"An invalid error occurred: {e}. Please try again.")


# Si el script se ejecuta como programa principal, llama a la función main().
if __name__ == "__main__":
    main()
