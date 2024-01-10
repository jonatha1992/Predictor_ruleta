# Función principal que ejecuta el programa.
import os
# from Entity.Predictor import Predictor
from Entity.Predictor import Predictor


def main():
    while True:
        excel_datos = input("\nIngresa el excel de datos: ")+ ".xlsx"
        carpeta = "Data/"+ excel_datos 
        if not os.path.exists(carpeta):
            print("El archivo Excel no existe. No se puede instanciar el Predictor.")
        else:
            predictor = Predictor(carpeta)
            numerosingresado = []

            while True:
                opcion = input(
                    "\nIngresa un nuevo número o 'salir' para terminar el programa: "
                )
                if opcion.lower() == "salir":
                    print(numerosingresado)
                    predictor.guardar_excel()
                    os.system("start excel Reportes.xlsx")
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

                    predictor.verificar_predecidos2(numero)
                    predictor.predecir()
                    predictor.actualizar_dataframe(numero)
                    predictor.mostrar_resultados()

                except ValueError:
                    print("Ocurrio un error  no válido. Inténtalo nuevamente.")


# Si el script se ejecuta como programa principal, llama a la función main().
if __name__ == "__main__":
    main()
