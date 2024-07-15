import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from Entity.Predictor import Predictor
from Entity.Parametro import Parametro_Juego, HiperParametros
import os


class RuletaPredictorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Predictor de Ruleta")
        self.master.geometry("800x600")

        self.predictor = None
        self.create_widgets()

    def create_widgets(self):
        # Frame para los parámetros de entrada
        input_frame = ttk.LabelFrame(self.master, text="Parámetros de entrada")
        input_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(input_frame, text="Archivo Excel:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.excel_entry = ttk.Entry(input_frame, width=40)
        self.excel_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Buscar", command=self.browse_file).grid(
            row=0, column=2, padx=5, pady=5
        )

        parameters = [
            ("Valor ficha inicial:", "valor_ficha_inicial", ""),
            (
                "Cantidad de vecinos:",
                "cantidad_vecinos",
                "Valores entre (1-4) (0 = sin vecinos)",
            ),
            ("Límite de juego:", "limite_juego", "Valores entre (1 al 5) "),
            ("Límite de pretendiente:", "limite_pretendiente", "Valores entre (1-5)"),
            (
                "Umbral de probabilidad:",
                "umbral_probabilidad",
                "Valores entre (20-100)",
            ),
        ]

        self.param_entries = {}
        for i, (label, key, restriction) in enumerate(parameters):
            ttk.Label(input_frame, text=label).grid(
                row=i + 1, column=0, sticky="w", padx=5, pady=5
            )
            self.param_entries[key] = ttk.Entry(input_frame, width=10)
            self.param_entries[key].grid(
                row=i + 1, column=1, sticky="w", padx=5, pady=5
            )
            ttk.Label(input_frame, text=restriction).grid(
                row=i + 1, column=2, sticky="w", padx=5, pady=5
            )

        ttk.Button(
            input_frame, text="Iniciar Predictor", command=self.iniciar_predictor
        ).grid(row=len(parameters) + 1, column=1, pady=10)

        # Frame para ingresar nuevos números
        input_number_frame = ttk.LabelFrame(self.master, text="Ingresar nuevo número")
        input_number_frame.pack(padx=10, pady=10, fill="x")

        self.number_entry = ttk.Entry(input_number_frame, width=10)
        self.number_entry.pack(side="left", padx=5, pady=5)
        ttk.Button(
            input_number_frame, text="Predecir", command=self.predict_number
        ).pack(side="left", padx=5, pady=5)
        ttk.Button(
            input_number_frame, text="Borrar último", command=self.delete_last
        ).pack(side="left", padx=5, pady=5)

        # Área de resultados
        result_frame = ttk.LabelFrame(self.master, text="Resultados")
        result_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.result_text = tk.Text(result_frame, height=10)
        self.result_text.pack(padx=5, pady=5, fill="both", expand=True)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if filename:
            self.excel_entry.delete(0, tk.END)
            self.excel_entry.insert(0, filename)

    def iniciar_predictor(self):
        try:
            excel_file = self.excel_entry.get()
            if not excel_file.endswith(".xlsx"):
                excel_file += ".xlsx"

            params = {}
            for key, entry in self.param_entries.items():
                value = int(entry.get())
                if key == "valor_ficha_inicial" and value <= 0:
                    raise ValueError(
                        "El valor de ficha inicial debe ser un entero positivo."
                    )
                elif key == "cantidad_vecinos" and not (1 <= value <= 4):
                    raise ValueError("La cantidad de vecinos debe estar entre 1 y 4.")
                elif key == "limite_juego" and not (500 <= value <= 100000):
                    raise ValueError(
                        "El límite de juego debe estar entre 500 y 100000."
                    )
                elif key == "limite_pretendiente" and not (5 <= value <= 20):
                    raise ValueError(
                        "El límite de pretendiente debe estar entre 5 y 20."
                    )
                elif key == "umbral_probabilidad" and not (1 <= value <= 100):
                    raise ValueError(
                        "El umbral de probabilidad debe estar entre 1 y 100."
                    )
                params[key] = value

            parametros_juego = Parametro_Juego(**params)

            self.predictor = Predictor(excel_file, parametros_juego)
            self.result_text.insert(tk.END, "Predictor iniciado correctamente.\n")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict_number(self):
        if not self.predictor:
            messagebox.showerror("Error", "Primero debes iniciar el predictor.")
            return

        try:
            number = int(self.number_entry.get())
            if 0 <= number <= 36:
                self.predictor.verificar_resultados(number)
                self.predictor.predecir()
                self.predictor.actualizar_dataframe(number)
                resultados = self.predictor.mostrar_resultados()
                self.result_text.insert(tk.END, f"Número ingresado: {number}\n")
                self.result_text.insert(tk.END, resultados + "\n")
            else:
                messagebox.showerror("Error", "El número debe estar entre 0 y 36.")
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingresa un número válido.")

    def delete_last(self):
        if self.predictor:
            self.predictor.borrar()
            self.result_text.insert(tk.END, "Último número borrado.\n")
        else:
            messagebox.showerror("Error", "El predictor no ha sido iniciado.")


if __name__ == "__main__":
    root = tk.Tk()
    app = RuletaPredictorGUI(root)
    root.mainloop()
