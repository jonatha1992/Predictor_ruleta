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
        self.filename = "bombay1.xlsx"
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
            (
                "Valor ficha inicial:",
                "valor_ficha_inicial",
                "Valor entre (1-1000)",
                "1000",
            ),
            (
                "Cantidad de vecinos:",
                "cantidad_vecinos",
                "Valores entre (1-4) (0 = sin vecinos)",
                "3",
            ),
            ("Límite de juego:", "limite_juego", "Valores entre (1 al 5) ", "5"),
            (
                "Límite de pretendiente:",
                "limite_pretendiente",
                "Valores entre (1-5)",
                "1",
            ),
            (
                "Umbral de probabilidad:",
                "umbral_probabilidad",
                "Valores entre (20-100)",
                "20",
            ),
        ]

        self.param_entries = {}
        for i, (label, key, restriction, value) in enumerate(parameters):
            ttk.Label(input_frame, text=label).grid(
                row=i + 1, column=0, sticky="w", padx=5, pady=5
            )
            self.param_entries[key] = ttk.Entry(input_frame, width=10)
            self.param_entries[key].grid(
                row=i + 1, column=1, sticky="w", padx=5, pady=5
            )
            self.param_entries[key].insert(0, value)  # Establecer el valor inicial
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
        self.number_entry.bind("<Return>", self.predict_number)
        ttk.Button(
            input_number_frame, text="Predecir", command=self.predict_number
        ).pack(side="left", padx=5, pady=5)
        ttk.Button(
            input_number_frame, text="Borrar último", command=self.delete_last
        ).pack(side="left", padx=5, pady=5)

        self.numeros_salidos_frame = ttk.LabelFrame(self.master, text="Números Salidos")
        self.numeros_salidos_frame.pack(padx=10, pady=5, fill="x")

        self.numeros_salidos_label = ttk.Label(
            self.numeros_salidos_frame, text="", wraplength=780
        )
        self.numeros_salidos_label.pack(padx=5, pady=5, fill="x")

        # Frame principal para resultados y estadísticas
        main_frame = ttk.Frame(self.master)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Área de resultados (mitad izquierda)
        result_frame = ttk.LabelFrame(main_frame, text="Resultados")
        result_frame.pack(side="left", padx=(0, 5), fill="both", expand=True)

        # Crear un widget Scrollbar para el área de resultados
        scrollbar = ttk.Scrollbar(result_frame)
        scrollbar.pack(side="right", fill="y")

        self.result_text = tk.Text(
            result_frame, height=10, yscrollcommand=scrollbar.set
        )
        self.result_text.pack(padx=5, pady=5, fill="both", expand=True)
        scrollbar.config(command=self.result_text.yview)

        # Tabla de estadísticas (mitad derecha)
        stats_frame = ttk.LabelFrame(main_frame, text="Estadísticas de Juego")
        stats_frame.pack(side="right", padx=(5, 0), fill="both", expand=True)

        self.stats_tree = ttk.Treeview(
            stats_frame, columns=("Estadística", "Valor"), show="headings", height=6
        )
        self.stats_tree.heading("Estadística", text="Estadística")
        self.stats_tree.heading("Valor", text="Valor")
        self.stats_tree.column("Estadística", width=150, anchor="w")
        self.stats_tree.column("Valor", width=100, anchor="center")
        self.stats_tree.pack(fill="both", expand=True)

        # Inicializar la tabla con filas vacías
        stats = [
            "Números Jugados",
            "Aciertos Totales",
            "Sin salir",
            "Ganancia Neta",
            "Valor de Ficha",
            "Predicciones",
        ]
        for stat in stats:
            self.stats_tree.insert("", "end", values=(stat, ""))
            # Área de resultados

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if filename:
            # Obtener solo el nombre del archivo
            base_name = os.path.basename(filename)
            self.excel_entry.delete(0, tk.END)
            self.excel_entry.insert(0, base_name)
            self.full_file_path = filename

    def iniciar_predictor(self):
        try:
            excel_file = self.excel_entry.get()
            if not excel_file.endswith(".xlsx"):
                excel_file += ".xlsx"
            if hasattr(self, "full_file_path"):
                excel_file = self.full_file_path
            else:
                excel_file = os.path.join(os.getcwd(), excel_file)
            params = {}
            for key, entry in self.param_entries.items():
                try:
                    value = int(entry.get())
                    if key == "valor_ficha_inicial" and not value > 0:
                        raise ValueError(
                            "El valor de ficha inicial debe ser un entero positivo."
                        )
                    elif key == "cantidad_vecinos" and not (0 <= value <= 4):
                        raise ValueError(
                            "La cantidad de vecinos debe estar entre 0 y 4."
                        )
                    elif key == "limite_juego" and not (1 <= value <= 5):
                        raise ValueError("El límite de juego debe estar entre 1 y 5.")
                    elif key == "limite_pretendiente" and not (1 <= value <= 5):
                        raise ValueError(
                            "El límite de pretendiente debe estar entre 1 y 5."
                        )
                    elif key == "umbral_probabilidad" and not (20 <= value <= 100):
                        raise ValueError(
                            "El umbral de probabilidad debe estar entre 20 y 100."
                        )
                    params[key] = value
                except ValueError as e:
                    messagebox.showerror("Error de entrada", str(e))
                    return None

            parametros_juego = Parametro_Juego(**params)
            self.predictor = Predictor(excel_file, parametros_juego)
            self.result_text.insert(tk.END, "Predictor iniciado correctamente.\n")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict_number(self, event=None):
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
                self.number_entry.delete(0, tk.END)
                if resultados != None:
                    self.result_text.insert(tk.END, str(resultados) + "\n")

                self.result_text.see(tk.END)
            else:
                messagebox.showerror("Error", "El número debe estar entre 0 y 36.")
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingresa un número válido.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

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
