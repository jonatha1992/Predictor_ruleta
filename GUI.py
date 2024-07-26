import tkinter as tk
from tkinter import ttk, messagebox
import os
from Entity.Predictor import Predictor
from Entity.Parametro import Parametro_Juego


class RuletaPredictorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Predictor de Ruleta")
        self.master.geometry("900x700")
        self.predictor = None
        self.create_widgets()

    def create_widgets(self):
        # Nuevo frame para contener input_frame y stats_frame
        top_frame = ttk.Frame(self.master)
        top_frame.pack(padx=10, pady=10, fill="x")

        # Frame para los parámetros de entrada (lado izquierdo)
        input_frame = ttk.LabelFrame(top_frame, text="Parámetros de entrada")
        input_frame.pack(side="left", padx=(0, 5), fill="x")

        ttk.Label(input_frame, text="Tipo de Ruleta:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.ruleta_type = ttk.Combobox(
            input_frame,
            values=["Electromecánica", "Virtual", "Crupier"],
            state="readonly",)

        self.ruleta_type.grid(row=0, column=1, padx=5, pady=5)
        self.ruleta_type.set("Electromecánica")

        parameters = [
            ("Valor ficha inicial:", "valor_ficha_inicial", "Valor entre (1-1000)", "1000",),
            ("Cantidad de vecinos:", "cantidad_vecinos", "Valores entre (1-4) (0 = sin vecinos)", "3",),
            ("Límite de juego:", "limite_juego", "Valores entre (1 al 5) ", "5"),
            ("Límite de pretendiente:", "limite_pretendiente", "Valores entre (1-5)", "1",),
            ("Umbral de probabilidad:", "umbral_probabilidad", "Valores entre (20-100)", "20",),
        ]

        self.param_entries = {}
        vcmd = (self.master.register(self.validate_entry), "%P")
        for i, (label, key, restriction, value) in enumerate(parameters):
            ttk.Label(input_frame, text=label).grid(row=i + 1, column=0, sticky="w", padx=5, pady=5)
            self.param_entries[key] = ttk.Entry(input_frame, width=10, validate="key", validatecommand=vcmd)
            self.param_entries[key].grid(row=i + 1, column=1, sticky="w", padx=5, pady=5)
            self.param_entries[key].insert(0, value)
            ttk.Label(input_frame, text=restriction).grid(row=i + 1, column=2, sticky="w", padx=5, pady=5)

        # Frame para los botones de control
        control_frame = ttk.Frame(input_frame)
        control_frame.grid(row=len(parameters) + 1, column=0, columnspan=3, pady=10)

        self.iniciar_button = ttk.Button(control_frame, text="Iniciar Predictor", command=self.iniciar_predictor)
        self.iniciar_button.pack(side="left", padx=(0, 5))

        self.reiniciar_button = ttk.Button(control_frame, text="Reiniciar", command=self.reset, state="disabled")
        self.reiniciar_button.pack(side="left")

        # Frame para ingresar nuevos números
        input_number_frame = ttk.LabelFrame(self.master, text="Ingresar nuevo número")
        input_number_frame.pack(padx=10, pady=10, fill="x")

        self.number_entry = ttk.Entry(input_number_frame, width=10, validate="key", validatecommand=vcmd)
        self.number_entry.pack(side="left", padx=5, pady=5)
        self.number_entry.bind("<Return>", self.predict_number)
        ttk.Button(input_number_frame, text="Predecir", command=self.predict_number).pack(side="left", padx=5, pady=5)
        ttk.Button(input_number_frame, text="Borrar último", command=self.delete_last).pack(side="left", padx=5, pady=5)
        ttk.Button(
            input_number_frame,
            text="Guardar Numeros",
            command=self.guardar_numeros).pack(
            side="left",
            padx=5,
            pady=5)

        # Frame para números salidos y estadísticas
        numeros_stats_frame = ttk.Frame(self.master)
        numeros_stats_frame.pack(padx=10, pady=5, fill="x")

        # Frame para números salidos (lado izquierdo)
        self.numeros_salidos_frame = ttk.LabelFrame(numeros_stats_frame, text="Números Salidos"
                                                    )
        self.numeros_salidos_frame.pack(
            side="left", padx=(0, 5), fill="both", expand=True
        )

        self.numeros_salidos_label = ttk.Label(
            self.numeros_salidos_frame, text="", wraplength=380
        )
        self.numeros_salidos_label.pack(padx=5, pady=5, fill="x", expand=True)

        # Tabla de estadísticas (lado derecho)
        stats_frame = ttk.LabelFrame(numeros_stats_frame, text="Estadísticas de Juego")
        stats_frame.pack(side="right", padx=(5, 0), fill="both", expand=True)

        self.stats_tree = ttk.Treeview(
            stats_frame, columns=("Estadística", "Valor"), show="headings", height=4
        )
        self.stats_tree.heading("Estadística", text="Estadística")
        self.stats_tree.heading("Valor", text="Valor")
        self.stats_tree.column("Estadística", width=100, anchor="w")
        self.stats_tree.column("Valor", width=40, anchor="center")
        self.stats_tree.pack(fill="both", expand=True)

        # Frame para resultados y estadísticas
        result_stats_frame = ttk.Frame(self.master)
        result_stats_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Área de resultados (lado izquierdo)
        result_frame = ttk.LabelFrame(result_stats_frame, text="Resultados")
        result_frame.pack(side="left", padx=(0, 5), fill="both", expand=True)

        scrollbar = ttk.Scrollbar(result_frame)
        scrollbar.pack(side="right", fill="y")

        self.result_text = tk.Text(result_frame, width=40, height=10, yscrollcommand=scrollbar.set)
        self.result_text.pack(padx=5, pady=5, fill="both", expand=True)
        scrollbar.config(command=self.result_text.yview)

        # Tabla de estadísticas (lado derecho)
        stats_frame2 = ttk.LabelFrame(result_stats_frame, text="Estadísticas de Juego")
        stats_frame2.pack(side="right", padx=(5, 0), fill="both", expand=True)

        self.stats_tree2 = ttk.Treeview(stats_frame2, columns=("Estadística", "Valor"), show="headings", height=4)
        self.stats_tree2.heading("Estadística", text="Estadística")
        self.stats_tree2.heading("Valor", text="Valor")
        self.stats_tree2.column("Estadística", width=100, anchor="w")
        self.stats_tree2.column("Valor", width=40, anchor="center")
        self.stats_tree2.pack(fill="both", expand=True)

        # Inicializar las tablas con filas vacías
        stats = ["Números ingresados", "Números Predecidos", "Aciertos Totales", "Sin salir"]
        for tree in [self.stats_tree, self.stats_tree2]:
            for stat in stats:
                tree.insert("", "end", values=(stat, ""))

        # Frame para ingresar nuevos números
        input_number_frame = ttk.LabelFrame(self.master, text="Ingresar nuevo número")
        input_number_frame.pack(padx=10, pady=10, fill="x")

        self.number_entry = ttk.Entry(input_number_frame, width=10, validate="key", validatecommand=vcmd)
        self.number_entry.pack(side="left", padx=5, pady=5)
        self.number_entry.bind("<Return>", self.predict_number)
        ttk.Button(input_number_frame, text="Predecir", command=self.predict_number).pack(side="left", padx=5, pady=5)
        ttk.Button(input_number_frame, text="Borrar último", command=self.delete_last).pack(side="left", padx=5, pady=5)
        ttk.Button(
            input_number_frame,
            text="Guardar Numeros",
            command=self.guardar_numeros).pack(
            side="left",
            padx=5,
            pady=5)

        # Frame para números salidos
        self.numeros_salidos_frame = ttk.LabelFrame(self.master, text="Números Salidos")
        self.numeros_salidos_frame.pack(padx=10, pady=5, fill="x")

        self.numeros_salidos_label = ttk.Label(self.numeros_salidos_frame, text="", wraplength=780)
        self.numeros_salidos_label.pack(padx=5, pady=5, fill="x")

        # Área de resultados
        result_frame = ttk.LabelFrame(self.master, text="Resultados")
        result_frame.pack(padx=10, pady=10, fill="both", expand=True)

        scrollbar = ttk.Scrollbar(result_frame)
        scrollbar.pack(side="right", fill="y")

        self.result_text = tk.Text(result_frame, width=60, height=10, yscrollcommand=scrollbar.set)
        self.result_text.pack(padx=5, pady=5, fill="both", expand=True)
        scrollbar.config(command=self.result_text.yview)

    def validate_entry(self, P):
        if P.strip() == "":
            return True
        try:
            int(P)
            return True
        except ValueError:
            return False

    def iniciar_predictor(self):
        if not self.ruleta_type.get():
            messagebox.showerror("Error", "Por favor, selecciona un tipo de ruleta.")
            return

        try:
            ruleta_tipo = self.ruleta_type.get()
            params = {}
            for key, entry in self.param_entries.items():
                try:
                    value = int(entry.get())
                    if key == "valor_ficha_inicial" and not value > 0:
                        raise ValueError("El valor de ficha inicial debe ser un entero positivo.")
                    elif key == "cantidad_vecinos" and not (0 <= value <= 4):
                        raise ValueError("La cantidad de vecinos debe estar entre 0 y 4.")
                    elif key == "limite_juego" and not (1 <= value <= 5):
                        raise ValueError("El límite de juego debe estar entre 1 y 5.")
                    elif key == "limite_pretendiente" and not (1 <= value <= 5):
                        raise ValueError("El límite de pretendiente debe estar entre 1 y 5.")
                    elif key == "umbral_probabilidad" and not (20 <= value <= 100):
                        raise ValueError("El umbral de probabilidad debe estar entre 20 y 100.")
                    params[key] = value
                except ValueError as e:
                    messagebox.showerror("Error de entrada", str(e))
                    return None

            parametros_juego = Parametro_Juego(**params)

            if ruleta_tipo == "Electromecánica":
                excel_file = "./Data/Electromecanica.xlsx"
            elif ruleta_tipo == "Electrónica":
                excel_file = "./Data/Electronica.xlsx"
            elif ruleta_tipo == "Crupier":
                excel_file = "./Data/Crupier.xlsx"
            else:
                raise ValueError("Tipo de ruleta no válido")

            if not os.path.exists(excel_file):
                raise FileNotFoundError(f"El archivo {excel_file} no se encuentra.")

            self.predictor = Predictor(excel_file, parametros_juego)
            self.limpiar_estadisticas()
            self.result_text.insert(
                tk.END, f"Predictor iniciado correctamente para ruleta {ruleta_tipo}.\n"
            )

        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {str(e)}")

    def predict_number(self, event=None):
        if not self.predictor:
            messagebox.showerror("Error", "Primero debes iniciar el predictor.")
            return

        number_str = self.number_entry.get().strip()
        if not number_str:
            messagebox.showerror("Error", "Por favor, ingresa un número.")
            return

        try:
            number = int(number_str)
            if 0 <= number <= 36:
                self.predictor.verificar_resultados(number)
                self.predictor.predecir()
                self.predictor.actualizar_dataframe(number)
                resultados = self.predictor.mostrar_resultados()
                self.result_text.insert(tk.END, f"Número ingresado: {number}")
                self.number_entry.delete(0, tk.END)
                if resultados is not None:
                    self.result_text.insert(tk.END, "\n" + str(resultados) + "\n")

                self.result_text.see(tk.END)
                self.actualizar_estadisticas()
            else:
                messagebox.showerror("Error", "El número debe estar entre 0 y 36.")
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingresa un número válido.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def delete_last(self):
        if not self.predictor:
            messagebox.showerror("Error", "El predictor no ha sido iniciado.")
            return

        if not self.predictor.contador.numeros_partida:
            messagebox.showinfo("Información", "No hay números para borrar.")
            return

        ultimo_numero = self.predictor.contador.numeros_partida[-1]
        self.result_text.insert(tk.END, f"Último número borrado: {ultimo_numero}\n")
        self.predictor.borrar()
        self.actualizar_estadisticas()

    def reset(self):
        confirmacion = messagebox.askyesno(
            "Confirmación", "¿Estás seguro que quieres reiniciar?"
        )

        if confirmacion:
            if self.predictor:
                self.predictor.guardar_reporte()
            self.result_text.insert(
                tk.END, "El juego ha sido reiniciado. Debe iniciar el predictor.\n"
            )
            self.limpiar_estadisticas()
            self.predictor = None
        else:
            self.result_text.insert(tk.END, "Reinicio cancelado.\n")

    def guardar_numeros(self):
        if not self.predictor or not self.predictor.contador.numeros_partida:
            messagebox.showinfo("Información", "No hay números para guardar.")
            return

        confirmacion = messagebox.askyesno("Confirmación", "¿Quieres guardar los números ingresados?")

        if confirmacion:
            self.predictor.guardar_excel()
            self.result_text.insert(tk.END, "Se han guardado los números ingresados.\n")
        else:
            self.result_text.insert(tk.END, "Guardado cancelado.\n")

    def actualizar_estadisticas(self):
        if self.predictor and self.predictor.contador:
            estadisticas = [
                ("Números Ingresados", self.predictor.contador.ingresados),
                ("Números Predecidos", self.predictor.contador.jugados),
                ("Aciertos Totales", self.predictor.contador.aciertos_totales),
                ("Sin salir", self.predictor.contador.Sin_salir_nada),
                ("Ganancia Neta", self.predictor.contador.ganancia_neta),
            ]

            for item in self.stats_tree.get_children():
                self.stats_tree.delete(item)

            for stat, value in estadisticas:
                self.stats_tree.insert("", "end", values=(stat, value))

            numeros_text = ", ".join(map(str, reversed(self.predictor.contador.numeros_partida)))
            self.numeros_salidos_label.config(text=numeros_text)

    def limpiar_estadisticas(self):
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        stats = ["Números ingresados", "Números Predecidos", "Aciertos Totales", "Sin salir"]
        for stat in stats:
            self.stats_tree.insert("", "end", values=(stat, ""))

        self.result_text.delete("1.0", tk.END)
        self.numeros_salidos_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = RuletaPredictorGUI(root)
    root.mainloop()
