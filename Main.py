import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import os
from Entity.predictor import Predictor
from Entity.parametro import HiperParametros, Parametro_Juego
from Entity.vecinos import colores_ruleta
from config import get_excel_file, get_ruleta_types


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

        try:
            tipos_ruleta = get_ruleta_types()
            ttk.Label(input_frame, text="Tipo de Ruleta:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.ruleta_type = ttk.Combobox(input_frame,
                                            values=tipos_ruleta,
                                            state="readonly")
            self.ruleta_type.grid(row=0, column=1, padx=5, pady=5)
            if tipos_ruleta:
                self.ruleta_type.set(tipos_ruleta[0])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron cargar los tipos de ruleta: {str(e)}")
            self.ruleta_type = ttk.Combobox(
                input_frame,
                values=["Error al cargar"],
                state="disabled")

        parameters = [
            ("Cantidad de vecinos:", "cantidad_vecinos", "Valores entre (1-4) (0 = sin vecinos)", "2",),
            ("Límite de tardancia:", "limite_tardancia", "Valores entre (1 al 15) ", "5"),
            ("Umbral de probabilidad:", "umbral_probabilidad", "Valores entre (0-100)", "50",),
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

        # Frame para números salidos y estadísticas
        numeros_stats_frame = ttk.Frame(self.master)
        numeros_stats_frame.pack(padx=10, pady=5, fill="x")

        # Frame para números salidos (lado izquierdo)
        self.numeros_salidos_frame = ttk.LabelFrame(numeros_stats_frame, text="Números Salidos")
        self.numeros_salidos_frame.pack(side="left", padx=(0, 5), fill="both", expand=True)

        self.numeros_salidos_label = ttk.Label(self.numeros_salidos_frame, text="", anchor="nw", justify="left", wraplength=500)
        self.numeros_salidos_label.pack(side="top", padx=5, pady=5, fill="x", expand=True)

        # Tabla de estadísticas (lado derecho)
        stats_frame = ttk.LabelFrame(numeros_stats_frame, text="Estadísticas de Juego")
        stats_frame.pack(side="right", padx=(5, 0))
        estadisticas = ["Nros. Ingresados", "Nros. Predecidos", "Aciertos ", "No acertados", "Efectividad (%)"]
        self.stats_tree = ttk.Treeview(stats_frame, columns=("Estadística", "Valor"), show="headings", height=5)
        self.stats_tree.heading("Estadística", text="Estadísticas")
        self.stats_tree.heading("Valor", text="Valor")
        self.stats_tree.column("Estadística", width=160, anchor="w")
        self.stats_tree.column("Valor", width=150, anchor="center")
        self.stats_tree.pack(fill="x", expand=False)

        for tree in [self.stats_tree, self.stats_tree]:
            for stat in estadisticas:
                tree.insert("", "end", values=(stat, "0"))

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
        probabildades_frame = ttk.LabelFrame(result_stats_frame, text="Probabilidades de Juego")
        probabildades_frame.pack(side="right", padx=(5, 0), fill="both", expand=True)

        # Crear un frame para contener el Treeview y el scrollbar
        tree_frame = ttk.Frame(probabildades_frame)
        tree_frame.pack(fill="both", expand=True)

        # Crear el Treeview
        self.probabilidades_tree = ttk.Treeview(
            tree_frame,
            columns=(
                "Número",
                "Probabilidad",
                "Tardanza",
                "Repetición"),
            show="tree headings",
            height=4)

        self.probabilidades_tree.column("#0", width=0, stretch=tk.NO)
        self.probabilidades_tree.heading("Número", text="Número")
        self.probabilidades_tree.heading("Probabilidad", text="Probabilidad")
        self.probabilidades_tree.heading("Tardanza", text="Tardanza")
        self.probabilidades_tree.heading("Repetición", text="Repetición")
        self.probabilidades_tree.column("Número", width=60, anchor="center")
        self.probabilidades_tree.column("Probabilidad", width=60, anchor="center")
        self.probabilidades_tree.column("Tardanza", width=60, anchor="center")
        self.probabilidades_tree.column("Repetición", width=60, anchor="center")

        # Crear y posicionar el scrollbar vertical
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.probabilidades_tree.yview)
        scrollbar.pack(side="right", fill="y")

        # Configurar el Treeview para usar el scrollbar y posicionarlo
        self.probabilidades_tree.configure(yscrollcommand=scrollbar.set)
        self.probabilidades_tree.pack(side="left", fill="both", expand=True)

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
                    if key == "cantidad_vecinos" and not (0 <= value <= 4):
                        raise ValueError("La cantidad de vecinos debe estar entre 0 y 4.")
                    elif key == "limite_tardancia" and not (1 <= value <= 15):
                        raise ValueError("El límite de juego debe estar entre 1 y 15.")
                    elif key == "umbral_probabilidad" and not (0 <= value <= 100):
                        raise ValueError("El umbral de probabilidad debe estar entre 0 y 100.")
                    params[key] = value
                except ValueError as e:
                    messagebox.showerror("Error de entrada", str(e))
                    return None

            parametros_juego = Parametro_Juego(**params)
            ruleta_tipo = self.ruleta_type.get()
            excel_file = get_excel_file(ruleta_tipo)
            self.predictor = Predictor(excel_file, parametros_juego, hiperparametros=HiperParametros())
            self.limpiar_estadisticas()
            self.result_text.insert(
                tk.END, f"¡Predictor iniciado correctamente!\n"
            )
            self.habilitar_juego()

        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {str(e)}")

    def habilitar_juego(self):
        self.reiniciar_button.config(state="normal")
        self.iniciar_button.config(state="disabled")
        self.ruleta_type.config(state="disabled")
        for entry in self.param_entries.values():
            entry.config(state="disabled")

    def desahabilitar_juego(self):
        self.reiniciar_button.config(state="disabled")
        self.iniciar_button.config(state="normal")
        self.ruleta_type.config(state="readonly")
        for entry in self.param_entries.values():
            entry.config(state="normal")

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
            self.result_text.insert(tk.END, "El juego ha sido reiniciado. Debe iniciar el predictor.\n")
            self.limpiar_estadisticas()
            self.predictor = None
            self.desahabilitar_juego()
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
                ("Nros Ingresados", self.predictor.contador.ingresados),
                ("Nros Predecidos", self.predictor.contador.jugados),
                ("Aciertos ", self.predictor.contador.aciertos_totales),
                ("No acertados", self.predictor.contador.Sin_salir_nada),
                ("Efectividad", self.predictor.contador.sacarEfectividad()),
            ]

            for item in self.stats_tree.get_children():
                self.stats_tree.delete(item)

            for stat, value in estadisticas:
                self.stats_tree.insert("", "end", values=(stat, value))

            numeros_text = ", ".join(map(str, reversed(self.predictor.contador.numeros_partida)))
            self.numeros_salidos_label.config(text=numeros_text)

        if self.predictor and len(self.predictor.numeros_a_jugar) > 0:
            for item in self.probabilidades_tree.get_children():
                self.probabilidades_tree.delete(item)

            for n in self.predictor.numeros_a_jugar:

                color = colores_ruleta.get(n.numero, 'black')

                item = self.probabilidades_tree.insert("", "end", values=(
                    n.numero,
                    f"{n.probabilidad}%",
                    n.tardancia,
                    n.repetido
                ))

                self.probabilidades_tree.tag_configure(f'color_{n.numero}', foreground=color)
                self.probabilidades_tree.item(item, tags=(f'color_{n.numero}',))

        else:
            for item in self.probabilidades_tree.get_children():
                self.probabilidades_tree.delete(item)

    def limpiar_estadisticas(self):
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        stats = ["Ingresados", "Predecidos", "Aciertos", "No acertados", "Efectividad"]
        for stat in stats:
            self.stats_tree.insert("", "end", values=(stat, ""))

        self.probabilidades_tree.delete(*self.probabilidades_tree.get_children())
        self.result_text.delete("1.0", tk.END)
        self.numeros_salidos_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = RuletaPredictorGUI(root)
    root.mainloop()
