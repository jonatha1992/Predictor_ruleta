from PyQt5 import QtWidgets, QtCore, QtGui
from Widget_ui import Ui_Gui_Predictor
from Predictor_logic import PredictorLogic
from Config import get_ruleta_types


class RuletaPredictorGUI(QtWidgets.QMainWindow, Ui_Gui_Predictor):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.logic = PredictorLogic()
        self.setup_connections()
        self.load_ruleta_types()

    def setup_connections(self):
        self.btnIniciar.clicked.connect(self.iniciar_predictor)
        self.btnReiniciar.clicked.connect(self.reset)
        self.btnPredecir.clicked.connect(self.predict_number)
        self.btnBorrar.clicked.connect(self.delete_last)
        self.btnGuardar.clicked.connect(self.guardar_numeros)

    def load_ruleta_types(self):
        try:
            tipos_ruleta = get_ruleta_types()
            self.comboBoxRuleta.addItems(tipos_ruleta)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudieron cargar los tipos de ruleta: {str(e)}")

    def iniciar_predictor(self):
        try:
            ruleta_tipo = self.comboBoxRuleta.currentText()
            cantidad_vecinos = int(self.lineVecinos.text())
            limite_juego = int(self.lineTardanza.text())
            umbral_probabilidad = int(self.lineProbabiliada.text())

            self.logic.iniciar_predictor(ruleta_tipo, cantidad_vecinos, limite_juego, umbral_probabilidad)
            self.limpiar_estadisticas()
            self.add_result("¡Predictor iniciado correctamente!")
            self.habilitar_juego()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def predict_number(self):
        try:
            number = int(self.lineIngresoNumero.text())
            resultados = self.logic.predict_number(number)
            self.add_result(f"Número ingresado: {number}")
            if resultados:
                self.add_result(str(resultados))
            self.lineIngresoNumero.clear()
            self.actualizar_estadisticas()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def delete_last(self):
        try:
            ultimo_numero = self.logic.delete_last()
            self.add_result(f"Último número borrado: {ultimo_numero}")
            self.actualizar_estadisticas()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def reset(self):
        confirmacion = QtWidgets.QMessageBox.question(self, "Confirmación",
                                                      "¿Estás seguro que quieres reiniciar?",
                                                      QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if confirmacion == QtWidgets.QMessageBox.Yes:
            self.logic.reset()
            self.add_result("El juego ha sido reiniciado. Debe iniciar el predictor.")
            self.limpiar_estadisticas()
            self.desahabilitar_juego()
        else:
            self.add_result("Reinicio cancelado.")

    def guardar_numeros(self):
        try:
            self.logic.guardar_numeros()
            self.add_result("Se han guardado los números ingresados.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def actualizar_estadisticas(self):
        estadisticas = self.logic.get_estadisticas()
        if estadisticas:
            self.tableWidget_2.setItem(0, 0, QtWidgets.QTableWidgetItem(str(estadisticas["ingresados"])))
            self.tableWidget_2.setItem(1, 0, QtWidgets.QTableWidgetItem(str(estadisticas["jugados"])))
            self.tableWidget_2.setItem(2, 0, QtWidgets.QTableWidgetItem(str(estadisticas["aciertos_totales"])))
            self.tableWidget_2.setItem(3, 0, QtWidgets.QTableWidgetItem(str(estadisticas["sin_salir_nada"])))

            numeros_text = ", ".join(map(str, reversed(estadisticas["numeros_partida"])))
            self.label_Numeros_salidos.setText(numeros_text)

            # Actualizar tabla de probabilidades
            model = QtGui.QStandardItemModel()
            model.setHorizontalHeaderLabels(["Número", "Probabilidad", "Tardanza", "Repetición"])
            for numero in estadisticas["numeros_a_jugar"]:
                model.appendRow([
                    QtGui.QStandardItem(str(numero.numero)),
                    QtGui.QStandardItem(f"{numero.probabilidad}%"),
                    QtGui.QStandardItem(str(numero.tardancia)),
                    QtGui.QStandardItem(str(numero.repetido))
                ])
            self.tableProbabilidad.setModel(model)

    def limpiar_estadisticas(self):
        for i in range(4):
            self.tableWidget_2.setItem(i, 0, QtWidgets.QTableWidgetItem(""))
        self.label_Numeros_salidos.setText("")
        self.tableProbabilidad.setModel(None)
        self.listResultados.clear()

    def add_result(self, text):
        self.listResultados.addItem(text)

    def habilitar_juego(self):
        self.btnReiniciar.setEnabled(True)
        self.btnIniciar.setEnabled(False)
        self.comboBoxRuleta.setEnabled(False)
        self.lineVecinos.setEnabled(False)
        self.lineTardanza.setEnabled(False)
        self.lineProbabiliada.setEnabled(False)

    def desahabilitar_juego(self):
        self.btnReiniciar.setEnabled(False)
        self.btnIniciar.setEnabled(True)
        self.comboBoxRuleta.setEnabled(True)
        self.lineVecinos.setEnabled(True)
        self.lineTardanza.setEnabled(True)
        self.lineProbabiliada.setEnabled(True)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui = RuletaPredictorGUI()
    gui.show()
    sys.exit(app.exec_())
