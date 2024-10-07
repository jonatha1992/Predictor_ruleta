import os
import sys


# def get_application_path():
#     if getattr(sys, 'frozen', False):
#         return os.path.dirname(sys.executable)
#     else:
#         return os.path.dirname(os.path.abspath(__file__))


def get_application_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))


def get_relative_path(relative_path):
    base_path = get_application_path()
    return os.path.join(base_path, relative_path)


# def get_ruleta_types():
#     data_folder = get_relative_path("Data")

#     if not os.path.exists(data_folder):
#         raise FileNotFoundError(f"La carpeta Data no existe en {data_folder}")

#     excel_files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')]

#     if not excel_files:
#         raise FileNotFoundError("No se encontraron archivos Excel en la carpeta Data")

#     tipos_ruleta = [os.path.splitext(f)[0] for f in excel_files]
#     return tipos_ruleta


def get_ruleta_types():
    data_folder = get_relative_path("Data")

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"La carpeta Data no existe en {data_folder}")

    excel_files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')]

    if not excel_files:
        raise FileNotFoundError("No se encontraron archivos Excel en la carpeta Data")

    tipos_ruleta = [
        os.path.splitext(f)[0] for f in excel_files if any(
            keyword in f for keyword in [
                "Electromecanica",
                "Crupier",
                "Electronica"])]
    return tipos_ruleta


def get_excel_file(ruleta_tipo):
    file_name = f"{ruleta_tipo}.xlsx"
    file_path = get_relative_path(os.path.join("Data", file_name))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")

    return file_path
