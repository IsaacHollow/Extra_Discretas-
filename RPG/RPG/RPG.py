import tkinter as tk
from tkinter import messagebox

# Definir las clases para el juego
class Personaje:
    def __init__(self, nombre, salud, ataque):
        self.nombre = nombre
        self.salud = salud
        self.ataque = ataque

    def recibir_dano(self, dano):
        self.salud -= dano

    def atacar(self, enemigo):
        enemigo.recibir_dano(self.ataque)
        return f"{self.nombre} ataco a {enemigo.nombre} y caus {self.ataque} de dano."

class Mapa:
    def __init__(self):
        self.lugares = {
            "Ciudad": ["Bosque", "Montana"],
            "Bosque": ["Ciudad", "Cueva"],
            "Montana": ["Ciudad"],
            "Cueva": ["Bosque"]
        }

    def mover(self, lugar_actual, nuevo_lugar):
        if nuevo_lugar in self.lugares[lugar_actual]:
            return f"Te has movido a {nuevo_lugar}."
        else:
            return "No puedes ir a ese lugar desde aqui."

# Crear la ventana principal
def crear_ventana():
    ventana = tk.Tk()
    ventana.title("RPG Basico")

    # Crear personajes
    jugador = Personaje("Heroe", 100, 20)
    enemigo = Personaje("Enemigo", 50, 10)
    mapa = Mapa()

    lugar_actual = "Ciudad"

    # Funciones para manejar los botones
    def atacar():
        mensaje = jugador.atacar(enemigo)
        etiqueta_resultado.config(text=mensaje)
        etiqueta_salud_enemigo.config(text=f"Salud del Enemigo: {enemigo.salud}")

    def mover():
        nonlocal lugar_actual
        lugar = entrada_lugar.get()
        if lugar:
            mensaje = mapa.mover(lugar_actual, lugar)
            lugar_actual = lugar if "Te has movido" in mensaje else lugar_actual
            etiqueta_resultado.config(text=mensaje)
            etiqueta_lugar_actual.config(text=f"Lugar actual: {lugar_actual}")
        else:
            messagebox.showerror("Error", "Por favor, ingresa un lugar valido.")

    # Etiquetas
    etiqueta_titulo = tk.Label(ventana, text="Bienvenido al RPG", font=("Arial", 16))
    etiqueta_titulo.pack(pady=10)

    etiqueta_resultado = tk.Label(ventana, text="Esperando accion...", font=("Arial", 12))
    etiqueta_resultado.pack(pady=10)

    etiqueta_salud_jugador = tk.Label(ventana, text=f"Salud de {jugador.nombre}: {jugador.salud}")
    etiqueta_salud_jugador.pack(pady=5)

    etiqueta_salud_enemigo = tk.Label(ventana, text=f"Salud del Enemigo: {enemigo.salud}")
    etiqueta_salud_enemigo.pack(pady=5)

    etiqueta_lugar_actual = tk.Label(ventana, text=f"Lugar actual: {lugar_actual}")
    etiqueta_lugar_actual.pack(pady=5)

    # Entrada para mover al jugador
    etiqueta_ingresar_lugar = tk.Label(ventana, text="Ingresa un lugar para moverte:")
    etiqueta_ingresar_lugar.pack(pady=5)

    entrada_lugar = tk.Entry(ventana)
    entrada_lugar.pack(pady=5)

    # Botones de accion
    boton_atacar = tk.Button(ventana, text="Atacar", command=atacar)
    boton_atacar.pack(pady=5)

    boton_mover = tk.Button(ventana, text="Mover", command=mover)
    boton_mover.pack(pady=5)

    # Ejecutar la ventana
    ventana.mainloop()

# Llamar la funcion para crear la ventana
crear_ventana()
