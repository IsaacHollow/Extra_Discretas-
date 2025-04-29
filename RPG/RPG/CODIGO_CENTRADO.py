import tkinter as tk
from tkinter import messagebox

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Juego Basico")
ventana.resizable(False, False)

# Tamaño deseado de la ventana
ancho_ventana = 600
alto_ventana = 400

# Obtener el tamaño de la pantalla
ancho_pantalla = ventana.winfo_screenwidth()
alto_pantalla = ventana.winfo_screenheight()

# Calcular coordenadas para centrar la ventana
x = (ancho_pantalla - ancho_ventana) // 2
y = (alto_pantalla - alto_ventana) // 2

# Aplicar tamaño y posicion centrada
ventana.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

# --- Funciones principales ---
def iniciar_juego():
    pantalla_principal.pack_forget()
    juego_frame.pack(expand=True)
    etiqueta_info.config(text="Salud: 100 de vida\nEnemigo frente a ti con 50 HP")

def atacar():
    messagebox.showinfo("Ganaste", "Enemigo derrotado. Has ganado.")
    ventana.quit()

def escapar():
    messagebox.showinfo("Perdiste", "No has podido escapar. Has muerto.")
    ventana.quit()

def moverse():
    etiqueta_info.config(text="No puedes moverte. Elige otra accion.")

def salir():
    ventana.quit()

# --- Pantalla principal ---
pantalla_principal = tk.Frame(ventana)
pantalla_principal.pack(expand=True)

boton_jugar = tk.Button(pantalla_principal, text="Jugar", width=20, command=iniciar_juego)
boton_jugar.pack(pady=10)

boton_reglas = tk.Button(
    pantalla_principal,
    text="Reglas",
    width=20,
    command=lambda: messagebox.showinfo(
        "Reglas",
        "Ataca al enemigo para ganar, o escapa para intentar huir."
    )
)
boton_reglas.pack(pady=10)

boton_salir = tk.Button(pantalla_principal, text="Salir", width=20, command=salir)
boton_salir.pack(pady=10)

# --- Pantalla de juego ---
juego_frame = tk.Frame(ventana)

etiqueta_info = tk.Label(juego_frame, text="", font=("Arial", 12), justify="center")
etiqueta_info.pack(pady=20)

boton_atacar = tk.Button(juego_frame, text="Atacar", width=15, command=atacar)
boton_atacar.pack(pady=5)

boton_moverse = tk.Button(juego_frame, text="Moverse", width=15, command=moverse)
boton_moverse.pack(pady=5)

boton_escapar = tk.Button(juego_frame, text="Escapar", width=15, command=escapar)
boton_escapar.pack(pady=5)

ventana.mainloop()
