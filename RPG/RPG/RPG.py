
import tkinter as tk
from tkinter import messagebox

# Funciones principales
def iniciar_juego():
    # Ocultar la pantalla inicial
    pantalla_principal.pack_forget()

    # Crear pantalla de juego
    juego_frame.pack()

    # Mostrar informacion inicial del juego
    etiqueta_info.config(text="Salud: 100 de vida\nEnemigo frente a ti con 50 HP.")
    
def atacar():
    mensaje = "Enemigo derrotado. Has ganado."
    messagebox.showinfo("Ganaste", mensaje)
    ventana.quit()

def escapar():
    mensaje = "No has podido escapar. Has muerto."
    messagebox.showinfo("Perdiste", mensaje)
    ventana.quit()

def moverse():
    mensaje = "No puedes moverte. Elige otra accion."
    etiqueta_info.config(text=mensaje)

def salir():
    ventana.quit()

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Juego Basico")

# Pantalla principal
pantalla_principal = tk.Frame(ventana)
pantalla_principal.pack()

# Botones principales
boton_jugar = tk.Button(pantalla_principal, text="Jugar", command=iniciar_juego)
boton_jugar.pack(pady=10)

boton_reglas = tk.Button(pantalla_principal, text="Reglas", command=lambda: messagebox.showinfo("Reglas", "Ataca al enemigo para ganar, o escapa para intentar huir."))
boton_reglas.pack(pady=10)

boton_salir = tk.Button(pantalla_principal, text="Salir", command=salir)
boton_salir.pack(pady=10)

# Pantalla de juego
juego_frame = tk.Frame(ventana)

etiqueta_info = tk.Label(juego_frame, text="Te encuentras frente a un enemigo con 50 HP.", font=("Arial", 12))
etiqueta_info.pack(pady=20)

# Botones de accion en el juego
boton_atacar = tk.Button(juego_frame, text="Atacar", command=atacar)
boton_atacar.pack(pady=10)

boton_moverse = tk.Button(juego_frame, text="Moverse", command=moverse)
boton_moverse.pack(pady=10)

boton_escapar = tk.Button(juego_frame, text="Escapar", command=escapar)
boton_escapar.pack(pady=10)

# Ejecutar la ventana
ventana.mainloop()
