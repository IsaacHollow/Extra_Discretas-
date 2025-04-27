def opcion1():
    print("Hola, has elegido la opcion 1")

def opcion2():
    print("Hola, has elegido la opcion 2")


opciones = {
    1: opcion1,
    2: opcion2,
}

print("Elige una opcion (1, 2):")
opcion = int(input())

opciones.get(opcion, lambda: print("Opcion no valida"))()


"""
import tkinter as tk
import random


WIDTH = 25  # cantidad de columnas
HEIGHT = 15  # cantidad de filas

def create_grid(root):
    for y in range(HEIGHT):
        for x in range(WIDTH):
            button = tk.Button(root, width=3, height=1, command=lambda x=x, y=y: click_cell(x, y))
            button.grid(row=y, column=x)
            buttons[(x, y)] = button

def click_cell(x, y):
    print(f"Click en celda ({x}, {y})")
    buttons[(x, y)].config(text="0")  # luego aqui puede mostrar mina o numero

# Crear ventana
root = tk.Tk()
root.title("Buscaminas Demo")

buttons = {}  # Diccionario para guardar los botones

create_grid(root)

root.mainloop()

"""