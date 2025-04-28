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
