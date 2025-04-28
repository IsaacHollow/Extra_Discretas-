import pygame
import sys

# Inicializar pygame
pygame.init()

# Configuraciones iniciales
WIDTH = 400
HEIGHT = 400
ROWS = 10
COLS = 10
CELL_SIZE = WIDTH // COLS

# Colores
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
BLACK = (0, 0, 0)

# Crear ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Buscaminas Demo (pygame)")

# Funcion para dibujar la cuadricula
def draw_grid():
    for x in range(COLS):
        for y in range(ROWS):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRAY, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  # borde negro

# Bucle principal
running = True
while running:
    screen.fill(WHITE)  # Fondo blanco
    draw_grid()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Detectar clics
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            grid_x = mouse_x // CELL_SIZE
            grid_y = mouse_y // CELL_SIZE
            print(f"Clic en celda ({grid_x}, {grid_y})")

    pygame.display.flip()  # Actualizar pantalla

pygame.quit()
sys.exit()



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