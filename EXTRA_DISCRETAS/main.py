# -*- coding: utf-8 -*-

import tkinter as tk
import random
from tkinter import messagebox

class GrafoJuego:
    def __init__(self, root):
        self.root = root
        self.root.title("Juego de Grafos")
        self.root.geometry("600x600")
        self.pantalla_inicio()

    def pantalla_inicio(self):
        # Borrar widgets anteriores si existen
        for widget in self.root.winfo_children():
            widget.destroy()

        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(fill="both", expand=True)

        # Coordenadas para los botones
        coords = {
            "jugar": (250, 100),
            "reglas": (300, 200),
            "salir": (250, 300)
        }

        # Crear los botones como óvalos con texto encima
        self.botones = {}

        for nombre, (x, y) in coords.items():
            ovalo = self.canvas.create_oval(x, y, x+100, y+40, fill="lightgray", outline="black")
            texto = self.canvas.create_text(x+50, y+20, text=nombre.upper(), font=("Arial", 12, "bold"))

            self.botones[nombre] = (ovalo, texto)
            self.canvas.tag_bind(ovalo, "<Button-1>", lambda e, n=nombre: self.boton_click(n))
            self.canvas.tag_bind(texto, "<Button-1>", lambda e, n=nombre: self.boton_click(n))

    def boton_click(self, nombre):
        if nombre == "salir":
            self.root.quit()
        elif nombre == "reglas":
            messagebox.showinfo("Reglas", "Aun no disponibles")
        elif nombre == "jugar":
            self.iniciar_juego()

    def iniciar_juego(self):
        JuegoGrafo(self.root)

class JuegoGrafo:
    def __init__(self, root):
        self.root = root
        self.tamano = 9
        self.matriz_botones = [[None for _ in range(self.tamano)] for _ in range(self.tamano)]
        self.visibles = set()
        self.meta = None
        self.puntaje = 1000
        self.botones = []

        self.crear_interfaz()

    def crear_interfaz(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.canvas = tk.Canvas(self.root, bg="white", width=600, height=600)
        self.canvas.pack(fill="both", expand=True)

        # Generar coordenadas de los botones
        self.coords = {}
        radio = 20
        for i in range(self.tamano):
            for j in range(self.tamano):
                x = 50 + i * 50
                y = 50 + j * 50
                self.coords[(i, j)] = (x, y)
                # Dibujar nodo (crculo) invisible al principio
                btn = self.canvas.create_oval(x - radio, y - radio, x + radio, y + radio, fill="white", outline="black")
                self.botones.append((btn, (i, j)))
                self.canvas.tag_bind(btn, "<Button-1>", lambda e, i=i, j=j: self.nodo_click(i, j))

        # Inicializar nodo de inicio
        self.inicio = (self.tamano // 2, self.tamano // 2)
        self.meta = (random.randint(0, self.tamano - 1), random.randint(0, self.tamano - 1))

        # Marcar nodo de inicio (rojo)
        self.marcar_visible(self.inicio[0], self.inicio[1], color="red")

        # Mostrar puntaje
        self.puntaje_label = tk.Label(self.root, text=f"Puntaje: {self.puntaje}", font=("Arial", 14))
        self.puntaje_label.pack(pady=20)

    def nodo_click(self, i, j):
        # Si es la meta
        if (i, j) == self.meta:
            self.canvas.itemconfig(self.botones[i * self.tamano + j][0], fill="blue")
            self.finalizar_juego(True)
            return

        # Cambiar color a blanco (visitado)
        self.canvas.itemconfig(self.botones[i * self.tamano + j][0], fill="white")
        self.marcar_visible(i, j)

        # Restar puntaje
        self.puntaje -= 12.5
        self.puntaje_label.config(text=f"Puntaje: {round(self.puntaje)}")

    def marcar_visible(self, i, j, color="gray"):
        # Hacer visible el nodo cambiando su color
        self.canvas.itemconfig(self.botones[i * self.tamano + j][0], fill=color)

    def finalizar_juego(self, gano):
        mensaje = f"Ganaste Encontraste el nodo azul. Puntaje final: {round(self.puntaje)}" if gano else f"Juego terminado. Puntaje final: {round(self.puntaje)}"
        etiqueta = tk.Label(self.root, text=mensaje, font=("Arial", 16), fg="green")
        etiqueta.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = GrafoJuego(root)
    root.mainloop()
