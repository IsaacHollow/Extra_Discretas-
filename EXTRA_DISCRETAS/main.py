# -*- coding: utf-8 -*-

import tkinter as tk
import random

class GrafoJuego:
    def __init__(self, root):
        self.root = root
        self.root.title("Juego de Grafos")
        self.root.geometry("600x700")
        self.pantalla_inicio()

    def pantalla_inicio(self):
        # Limpiar todo
        for w in self.root.winfo_children():
            w.destroy()

        self.canvas = tk.Canvas(self.root, bg="white", width=600, height=600)
        self.canvas.pack()

        # Definimos centros
        coords = {
            "jugar": (250, 150),
            "reglas": (350, 300),
            "salir": (250, 450)
        }

        # Dibujar líneas
        self.canvas.create_line(*coords["jugar"], *coords["reglas"], width=3)
        self.canvas.create_line(*coords["reglas"], *coords["salir"], width=3)

        # Dibujar óvalos con texto
        for name, (cx, cy) in coords.items():
            oval = self.canvas.create_oval(cx-50, cy-25, cx+50, cy+25,
                                          fill="lightgray", outline="black", width=2)
            txt  = self.canvas.create_text(cx, cy, text=name.upper(),
                                          font=("Arial", 14, "bold"))
            for tag in (oval, txt):
                self.canvas.tag_bind(tag, "<Button-1>",
                                     lambda e, n=name: self._click_inicio(n))

    def _click_inicio(self, name):
        if name == "salir":
            self.root.quit()
        elif name == "reglas":
            tk.messagebox.showinfo("Reglas", "¡Aún no disponibles!")
        else:  # jugar
            JuegoGrafo(self.root)


class JuegoGrafo:
    def __init__(self, root):
        self.root = root
        self.N = 5
        self.radio = 20
        self.score = 5000.0

        self._setup_ui()
        self._init_game()

    def _setup_ui(self):
        for w in self.root.winfo_children():
            w.destroy()

        # Label de puntuación
        self.lbl_score = tk.Label(self.root, text=f"Puntaje: {int(self.score)}",
                                  font=("Arial", 16))
        self.lbl_score.pack(pady=5)

        # Canvas para el grid
        self.canvas = tk.Canvas(self.root, bg="white", width=600, height=600)
        self.canvas.pack()

        # Coordenadas de cada celda
        self.cell_coords = {}
        for i in range(self.N):
            for j in range(self.N):
                x = 50 + i * 60
                y = 50 + j * 60
                self.cell_coords[(i,j)] = (x, y)
        # Dibujar todos los nodos grises
        self.ovals = {}
        for (i,j), (x,y) in self.cell_coords.items():
            oid = self.canvas.create_oval(x-self.radio, y-self.radio,
                                          x+self.radio, y+self.radio,
                                          fill="gray", outline="black", width=2)
            self.ovals[(i,j)] = oid
            self.canvas.tag_bind(oid, "<Button-1>",
                                 lambda e, ii=i, jj=j: self._on_click(ii,jj))

    def _init_game(self):
        # Posición inicial en el centro
        c = self.N // 2
        self.current = (c, c)
        # Meta aleatoria distinta del centro
        # Crear dos metas diferentes
        positions = [(i, j) for i in range(self.N) for j in range(self.N)]
        positions.remove(self.current)
        self.goal_azul = random.choice(positions)
        positions.remove(self.goal_azul)
        self.goal_verde = random.choice(positions)

        # Pintar inicio y meta
        self._color_cell(self.current, "red")
        self._color_cell(self.goal_azul, "gray")
        self._color_cell(self.goal_verde, "gray")
        # meta oculta en gris hasta llegar

    def _on_click(self, i, j):
        # Sólo permitimos movernos a vecinos
        ci, cj = self.current
        if abs(ci - i) + abs(cj - j) != 1:
            return

        # Si es meta, gana
        if (i,j) == self.goal_azul:
             self._color_cell((i,j), "blue")
             self.score += 100  # gana 100 puntos extra
             self._end_game("¡Ganaste! Meta azul")
             return
        elif (i,j) == self.goal_verde:
            self._color_cell((i,j), "green")
            self.score += 250  # gana 250 puntos extra
            self._end_game("¡Ganaste! Meta verde")
            return

        # Mover: antiguo rojo → blanco, nuevo → rojo
        self._color_cell(self.current, "white")
        self.current = (i,j)
        self._color_cell(self.current, "red")

        # Restar puntaje
        self.score -= 12.5
        self.lbl_score.config(text=f"Puntaje: {int(self.score)}")

    def _color_cell(self, pos, color):
        oid = self.ovals[pos]
        self.canvas.itemconfig(oid, fill=color)


    def _end_game(self, mensaje):

        for w in self.root.winfo_children():
            w.destroy()

        label = tk.Label(self.root, text=f"{mensaje}\nPuntaje final: {int(self.score)}",
                     font=("Arial", 20), fg="green")
        label.pack(pady=30)

        btn_retry = tk.Button(self.root, text="Volver a jugar",
                          font=("Arial", 16),
                          command=lambda: GrafoJuego(self.root))
        btn_retry.pack(pady=10)

        btn_salir = tk.Button(self.root, text="Salir", font=("Arial", 16),
                          command=self.root.quit)
        btn_salir.pack(pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    import tkinter.messagebox  # para reglas
    app = GrafoJuego(root)
    root.mainloop()

