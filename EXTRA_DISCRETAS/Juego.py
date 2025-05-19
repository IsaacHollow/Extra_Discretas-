import tkinter as tk #Libreria thinker
import random  #Libreria random para la salida

class Juego: 
    def __init__(self, root): #Constructor
        self.root = root # Guardamos ROOT
        self.N = 5 #Se crea el tamano de la matricula
        self.radio = 20 #Radio de los circulos
        self.score = 5000.0 #Puntaje Inicial del juego

        self._setup_ui() # Metodo para Dibujar la interfaz del juego
        self._init_game() # Metodo para colocar la posicion del juego

    def _setup_ui(self): #Metodo para la interfaz grafica
        for w in self.root.winfo_children():
            w.destroy()  #Borra lo que hay en la ventana (Ventana de inicio)

        self.lbl_score = tk.Label(self.root, text=f"Puntaje: {int(self.score)}",
                                  font=("Arial", 16))
        self.lbl_score.pack(pady=5) #Mostramos el puntaje, eligiendo el tipo de texto, color y tamanio

        self.canvas = tk.Canvas(self.root, bg="white", width=600, height=400) #bg= color del fondo, width = ancho,  height=600. 
        self.canvas.pack() #canvas es el area donde se dibuja, el canvas, pack() muestra en la pantalla

        self.cell_coords = {} #Llamamos a un diccionario, Cuadricula(i,j) y coordenadas (x,y)
        for i in range(self.N): 
            for j in range(self.N):
                x = 50 + i * 60
                y = 50 + j * 60
                self.cell_coords[(i,j)] = (x, y)

        self.ovals = {}
        for (i,j), (x,y) in self.cell_coords.items():
            oid = self.canvas.create_oval(x-self.radio, y-self.radio,
                                          x+self.radio, y+self.radio,
                                          fill="gray", outline="black", width=2)
            self.ovals[(i,j)] = oid
            self.canvas.tag_bind(oid, "<Button-1>",
                                 lambda e, ii=i, jj=j: self._on_click(ii,jj))

    def _init_game(self):
        c = self.N // 2
        self.current = (c, c)
        positions = [(i, j) for i in range(self.N) for j in range(self.N)]
        positions.remove(self.current)
        self.goal_azul = random.choice(positions)
        positions.remove(self.goal_azul)
        self.goal_verde = random.choice(positions)

        self._color_cell(self.current, "red")
        self._color_cell(self.goal_azul, "gray")
        self._color_cell(self.goal_verde, "gray")

    def _on_click(self, i, j):
        ci, cj = self.current
        if abs(ci - i) + abs(cj - j) != 1:
            return

        if (i,j) == self.goal_azul:
            self._color_cell((i,j), "blue")
            self.score += 100
            self._end_game("Ganaste Meta azul")
            return
        elif (i,j) == self.goal_verde:
            self._color_cell((i,j), "green")
            self.score += 250
            self._end_game("Ganaste Meta verde")
            return

        self._color_cell(self.current, "white")
        self.current = (i,j)
        self._color_cell(self.current, "red")

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
                              command=lambda: Juego(self.root))
        btn_retry.pack(pady=10)

        btn_salir = tk.Button(self.root, text="Salir", font=("Arial", 16),
                              command=self.root.quit)
        btn_salir.pack(pady=5)
