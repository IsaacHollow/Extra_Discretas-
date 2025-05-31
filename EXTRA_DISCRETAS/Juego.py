import tkinter as tk #Libreria thinker
import random  #Libreria random para la salida

class Juego: 
    def __init__(self, root): #Constructor
        self.root = root # Guardamos ROOT
        self.N = 7 #Se crea el tamano de la matricula
        self.radio = 20 
        self.score = 5000.0 
        self.recorrido = 1 

        self.dibujar_Interfaz() # Metodo para Dibujar la interfaz del juego
        self.iniciarJuego() # Metodo para colocar la posicion del juego

    def dibujar_Interfaz(self): #Metodo para la interfaz grafica
        for w in self.root.winfo_children():
            w.destroy()  #Borra lo que hay en la ventana (Ventana de inicio)
        
      
        self.lbl_recorrido = tk.Label(self.root, text=f"Find the Exit: {int(self.recorrido)}", 
                              font=("Arial", 16))
        self.lbl_recorrido.pack(pady=5) #Mostramos cuantos nodos ha recorrido con texto, color y tamanio


        self.canvas = tk.Canvas(self.root, bg="white", width=1000, height=1000) #bg= color del fondo, width = ancho,  height=600. 
        self.canvas.pack() #canvas es el area donde se dibuja, el canvas, pack() muestra en la pantalla

        self.coordenadas = {} #Llamamos a un diccionario, Cuadricula(i,j) y coordenadas (x,y)
        for i in range(self.N): 
            for j in range(self.N):
                x = 50 + i * 60
                y = 50 + j * 60
                self.coordenadas[(i,j)] = (x, y)
                for (i, j), (x, y) in self.coordenadas.items():
                 neighbors = [ (i-1, j), (i+1, j), (i, j-1), (i, j+1) ]
                 for ni, nj in neighbors:
                    if (ni, nj) in self.coordenadas:
                     x2, y2 = self.coordenadas[(ni, nj)]
                     self.canvas.create_line(x, y, x2, y2, fill="lightgray")
  
        self.ovalos = {} #Diccionario que guarda los circulos
        for (i,j), (x,y) in self.coordenadas.items():
            oid = self.canvas.create_oval(x-self.radio, y-self.radio,
                                          x+self.radio, y+self.radio,
                                          fill="gray", outline="black", width=2)
            self.ovalos[(i,j)] = oid
            self.canvas.tag_bind(oid, "<Button-1>",
                                 lambda e, ii=i, jj=j: self.click_Jugador(ii,jj))


    def iniciarJuego(self): #Metodo para Inicializar Juego
        c = self.N // 2  # c = Mitad del tablero
        self.current = (c, c) #Posicion Inicial del jugador (Boton Rojo, en el centro)
        positions = [(i, j) for i in range(self.N) for j in range(self.N)]
        positions.remove(self.current)
        self.goal_azul = random.choice(positions) # Posicion de la Meta Azul
        positions.remove(self.goal_azul)
        self.goal_verde = random.choice(positions) #Posicion de la meta Verde

        self.colorNODO(self.current, "red") #Jugador en rojo
        self.colorNODO(self.goal_azul, "gray") # Meta azul en gris (Oculta)
        self.colorNODO(self.goal_verde, "gray") #Meta verde en gris (Oculta)

    def click_Jugador(self, i, j): # Este metodo se "activa" cuando se hace click en un boton
        ci, cj = self.current # Posición actual del jugador
        if abs(ci - i) + abs(cj - j) != 1:
            return # Solo permite moverse a una celda que este arriba, abajo, derecha o izquierda. 

        if (i,j) == self.goal_azul: #Metodo cuando se llega a la meta azul
            self.colorNODO((i,j), "blue") #Color del boton
            self.score += 100 #Se le suma el puntaje
            self.root.after(2000, lambda: self.end_Game("Ganaste Meta azul")) #Se espera 2 segundos y se llama al metodo end_game 
            return
        elif (i,j) == self.goal_verde: #Metodo cuando se llega a la meta verde
            self.colorNODO((i,j), "green") #Color 
            self.score += 250 #Se le suma mas puntaje por ser salida verde
            self.root.after(2000, lambda: self.end_Game("Ganaste Meta azul")) #Se espera 2 segundos y se llama al metodo end_game 
            return

        self.colorNODO(self.current, "white") #Borra la posicion donde eetaba el jugador anterior
        self.current = (i,j) #Actualiza la posicion
        self.colorNODO(self.current, "red") # Crea la nueva posicion del jugador

        self.score -= 12.5  #El puntaje que se quita durante el camino
        self.recorrido = self.recorrido + 1
        self.lbl_recorrido.config(text=f"Nodes Visited: {int(self.recorrido)}")

    def colorNODO(self, pos, color):  # Cambia el color de una celda
        oid = self.ovalos[pos]  # Obtiene el ID del ovalo en esa celda
        self.canvas.itemconfig(oid, fill=color)  # Cambia el color del óvalo en el canvas

    def end_Game(self, mensaje):  # Finaliza el juego y muestra el resultado
        for w in self.root.winfo_children():  # Recorre todos los elementos en la ventana
            w.destroy()  # Los elimina de la interfaz

        label = tk.Label(self.root, text=f"{mensaje}\nPuntaje final: {int(self.score)}",  # Crea etiqueta con mensaje final y puntaje
                     font=("Arial", 20), fg="green")  # Usa letra grande y color verde
        label.pack(pady=30)  # Muestra la etiqueta con margen superior

        btn_retry = tk.Button(self.root, text="Volver a jugar",  # Botón para reiniciar el juego
                        font=("Arial", 16),
                          command=lambda: Juego(self.root))  # Reinicia creando nuevo objeto Juego
        btn_retry.pack(pady=10)  # Muestra el botón con un poco de espacio

        btn_salir = tk.Button(self.root, text="Salir", font=("Arial", 16),  # Botón para cerrar el juego
                          command=self.root.quit)  # Cierra la ventana al hacer clic
        btn_salir.pack(pady=5)  # Muestra el botón con un pequeño margen


