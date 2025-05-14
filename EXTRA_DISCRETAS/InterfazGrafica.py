import tkinter as tk
import tkinter.messagebox
from Juego import Juego

def iniciar_juego():
    root = tk.Tk()   #Crea la ventana
    root.title("Juego de Grafos") #El titulo de la ventana
    root.geometry("600x700") #El tamanio de la ventana
    GrafoJuego(root) #Crea la pantalla de inicio
    root.mainloop() #Hace que no se cierre inmediatamente la ventana

class GrafoJuego:  #Clase
    def __init__(self, root):  #Constructor
        self.root = root #Se "Guarda la ventana"
        self.pantalla_inicio() #Llama al metodo 

    def pantalla_inicio(self):
        for w in self.root.winfo_children():   #Borra todo lo que hay en pantalla
            w.destroy()

        self.canvas = tk.Canvas(self.root, bg="white", width=600, height=600)
        self.canvas.pack()  #Canvas es una clase
        #Sirve para dibujar formar dentro de una Ventana

        coords = {
            "jugar": (250, 150),
            "reglas": (350, 300),
            "salir": (250, 450)
        } #Creamos las cordenadas para luego hacer los ovalos (Botones)

        self.canvas.create_line(*coords["jugar"], *coords["reglas"], width=3) #Crea linea, width es el ancho
        self.canvas.create_line(*coords["reglas"], *coords["salir"], width=3) #Crea linea

        for name, (cx, cy) in coords.items():
            oval = self.canvas.create_oval(cx-50, cy-25, cx+50, cy+25,
                                           fill="lightgray", outline="black", width=2) #Crea el ovalo
            txt = self.canvas.create_text(cx, cy, text=name.upper(), #Agrega el nombre del boton
                                          font=("Arial", 14, "bold"))
            for tag in (oval, txt): #Recorremos el ovalo y el txt
                self.canvas.tag_bind(tag, "<Button-1>", #Cuando el usuario marca 
                                     lambda e, n=name: self.click_inicio(n)) #Llamamos al metodo click_inicio
                                     
    def click_inicio(self, name): #Metodo que resive name por parametros
        if name == "salir": #Si name es salir 
            self.root.quit() #Entonces se llama al metodo quit() que cierra la ventana 
        elif name == "reglas": # Si se toca Reglas 
            tk.messagebox.showinfo("Reglas", "Debes llegar a una de las metas moviendote solo a celdas vecinas.")
        #Muestra las reglas, usando  tk.messagebox.showinfo
        else:
            Juego(self.root)  #Si no es de las opciones anteriores, Llama al metodo Juego de la clase Juego
