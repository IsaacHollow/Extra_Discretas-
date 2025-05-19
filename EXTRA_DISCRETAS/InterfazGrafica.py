import tkinter as tk #Libreri tkinter
import tkinter.messagebox 
from Juego import Juego #importa la clase juego

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
            "play": (300, 150),
            "rules": (450, 300),
            "exit": (300, 400),
            "credits": (150,300),
            "C" : (100,500)
        } #Creamos las cordenadas para luego hacer los ovalos (Botones)
        
        coords2 = {
            "." : (300,110),
            " " : (110,300),
        }

        for name, (cx, cy) in coords2.items():
            circule = self.canvas.create_oval(cx-45, cy-45, cx+45, cy+45,
                                           fill="white", outline="black", width=2) #Crea el circulo
            txt = self.canvas.create_text(cx, cy, text=name.upper(), #Agrega el nombre del boton
                                          font=("Arial", 14, "bold"))
            for tag in (circule, txt): #Recorremos el ovalo y el txt
                self.canvas.tag_bind(tag, "<Button-1>", #Cuando el usuario marca 
                                     lambda e, n=name: self.click_inicio(n)) #Llamamos al metodo click_inicio


        self.canvas.create_line(*coords["play"], *coords["rules"], width=3) #Crea linea, width es el ancho
        #self.canvas.create_line(*coords["rules"], *coords["exit"], width=3) #Crea linea
        self.canvas.create_line(*coords["play"], *coords["credits"], width=3)
        #self.canvas.create_line(*coords["credits"], *coords["exit"], width=3)
        self.canvas.create_line(*coords["play"], *coords["exit"], width=3)

        for name, (cx, cy) in coords.items():
            circule = self.canvas.create_oval(cx-45, cy-45, cx+45, cy+45,
                                           fill="orange", outline="black", width=2) #Crea el circulo
            txt = self.canvas.create_text(cx, cy, text=name.upper(), #Agrega el nombre del boton
                                          font=("Arial", 14, "bold"))
            for tag in (circule, txt): #Recorremos el ovalo y el txt
                self.canvas.tag_bind(tag, "<Button-1>", #Cuando el usuario marca 
                                     lambda e, n=name: self.click_inicio(n)) #Llamamos al metodo click_inicio
                                     
    def click_inicio(self, name): #Metodo que resive name por parametros
        if name == "exit": #Si name es salir 
            self.root.quit() #Entonces se llama al metodo quit() que cierra la ventana 
        elif name == "rules": # Si se toca Reglas 
            tk.messagebox.showinfo("Reglas", "Debes llegar a una de las metas moviendote solo a celdas vecinas.")
        #Muestra las reglas, usando  tk.messagebox.showinfo
        elif name == "credits":
            tk.messagebox.showinfo(
            "Creditos",
            "Programadores:\nIsaac Sibaja\nJose Chavarria\nCodigo hecho en Python\nInterfaz Grafica: Tkinter")
        elif name == "play":
            Juego(self.root)  #Si no es de las opciones anteriores, Llama al metodo Juego de la clase Juego


         
