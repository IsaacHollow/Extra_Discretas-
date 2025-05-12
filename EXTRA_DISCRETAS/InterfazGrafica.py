import tkinter as tk
import tkinter.messagebox
from Juego import JuegoGrafo

def iniciar_juego():
    root = tk.Tk()
    root.title("Juego de Grafos")
    root.geometry("600x700")
    GrafoJuego(root)
    root.mainloop()

class GrafoJuego:
    def __init__(self, root):
        self.root = root
        self.pantalla_inicio()

    def pantalla_inicio(self):
        for w in self.root.winfo_children():
            w.destroy()

        self.canvas = tk.Canvas(self.root, bg="white", width=600, height=600)
        self.canvas.pack()

        coords = {
            "jugar": (250, 150),
            "reglas": (350, 300),
            "salir": (250, 450)
        }

        self.canvas.create_line(*coords["jugar"], *coords["reglas"], width=3)
        self.canvas.create_line(*coords["reglas"], *coords["salir"], width=3)

        for name, (cx, cy) in coords.items():
            oval = self.canvas.create_oval(cx-50, cy-25, cx+50, cy+25,
                                           fill="lightgray", outline="black", width=2)
            txt = self.canvas.create_text(cx, cy, text=name.upper(),
                                          font=("Arial", 14, "bold"))
            for tag in (oval, txt):
                self.canvas.tag_bind(tag, "<Button-1>",
                                     lambda e, n=name: self._click_inicio(n))

    def _click_inicio(self, name):
        if name == "salir":
            self.root.quit()
        elif name == "reglas":
            tk.messagebox.showinfo("Reglas", "Debes llegar a una de las metas moviendote solo a celdas vecinas.")
        else:
            JuegoGrafo(self.root)
