
class MiClase:
  
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad


    def saludar(self):
        print(f"Hola, mi nombre es {self.nombre} y tengo {self.edad} anos.")


if __name__ == "__main__":
    
    persona = MiClase("Isaac", 21)
    
   
    persona.saludar()