# The karate dataset is built-in in networkx
import networkx as nx
import torch
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim
from sklearn.metrics import classification_report

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

G = nx.karate_club_graph() #Karate_club es un grafo ya predeterminado
"""
print(G)
"""
# Known ids of the instructor, admin and members
ID_INSTR = 0    #le da el ID y el valo 0 representa que es nodo instructor
ID_ADMIN = 33   #el valor 33 representa el administrador
ID_MEMBERS = set(G.nodes()) - {ID_ADMIN, ID_INSTR} #contiene todos los nodos excepto los lideres 0 y 33
""""
print(ID_MEMBERS)
print(f'{G.name}: {len(G.nodes)} vertices, {len(G.edges)} edges')

print(G.number_of_nodes())
"""
# Input featuers (no information on nodes):
X = torch.eye(G.number_of_nodes()) #hace una matriz identidad del tamano de la cantidad de nodos
#print(X)

# Create ground-truth labels
# - Assign the label "0" to the "Mr. Hi" community
# - Assign the label "1" to the "Officer" community
labels = [int(not d['club']=='Mr. Hi') for _, d in G.nodes().data()]
"""
G.nodes().data() contiene la cantidad de nodos y que mensaje tiene
labels es un diccionario de quienes son MR.hi=0 y officer=1
"""

labels = torch.tensor(labels, dtype=torch.long) #convierte labels en un tensor largo

print (labels)

# Let's check the nodes metadata
"""
for (node_id, node_data), label_id in zip(G.nodes().data(), labels):
    print(f'Node id: {node_id},\tClub: {node_data["club"]},\t\tLabel: {label_id.item()}')
"""

# Adjacency matrix, binary
A = nx.to_numpy_array(G, weight=None)
A = np.array(A) #esto vuelve A en una matriz adyacente

# Degree matrix
dii = np.sum(A, axis=1, keepdims=False)  # suma las columnas
D = np.diag(dii) #la vuelve una matriz diagonal

L = D - A

#verifica si es simetrica
print((L.transpose() == L).all())

print(np.trace(L) == 2 * G.number_of_edges()) #np.trace devuelve la suma de los elementos diagonales en L
#verifica si la suma de los grados es igual cocal doble del numero de aristas, lo cual debe cumplirse siempre para un grafo no dirigido.

# Sum of colums/rows is zero
print(np.sum(L, axis=1))#imprime la suma de las fila de la matriz laplaciana
print(np.sum(L, axis=0))#imprime la suma de las fila de la matriz laplaciana
#las matrices Lapicianas siempre deben de dar 0

# Compute the eigevanlues and eigenvector
w, Phi = np.linalg.eigh(L) #w calcula los auto valores y phi el primer auto vector

plt.plot(w); plt.xlabel(r'$\lambda$'); plt.title('Spectrum')
plt.show()

# Adjacency matrix
# Obtiene la matriz de adyacencia del grafo G como una matriz dispersa (sparse) de SciPy.
A = nx.to_scipy_sparse_array(G, weight=None)
# Convierte la matriz dispersa A a una matriz densa (NumPy array).
A = np.array(A.todense())
# Crea una matriz identidad del mismo tamano que A (diagonal de 1s, resto ceros).
I = np.eye(A.shape[0])
# Suma la matriz identidad a A: esto agrega bucles en los nodos (autoconexiones).
A = A + I


dii = np.sum(A, axis=1, keepdims=False)#devuelve un array de una dimension de la suma de las filas de A


# normalizacion 
D_inv_h = np.diag(dii**(-0.5)) # convierte a dii en inversa 
L =  D_inv_h @ A @ D_inv_h #realiza la operacion de la normalizacion


import torch.nn as nn
from typing import List

# aplica una capa de convolucion en un grafo
class GCNLayer(nn.Module):
    def __init__(self, 
                 graph_L: torch.Tensor, 
                 in_features: int, 
                 out_features: int, 
                 max_deg: int = 1
        ):
        """
        :param graph_L: el laplaciano del grafo normalizado. Contiene toda la informacion que necesitamos sobre el grafo.
        :param in_features: el numero de caracteristicas de entrada para cada nodo.
        :param out_features: el numero de caracteristicas de salida para cada nodo.
        :param max_deg: la potencia del laplaciano a considerar, es decir, la q en la formula espacial.
        """
        super().__init__()
        
        # Cada FC es como la matriz alpha_k, donde la ultima incluye la bias
        self.fc_layers = nn.ModuleList()
        for i in range(max_deg - 1):
            self.fc_layers.append(nn.Linear(in_features, out_features, bias=False))     # q - 1 capas sin bias
        self.fc_layers.append(nn.Linear(in_features, out_features, bias=True))          # el ultimo con bias
        
        # Pre-calcular beta_k(L) para cada una
        self.laplacians = self.calc_laplacian_functions(graph_L, max_deg)
        
    def calc_laplacian_functions(self, L: torch.Tensor, max_deg: int) -> List[torch.Tensor]:
        """
        Calcular todas las potencias de L desde 1 hasta max_deg

        :parametro L: una matriz cuadrada
        :parametro max_deg: numero de potencias a calcular

        :devuelve: una lista de tensores, donde el elemento i es L^{i+1} (es decir, se empieza a contar desde 1)
        """
        res = [L]
        for _ in range(max_deg-1):
            res.append(torch.mm(res[-1], L))
        return res
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Realizar un paso hacia adelante en la convolucion del grafo

        :params X: mapas de caracteristicas de entrada [vertices, caracteristicas_de_entrada]
        :returns: mapas de caracteristicas de salida [vertices, caracteristicas_de_salida]
        """

        Z = torch.tensor(0.)
        for k, fc in enumerate(self.fc_layers):
            L = self.laplacians[k]
            LX = torch.mm(L, X)  # Aplica la propagacion
            Z = fc(LX) + Z       # Aplica transformacion lineal y acumula
        
        return torch.relu(Z)
    
in_features, out_features = X.shape[1], 2
graph_L = torch.tensor(L, dtype=torch.float)
max_deg = 2
hidden_dim = 5

# Stack two GCN layers as our model
gcn2 = nn.Sequential(
    GCNLayer(graph_L, in_features, hidden_dim, max_deg),
    GCNLayer(graph_L, hidden_dim, out_features, max_deg),
    nn.LogSoftmax(dim=1)
)
"""
plot_de_perdidas codigo propio de 
Jose David Chavarria Villalobos
Isaac Sibaja Cortez
Erick Zamora Cruz
"""
def plot_de_perdidas(losses):

    plt.plot(losses)                    # Dibuja la curva de perdidas
    plt.xlabel("epoch")                # Etiqueta del eje X
    plt.ylabel("perdida")                 # Etiqueta del eje Y
    plt.title("perdida")                   
    plt.grid(True)                     # Activa cuadricula
    plt.show()                         


def train_node_classifier(model, optimizer, X, y, epochs=60, print_every=10):
    y_pred_epochs = []
    loss_values = []
    for epoch in range(epochs+1):
        y_pred = model(X)  # Compute on all the graph
        y_pred_epochs.append(y_pred.detach())
        # Semi-supervised: only use labels of the Instructor and Admin nodes
        labelled_idx = [ID_ADMIN, ID_INSTR]
        loss = F.nll_loss(y_pred[labelled_idx], y[labelled_idx])  # loss on only two nodes
        loss_values.append(loss.item()) #nuevo
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            print(f'Epoch {epoch:2d}, loss={loss.item():.5f}')
    print("-"*50)
    print(loss_values)
    print("-"*50)
    return y_pred_epochs, loss_values 

"""
codigo propio de 
Jose David Chavarria Villalobos
Isaac Sibaja Cortez
Erick Zamora Cruz
"""

def ver_predicciones(G, y_true, y_pred): #G es el grafo, y_true son las clases verdaderas, y_pred son las clases predichas.
    pos = nx.spring_layout(G)  # Posiciones de los nodos

    # Dibujar nodos correctamente clasificados (verde) y mal clasificados (rojo)
    correctos = [n for n in G.nodes if y_true[n] == y_pred[n]]
    incorrectos = [n for n in G.nodes if y_true[n] != y_pred[n]]

    nx.draw_networkx_nodes(G, pos, nodelist=correctos, node_color='lightgreen')
    nx.draw_networkx_nodes(G, pos, nodelist=incorrectos, node_color='salmon')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.title("Predicciones")
    plt.axis("off")
    plt.show()
    
# ENTRENAMIENTO del modelo GCN
epochs = 100  # Aumenta el numero de epocas para tener una curva significativa
all_losses = []  # Aqui se almacenaran las listas de perdidas de cada ejecucion
i=0
for i in range(1):
    print(f"\n--- Entrenamiento {i + 1} ---")
    # Re-crear modelo y optimizador en cada iteracion para que no use pesos anteriores
    gcn_model = nn.Sequential(
        GCNLayer(graph_L, in_features, hidden_dim, max_deg),
        GCNLayer(graph_L, hidden_dim, out_features, max_deg),
        nn.LogSoftmax(dim=1)
    )
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)

    _, loss_values = train_node_classifier(gcn_model, optimizer, X, labels, epochs=100)
    all_losses.append(loss_values)

y_pred = torch.argmax(gcn2(X), dim=1).detach().numpy()
y_true = labels.numpy()
print(classification_report(y_true, y_pred, target_names=['Mr. Hi', 'Officer']))

plot_de_perdidas(loss_values)
ver_predicciones(G, y_true, y_pred)



