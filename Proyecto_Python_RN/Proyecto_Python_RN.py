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
X = torch.eye(G.number_of_nodes()) #hace una matriz identidad del tamaño de la cantidad de nodos
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
#verifica si la suma de los grados es igual cocal doble del número de aristas, lo cual debe cumplirse siempre para un grafo no dirigido.

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
# Crea una matriz identidad del mismo tamaño que A (diagonal de 1s, resto ceros).
I = np.eye(A.shape[0])
# Suma la matriz identidad a A: esto agrega bucles en los nodos (autoconexiones).
A = A + I


dii = np.sum(A, axis=1, keepdims=False)#devuelve un array de una dimension de la suma de las filas de A


# normalizacion 
D_inv_h = np.diag(dii**(-0.5)) # convierte a dii en inversa 
L =  D_inv_h @ A @ D_inv_h #realiza la operacion de la normalizacion


import torch.nn as nn
from typing import List

class GCNLayer(nn.Module):
    def __init__(self, 
                 graph_L: torch.Tensor, 
                 in_features: int, 
                 out_features: int, 
                 max_deg: int = 1
        ):
        """
        :param graph_L: the normalized graph laplacian. It is all the information we need to know about the graph
        :param in_features: the number of input features for each node
        :param out_features: the number of output features for each node
        :param max_deg: how many power of the laplacian to consider, i.e. the q in the spacial formula
        """
        super().__init__()
        
        # Each FC is like the alpha_k matrix, with the last one including bias
        self.fc_layers = nn.ModuleList()
        for i in range(max_deg - 1):
            self.fc_layers.append(nn.Linear(in_features, out_features, bias=False))     # q - 1 layers without bias
        self.fc_layers.append(nn.Linear(in_features, out_features, bias=True))          # last one with bias
        
        # Pre-calculate beta_k(L) for every key
        self.laplacians = self.calc_laplacian_functions(graph_L, max_deg)
        
    def calc_laplacian_functions(self, 
                                 L: torch.Tensor, 
                                 max_deg: int
        ) -> List[torch.Tensor]:
        """
        Compute all the powers of L from 1 to max_deg

        :param L: a square matrix
        :param max_deg: number of powers to compute

        :returns: a list of tensors, where the element i is L^{i+1} (i.e. start counting from 1)
        """
        res = [L]
        for _ in range(max_deg-1):
            res.append(torch.mm(res[-1], L))
        return res
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform one forward step of graph convolution

        :params X: input features maps [vertices, in_features]
        :returns: output features maps [vertices, out_features]
        """
        Z = torch.tensor(0.)
        for k, fc in enumerate(self.fc_layers):
            L = self.laplacians[k]
            LX = torch.mm(L, X)
            Z = fc(LX) + Z
        
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



def train_node_classifier(model, optimizer, X, y, epochs=60, print_every=10):
    y_pred_epochs = []
    for epoch in range(epochs+1):
        y_pred = model(X)  # Compute on all the graph
        y_pred_epochs.append(y_pred.detach())

        # Semi-supervised: only use labels of the Instructor and Admin nodes
        labelled_idx = [ID_ADMIN, ID_INSTR]
        loss = F.nll_loss(y_pred[labelled_idx], y[labelled_idx])  # loss on only two nodes

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            print(f'Epoch {epoch:2d}, loss={loss.item():.5f}')
    return y_pred_epochs

optimizer = torch.optim.Adam(gcn2.parameters(), lr=0.01)

y_pred_epochs = train_node_classifier(gcn2, optimizer, X, labels)

y_pred = torch.argmax(gcn2(X), dim=1).detach().numpy()
y = labels.numpy()
print(classification_report(y, y_pred, target_names=['I','A']))

mlp = nn.Sequential(
    nn.Linear(in_features, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, out_features),
    nn.ReLU(),
    nn.LogSoftmax(dim=1)
)

optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
_ = train_node_classifier(mlp, optimizer, X, labels, epochs=2000, print_every=500)

print(classification_report(labels.numpy(), torch.argmax(mlp(X), dim=1).detach().numpy(), target_names=['I','A']))


