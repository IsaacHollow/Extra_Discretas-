"""
Prompts ChatGPT (ezamc26@gmail.com):
1) Optimiza el código de la aplicación dash Cytoscape para visualizar grafos de mayor tamaño

2) Revisa funcionalidades para agregar, eliminar y conectar nodos, así como encontrar caminos entre nodos utilizando algoritmos de recorrido BFS, DFS y Dijkstra

3) Implementa un codigo de css para visualizar los nodos y aristas del grafo de manera más clara
"""

#pip install -r Dependencias.txt 

# Importación de librerías necesarias
import dash
import json
import random
from dash import html, dcc, State, ctx
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
import networkx as nx

cyto.load_extra_layouts() 
# Carga de layouts adicionales para Cytoscape (especificamente para utilizar cose-bilkent en vez de cose)


# ---------- Cargar y preparar datos ----------

""" Cargar nodos desde archivo JSON y genera relaciones"""
def cargar_datos(ruta="data.json"):
    with open(ruta, 'r', encoding='utf-8') as archivo:
        datos = json.load(archivo)
        personas = set(datos.get('persons', []))
        libros = set(datos.get('books', []))
        return personas, libros

def generar_relaciones_aleatorias(personas, libros, min_enlaces=1, max_enlaces=2):
    """Genera relaciones aleatorias entre personas y libros (utilizando random)
    Asegura que las relaciones no excedan un límite (250)

    """
    relaciones = set()
    while len(relaciones) <= 250:
        for persona in personas:
            num_enlaces = random.randint(min_enlaces, max_enlaces)
            libros_relacionados = random.sample(sorted(libros), min(num_enlaces, len(libros)))
            for libro in libros_relacionados:
                relaciones.add((persona, libro))
    return relaciones

def tipo_nodo(nodo, personas):
    """tipo de nodo (persona o libro)"""
    return "person" if nodo in personas else "book"

def construir_grafo():
    """Construye el grafo con nodos y aristas utilizando NetworkX"""
    grafo = nx.Graph()
    for n in Personas | Libros:
        grafo.add_node(n, type=tipo_nodo(n, Personas))
    relaciones = generar_relaciones_aleatorias(Personas, Libros) # Genera relaciones aleatorias
    for persona, libro in relaciones:
        grafo.add_edge(persona, libro)
    return grafo

def nx_a_cytoscape(grafo): 
    """Convierte el grafo de NetworkX a un formato compatible con Cytoscape"""
    elementos = []
    for id_nodo, datos_nodo in grafo.nodes(data=True):
        elementos.append({
            "data": {"id": id_nodo, "label": id_nodo, "type": datos_nodo.get('type', 'unknown')}
        })
    for origen, destino in grafo.edges:
        elementos.append({
            "data": {"source": origen, "target": destino}
        })
    return elementos

# ---------- Inicializar datos ----------
Personas, Libros = cargar_datos()
G = construir_grafo()

# ---------- Construcción de la app de Dash ----------
app = dash.Dash(__name__)


"""
Layout de la aplicación Dash que incluye botones para mostrar integrantes, opciones y estadísticas en HTML
y usando css para los estilos
"""
app.layout = html.Div([
    html.H2("[ NetworkX and Dash Cytoscape Demo ]"),

    # Botones principales
    html.Div([
        html.Button("Integrantes", id="btn-integrantes", n_clicks=0),
        html.Button("Opciones", id="toggle-options", n_clicks=0),
        html.Button("Estadísticas", id="toggle-stats", n_clicks=0),
    ], className="button-row"),   #classname='button-row_options' para aplicar estilos css

    html.Div(id="members-container", style={"display": "none"}, children=[
        html.P("Isaac Sibaja Cortes, Jose david Chavarria, Erick Zamora Cruz")
    ]),

    # Contenedor de opciones de manipulación del grafo
  
    html.Div(id="options-container", style={"display": "none"}, children=[
        html.Div([
            html.Label("Agregar nodo:"),
            dcc.Input(id='input-node', type='text'),
            dcc.Dropdown(
                id='input-node-type',
                options=[
                    {'label': 'Persona', 'value': 'person'},
                    {'label': 'Libro', 'value': 'book'}
                ],
                placeholder='Tipo de nodo'
            ),
            html.Button('Agregar Nodo', id='boton_añadir_nodo', n_clicks=0)
        ]),
            html.Div([
                html.Label("Conectar nodos: "),
                dcc.Input(id='input-source-node', type='text', placeholder='Nodo tipo persona'),
                dcc.Input(id='input-target-node', type='text', placeholder='Nodo tipo libro'),
                html.Button('Conectar nodos', id='boton_conectar_nodos', n_clicks=0)
            ]),
            html.Div([
                html.Label("Eliminar nodo:"),
                dcc.Input(id='entrada_borrar_nodo', type='text'),
                html.Button('Eliminar nodo', id='boton_eliminar_nodo')
            ]),
            html.Div([
                html.Label("Buscar ruta entre:"),
                dcc.Input(id='input-source', type='text'),
                dcc.Input(id='input-target', type='text'),
                html.Label("Seleccione algoritmo de recorrido:"),
                
            dcc.Dropdown(id='algoritmo', options=[
                    {'label': 'Dijkstra', 'value': 'dijkstra'},
                    {'label': 'BFS', 'value': 'bfs'},
                    {'label': 'DFS', 'value': 'dfs'}
                ],
                value='dijkstra',
                clearable=False
            ),

        ]),

            html.Div(className='button-row_options', children=[ #botones en orden
                html.Button("Encontrar camino", id='boton_camino'),
                html.Button("Reiniciar", id='boton_regenerar')
            ]),

    ]),

        html.Div(id="stats-container", style={"display": "none"},
            children=[
                html.H3("Estadísticas del grafo"),
                html.Div(id="stats-output")
        ]
    ),

    html.Br(),

# Visualización del grafo con Cytoscape

    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=nx_a_cytoscape(G),
        layout={
            'name': 'cose-bilkent', # Utiliza el layout cose-bilkent para una mejor disposición de los nodos en grafos grandes
            'nodeRepulsion': 10000000, #Gran repulsion entre nodos para evitar superposiciones
            'idealEdgeLength': 120, # Longitud ideal de las aristas
            'nodeOverlap': 0,  # Evita superposiciones de nodos
            'gravity': 0.3, # Gravedad para mantener los nodos juntos
            'numIter': 3000 # Número de iteraciones para el layout para ajustar bien los nodos y el grafo en general
        },
        style={'width': '100%', 'height': '1000px'}, # Estilo del grafo
        minZoom=0.1, #rangos de zoom
        maxZoom=2,
        userZoomingEnabled=True, # Permite hacer zoom en el grafo
        userPanningEnabled=True, # Permite desplazar el grafo
        stylesheet=[
            {
                'selector': 'node', # Estilo de los nodos
                'style': { #forma de los nodos
                    'label': 'data(label)',
                    'font-size': '14px',
                    'width': 'label',
                    'height': 'label',
                    'padding': '6px',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'color': 'black',
                    'text-wrap': 'wrap',
                    'text-max-width': '80px',
                    'background-color': "#000000"
                }
            },
            {
                'selector': 'node[type="book"]', #nodos tipo libro
                'style': {
                    'shape': 'square',
                    'background-color': '#2ECC71'
                }
            },
            {
                'selector': 'node[type="person"]', #nodos tipo persona
                'style': {
                    'background-color': '#3498DB'
                }
            },
            {
                'selector': 'edge', # Estilo de las aristas
                'style': {
                    'line-color': '#B3B3B3',
                    'width': 0.5
                }
            },
            {
                'selector': '.highlighted', # Estilo de las aristas resaltadas para algoritmos como BFS, DFS o Djkstra
                'style': {
                    'line-color': 'red',
                    'target-arrow-color': 'red',
                    'width': 2
                }
            }
        ]
    ),


    html.Br(),
    html.Div(id='click-output'),
])

# ---------- Callbacks para la interacción con la app ---------- llamadas
@app.callback( 
    # Mostrar/ocultar contenedor de opciones
    Output('options-container', 'style'), 
    Input('toggle-options', 'n_clicks'),
)
def alternar_opciones(n_clicks):
    # Mostrar/ocultar contenedor de opciones
    # Si el número de clicks es impar, se muestra el contenedor, si es par, se oculta
    return {"display": "block" if n_clicks % 2 == 1 else "none"}

@app.callback(
    # Mostrar/ocultar contenedor de integrantes
    Output('members-container', 'style'),
    Input('btn-integrantes', 'n_clicks'),
)
def alternar_integrantes(n_clicks):
    return {"display": "block" if n_clicks % 2 == 1 else "none"}


# Actualizar el contenido con las estadísticas del grafo

@app.callback(
    # Mostrar/ocultar contenedor de estadísticas
    Output('stats-container', 'style'),
    Input('toggle-stats', 'n_clicks')
)
def alternar_estadisticas(n_clicks):
    return {"display": "block" if n_clicks % 2 == 1 else "none"}

@app.callback(
    # Actualizar el contenido de estadísticas del grafo
    Output('stats-output', 'children'),
    Input('toggle-stats', 'n_clicks'),
    prevent_initial_call=True # Evita que se ejecute al iniciar la app
)

def actualizar_salida_estadisticas(n_clicks):
    # Este metodo se llama cuando se hace click en el botón de estadísticas
    return mostrar_estadisticas_grafo(n_clicks)

def mostrar_estadisticas_grafo(_n_clicks):
    """Muestra las estadísticas del grafo haciendo uso de metodos predefinidos de NetworkX"""

    num_nodos = G.number_of_nodes()
    num_aristas = G.number_of_edges()
    # Calcula el grado máximo de los nodos
    # Si no hay nodos, el grado máximo es 0
    grado_maximo = max(dict(G.degree()).values()) if num_nodos else 0
    # Calcula el grado promedio de los nodos
    grado_promedio = round(sum(dict(G.degree()).values()) / num_nodos, 2) if num_nodos else 0
    # Calcula el número de componentes conexas
    num_componentes = nx.number_connected_components(G) if not G.is_directed() else "No hay componentes conexas"

    return html.Div([
        # Retorna las estadísticas del grafo en formato HTML
        html.P(f"Nodos: {num_nodos}"),
        html.P(f"Aristas: {num_aristas}"),
        html.P(f"Grado máximo: {grado_maximo}"),
        html.P(f"Grado promedio: {grado_promedio}"),
        html.P(f"Componentes conexas: {num_componentes}") 
])


@app.callback(
    # Actualizar los elementos del grafo en Cytoscape
    # Dependiendo de las acciones (agregar, eliminar, conectar nodos, encontrar caminos o regenerar el grafo)
        
        #Outputs
        Output('cytoscape-graph', 'elements'),
    [
        # Inputs
        Input('boton_añadir_nodo', 'n_clicks'),
        Input('boton_eliminar_nodo', 'n_clicks'),
        Input('boton_conectar_nodos', 'n_clicks'),
        Input('boton_camino', 'n_clicks'),
        Input('boton_regenerar', 'n_clicks')
    ],
    [
        # States
        State('cytoscape-graph', 'elements'),
        State('input-node', 'value'),
        State('entrada_borrar_nodo', 'value'),
        State('input-source-node', 'value'),
        State('input-target-node', 'value'),
        State('input-source', 'value'),        # Para buscar ruta
        State('input-target', 'value'),       
        State('input-node-type', 'value'),
        State('algoritmo', 'value')
    ],
    prevent_initial_call=True 
)
def actualizar_grafo(_agregar, _eliminar, _conectar, _camino, _reiniciar,elementos_actuales, nuevo_nodo, nodo_eliminar,fuente, destino, fuente_busqueda, destino_busqueda,nuevo_tipo, algoritmo):

    global G # Grafo global para mantener el estado entre callbacks
    trigger = ctx.triggered_id # Obtener el ID del botón que activó el callback

    if trigger == 'boton_añadir_nodo' and nuevo_nodo and nuevo_tipo:
        """Agregar un nuevo nodo al grafo
            Verifica si el nodo ya existe y si no, lo agrega"""
        if nuevo_nodo not in G:
            G.add_node(nuevo_nodo, type=nuevo_tipo) # Agrega el nodo al grafo
            # Agrega el nodo a los elementos de Cytoscape
            elementos_actuales.append({
                "data": {"id": nuevo_nodo, "label": nuevo_nodo, "type": nuevo_tipo}
            })
        return elementos_actuales #retorna los elementos actualizados

    elif trigger == 'boton_eliminar_nodo' and nodo_eliminar:
        """Eliminar un nodo del grafo"""
        if nodo_eliminar in G:
            G.remove_node(nodo_eliminar)
            elementos_actuales = [_ for _ in elementos_actuales
                                if _['data'].get('id') != nodo_eliminar and _['data'].get('source') != nodo_eliminar and _['data'].get('target') != nodo_eliminar]
        return elementos_actuales
    
    elif trigger == 'boton_conectar_nodos' and fuente and destino:
            """Conectar nodos existentes en el grafo
            Verifica que el nodo "fuente" sea una persona y el nodo destino un libro"""
            
            if (fuente in G and destino in G and
                G.nodes[fuente].get('type') == 'person' and G.nodes[destino].get('type') == 'book' and not G.has_edge(fuente, destino)): # Verifica que la arista no exista
                G.add_edge(fuente, destino)
                elementos_actuales.append({
                    "data": {"source": fuente, "target": destino}
                })
            return elementos_actuales

    elif trigger == 'boton_regenerar':
        """Regenera el grafo desde cero, eliminando todos los nodos y aristas"""
        G = construir_grafo()
        return nx_a_cytoscape(G)  #regenera todo el layout

    elif trigger == 'boton_camino' and fuente_busqueda in G and destino_busqueda in G: # Verifica que los nodos fuente y destino existan
        try:
            """Encuentra el camino más corto entre dos nodos utilizando metodos predefinidos de NetworkX"""
            if algoritmo == 'dijkstra':
                camino = nx.shortest_path(G, source=fuente_busqueda, target=destino_busqueda) # Utiliza Djkstra para encontrar el camino más corto

            elif algoritmo == 'bfs': #por BFS
                arbol_bfs = nx.bfs_tree(G, fuente_busqueda) 
                camino = nx.shortest_path(arbol_bfs, source=fuente_busqueda, target=destino_busqueda)

            elif algoritmo == 'dfs':
                arbol_dfs = nx.dfs_tree(G, fuente_busqueda)
                camino = nx.shortest_path(arbol_dfs, source=fuente_busqueda, target=destino_busqueda)
            else:
                camino = nx.shortest_path(G, source=fuente_busqueda, target=destino_busqueda) # Por defecto NetworkX utiliza Djkstra

            aristas_camino = set(zip(camino, camino[1:]))  # por BFS 

            #"zip" crea una lista de tuplas con las aristas del camino encontrado
            # Obtiene las aristas del camino encontrado
            nuevos_elementos = []
            for el in elementos_actuales:
                if 'source' in el['data']:
                    src, tgt = el['data']['source'], el['data']['target']
                    if (src, tgt) in aristas_camino or (tgt, src) in aristas_camino:
                        el['classes'] = 'highlighted'
                    else:
                        el.pop('classes', None)
                else:
                    el['classes'] = 'highlighted' if el['data']['id'] in camino else ''
                nuevos_elementos.append(el)
            return nuevos_elementos
        except:
            return elementos_actuales

    return elementos_actuales


@app.callback(
    Output('click-output', 'children'),
    Input('cytoscape-graph', 'tapNodeData')
)
def mostrar_click(datos_nodo):
    if datos_nodo:
        return f"Click en nodo: '{datos_nodo['label']}'"
    return "Haga click en un nodo para ver su información"


# ---------- Ejecutar la aplicación ----------
if __name__ == '__main__':
    app.run(debug=True)