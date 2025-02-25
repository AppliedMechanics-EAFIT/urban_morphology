import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
import os
import plotly.graph_objects as go
import numpy as np
from matplotlib.colors import Normalize, rgb2hex

def plot_centrality(graph, metric, place_name, cmap=plt.cm.viridis):
    """
    Calcula la métrica de centralidad especificada para un grafo y guarda el gráfico en un formato interactivo (HTML).
    Parameters:
        graph (networkx.Graph): Grafo de entrada.
        metric (str): Métrica de centralidad ('closeness', 'eigenvector', 'pagerank', etc.).
        place_name (str): Nombre de la ciudad o lugar para nombrar el archivo.
        cmap (matplotlib colormap, optional): Colormap para el gradiente. Por defecto: plt.cm.viridis.
    Returns:
        None
    """
    # Cálculo de la métrica de centralidad
    if metric == "degree":
        centrality = dict(graph.degree())
    elif metric == "eigenvector":
        # Convertir MultiDiGraph a Graph (no dirigido) consolidando aristas
        undirected_graph = nx.Graph()
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 1.0)  # Usa 'length' para calles
            if undirected_graph.has_edge(u, v):
                undirected_graph[u][v]['weight'] += weight
            else:
                undirected_graph.add_edge(u, v, weight=weight)
        
        # Verificar conectividad
        if not nx.is_connected(undirected_graph):
            # Tomar el componente conexo más grande (LCC)
            lcc = max(nx.connected_components(undirected_graph), key=len)
            undirected_graph = undirected_graph.subgraph(lcc)
        
        # Calcular eigenvector centrality con parámetros ajustados
        try:
            centrality = nx.eigenvector_centrality(
                undirected_graph, 
                max_iter=10_000,  # Aumentar iteraciones
                tol=1e-9,         # Tolerancia muy baja
                weight='weight'
            )
        except nx.PowerIterationFailedConvergence:
            # Si falla, relajar tolerancia
            centrality = nx.eigenvector_centrality(
                undirected_graph, 
                max_iter=10_000,
                tol=1e-3,
                weight='weight'
            )
        # Mapear al grafo original (no LCC tendrán 0)
        centrality = {node: centrality.get(node, 0.0) for node in graph.nodes()}
        # Normalizar para visualización
        max_c = max(centrality.values(), default=1)
        centrality = {node: val / max_c for node, val in centrality.items()}
    elif metric == "pagerank":
        centrality = nx.pagerank(graph)
    elif metric == "betweenness":
        centrality = nx.betweenness_centrality(graph)
    elif metric == "closeness":
        centrality = nx.closeness_centrality(graph)
    elif metric == "slc":
        centrality = calculate_slc(graph)
    elif metric == "lsc":
        centrality = calculate_lsc(graph, alpha=0.5)
    else:
        raise ValueError(f"Métrica desconocida: {metric}")




    # Extraer valores de centralidad
    node_colors = [centrality[node] for node in graph.nodes()]

      # Verificación crítica
    missing_nodes = [node for node in graph.nodes() if node not in centrality]
    if missing_nodes:
        raise ValueError(f"Centralidad no asignada a {len(missing_nodes)} nodos")
    
    # =============================================
    # 2. Optimización de rendimiento
    # =============================================
    # Simplificación de geometrías
    
    graph = ox.project_graph(graph)
    
    # Muestreo estratégico para grafos grandes
    if len(graph.nodes()) > 5000:
        nodes_to_keep = np.random.choice(list(graph.nodes()), 5000, replace=False)
        graph = graph.subgraph(nodes_to_keep).copy()
        print(f"¡Grafo muy grande! Muestreado a 5000 nodos para rendimiento")

    # Procesamiento eficiente de posiciones
    nodes_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    positions = np.array([(geom.y, geom.x) for geom in nodes_gdf.geometry])
    
    # Conversión vectorizada de colores
    node_colors = np.array([centrality[node] for node in graph.nodes()])
    if not hasattr(graph, 'simplified') or not graph.simplified:
        try:
            graph = ox.simplify_graph(graph.copy())
            graph = ox.project_graph(graph)
        except ox._errors.GraphSimplificationError:
            pass  # Si falla, trabajar con el grafo original

    # =============================================
    # 2. Validación de asignación de centralidad
    # =============================================
    # [Tu código original para calcular 'centrality'...]
    
    # Verificación mejorada
    missing_nodes = [node for node in graph.nodes() if node not in centrality]
    if missing_nodes:
        print(f"Advertencia: {len(missing_nodes)} nodos sin valores de centralidad")
        for node in missing_nodes:
            centrality[node] = 0.0  # Asignar valor por defecto

    # =============================================
    # 3. Optimización de rendimiento mejorada
    # =============================================
    # Procesamiento eficiente de nodos
    nodes = list(graph.nodes())
    positions = np.array([(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in nodes])
    
    # Manejo de grafos grandes con muestreo inteligente
    if len(nodes) > 5000:
        sample_indices = np.random.choice(len(nodes), 5000, replace=False)
        nodes = [nodes[i] for i in sample_indices]
        positions = positions[sample_indices]
        print(f"Muestreo aplicado: 5000 nodos de {len(nodes)} originales")

    # Conversión de colores optimizada
    node_colors = np.array([centrality[node] for node in nodes])
     
    # 1. Tamaño de nodos y bordes aumentado
    NODE_SIZE = 5  # Aumentado de 4 a 8
    EDGE_WIDTH = 1  # Aumentado de 0.3 a 1.5
    
    # 2. Configuración de la barra de colores
    norm = Normalize(vmin=node_colors.min(), vmax=node_colors.max())
    colorscale = [[norm(val), rgb2hex(cmap(norm(val)))] for val in np.linspace(0, 1, 256)]
    
    # 3. Traza de nodos con barra de colores
    node_trace = go.Scatter(
        x=positions[:, 1],
        y=positions[:, 0],
        mode='markers',
        marker=dict(
            size=NODE_SIZE,
            color=node_colors,  # Usar valores directos
            colorscale='Rainbow',  # Mapeo directo a colorscale
            colorbar=dict(
                title=dict(text=f'Centralidad {metric}', side='right'),
                thickness=25,
                x=1.05,
                len=0.8,
                ticksuffix='   '
            ),
            line=dict(width=0.6, color='rgba(0,0,0,0.8)'),  # Borde más visible
            opacity=0.85
        ),
        hoverinfo='text+name',
        text=[f"Nodo: {node}<br>Centralidad: {val:.4f}" for node, val in zip(nodes, node_colors)]
    )



    # Traza de bordes optimizada
    edge_x = []
    edge_y = []
    for u, v in graph.edges():
        if u in nodes and v in nodes:
            u_idx = nodes.index(u)
            v_idx = nodes.index(v)
            edge_x.extend([positions[u_idx][1], positions[v_idx][1], None])
            edge_y.extend([positions[u_idx][0], positions[v_idx][0], None])

    # 4. Traza de bordes más visibles
    edge_trace = go.Scattergl(
        x=edge_x,
        y=edge_y,
        line=dict(
            width=EDGE_WIDTH,
            color='rgba(50,50,50,0.5)'  # Color más oscuro y opaco
        ),
        hoverinfo='none',
        mode='lines'
    )
    # =============================================
    # 5. Mecanismo de reset robusto
    # =============================================
    x_range = [np.min(positions[:,1]), np.max(positions[:,1])]
    y_range = [np.min(positions[:,0]), np.max(positions[:,0])]

    layout = go.Layout(
        title=f'Centralidad {metric} - {place_name}',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(
            range=x_range,
            autorange=False,
            scaleanchor="y",
            constrain='domain',
            showgrid=False, 
            zeroline=False, 
            showticklabels=False
        ),
        yaxis=dict(
            range=y_range,
            autorange=False,
            showgrid=False, 
            zeroline=False, 
            showticklabels=False
        ),
        template='plotly_white',
        height=800,
    )
    # =============================================
    # 6. Exportación optimizada
    # =============================================
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    # Configuración de rendimiento
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
        'toImageButtonOptions': {
            'format': 'svg',
            'width': 1600,
            'height': 1600
        }
    }
    # Guardar HTML
    ruta_carpeta = f"Graphs_Cities/Graphs_for_{place_name}"
    os.makedirs(ruta_carpeta, exist_ok=True)
    nombre_archivo = f"Centrality_{metric}_{place_name}.html"
    ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
    
    fig.write_html(
        ruta_completa,
        config=config,
        include_plotlyjs='cdn',
        auto_open=False
    )
    layout.margin.r = 120  # Aumentado de 20 a 120

    print(f"Visualización optimizada guardada en: {ruta_completa}")

def calculate_eigenvector_centrality(graph):
    # Convertir MultiDiGraph a Graph (no dirigido) consolidando aristas
    undirected_graph = nx.Graph()
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)  # Usa 'weight' para las aristas
        if undirected_graph.has_edge(u, v):
            undirected_graph[u][v]['weight'] += weight
        else:
            undirected_graph.add_edge(u, v, weight=weight)
    
    # Verificar conectividad
    if not nx.is_connected(undirected_graph):
        # Tomar el componente conexo más grande (LCC)
        lcc = max(nx.connected_components(undirected_graph), key=len)
        undirected_graph = undirected_graph.subgraph(lcc)
    
    # Calcular eigenvector centrality con parámetros ajustados
    try:
        centrality = nx.eigenvector_centrality(
            undirected_graph, 
            max_iter=10_000,  # Aumentar iteraciones
            tol=1e-9,         # Tolerancia muy baja
            weight='weight'
        )
    except nx.PowerIterationFailedConvergence:
        # Si falla, relajar tolerancia
        centrality = nx.eigenvector_centrality(
            undirected_graph, 
            max_iter=10_000,
            tol=1e-3,
            weight='weight'
        )
    
    # Mapear al grafo original (nodos no en LCC tendrán 0)
    centrality = {node: centrality.get(node, 0.0) for node in graph.nodes()}
    
    # Normalizar para visualización
    max_c = max(centrality.values(), default=1)
    centrality = {node: val / max_c for node, val in centrality.items()}
    
    return centrality

def coefficient_centrality(graph, metric, ciudad):
    # Calcular todas las métricas
    metricas = {
        "degree": dict(graph.degree()),
        "betweenness": nx.betweenness_centrality(graph),
        "closeness": nx.closeness_centrality(graph),
        "pagerank": nx.pagerank(graph, alpha=0.85),
        "eigenvector": calculate_eigenvector_centrality(graph) , 
        "slc": calculate_slc(graph),
        "lsc": calculate_lsc(graph, alpha=0.5)
    }
    
    # Crear libro de Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Centralidades"
    
    # Configurar cabeceras
    columnas = [
        ("Grado", 2),
        ("Intermediación", 2),
        ("Cercanía", 2),
        ("PageRank", 2),
        ("Vector Propio", 2),
        ("Centralidad semilocal",2),
        ("Centraldad local ponderada", 2)
    ]
    
    # Escribir cabecera principal
    ws.merge_cells('A1:N1')
    ws['A1'] = ciudad
    ws['A1'].alignment = Alignment(horizontal='center')
    
    # Escribir cabeceras de métricas
    start_col = 1
    for nombre, span in columnas:
        ws.merge_cells(start_row=2, start_column=start_col, end_row=2, end_column=start_col+1)
        ws.cell(row=2, column=start_col, value=nombre)
        ws.cell(row=2, column=start_col).alignment = Alignment(horizontal='center')
        
        # Subcabeceras
        ws.cell(row=3, column=start_col, value="Nodo")
        ws.cell(row=3, column=start_col+1, value="Valor")
        start_col += 2
    
    # Escribir datos
    fila = 4
    for nodo in graph.nodes():
        col = 1
        for metrica in ['degree', 'betweenness', 'closeness', 'pagerank', 'eigenvector', "slc" , "lsc"]:
            try:
                valor = metricas[metrica].get(nodo, 0)
            except KeyError:
                valor = 0
                
            ws.cell(row=fila, column=col, value=nodo)
            ws.cell(row=fila, column=col+1, value=valor)
            col += 2
        fila += 1
    
    # Ajustar anchos de columna
    for idx in range(len(columnas)*2):
        ws.column_dimensions[get_column_letter(idx+1)].width = 15
    
    # Crear la carpeta si no existe
    ruta_carpeta = "Metrics_City"
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)
    
    # Guardar archivo en la ruta especificada
    nombre_archivo = f"Centralidades_{ciudad}.xlsx"
    ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
    wb.save(ruta_completa)
    
    return ruta_completa


def calculate_slc(graph):
    """
    Calcula la Centralidad Semilocal Clásica (SLC) para cada nodo en el grafo dirigido.
    
    Parámetros:
    - graph: Grafo dirigido de NetworkX.

    Retorna:
    - Un diccionario con la SLC de cada nodo.
    """
    slc = {}

    for node in graph.nodes():
        # Obtener vecinos a distancia 1
        neighbors_1 = set(graph.successors(node))  # Vecinos directos salientes

        # Calcular SLC usando la fórmula:
        slc_value = sum(len(set(graph.successors(w))) for w in neighbors_1)  # N2(w)
        slc[node] = slc_value

    # Normalizar valores entre 0 y 1
    max_slc = max(slc.values(), default=1)
    slc = {node: val / max_slc for node, val in slc.items()}

    return slc


def calculate_lsc(graph, alpha=0.5):
    """
    Calcula la Centralidad Semilocal Mejorada (LSC) para cada nodo en el grafo dirigido.
    
    Parámetros:
    - graph: Grafo dirigido de NetworkX.
    - alpha: Parámetro de ajuste (0 <= alpha <= 1).

    Retorna:
    - Un diccionario con la LSC de cada nodo.
    """
    # Obtener la centralidad SLC primero
    slc = calculate_slc(graph)
    lsc = {}

    for node in graph.nodes():
        # Obtener vecinos a distancia 1
        neighbors_1 = set(graph.successors(node))

        # Obtener vecinos a distancia 2
        neighbors_2 = set()
        for neighbor in neighbors_1:
            neighbors_2.update(set(graph.successors(neighbor)))

        # Calcular LSC usando la fórmula:
        lsc_value = sum(alpha * len(set(graph.successors(u))) + 
                        (1 - alpha) * sum(slc.get(w, 0) for w in set(graph.successors(u)))
                        for u in neighbors_1)

        lsc[node] = lsc_value

    # Normalizar valores entre 0 y 1
    max_lsc = max(lsc.values(), default=1)
    lsc = {node: val / max_lsc for node, val in lsc.items()}

    return lsc

