import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, MultiPoint, Polygon
from shapely.ops import split
import itertools
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='osmnx')

# O para suprimir todas las FutureWarnings:
warnings.filterwarnings('ignore', category=FutureWarning)

# --- GEOJSON SAVING FUNCTION ---

def save_graph_to_geojson(G, graph_type, base_folder, counters):
    """Saves a NetworkX graph with positional data to a GeoJSON file."""
    if 'pos' not in G.graph:
        print(f"Error: The graph of type '{graph_type}' has no position data and cannot be saved.")
        return
    
    # Increment the counter for the specific graph type
    counters[graph_type] += 1
    file_number = counters[graph_type]
    filename = f"{graph_type}_{file_number}.geojson"
    full_path = os.path.join(base_folder, filename)
    
    pos = G.graph['pos']
    features = []
    
    # Create a LineString feature for each edge in the graph
    for u, v in G.edges():
        coord_u = [float(c) for c in pos[u]]
        coord_v = [float(c) for c in pos[v]]
        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [coord_u, coord_v]},
            "properties": {}
        }
        features.append(feature)
        
    geojson_structure = {"type": "FeatureCollection", "features": features}
    
    # Create the output directory if it doesn't exist
    os.makedirs(base_folder, exist_ok=True)
    
    with open(full_path, 'w') as f:
        json.dump(geojson_structure, f, indent=2)
        
    print(f"Graph successfully saved to: '{full_path}'")

def normalize_graph_size(G, target_width_meters):
    """
    Re-escala y traslada un grafo para que encaje en un bounding box
    con un ancho específico en metros, manteniendo la proporción.
    """
    pos = G.graph.get('pos')
    if not pos or len(pos) < 2:
        # No se puede normalizar un grafo vacío o con un solo punto
        return G

    # Convertir las posiciones a un array de numpy para cálculos fáciles
    coords = np.array(list(pos.values()))

    # Encontrar el bounding box actual
    min_coords = coords.min(axis=0)  # [min_x, min_y]
    max_coords = coords.max(axis=0)  # [max_x, max_y]
    
    # Calcular las dimensiones actuales
    current_dims = max_coords - min_coords
    current_width = current_dims[0]
    current_height = current_dims[1]

    # Evitar división por cero si el grafo es una línea vertical o un punto
    if current_width == 0 and current_height == 0:
        return G # Es un solo punto
    
    # Calcular el factor de escala para mantener la proporción
    if current_width > 0:
        scale_factor = target_width_meters / current_width
    else: # Si el grafo es una línea perfectamente vertical
        # Usamos la altura como referencia, asumiendo que el target_width también aplica para la altura
        scale_factor = target_width_meters / current_height

    # Aplicar la normalización a cada nodo
    new_pos = {}
    for node, (x, y) in pos.items():
        # 1. Trasladar a (0,0) restando el mínimo
        # 2. Escalar con el factor calculado
        new_x = (x - min_coords[0]) * scale_factor
        new_y = (y - min_coords[1]) * scale_factor
        new_pos[node] = (new_x, new_y)
    
    # Actualizar el diccionario de posiciones del grafo
    G.graph['pos'] = new_pos
    
    print(f"Graph '{G.graph.get('type', 'unknown')}' normalizado a un ancho de ~{target_width_meters:.0f} metros.")
    return G

# --- GRAPH GENERATION FUNCTIONS ---

# --- CATEGORY 1: GRID-BASED PATTERNS ---

def generate_perfect_grid(target_nodes):
    """Generates a perfect square grid graph."""
    side = int(np.sqrt(target_nodes))
    G = nx.grid_2d_graph(side, side)
    pos = {(x, y): (x, y) for x, y in G.nodes()}
    G.graph['pos'] = pos
    G.graph['type'] = 'perfect_grid'
    return G

def generate_irregular_grid(target_nodes, jitter=0.25):
    """Generates a grid graph with random positional jitter for each node."""
    side = int(np.sqrt(target_nodes))
    G = nx.grid_2d_graph(side, side)
    pos = {}
    for x, y in G.nodes():
        pos[(x, y)] = (x + random.uniform(-jitter, jitter), y + random.uniform(-jitter, jitter))
    G.graph['pos'] = pos
    G.graph['type'] = 'irregular_grid'
    return G

# --- CROSSED GRID SECTION ---

def generate_crossed_grid(target_nodes, config=None):
    """
    Main function to generate a crossed grid.
    It creates a configuration of overlapping grids based on target_nodes if none is provided.
    """
    if config is None:
        # Create a default configuration based on target_nodes
        # Split nodes between two grids. The first grid is slightly larger.
        side1 = int(np.sqrt(target_nodes / 1.8)) 
        side2 = int(np.sqrt(target_nodes / 2.5)) 

        config = [
            {
                'n_rows': side1, 
                'n_cols': side1, 
                'angle_deg': random.uniform(-10, 10), 
                'center': (side1/2, side1/2), 
                'spacing': 1
            },
            {
                'n_rows': side2, 
                'n_cols': side2, 
                'angle_deg': random.uniform(35, 55), 
                'center': (side1/2, side1/2), # Use the same center to ensure they cross
                'spacing': 1.2
            },
        ]

    # Generate the graph using the internal helper function
    G = _generate_clipped_grid_from_config(config)

    # Adapt the resulting graph to the standard format
    # Get the positions that were stored as node attributes
    pos = nx.get_node_attributes(G, 'pos')
    if not pos: # If the graph is empty, prevent an error
        pos = {}

    G.graph['pos'] = pos
    G.graph['type'] = 'crossed_grid'
    
    return G

# Internal helper functions for the crossed grid
def _create_grid_geometries(n_rows, n_cols, center=(0, 0), angle_deg=0, spacing=1.0):
    """Creates shapely LineStrings and a bounding Polygon for a single grid."""
    lines = []
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    width = (n_cols - 1) * spacing
    height = (n_rows - 1) * spacing
    x_coords = np.linspace(-width / 2, width / 2, n_cols)
    y_coords = np.linspace(-height / 2, height / 2, n_rows)
    
    for x in x_coords:
        p1 = np.dot(rotation_matrix, np.array([x, -height / 2])) + center
        p2 = np.dot(rotation_matrix, np.array([x, height / 2])) + center
        lines.append(LineString([p1, p2]))
    for y in y_coords:
        p1 = np.dot(rotation_matrix, np.array([-width / 2, y])) + center
        p2 = np.dot(rotation_matrix, np.array([width / 2, y])) + center
        lines.append(LineString([p1, p2]))
        
    corners = [
        np.dot(rotation_matrix, np.array([-width / 2, -height / 2])) + center,
        np.dot(rotation_matrix, np.array([width / 2, -height / 2])) + center,
        np.dot(rotation_matrix, np.array([width / 2, height / 2])) + center,
        np.dot(rotation_matrix, np.array([-width / 2, height / 2])) + center,
    ]
    return lines, Polygon(corners)

def _build_graph_from_lines(lines_list):
    """Builds a NetworkX graph from a list of shapely LineStrings."""
    if not lines_list: return nx.Graph()
    
    junctions = set()
    # Find all intersections between pairs of lines
    for line1, line2 in itertools.combinations(lines_list, 2):
        if line1.intersects(line2):
            intersection = line1.intersection(line2)
            if isinstance(intersection, Point):
                junctions.add(intersection.coords[0])
            elif hasattr(intersection, 'geoms'): # Handle MultiPoint intersections
                for pt in intersection.geoms:
                    if isinstance(pt, Point): junctions.add(pt.coords[0])

    # Add start and end points of all lines as junctions
    for line in lines_list:
        if len(line.coords) > 0:
            junctions.add(line.coords[0])
            junctions.add(line.coords[-1])
            
    if not junctions: return nx.Graph()
    
    # Split all lines at the identified junction points
    junction_points = MultiPoint([Point(j) for j in junctions])
    all_segments = []
    for line in lines_list:
        split_result = split(line, junction_points)
        if hasattr(split_result, 'geoms'):
            all_segments.extend(list(split_result.geoms))
        else:
            all_segments.append(split_result)
            
    # Build the graph from the resulting segments
    G = nx.Graph()
    for segment in all_segments:
        if isinstance(segment, LineString) and not segment.is_empty and len(segment.coords) >= 2:
            p1 = tuple(np.round(segment.coords[0], 5))
            p2 = tuple(np.round(segment.coords[-1], 5))
            G.add_node(p1, pos=p1)
            G.add_node(p2, pos=p2)
            if p1 != p2:
                G.add_edge(p1, p2)
    return G

def _generate_clipped_grid_from_config(grids_config):
    """Generates multiple grids and clips them against each other."""
    if not grids_config: return nx.Graph()
    
    all_grid_data = [_create_grid_geometries(**c) for c in grids_config]
    final_lines = []
    
    # For each grid, clip its lines against the polygons of subsequent grids
    for i, (lines_i, poly_i) in enumerate(all_grid_data):
        lines_to_process = lines_i
        for j in range(i + 1, len(all_grid_data)):
            poly_top = all_grid_data[j][1]
            temp_clipped_lines = []
            for line in lines_to_process:
                clipped_geom = line.difference(poly_top)
                if not clipped_geom.is_empty:
                    if clipped_geom.geom_type == 'MultiLineString':
                        temp_clipped_lines.extend(list(clipped_geom.geoms))
                    elif clipped_geom.geom_type == 'LineString':
                        temp_clipped_lines.append(clipped_geom)
            lines_to_process = temp_clipped_lines
        final_lines.extend(lines_to_process)
        
    return _build_graph_from_lines(final_lines)

# --- CATEGORY 2: DENDRITIC WITH CUL-DE-SACS ---

def generate_cul_de_sac(target_nodes, config=None):
    """Generates a dendritic network with cul-de-sacs and hammerhead turns."""
    main_road_nodes = max(5, int(target_nodes / 6.5))
    if config is None:
        config = {'main_road_length': 4.0, 'main_angle_variance': 15, 'branch_prob': 0.6, 'branch_max_depth_range': (2, 5), 'branch_length': 3.0, 'sub_branch_prob': 0.3, 'sub_branch_depth_range': (2, 4), 'angle_variance': 25, 'length_decay': 0.9}
    
    G = nx.Graph()
    G.add_node(0)
    pos = {0: (0, 0)}
    last_pos, last_angle, next_node_id = np.array([0.0, 0.0]), 0.0, 1
    
    # Create the main road
    for i in range(1, main_road_nodes):
        angle = last_angle + random.uniform(-config['main_angle_variance'], config['main_angle_variance'])
        length = config['main_road_length'] * random.uniform(0.8, 1.2)
        last_pos += np.array([length * np.cos(np.radians(angle)), length * np.sin(np.radians(angle))])
        pos[next_node_id] = tuple(last_pos)
        G.add_node(next_node_id)
        G.add_edge(next_node_id - 1, next_node_id)
        last_angle = angle
        next_node_id += 1
        
    # Create branches off the main road
    side = 1
    for i in range(1, main_road_nodes - 1):
        if random.random() < config['branch_prob']:
            vec = np.array(pos[i]) - np.array(pos[i-1])
            p_angle = np.degrees(np.arctan2(vec[1], vec[0]))
            b_angle = p_angle + 90 * side + random.uniform(-20, 20)
            b_depth = random.randint(*config['branch_max_depth_range'])
            _create_hierarchical_branch_hammerhead(G, pos, i, 1, b_depth, b_angle, config['branch_length'], config)
            side *= -1 # Alternate sides
            
    G.graph['pos'] = pos
    G.graph['type'] = 'cul_de_sac'
    return G

def _create_hierarchical_branch_hammerhead(G, pos, parent_node, depth, max_depth, angle_deg, length, config):
    """Recursively creates branches, ending in a 'hammerhead' turn-around."""
    # Base case: at max depth, create the hammerhead
    if depth >= max_depth:
        parent_pos = np.array(pos[parent_node])
        base_angle_rad = np.radians(angle_deg)
        stem_length = random.uniform(0.8, 1.2)
        stem_pos = parent_pos + np.array([stem_length * np.cos(base_angle_rad), stem_length * np.sin(base_angle_rad)])
        stem_node = f"{parent_node}_h_stem_{random.randint(1000,9999)}"
        pos[stem_node] = tuple(stem_pos)
        G.add_node(stem_node)
        G.add_edge(parent_node, stem_node)
        
        # Create the two prongs of the hammerhead
        for i, side_angle in enumerate([-120, 120]):
            hammer_length = random.uniform(1.0, 1.5)
            angle_rad = np.radians(angle_deg + side_angle + random.uniform(-10, 10))
            hammer_pos = stem_pos + np.array([hammer_length * np.cos(angle_rad), hammer_length * np.sin(angle_rad)])
            hammer_node = f"{parent_node}_h_{i}_{random.randint(1000,9999)}"
            pos[hammer_node] = tuple(hammer_pos)
            G.add_node(hammer_node)
            G.add_edge(stem_node, hammer_node)
        return
        
    # Recursive step: grow the branch
    node_id = f"{parent_node}_d{depth}_{random.randint(1000,9999)}"
    angle_rad = np.radians(angle_deg)
    new_pos = np.array(pos[parent_node]) + np.array([length * np.cos(angle_rad), length * np.sin(angle_rad)])
    pos[node_id] = tuple(new_pos)
    G.add_node(node_id)
    G.add_edge(parent_node, node_id)
    
    new_angle = angle_deg + random.uniform(-config['angle_variance'], config['angle_variance'])
    new_length = length * config['length_decay']
    _create_hierarchical_branch_hammerhead(G, pos, node_id, depth + 1, max_depth, new_angle, new_length, config)
    
    # Chance to create a sub-branch
    if random.random() < config['sub_branch_prob'] and depth < max_depth - 1:
        sub_angle = angle_deg + 90 * random.choice([-1, 1]) + random.uniform(-20, 20)
        sub_depth = random.randint(*config['sub_branch_depth_range'])
        _create_hierarchical_branch_hammerhead(G, pos, node_id, 1, sub_depth, sub_angle, length * 0.8, config)

# --- CATEGORY 3: ORGANIC GROWTH ---

def generate_organic_agent_based(target_nodes, step_length=2.0, min_dist=1.5, branch_prob=0.05, angle_variation=0.2):
    """Generates an 'organic' network using an agent-based growth model."""
    num_iterations = int(target_nodes * 1.1)
    G = nx.Graph()
    pos = {0: (0, 0), 1: (step_length, 0)}
    G.add_edge(0, 1)
    active_growth_tips = [(1, 0.0)] # (node_id, angle)
    next_node_id = 2
    
    for _ in range(num_iterations):
        if not active_growth_tips: break
        
        # Pick a random growth tip
        tip_idx = random.randint(0, len(active_growth_tips) - 1)
        parent_node, parent_angle = active_growth_tips[tip_idx]
        
        # Propose a new node position
        angle = parent_angle + random.uniform(-np.pi * angle_variation, np.pi * angle_variation)
        new_pos = np.array(pos[parent_node]) + np.array([step_length * np.cos(angle), step_length * np.sin(angle)])
        
        all_nodes_pos = np.array(list(pos.values()))
        kdtree = KDTree(all_nodes_pos)
        
        # Check if the new position is too close to existing nodes
        nearby_indices = kdtree.query_ball_point(new_pos, r=min_dist * 1.2)
        if len(nearby_indices) > 0:
            # If too close, connect to the nearest existing node and terminate this branch
            closest_node_idx = kdtree.query(new_pos)[1]
            closest_node_id = list(pos.keys())[closest_node_idx]
            if closest_node_id != parent_node:
                G.add_edge(parent_node, closest_node_id)
            active_growth_tips.pop(tip_idx) # Stop this growth tip
            continue
            
        # If position is valid, add the new node and edge
        pos[next_node_id] = tuple(new_pos)
        G.add_node(next_node_id)
        G.add_edge(parent_node, next_node_id)
        
        # Update the growth tip
        active_growth_tips[tip_idx] = (next_node_id, angle)
        
        # Randomly create a new branch
        if random.random() < branch_prob:
            branch_angle = angle + np.pi / 2 * random.choice([-1, 1])
            active_growth_tips.append((next_node_id, branch_angle))
            
        next_node_id += 1
        
    G.graph['pos'] = pos
    G.graph['type'] = 'organic'
    return G

# --- VISUALIZATION FUNCTION ---

def plot_graph(G, ax, title):
    """Plots a graph on a given matplotlib axis."""
    pos = G.graph.get('pos')
    if not pos:
        print(f"Warning: No position data found for graph '{title}'.")
        # If there are no nodes, just set the title and background
        if len(G) == 0:
            ax.set_title(title, fontsize=14, color='black', pad=15)
            ax.set_facecolor('black')
            ax.set_xticks([]); ax.set_yticks([])
            return
        # If nodes exist but no positions, calculate a layout
        pos = nx.spring_layout(G)
        G.graph['pos'] = pos
        
    ax.set_title(title, fontsize=14, color='black', pad=15)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=0)
    edge_color = 'black'
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, edge_color=edge_color)
    
    ax.set_facecolor('white')
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis('equal')




# --- MAIN SCRIPT ---

# if __name__ == '__main__':
#     # --- PARÁMETROS GLOBALES Y CONFIGURACIÓN ---
#     GLOBAL_PARAMETERS = {
#         'APPROX_NODE_COUNT': 400,
#         'IRREGULAR_GRID_JITTER': 0.2
#     }
#     OUTPUT_FOLDER = "Synthetic_Street_Networks"

#     FILE_COUNTERS = {
#         'perfect_grid': 0,
#         'irregular_grid': 0,
#         'crossed_grid': 0,
#         'cul_de_sac': 0,
#         'organic': 0,
#         'organic2': 0
#     }

#     target_node_count = GLOBAL_PARAMETERS['APPROX_NODE_COUNT']
#     print(f"--- Starting graph generation with ~{target_node_count} nodes per graph ---")

#     fig, axes = plt.subplots(2, 3, figsize=(16, 8))
#     fig.patch.set_facecolor('white')
#     fig.suptitle(f'Synthetic Street Network Generator (N ≈ {target_node_count})',
#                  color='black', fontsize=24)
#     axes_flat = axes.flatten()

#     print("\n1. Generating Perfect Grid...")
#     g_perfect_grid = generate_perfect_grid(target_nodes=target_node_count)
#     plot_graph(g_perfect_grid, axes_flat[0],
#                f"Perfect Grid\n(Nodes: {len(g_perfect_grid.nodes())})")
#     save_graph_to_geojson(g_perfect_grid, 'perfect_grid', OUTPUT_FOLDER, FILE_COUNTERS)

#     print("2. Generating Irregular Grid...")
#     jitter_value = GLOBAL_PARAMETERS['IRREGULAR_GRID_JITTER']
#     g_irregular = generate_irregular_grid(target_nodes=target_node_count, jitter=jitter_value)
#     plot_graph(g_irregular, axes_flat[1],
#                f"Irregular Grid (Jitter: {jitter_value})\n(Nodes: {len(g_irregular.nodes())})")
#     save_graph_to_geojson(g_irregular, 'irregular_grid', OUTPUT_FOLDER, FILE_COUNTERS)

#     print("3. Generating Crossed Grid...")
#     config1 = [
#         {
#             'n_rows': 10,
#             'n_cols': 10,
#             'angle_deg': 0,
#             'center': (6, 6),
#             'spacing': 0.7
#         },
#         {
#             'n_rows': 8,
#             'n_cols': 8,
#             'angle_deg': 0,
#             'center': (14, 14),
#             'spacing': 1.0
#         },
#         {
#             'n_rows': 10,
#             'n_cols': 12,
#             'angle_deg': 45,
#             'center': (10, 10),
#             'spacing': 0.8
#         }
#     ]
#     g_crossed = generate_crossed_grid(target_nodes=target_node_count, config=config1)
#     plot_graph(g_crossed, axes_flat[2],
#                f"Crossed Grid\n(Nodes: {len(g_crossed.nodes())})")
#     save_graph_to_geojson(g_crossed, 'crossed_grid', OUTPUT_FOLDER, FILE_COUNTERS)

#     print("4. Generating Cul-de-Sac Network...")
#     g_culdesac = generate_cul_de_sac(target_nodes=target_node_count)
#     plot_graph(g_culdesac, axes_flat[3],
#                f"Cul-de-Sac Network\n(Nodes: {len(g_culdesac.nodes())})")
#     save_graph_to_geojson(g_culdesac, 'cul_de_sac', OUTPUT_FOLDER, FILE_COUNTERS)

#     print("5. Generating Organic Network...")
#     g_organic = generate_organic_agent_based(target_nodes=target_node_count, branch_prob=0.08)
#     plot_graph(g_organic, axes_flat[4],
#                f'Organic Growth\n(Nodes: {len(g_organic.nodes())})')
#     save_graph_to_geojson(g_organic, 'organic', OUTPUT_FOLDER, FILE_COUNTERS)

#     print("6. Generating Organic Network v2...")
#     g_organic2 = generate_organic_agent_based(target_nodes=target_node_count, branch_prob=0.1)
#     plot_graph(g_organic2, axes_flat[5],
#                f'Organic Growth_v2\n(Nodes: {len(g_organic2.nodes())})')
#     save_graph_to_geojson(g_organic2, 'organic2', OUTPUT_FOLDER, FILE_COUNTERS)

#     print(f"\n--- Process complete. Check the '{OUTPUT_FOLDER}' folder. ---")

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     fig.savefig("Synthetic_Street_Networks/output_plot.png", dpi=320, bbox_inches='tight')
#     plt.show()









    

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, LineString
import json
import os

def geojson_to_networkx_graph(geojson_data):
    """
    Convierte un GeoJSON con LineStrings a un grafo NetworkX compatible con OSMnx
    """
    # Crear GeoDataFrame con CRS explícito
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'], crs='EPSG:4326')
    
    # Crear grafo dirigido
    G = nx.MultiDiGraph()
    
    # Diccionario para almacenar nodos únicos con tolerancia
    nodes = {}
    node_id = 0
    tolerance = 1e-10  # Tolerancia para coordenadas muy cercanas
    
    # Función para obtener o crear ID de nodo con tolerancia
    def get_node_id(coord):
        # Redondear coordenadas para evitar problemas de precisión
        rounded_coord = (round(coord[0], 10), round(coord[1], 10))
        
        # Buscar nodo existente dentro de la tolerancia
        for existing_coord, existing_id in nodes.items():
            if (abs(existing_coord[0] - rounded_coord[0]) < tolerance and 
                abs(existing_coord[1] - rounded_coord[1]) < tolerance):
                return existing_id
        
        # Si no existe, crear nuevo nodo
        nonlocal node_id
        current_id = node_id
        nodes[rounded_coord] = current_id
        
        # Agregar nodo al grafo con atributos requeridos por OSMnx
        G.add_node(current_id, 
                  x=coord[0], 
                  y=coord[1],
                  osmid=current_id,
                  street_count=0)  # Agregar atributo requerido por OSMnx
        
        node_id += 1
        return current_id
    
    # Procesar cada LineString
    edge_id = 0
    
    for idx, row in gdf.iterrows():
        line = row.geometry
        coords = list(line.coords)
        
        # Para cada segmento en la línea
        for i in range(len(coords) - 1):
            start_coord = coords[i]
            end_coord = coords[i + 1]
            
            # Obtener IDs de nodos
            start_node = get_node_id(start_coord)
            end_node = get_node_id(end_coord)
            
            # Solo agregar arista si los nodos son diferentes
            if start_node != end_node:
                # Calcular longitud del segmento
                segment_line = LineString([start_coord, end_coord])
                length = segment_line.length
                
                # Agregar arista con atributos requeridos por OSMnx
                G.add_edge(start_node, end_node, 
                          key=edge_id,
                          osmid=edge_id,
                          length=length,
                          geometry=segment_line,
                          highway='unclassified',  # Tipo de vía requerido
                          oneway=False,           # Dirección
                          name=f'edge_{edge_id}') # Nombre de la calle
                
                edge_id += 1
    
    # Calcular street_count para cada nodo
    for node in G.nodes():
        G.nodes[node]['street_count'] = len(list(G.edges(node)))
    
    # Agregar atributos faltantes al grafo
    G.graph['crs'] = 'EPSG:4326'
    G.graph['name'] = 'Synthetic Network'
    
    return G

def networkx_to_geojson(G):
    """
    Convierte un grafo NetworkX de vuelta a formato GeoJSON
    """
    features = []
    
    # Convertir aristas a LineStrings
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' in data:
            geometry = data['geometry']
        else:
            # Si no hay geometría, crear LineString desde coordenadas de nodos
            start_point = [G.nodes[u]['x'], G.nodes[u]['y']]
            end_point = [G.nodes[v]['x'], G.nodes[v]['y']]
            geometry = LineString([start_point, end_point])
        
        # Crear feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": list(geometry.coords)
            },
            "properties": {
                "osmid": data.get('osmid', key),
                "length": data.get('length', geometry.length),
                "u": u,
                "v": v,
                "key": key
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

def process_geojson_file(input_filepath):
    """
    Procesa un archivo GeoJSON individual
    """
    print(f"Procesando: {input_filepath}")
    
    try:
        # Cargar GeoJSON original con CRS explícito
        gdf = gpd.read_file(input_filepath)
        
        # Asegurar que tiene CRS
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        
        # Convertir a formato de diccionario
        geojson_dict = {
            "type": "FeatureCollection",
            "features": json.loads(gdf.to_json())['features']
        }
        
        # Convertir a grafo NetworkX
        G = geojson_to_networkx_graph(geojson_dict)
        
        # Verificar que el grafo se creó correctamente
        print(f"  - Nodos: {G.number_of_nodes()}")
        print(f"  - Aristas: {G.number_of_edges()}")
        
        # Calcular estadísticas básicas como prueba
        try:
            stats = ox.basic_stats(G)
            print(f"  - Longitud total: {stats.get('street_length_total', 'N/A'):.4f}")
            print(f"  - Circuidad promedio: {stats.get('circuity_avg', 'N/A'):.4f}")
        except Exception as e:
            print(f"  - Advertencia: No se pudieron calcular estadísticas: {e}")
        
        # Convertir de vuelta a GeoJSON con CRS
        new_geojson = networkx_to_geojson(G)
        
        # Crear GeoDataFrame temporal para asegurar formato correcto
        temp_gdf = gpd.GeoDataFrame.from_features(new_geojson['features'], crs='EPSG:4326')
        
        # Guardar sobrescribiendo el archivo original
        temp_gdf.to_file(input_filepath, driver='GeoJSON')
        
        print(f"  ✓ Archivo actualizado exitosamente")
        return True
        
    except Exception as e:
        print(f"  ✗ Error procesando {input_filepath}: {e}")
        return False

def batch_process_geojson_files():
    """
    Procesa todos los archivos GeoJSON especificados
    """
    # Diccionario de archivos a procesar
    files_to_process = {
        "Synthetic_Street_Networks/perfect_grid_1.geojson": "perfect_grid_1",
        "Synthetic_Street_Networks/crossed_grid_1.geojson": "crossed_grid_1",
        "Synthetic_Street_Networks/cul_de_sac_1.geojson": "cul_de_sac_1",
        "Synthetic_Street_Networks/irregular_grid_1.geojson": "irregular_grid_1",
        "Synthetic_Street_Networks/organic_1.geojson": "organic_1",
        "Synthetic_Street_Networks/organic2_1.geojson": "organic2_1"
    }
    
    print("=== CONVERSIÓN POR LOTES DE GEOJSON ===")
    print(f"Procesando {len(files_to_process)} archivos...\n")
    
    successful = 0
    failed = 0
    
    for filepath, name in files_to_process.items():
        # Verificar que el archivo existe
        if not os.path.exists(filepath):
            print(f"⚠️  Archivo no encontrado: {filepath}")
            failed += 1
            continue
        
        # Procesar archivo
        if process_geojson_file(filepath):
            successful += 1
        else:
            failed += 1
        
        print()  # Línea en blanco para separar
    
    # Resumen final
    print("=== RESUMEN ===")
    print(f"✓ Archivos procesados exitosamente: {successful}")
    print(f"✗ Archivos con errores: {failed}")
    print(f"📁 Total archivos: {len(files_to_process)}")
    
    if successful > 0:
        print(f"\n🎉 Los archivos han sido convertidos y sobrescritos.")
        print("Ahora puedes usar ox.basic_stats() directamente con estos archivos:")
        print("\n# Ejemplo de uso:")
        print("import osmnx as ox")
        print("import geopandas as gpd")
        print("import json")
        print("from tu_script import geojson_to_networkx_graph")
        print("")
        print("# Cargar y analizar")
        print("with open('Synthetic_Street_Networks/perfect_grid_1.geojson', 'r') as f:")
        print("    geojson_data = json.load(f)")
        print("G = geojson_to_networkx_graph(geojson_data)")
        print("stats = ox.basic_stats(G)")

# Función auxiliar para uso posterior
def load_processed_geojson_as_graph(filepath):
    """
    Carga un GeoJSON procesado y lo convierte a grafo NetworkX
    Versión robusta con manejo de errores CRS
    """
    try:
        # Cargar con geopandas para manejar CRS correctamente
        gdf = gpd.read_file(filepath)
        
        # Verificar y asignar CRS si es necesario
        if gdf.crs is None:
            print(f"Asignando CRS EPSG:4326 a {filepath}")
            gdf = gdf.set_crs('EPSG:4326')
        
        # Convertir a diccionario manualmente para evitar problemas de CRS
        features = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Crear feature manualmente
            if geom.geom_type == "LineString":
                coordinates = list(geom.coords)
            else:
                coordinates = []
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {k: v for k, v in row.items() if k != 'geometry'}
            }
            features.append(feature)
        
        geojson_dict = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return geojson_to_networkx_graph(geojson_dict)
        
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        # Intentar carga alternativa con JSON puro
        try:
            with open(filepath, 'r') as f:
                geojson_data = json.load(f)
            return geojson_to_networkx_graph(geojson_data)
        except Exception as e2:
            print(f"Error también con carga JSON: {e2}")
            raise e

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, LineString
import json
import os

def geojson_to_networkx_graph(geojson_data):
    """
    Convierte un GeoJSON con LineStrings a un grafo NetworkX compatible con OSMnx
    """
    # Crear GeoDataFrame con CRS explícito
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'], crs='EPSG:4326')
    
    # Crear grafo dirigido
    G = nx.MultiDiGraph()
    
    # Diccionario para almacenar nodos únicos con tolerancia
    nodes = {}
    node_id = 0
    tolerance = 1e-10  # Tolerancia para coordenadas muy cercanas
    
    # Función para obtener o crear ID de nodo con tolerancia
    def get_node_id(coord):
        # Redondear coordenadas para evitar problemas de precisión
        rounded_coord = (round(coord[0], 10), round(coord[1], 10))
        
        # Buscar nodo existente dentro de la tolerancia
        for existing_coord, existing_id in nodes.items():
            if (abs(existing_coord[0] - rounded_coord[0]) < tolerance and 
                abs(existing_coord[1] - rounded_coord[1]) < tolerance):
                return existing_id
        
        # Si no existe, crear nuevo nodo
        nonlocal node_id
        current_id = node_id
        nodes[rounded_coord] = current_id
        
        # Agregar nodo al grafo con atributos requeridos por OSMnx
        G.add_node(current_id, 
                  x=coord[0], 
                  y=coord[1],
                  osmid=current_id,
                  street_count=0)  # Agregar atributo requerido por OSMnx
        
        node_id += 1
        return current_id
    
    # Procesar cada LineString
    edge_id = 0
    
    for idx, row in gdf.iterrows():
        line = row.geometry
        coords = list(line.coords)
        
        # Para cada segmento en la línea
        for i in range(len(coords) - 1):
            start_coord = coords[i]
            end_coord = coords[i + 1]
            
            # Obtener IDs de nodos
            start_node = get_node_id(start_coord)
            end_node = get_node_id(end_coord)
            
            # Solo agregar arista si los nodos son diferentes
            if start_node != end_node:
                # Calcular longitud del segmento
                segment_line = LineString([start_coord, end_coord])
                length = segment_line.length
                
                # Agregar arista con atributos requeridos por OSMnx
                G.add_edge(start_node, end_node, 
                          key=edge_id,
                          osmid=edge_id,
                          length=length,
                          geometry=segment_line,
                          highway='unclassified',  # Tipo de vía requerido
                          oneway=False,           # Dirección
                          name=f'edge_{edge_id}') # Nombre de la calle
                
                edge_id += 1
    
    # Calcular street_count para cada nodo
    for node in G.nodes():
        G.nodes[node]['street_count'] = len(list(G.edges(node)))
    
    # Agregar atributos faltantes al grafo
    G.graph['crs'] = 'EPSG:4326'
    G.graph['name'] = 'Synthetic Network'
    
    return G

def networkx_to_geojson(G):
    """
    Convierte un grafo NetworkX de vuelta a formato GeoJSON
    """
    features = []
    
    # Convertir aristas a LineStrings
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' in data:
            geometry = data['geometry']
        else:
            # Si no hay geometría, crear LineString desde coordenadas de nodos
            start_point = [G.nodes[u]['x'], G.nodes[u]['y']]
            end_point = [G.nodes[v]['x'], G.nodes[v]['y']]
            geometry = LineString([start_point, end_point])
        
        # Crear feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": list(geometry.coords)
            },
            "properties": {
                "osmid": data.get('osmid', key),
                "length": data.get('length', geometry.length),
                "u": u,
                "v": v,
                "key": key
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

def process_geojson_file(input_filepath):
    """
    Procesa un archivo GeoJSON individual
    """
    print(f"Procesando: {input_filepath}")
    
    try:
        # Cargar GeoJSON original con CRS explícito
        gdf = gpd.read_file(input_filepath)
        
        # Asegurar que tiene CRS
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        
        # Convertir a formato de diccionario
        geojson_dict = {
            "type": "FeatureCollection",
            "features": json.loads(gdf.to_json())['features']
        }
        
        # Convertir a grafo NetworkX
        G = geojson_to_networkx_graph(geojson_dict)
        
        # Verificar que el grafo se creó correctamente
        print(f"  - Nodos: {G.number_of_nodes()}")
        print(f"  - Aristas: {G.number_of_edges()}")
        
        # Calcular estadísticas básicas como prueba
        try:
            stats = ox.basic_stats(G)
            print(f"  - Longitud total: {stats.get('street_length_total', 'N/A'):.4f}")
            print(f"  - Circuidad promedio: {stats.get('circuity_avg', 'N/A'):.4f}")
        except Exception as e:
            print(f"  - Advertencia: No se pudieron calcular estadísticas: {e}")
        
        # Convertir de vuelta a GeoJSON con CRS
        new_geojson = networkx_to_geojson(G)
        
        # Crear GeoDataFrame temporal para asegurar formato correcto
        temp_gdf = gpd.GeoDataFrame.from_features(new_geojson['features'], crs='EPSG:4326')
        
        # Guardar sobrescribiendo el archivo original
        temp_gdf.to_file(input_filepath, driver='GeoJSON')
        
        print(f"  ✓ Archivo actualizado exitosamente")
        return True
        
    except Exception as e:
        print(f"  ✗ Error procesando {input_filepath}: {e}")
        return False

def batch_process_geojson_files():
    """
    Procesa todos los archivos GeoJSON especificados
    """
    # Diccionario de archivos a procesar
    files_to_process = {
        "Synthetic_Street_Networks/perfect_grid_1.geojson": "perfect_grid_1",
        "Synthetic_Street_Networks/crossed_grid_1.geojson": "crossed_grid_1",
        "Synthetic_Street_Networks/cul_de_sac_1.geojson": "cul_de_sac_1",
        "Synthetic_Street_Networks/irregular_grid_1.geojson": "irregular_grid_1",
        "Synthetic_Street_Networks/organic_1.geojson": "organic_1",
        "Synthetic_Street_Networks/organic2_1.geojson": "organic2_1"
    }
    
    print("=== CONVERSIÓN POR LOTES DE GEOJSON ===")
    print(f"Procesando {len(files_to_process)} archivos...\n")
    
    successful = 0
    failed = 0
    
    for filepath, name in files_to_process.items():
        # Verificar que el archivo existe
        if not os.path.exists(filepath):
            print(f"⚠️  Archivo no encontrado: {filepath}")
            failed += 1
            continue
        
        # Procesar archivo
        if process_geojson_file(filepath):
            successful += 1
        else:
            failed += 1
        
        print()  # Línea en blanco para separar
    
    # Resumen final
    print("=== RESUMEN ===")
    print(f"✓ Archivos procesados exitosamente: {successful}")
    print(f"✗ Archivos con errores: {failed}")
    print(f"📁 Total archivos: {len(files_to_process)}")
    
    if successful > 0:
        print(f"\n🎉 Los archivos han sido convertidos y sobrescritos.")
        print("Ahora puedes usar ox.basic_stats() directamente con estos archivos:")
        print("\n# Ejemplo de uso:")
        print("import osmnx as ox")
        print("import geopandas as gpd")
        print("import json")
        print("from tu_script import geojson_to_networkx_graph")
        print("")
        print("# Cargar y analizar")
        print("with open('Synthetic_Street_Networks/perfect_grid_1.geojson', 'r') as f:")
        print("    geojson_data = json.load(f)")
        print("G = geojson_to_networkx_graph(geojson_data)")
        print("stats = ox.basic_stats(G)")

# Función auxiliar para uso posterior
def load_processed_geojson_as_graph(filepath):
    """
    Carga un GeoJSON procesado y lo convierte a grafo NetworkX
    Versión robusta con manejo de errores CRS
    """
    try:
        # Cargar con geopandas para manejar CRS correctamente
        gdf = gpd.read_file(filepath)
        
        # Verificar y asignar CRS si es necesario
        if gdf.crs is None:
            print(f"Asignando CRS EPSG:4326 a {filepath}")
            gdf = gdf.set_crs('EPSG:4326')
        
        # Convertir a diccionario manualmente para evitar problemas de CRS
        features = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Crear feature manualmente
            if geom.geom_type == "LineString":
                coordinates = list(geom.coords)
            else:
                coordinates = []
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {k: v for k, v in row.items() if k != 'geometry'}
            }
            features.append(feature)
        
        geojson_dict = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return geojson_to_networkx_graph(geojson_dict)
        
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        # Intentar carga alternativa con JSON puro
        try:
            with open(filepath, 'r') as f:
                geojson_data = json.load(f)
            return geojson_to_networkx_graph(geojson_data)
        except Exception as e2:
            print(f"Error también con carga JSON: {e2}")
            raise e

# if __name__ == "__main__":
#     # Lista de archivos a analizar
#     files_to_analyze = {
#         "Synthetic_Street_Networks/perfect_grid_1.geojson": "Perfect Grid",
#         "Synthetic_Street_Networks/crossed_grid_1.geojson": "Crossed Grid",
#         "Synthetic_Street_Networks/cul_de_sac_1.geojson": "Cul-de-sac",
#         "Synthetic_Street_Networks/irregular_grid_1.geojson": "Irregular Grid",
#         "Synthetic_Street_Networks/organic_1.geojson": "Organic 1",
#         "Synthetic_Street_Networks/organic2_1.geojson": "Organic 2"
#     }
    
#     print("=== ANÁLISIS DE BASIC STATS PARA REDES SINTÉTICAS ===\n")
    
#     for filepath, name in files_to_analyze.items():
#         if not os.path.exists(filepath):
#             print(f"❌ {name}: Archivo no encontrado - {filepath}")
#             continue
        
#         try:
#             print(f"🔍 {name} ({filepath}):")
            
#             # Cargar GeoJSON
#             gdf = gpd.read_file(filepath)
#             if gdf.crs is None:
#                 gdf = gdf.set_crs('EPSG:4326')
            
#             # Convertir a diccionario manualmente
#             features = []
#             for idx, row in gdf.iterrows():
#                 geom = row.geometry
#                 feature = {
#                     "type": "Feature",
#                     "geometry": {
#                         "type": geom.geom_type,
#                         "coordinates": list(geom.coords) if geom.geom_type == "LineString" else []
#                     },
#                     "properties": dict(row.drop('geometry'))
#                 }
#                 features.append(feature)
            
#             geojson_dict = {
#                 "type": "FeatureCollection",
#                 "features": features
#             }
            
#             # Convertir a grafo NetworkX
#             G = geojson_to_networkx_graph(geojson_dict)
            
#             # Calcular basic_stats
#             stats = ox.basic_stats(G)
#             print(stats)
           
#             # Mostrar todas las propiedades organizadamente
#             print("=" * 60)
#             print("ESTADÍSTICAS BÁSICAS DEL GRAFO")
#             print("=" * 60)

#             # Información básica del grafo
#             print("\n📊 INFORMACIÓN BÁSICA:")
#             print(f"   • Número de nodos (n): {stats['n']}")
#             print(f"   • Número de aristas (m): {stats['m']}")
#             print(f"   • Grado promedio (k_avg): {stats['k_avg']:.4f}")

           
#             # Información de calles por nodo
#             print("\n🛣️ CALLES POR NODO:")
#             print(f"   • Promedio de calles por nodo: {stats['streets_per_node_avg']:.4f}")
#             print(f"   • Conteo de calles por nodo:")
#             for calles, cantidad in stats['streets_per_node_counts'].items():
#                 print(f"     - {calles} calles: {cantidad} nodos")
#             print(f"   • Proporciones de calles por nodo:")
#             for calles, proporcion in stats['streets_per_node_proportions'].items():
#                 print(f"     - {calles} calles: {proporcion:.4f} ({proporcion*100:.2f}%)")

#             # Información adicional
#             print("\n🔍 INFORMACIÓN ADICIONAL:")
#             print(f"   • Número de intersecciones: {stats['intersection_count']}")
#             print(f"   • Número de segmentos de calle: {stats['street_segment_count']}")
#             print(f"   • Circuidad promedio: {stats['circuity_avg']:.10f}")

#             print("\n" + "=" * 60)
            
#         except Exception as e:
#             print(f"   ❌ Error: {e}")
#             print()




import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
from math import degrees, atan2

# =============================================================================
# PASO 1: LA FUNCIÓN DE ÁNGULOS CON LA LÓGICA DEFINITIVA Y CORRECTA
# =============================================================================

def calculate_intersection_corner_angles(G):
    """
    Calcula los ángulos de las "esquinas" en cada intersección.
    Este es el método conceptualmente correcto.
    """
    if G is None or G.number_of_nodes() == 0:
        return 0, 0, 0, 0
    
    corner_angles = []
    ortho_corners = 0
    
    for node, degree in G.degree():
        if degree > 2:  # Solo analizamos intersecciones reales
            neighbors = list(G.neighbors(node))
            if 'x' not in G.nodes[node] or 'y' not in G.nodes[node]: continue
            
            # 1. Obtener los ángulos de todas las calles salientes
            outgoing_angles = []
            for neighbor in neighbors:
                if 'x' not in G.nodes[neighbor] or 'y' not in G.nodes[neighbor]: continue
                
                x1, y1 = G.nodes[node]['x'], G.nodes[node]['y']
                x2, y2 = G.nodes[neighbor]['x'], G.nodes[neighbor]['y']
                angle = degrees(atan2(y2 - y1, x2 - x1))
                # Normalizar a 0-360
                if angle < 0:
                    angle += 360
                outgoing_angles.append(angle)
            
            # 2. Ordenar los ángulos para procesarlos circularmente
            outgoing_angles.sort()
            
            if len(outgoing_angles) > 1:
                # 3. Calcular la diferencia entre ángulos adyacentes ("esquinas")
                for i in range(len(outgoing_angles) - 1):
                    angle_diff = outgoing_angles[i+1] - outgoing_angles[i]
                    corner_angles.append(angle_diff)
                    if 80 <= angle_diff <= 100:
                        ortho_corners += 1

                # 4. Calcular la última esquina (entre el último y el primer ángulo)
                last_angle_diff = (360 - outgoing_angles[-1]) + outgoing_angles[0]
                corner_angles.append(last_angle_diff)
                if 80 <= last_angle_diff <= 100:
                    ortho_corners += 1

    if not corner_angles:
        return 0, 0, 0, 0
    
    mean_angle = np.mean(corner_angles)
    std_angle = np.std(corner_angles)
    # La proporción ahora es sobre el número total de esquinas
    ortho_proportion = ortho_corners / len(corner_angles) if corner_angles else 0
    cv_angle = std_angle / mean_angle if mean_angle > 0 else 0
    
    return mean_angle, std_angle, ortho_proportion, cv_angle


# La función de calles sin salida no cambia.
def calculate_dead_end_features(G):
    if G is None or G.number_of_nodes() == 0: return 0, 0
    dead_ends = [node for node, degree in G.degree() if degree == 1]
    dead_end_count = len(dead_ends)
    total_nodes = G.number_of_nodes()
    dead_end_ratio = dead_end_count / total_nodes if total_nodes > 0 else 0
    distances = []
    if len(dead_ends) > 1:
        for i, d1 in enumerate(dead_ends):
            x1, y1 = G.nodes[d1]['x'], G.nodes[d1]['y']
            for d2 in dead_ends[i+1:]:
                x2, y2 = G.nodes[d2]['x'], G.nodes[d2]['y']
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                distances.append(dist)
    cv_distances = 0
    if distances:
        mean_dist = np.mean(distances)
        if mean_dist > 0: cv_distances = np.std(distances) / mean_dist
    return dead_end_ratio, cv_distances

# =============================================================================
# EL RESTO DEL SCRIPT (NO NECESITA CAMBIOS)
# =============================================================================

files_to_analyze = {
    "Synthetic_Street_Networks/perfect_grid_1.geojson": "Perfect Grid",
    "Synthetic_Street_Networks/crossed_grid_1.geojson": "Crossed Grid",
    "Synthetic_Street_Networks/cul_de_sac_1.geojson": "Cul-de-sac",
    "Synthetic_Street_Networks/irregular_grid_1.geojson": "Irregular Grid",
    "Synthetic_Street_Networks/organic_1.geojson": "Organic 1",
    "Synthetic_Street_Networks/organic2_1.geojson": "Organic 2"
}
results_list = []

print("Iniciando análisis de redes viales sintéticas...")

for file_path, name in files_to_analyze.items():
    try:
        print(f"\nProcesando: {name} ({file_path})")
        gdf = gpd.read_file(file_path)
        gdf = gdf.to_crs("EPSG:3857")
        if gdf.empty: continue
        G = momepy.gdf_to_nx(gdf, approach='primal')
        print(f"  -> Grafo creado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")
        
        # LLAMAMOS A LA FUNCIÓN CON LA LÓGICA FINAL Y CORRECTA
        mean_angle, std_angle, ortho_prop, cv_angle = calculate_intersection_corner_angles(G)
        dead_end_ratio, cv_distances = calculate_dead_end_features(G)
        
        result_data = {
            "Network Type": name,
            "Mean Angle": mean_angle,
            "Std Dev Angle": std_angle,
            "Ortho Proportion": ortho_prop,
            "CV Angle": cv_angle,
            "Dead-end Ratio": dead_end_ratio,
            "CV Dead-end Dist.": cv_distances
        }
        results_list.append(result_data)
        print(f"  -> Análisis completado.")
    except FileNotFoundError:
        print(f"  -> ERROR: El archivo no se encontró en la ruta: {file_path}")
    except Exception as e:
        print(f"  -> ERROR al procesar {file_path}: {e}")

if results_list:
    df_results = pd.DataFrame(results_list)
    pd.options.display.float_format = '{:.4f}'.format
    print("\n\n--- TABLA DE RESULTADOS COMPARATIVOS ---")
    print(df_results.to_string())
else:
    print("\nNo se pudieron generar resultados.")