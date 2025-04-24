import geopandas as gpd
import os
import osmnx as ox
import json
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import re
import shapely
from shapely.geometry import Polygon, MultiPolygon
import plotly.graph_objects as go
import numpy as np
import matplotlib.patches as mpatches
import sys
import ast
import seaborn as sns
from scipy.stats import f_oneway, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report , calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import colors as mcolors
from shapely.wkt import loads
from shapely.geometry import shape
import scipy.stats as stats
from matplotlib.colors import ListedColormap
from networkx.algorithms.community import modularity
from matplotlib.patches import Patch
from math import degrees, atan2
import fiona
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)





def process_selected_layers(gdb_path, layer_indices=None, layer_names=None, save_geojson=True, visualize=True):
    """
    Process only selected layers from a geodatabase.
    
    Parameters:
    -----------
    gdb_path : str
        Path to the geodatabase folder
    layer_indices : list of int, optional
        List of layer indices to process (1-indexed as shown in the listing)
    layer_names : list of str, optional
        List of specific layer names to process
    save_geojson : bool, default True
        Whether to save layers as GeoJSON files
    visualize : bool, default True
        Whether to visualize the layers
        
    Returns:
    --------
    dict
        Dictionary of loaded GeoDataFrames with layer names as keys
    """
  
    
    # Create output folder for geojson if needed
    if save_geojson:
        output_dir = "GeoJSON_Export"
        os.makedirs(output_dir, exist_ok=True)
    
    # List available layers
    print("\nSearching for layers in the GDB...")
    all_layers = fiona.listlayers(gdb_path)
    print(f"Layers found ({len(all_layers)}):")
    for i, layer in enumerate(all_layers):
        print(f"{i+1}. {layer}")
    
    # Determine which layers to process
    selected_layers = []
    
    if layer_indices:
        # Convert 1-indexed to 0-indexed
        selected_layers.extend([all_layers[i-1] for i in layer_indices if 1 <= i <= len(all_layers)])
    
    if layer_names:
        # Add layers specified by name
        selected_layers.extend([layer for layer in layer_names if layer in all_layers])
    
    # Remove duplicates while preserving order
    selected_layers = list(dict.fromkeys(selected_layers))
    
    if not selected_layers:
        print("No valid layers selected.")
        return {}
    
    # Process selected layers
    print(f"\nProcessing {len(selected_layers)} selected layers:")
    results = {}
    
    for layer in selected_layers:
        print(f"\nReading layer: {layer}")
        gdf = gpd.read_file(gdb_path, layer=layer)
        results[layer] = gdf
        
        # Show properties
        print("Columns:", gdf.columns.tolist())
        print("Total geometries:", len(gdf))
        print("Coordinate Reference System (CRS):", gdf.crs)
        print(gdf.head())
        
        # Visualize if requested
        if visualize:
            gdf.plot(figsize=(8, 6), edgecolor="black", cmap="Set2")
            plt.title(f"Layer: {layer}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.tight_layout()
            plt.show()
        
        # Save as GeoJSON if requested
        if save_geojson:
            output_path = os.path.join(output_dir, f"{layer}.geojson")
            gdf.to_file(output_path, driver="GeoJSON")
            print(f"Saved as GeoJSON: {output_path}")
    
    return results

# Example 1: Process layers by index numbers (as shown in the listing)
gdb_path = "Pasto/AJUSTE_POT_PASTO_2023.gdb"
layer_data = process_selected_layers(gdb_path, layer_indices=[1, 2, 5, 6])

# # Example 2: Process layers by name
# layer_data = process_selected_layers(gdb_path, layer_names=["Poblaciones", "Canales"])

# # Example 3: Mix of indices and names
# layer_data = process_selected_layers(gdb_path, 
#                                     layer_indices=[1, 5], 
#                                     layer_names=["Vias_contexto"])

# # Example 4: No visualization, just data loading
# layer_data = process_selected_layers(gdb_path, 
#                                     layer_indices=[1, 2, 5, 6], 
#                                     visualize=False)

# # Example 5: No GeoJSON saving
# layer_data = process_selected_layers(gdb_path, 
#                                     layer_indices=[1, 2, 5, 6], 
#                                     save_geojson=False)

def convert_shapefile_to_geojson(shapefile_paths, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    geojson_data = {}

    for shapefile_path in shapefile_paths:
        gdf = gpd.read_file(shapefile_path)

        # Convert to EPSG:4326 if needed
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        # Convert the geodataframe to GeoJSON format
        geojson_string = gdf.to_json()

        # Define the output JSON file name
        shapefile_name = os.path.basename(shapefile_path).replace(".shp", ".geojson")
        output_path = os.path.join(output_directory, shapefile_name)

        # Save the GeoJSON file
        with open(output_path, "w") as f:
            f.write(geojson_string)

        # Store the file path in the result dictionary
        geojson_data[shapefile_path] = output_path

    return geojson_data







def plot_road_network_from_geojson(geojson_path, network_type, simplify=True):
    """
    Loads a GeoJSON file, extracts the polygon boundary, retrieves the road network,
    and plots the network with distinct colors for polygons, road links, and nodes.

    Parameters:
    - geojson_path (str): Path to the GeoJSON file.
    - network_type (str): Type of roads to retrieve ('all', 'drive', 'walk', etc.).
    - simplify (bool): Whether to simplify the graph for better visualization.

    Returns:
    - None (plots the figure).
    """

    # Load the GeoJSON file
    gdf = gpd.read_file(geojson_path)

    # Get the exact polygon boundary (avoid convex hull if possible)
    polygon = gdf.unary_union

    # Retrieve the road network from OSMnx
    G = ox.graph_from_polygon(polygon, network_type=network_type, simplify=simplify)

    # Convert graph to GeoDataFrame for better node handling
    nodes, edges = ox.graph_to_gdfs(G)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the boundary polygons with distinct colors
    gdf.plot(ax=ax, facecolor="lightgray", edgecolor="blue", linewidth=2, alpha=0.5, linestyle="--", label="Boundary Polygons")

    # Plot the road network edges in black
    edges.plot(ax=ax, edgecolor="black", linewidth=0.3, alpha=0.6, label="Road Links")

    # Plot the road network (links) in black
    ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.1, edge_color="black")

def get_street_network_metrics_per_polygon(
    geojson_path,
    network_type='drive',
    output_txt='stats_output.txt'
    ):
    """
    Lee un archivo GeoJSON que contiene uno o varios polígonos (o multipolígonos)
    y, para cada polígono/sub-polígono, calcula estadísticas de la red vial usando OSMnx.
    Luego, almacena los resultados en un archivo de texto, con un índice que
    identifica cada polígono procesado.

    Parámetros:
    -----------
    geojson_path : str
        Ruta al archivo GeoJSON.
    network_type : str
        Tipo de vías a recuperar ('all', 'drive', 'walk', etc.). Por defecto 'drive'.
    output_txt : str
        Nombre o ruta del archivo .txt donde se guardarán los resultados.

    Retorna:
    --------
    None. Escribe un archivo .txt con las estadísticas de cada polígono.
    """

    # ---------------------------------------------------------------------
    # 0. Si ya existe el .txt, leer su contenido previo
    #    y detectar qué Polígono/SubPolígono se han calculado.
    # ---------------------------------------------------------------------
    old_lines = []
    processed_pairs = set()  # para almacenar (idx, sub_idx)

    if os.path.exists(output_txt):
        with open(output_txt, 'r', encoding='utf-8') as old_file:
            old_lines = old_file.readlines()

        # Buscar líneas con el patrón: "=== Polígono X - SubPolígono Y ==="
        pattern = r"=== Polígono (\d+) - SubPolígono (\d+) ==="
        for line in old_lines:
            match = re.search(pattern, line)
            if match:
                i_str, s_str = match.groups()
                processed_pairs.add((int(i_str), int(s_str)))

    # ---------------------------------------------------------------------
    # 1. Cargar el GeoJSON como un GeoDataFrame
    # ---------------------------------------------------------------------
    gdf = gpd.read_file(geojson_path)

    # ---------------------------------------------------------------------
    # 2. Abrimos el archivo de salida (modo 'w') para sobrescribir
    #    pero primero volvemos a escribir lo que había antes
    # ---------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_txt) or '.', exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:

        # Reescribir el contenido previo (si existía)
        for line in old_lines:
            f.write(line)

        # -----------------------------------------------------------------
        # 3. Iterar sobre cada fila (cada 'feature') del GeoDataFrame
        # -----------------------------------------------------------------
        for idx, row in gdf.iterrows():
            geometry = row.geometry

            if geometry is None or geometry.is_empty:
                # Si la geometría está vacía, la ignoramos
                f.write(f"\n=== Polígono {idx}: GEOMETRÍA VACÍA ===\n")
                continue

            # Determinar si es Polygon o MultiPolygon
            if geometry.geom_type == 'Polygon':
                polygons_list = [geometry]
            elif geometry.geom_type == 'MultiPolygon':
                polygons_list = list(geometry.geoms)
            else:
                # Si es otro tipo de geometría (LineString, Point, etc.), saltar
                f.write(f"\n=== Polígono {idx}: Tipo de geometría no válido ({geometry.geom_type}) ===\n")
                continue

            # -----------------------------------------------------------------
            # 4. Procesar cada sub-polígono
            # -----------------------------------------------------------------
            for sub_idx, poly in enumerate(polygons_list):

                # Si ya está en processed_pairs, no recalculamos
                if (idx, sub_idx) in processed_pairs:
                    print(f"Saltando Polígono {idx} - SubPolígono {sub_idx}: ya existe en {output_txt}")
                    continue

                try:
                    G = ox.graph_from_polygon(
                        poly,
                        network_type=network_type,
                        simplify=True
                    )
                except Exception as e:
                    f.write(f"\n--- Polígono {idx}-{sub_idx}: ERROR al crear la red ---\n{e}\n")
                    continue

                # Verificar si el grafo tiene aristas
                if len(G.edges()) == 0:
                    f.write(f"\n--- Polígono {idx}-{sub_idx}: Grafo vacío (sin vías) ---\n")
                    continue

                # Calcular estadísticas
                stats = ox.stats.basic_stats(G)

                # Calcular área de este sub-polígono en km²
                area_km2 = (
                    gpd.GeoDataFrame(geometry=[poly], crs=gdf.crs)
                    .to_crs(epsg=3395)
                    .geometry.area.sum() / 1e6
                )

                intersection_count = stats.get("intersection_count", 0)
                street_length_total = stats.get("street_length_total", 0.0)

                if area_km2 > 0:
                    intersection_density_km2 = intersection_count / area_km2
                    street_density_km2 = street_length_total / area_km2
                else:
                    intersection_density_km2 = 0
                    street_density_km2 = 0

                stats["intersection_density_km2"] = intersection_density_km2
                stats["street_density_km2"] = street_density_km2
                stats["area_km2"] = area_km2

                # -----------------------------------------------------------------
                # 4.5 Guardar resultados en el archivo de texto
                # -----------------------------------------------------------------
                f.write(f"\n=== Polígono {idx} - SubPolígono {sub_idx} ===\n")
                for k, v in stats.items():
                    f.write(f"{k}: {v}\n")

    print(f"Resultados guardados en: {output_txt}")

def ordenar_y_limpiar_txt(input_txt, output_txt):
    """
    Lee un archivo .txt con bloques relacionados a "Polígono X - SubPolígono Y",
    o líneas de red vacía y errores, los ordena, elimina duplicados
    y genera un nuevo archivo .txt.

    Parámetros:
    -----------
    input_txt : str
        Ruta al archivo original desordenado.
    output_txt : str
        Ruta al archivo de salida ya ordenado y sin duplicados.

    Ejemplo de bloques reconocidos:
      "=== Polígono 181 - SubPolígono 0 ==="
      "--- Polígono 24-0: Grafo vacío (sin vías) ---"
      "=== Polígono 12: GEOMETRÍA VACÍA ==="
    """

    # 1. Leer todas las líneas
    if not os.path.exists(input_txt):
        print(f"Archivo {input_txt} no existe.")
        return

    with open(input_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. Expresiones regulares para reconocer "cabeceras"
    #    Bloques que inician con "=== Polígono X - SubPolígono Y ==="
    #    o "=== Polígono X: ..." (geom vacía) o "--- Polígono X-Y: Grafo vacío..."
    # ------------------------------------------------------------------
    #    Vamos a capturar:
    #      pol_idx -> \d+     (grupo 1)
    #      sub_idx -> \d+     (grupo 2), opcional
    #      type/block -> para distinguir "estadísticas", "vacío", "error", etc.
    #    
    #    Para simplificar, usaremos varios patrones y vemos cuál coincide.
    # ------------------------------------------------------------------

    # Caso A: "=== Polígono 181 - SubPolígono 0 ==="
    pattern_stats = re.compile(r'^=== Polígono\s+(\d+)\s*-\s*SubPolígono\s+(\d+)\s*===\s*$')
    # Caso B: "--- Polígono 24-0: Grafo vacío (sin vías) ---"
    pattern_empty = re.compile(r'^--- Polígono\s+(\d+)-(\d+):\s*(.+)---$')
    # Caso C: "=== Polígono 181: GEOMETRÍA VACÍA ==="
    pattern_geom = re.compile(r'^=== Polígono\s+(\d+):\s*(.+)===\s*$')

    # Vamos a recorrer las líneas y construir "bloques" de texto.
    # Cada vez que detectamos una cabecera, creamos una nueva entrada.
    data_dict = {}  
    # clave -> (pol_idx, sub_idx, tipo) 
    # valor -> list of lines (contenido)

    current_key = None
    current_block = []

    def save_block(key, block):
        """Guarda el bloque en data_dict, sobrescribiendo si la clave ya existía."""
        if key is None or not block:
            return
        # Sobrescribimos => la "última" versión de un polígono es la que prevalece
        data_dict[key] = block

    for line in lines:
        line_stripped = line.strip('\n')

        # ¿Coincide con pattern_stats?
        match_stats = pattern_stats.match(line_stripped)
        if match_stats:
            # Antes de iniciar nuevo bloque, guardamos el anterior
            save_block(current_key, current_block)
            current_block = [line]  # empezamos un nuevo bloque con esta línea
            pol = int(match_stats.group(1))
            sub = int(match_stats.group(2))
            current_key = (pol, sub, "stats") 
            continue

        # ¿Coincide con pattern_empty? (ej: Grafo vacío)
        match_empty = pattern_empty.match(line_stripped)
        if match_empty:
            save_block(current_key, current_block)
            current_block = [line]
            pol = int(match_empty.group(1))
            sub = int(match_empty.group(2))
            info = match_empty.group(3).strip()  # "Grafo vacío (sin vías)"
            current_key = (pol, sub, "empty:" + info)
            continue

        # ¿Coincide con pattern_geom? (ej: GEOMETRÍA VACÍA)
        match_geom = pattern_geom.match(line_stripped)
        if match_geom:
            save_block(current_key, current_block)
            current_block = [line]
            pol = int(match_geom.group(1))
            info = match_geom.group(2).strip()  # "GEOMETRÍA VACÍA" u otro
            # sub_idx = None => usaremos -1 para indicar "sin subpolígono"
            current_key = (pol, -1, "geom:" + info)
            continue

        # Si no coincide, es una línea dentro del bloque actual
        if current_key is not None:
            current_block.append(line)
        else:
            # No estamos dentro de un bloque (puede ser texto suelto),
            # lo ignoramos o guardamos si quieres.
            pass

    # Al final, guardar el último bloque (si existiera)
    save_block(current_key, current_block)

    # 3. Ahora data_dict tiene las entradas clave -> [lines_de_bloque].
    #    Queremos ordenarlas por polígono, subpolígono.
    #    Nota: sub_idx podría ser -1 en caso de "GEOMETRÍA VACÍA" (sin subpolígono).
    sorted_keys = sorted(data_dict.keys(), key=lambda x: (x[0], x[1]))

    # 4. Escribir el resultado ordenado y sin duplicados en output_txt
    with open(output_txt, 'w', encoding='utf-8') as out:
        for key in sorted_keys:
            block_lines = data_dict[key]
            for bl in block_lines:
                out.write(bl)
        out.write("\n")  # Salto de línea final

    print(f"Archivo ordenado y limpio guardado en: {output_txt}")

# if __name__ == "__main__":
#     input_file = "Poligonos_Medellin/Resultados/poligonos_stats.txt"   # El .txt caótico
#     output_file = "Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt"
#     ordenar_y_limpiar_txt(input_file, output_file)

def load_polygon_stats_from_txt(stats_txt):
    """
    Lee un archivo de estadísticas (.txt) y devuelve un diccionario
    { (pol_idx, sub_idx): {"k_avg": val, "m": val, ...}, ... }.
    Ignora los casos de 'Grafo vacío' o 'GEOMETRÍA VACÍA'.
    """

    with open(stats_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pattern_header = re.compile(r'^=== Polígono\s+(\d+)\s*-\s*SubPolígono\s+(\d+)\s*===\s*$')
    pattern_kv = re.compile(r'^([^:]+):\s+(.*)$')  # clave: valor

    stats_dict = {}
    current_key = None

    for line in lines:
        line_stripped = line.strip()
        # Detectar encabezado
        match_header = pattern_header.match(line_stripped)
        if match_header:
            pol_idx = int(match_header.group(1))
            sub_idx = int(match_header.group(2))
            current_key = (pol_idx, sub_idx)
            stats_dict[current_key] = {}
            continue

        # Si estamos dentro de un bloque, buscar clave: valor
        if current_key is not None:
            match_kv = pattern_kv.match(line_stripped)
            if match_kv:
                k = match_kv.group(1)
                v = match_kv.group(2)
                # Intentar convertir a float o int
                try:
                    v_num = float(v)
                    # Si es entero sin decimales, lo pasamos a int
                    if v_num.is_integer():
                        v_num = int(v_num)
                    stats_dict[current_key][k] = v_num
                except ValueError:
                    # no se pudo convertir -> guardar como string
                    stats_dict[current_key][k] = v

    return stats_dict

def classify_polygon(poly_stats):
    """
    Clasifica un polígono (o sub-polígono) en:
      'cul_de_sac', 'gridiron', 'organico' o 'hibrido'
    usando reglas de decisión basadas en diversas estadísticas.

    Parámetros:
    -----------
    poly_stats : dict con claves:
      - "streets_per_node_avg" (float)
      - "streets_per_node_counts" (str o dict)
      - "streets_per_node_proportions" (str o dict)
      - "intersection_density_km2" (float)
      - "circuity_avg" (float)
      - "k_avg" (float)
      - "street_density_km2" (float)
      - etc.

    Retorna:
    --------
    str : 'cul_de_sac', 'gridiron', 'organico' o 'hibrido'
    """

    # -------------------------------------------------------------------
    # 1. Parsear fields que podrían venir como string en lugar de dict
    # -------------------------------------------------------------------
    spn_counts_str = poly_stats.get("streets_per_node_counts", "{}")
    spn_props_str = poly_stats.get("streets_per_node_proportions", "{}")

    if isinstance(spn_counts_str, str):
        try:
            spn_counts = ast.literal_eval(spn_counts_str)
        except:
            spn_counts = {}
    else:
        spn_counts = spn_counts_str

    if isinstance(spn_props_str, str):
        try:
            spn_props = ast.literal_eval(spn_props_str)
        except:
            spn_props = {}
    else:
        spn_props = spn_props_str

    # -------------------------------------------------------------------
    # 2. Extraer Variables Numéricas Principales
    # -------------------------------------------------------------------
    streets_per_node = float(poly_stats.get("streets_per_node_avg", 0.0))
    intersection_density = float(poly_stats.get("intersection_density_km2", 0.0))
    circuity = float(poly_stats.get("circuity_avg", 1.0))
    k_avg = float(poly_stats.get("k_avg", 0.0))
    street_density = float(poly_stats.get("street_density_km2", 0.0))

    # Ejemplo de proporciones de nodos grado 1 y 4
    prop_deg1 = float(spn_props.get(1, 0.0))  # nodos con 1 calle
    prop_deg4 = float(spn_props.get(4, 0.0))  # nodos con 4 calles

    # -------------------------------------------------------------------
    # 3. Árbol de Decisión (umbrales orientativos)
    # -------------------------------------------------------------------
    #
    # A. Detectar cul-de-sac
    #    Reglas ejemplo:
    #    - Proporción de nodos grado 1 alta (>= 0.40)
    #    - O (streets_per_node < 2.1 y intersection_density < 30)
    #
    if prop_deg1 >= 0.40 or (streets_per_node < 2.1 and intersection_density < 30):
        return "cul_de_sac"

    # B. Detectar gridiron
    #    - Poca sinuosidad: circuity < 1.03
    #    - Buena conectividad local: streets_per_node >= 3.0
    #    - intersection_density >= 50 (bastante intersecciones)
    #    - prop_deg4 > 0.30 ó 0.40 => muchos nodos con 4 salidas
    #
    if (circuity < 1.03) and (streets_per_node >= 3.0 or intersection_density >= 50 or prop_deg4 >= 0.30):
        return "gridiron"

    # C. Detectar orgánico
    #    - Calles sinuosas => circuity > 1.05
    #    - k_avg > 3.0 => muchos nodos con 3+ aristas
    #    - street_density > 3000 => malla densa
    #
    if (circuity > 1.05) and (k_avg > 3.0) and (street_density > 3000):
        return "organico"

    # D. Caso general => híbrido
    return "hibrido"


# def classify_polygon(poly_stats):
#     """
#     Clasifica un polígono (o sub-polígono) en:
#       'cul_de_sac', 'gridiron', 'organico' o 'hibrido'
#     basado en la teoría de patrones urbanos y métricas morfológicas.

#     Parámetros:
#     -----------
#     poly_stats : dict con claves:
#       - "streets_per_node_avg" (float): Promedio de calles por nodo
#       - "streets_per_node_counts" (str o dict): Conteo de nodos por número de calles
#       - "streets_per_node_proportions" (str o dict): Proporción de nodos por número de calles
#       - "intersection_density_km2" (float): Densidad de intersecciones por km²
#       - "circuity_avg" (float): Sinuosidad promedio de segmentos
#       - "k_avg" (float): Grado promedio de nodos
#       - "street_density_km2" (float): Densidad de calles por km²
#       - "orientation_entropy" (float, opcional): Entropía de orientación de segmentos
#       - "edge_length_avg" (float, opcional): Longitud promedio de aristas
#       - "street_length_avg" (float, opcional): Longitud promedio de calles

#     Retorna:
#     --------
#     str : 'cul_de_sac', 'gridiron', 'organico' o 'hibrido'
#     """
    
#     # -------------------------------------------------------------------
#     # 1. Parsear fields que podrían venir como string en lugar de dict
#     # -------------------------------------------------------------------
#     spn_counts_str = poly_stats.get("streets_per_node_counts", "{}")
#     spn_props_str = poly_stats.get("streets_per_node_proportions", "{}")

#     if isinstance(spn_counts_str, str):
#         try:
#             spn_counts = ast.literal_eval(spn_counts_str)
#         except:
#             spn_counts = {}
#     else:
#         spn_counts = spn_counts_str

#     if isinstance(spn_props_str, str):
#         try:
#             spn_props = ast.literal_eval(spn_props_str)
#         except:
#             spn_props = {}
#     else:
#         spn_props = spn_props_str

#     # -------------------------------------------------------------------
#     # 2. Extraer Variables Numéricas Principales
#     # -------------------------------------------------------------------
#     # Conectividad y estructura nodal
#     streets_per_node = float(poly_stats.get("streets_per_node_avg", 0.0))
#     k_avg = float(poly_stats.get("k_avg", 0.0))
    
#     # Densidades
#     intersection_density = float(poly_stats.get("intersection_density_km2", 0.0))
#     street_density = float(poly_stats.get("street_density_km2", 0.0))
    
#     # Geometría
#     circuity = float(poly_stats.get("circuity_avg", 1.0))
#     edge_length_avg = float(poly_stats.get("edge_length_avg", 0.0))
#     street_length_avg = float(poly_stats.get("street_length_avg", 0.0))
    
#     # Entropía de orientación (0=alineado, 1=diverso)
#     orientation_entropy = float(poly_stats.get("orientation_entropy", 0.5))
    
#     # Proporciones de nodos por grado
#     prop_deg1 = float(spn_props.get('1', 0.0))  # callejones sin salida
#     prop_deg3 = float(spn_props.get('3', 0.0))  # intersecciones en T
#     prop_deg4 = float(spn_props.get('4', 0.0))  # intersecciones en cruz

#     # -------------------------------------------------------------------
#     # 3. Clasificación por Patrones Urbanos
#     # -------------------------------------------------------------------
    
#     # A. Patrón Cul-de-sac / Suburban
#     # Características: Alta proporción de callejones sin salida, baja conectividad,
#     # estructura jerárquica y arborescente, baja densidad de intersecciones
#     if (prop_deg1 >= 0.35 or 
#         (streets_per_node < 2.4 and intersection_density < 40) or
#         (prop_deg1 >= 0.25 and circuity > 1.1 and streets_per_node < 2.5)):
#         return "cul_de_sac"

#     # B. Patrón Gridiron / Reticular
#     # Características: Baja sinuosidad, alta proporción de cruces (nodos grado 4),
#     # buena conectividad, orientación consistente (baja entropía de orientación)
#     if ((circuity < 1.05 and prop_deg4 >= 0.25) or
#         (streets_per_node >= 2.8 and orientation_entropy < 0.6 and prop_deg4 > prop_deg3) or
#         (intersection_density >= 70 and circuity < 1.08 and prop_deg1 < 0.15)):
#         return "gridiron"

#     # C. Patrón Orgánico / Irregular
#     # Características: Alta sinuosidad, predominio de intersecciones en T,
#     # alta entropía de orientación, irregularidad geométrica
#     if ((circuity > 1.08 and orientation_entropy > 0.7) or
#         (prop_deg3 > prop_deg4 * 1.5 and circuity > 1.05) or
#         (orientation_entropy > 0.8 and street_density > 15000 and circuity > 1.05)):
#         return "organico"

#     # D. Patrón Híbrido (mezcla de tipos o casos especiales)
#     # Incluyendo subdivisión si se detectan características específicas
#     if (streets_per_node > 2.7 and intersection_density > 50 and 
#         0.15 < prop_deg1 < 0.25 and 0.2 < prop_deg4 < 0.3):
#         # Híbrido con tendencia a retícula
#         return "hibrido"
    
#     # Caso general - Híbrido no específico
#     return "hibrido"

def add_classification_to_gdf(geojson_path, stats_dict):
    """
    Carga el geojson como gdf, crea una columna 'class' con
    la categoría del polígono, o None si no hay datos en stats_dict.
    """
    gdf = gpd.read_file(geojson_path).copy()

    # Crear nueva columna
    gdf["class"] = None

    for idx in gdf.index:
        # Si usas "Polígono idx - SubPolígono 0" => la clave es (idx, 0)
        key = (idx, 0)
        if key in stats_dict:
            # tenemos datos
            poly_stats = stats_dict[key]
            cat = classify_polygon(poly_stats)
            gdf.at[idx, "class"] = cat
        else:
            # No hay datos => se queda como None
            pass

    return gdf

def plot_polygons_classification_png(
    geojson_path,
    stats_dict,
    classify_func,
    output_png="polygons_classification.png"
):
    """
    Lee un GeoDataFrame (geojson_path), asigna una 'clase' a cada polígono
    según las estadísticas en 'stats_dict' y la 'classify_func',
    y dibuja en un PNG (Matplotlib) con colores distintos por clase.

    Se asume que, en 'stats_dict', las claves son (idx, sub_idx).
    Aquí, tomamos solamente sub_idx=0, por ejemplo, si cada fila
    corresponde a (idx, 0). Ajusta si es distinto.
    """

    gdf = gpd.read_file(geojson_path)

    # Crear columna 'pattern' con la clase
    patterns = []
    for idx, row in gdf.iterrows():
        key = (idx, 0)  # si cada fila = sub_poligono 0
        if key in stats_dict:
            poly_stats = stats_dict[key]
            category = classify_func(poly_stats)
        else:
            category = "desconocido"
        patterns.append(category)

    gdf["pattern"] = patterns

    # Mapear cada clase a un color
    color_map = {
        'cul_de_sac': '#FF6B6B',   # Rojo suave
        'gridiron': '#4ECDC4',     # Verde azulado
        'organico': '#45B7D1',     # Azul claro
        'hibrido': '#FDCB6E', 
        "desconocido": "gray"
    }

    def get_color(cat):
        return color_map.get(cat, "black")

    plot_colors = [get_color(cat) for cat in gdf["pattern"]]

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(
        ax=ax,
        color=plot_colors,
        edgecolor="black",
        linewidth=0.5
    )

    # Leyenda manual
    legend_patches = []
    for cat, col in color_map.items():
        patch = mpatches.Patch(color=col, label=cat)
        legend_patches.append(patch)
    ax.legend(handles=legend_patches, title="Tipo de polígono")

    ax.set_title("Clasificación de Polígonos", fontsize=14)
    ax.set_axis_off()

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Imagen guardada en: {output_png}")

# # =============================================================================
# # Ejemplo de uso
# if __name__ == "__main__":
#     geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA.geojson"
#     stats_txt = "Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt"

#     # 1. Cargar stats desde .txt
#     stats_dict = load_polygon_stats_from_txt(stats_txt)

#     # 2. Generar PNG con cada fila = (idx, 0)
#     plot_polygons_classification_png(
#         geojson_path=geojson_file,
#         stats_dict=stats_dict,
#         classify_func=classify_polygon,
#         output_png="polygon_classification.png"
#     )

def normalize_edge(x0, y0, x1, y1, tol=4):
    """
    Redondea las coordenadas a 'tol' decimales y retorna una tupla
    ordenada con los dos puntos. De esta forma, el segmento (A,B) es igual a (B,A).
    """
    p1 = (round(x0, tol), round(y0, tol))
    p2 = (round(x1, tol), round(y1, tol))
    return tuple(sorted([p1, p2]))

def plot_street_patterns_classification(
    geojson_path,
    classify_func,
    stats_dict,
    place_name="MyPlace",
    network_type="drive",
    output_folder="Graphs_Cities",
    simplify=True
):
    """
    Genera un HTML interactivo en Plotly con DOS capas:
      1) Capa base vectorial (en gris) de la red completa a partir de la unión de
         todos los polígonos, filtrando aquellos segmentos que se sobreponen con los
         de la clasificación.
      2) Capas vectoriales de sub-polígonos coloreados según su clasificación.
    
    De esta forma, los segmentos repetidos se muestran sólo en la capa clasificada.
    """
    print("Construyendo la capa base (red completa) a partir de la unión de polígonos...")
    gdf = gpd.read_file(geojson_path)
    try:
        poly_union = gdf.union_all()  # Si tu geopandas es reciente
    except AttributeError:
        poly_union = gdf.unary_union

    try:
        G_full = ox.graph_from_polygon(poly_union, network_type=network_type, simplify=simplify)
    except Exception as e:
        print(f"Error al crear la red base: {e}")
        return

    if len(G_full.edges()) == 0:
        print("La red base (unión) está vacía. Revisa tu GeoJSON y tipo de red.")
        return

    G_full = ox.project_graph(G_full)
    base_nodes = list(G_full.nodes())
    base_positions = np.array([(G_full.nodes[n]['y'], G_full.nodes[n]['x']) for n in base_nodes])
    x_vals = base_positions[:, 1]
    y_vals = base_positions[:, 0]
    global_x_min, global_x_max = x_vals.min(), x_vals.max()
    global_y_min, global_y_max = y_vals.min(), y_vals.max()

    # --------------------------------------------------------------------------------
    # B) CAPAS DE SUB-POLÍGONOS: Genera trazas vectoriales clasificadas y almacena
    # sus segmentos en un set para filtrar la capa base
    # --------------------------------------------------------------------------------
    print("Generando capas vectoriales para sub-polígonos clasificados...")
    pattern_colors = {
        "cul_de_sac":   "red",
        "gridiron":     "green",
        "organico":     "blue",
        "hibrido":      "orange"
    }
    default_color = "gray"
    classification_traces = []
    classified_edges_set = set()

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for sub_idx, poly in enumerate(polys):
            key = (idx, sub_idx)
            if key in stats_dict:
                poly_stats = stats_dict[key]
                poly_class = classify_func(poly_stats)
            else:
                poly_class = None
            color = pattern_colors.get(poly_class, default_color)
            try:
                G_sub = ox.graph_from_polygon(poly, network_type=network_type, simplify=simplify)
            except Exception:
                continue
            if len(G_sub.edges()) == 0:
                continue
            G_sub = ox.project_graph(G_sub)
            sub_nodes = list(G_sub.nodes())
            sub_positions = np.array([(G_sub.nodes[n]['y'], G_sub.nodes[n]['x']) for n in sub_nodes])
            if sub_positions.size == 0:
                continue

            # Actualizar bounding box global
            lx_min, lx_max = sub_positions[:,1].min(), sub_positions[:,1].max()
            ly_min, ly_max = sub_positions[:,0].min(), sub_positions[:,0].max()
            global_x_min = min(global_x_min, lx_min)
            global_x_max = max(global_x_max, lx_max)
            global_y_min = min(global_y_min, ly_min)
            global_y_max = max(global_y_max, ly_max)

            # Procesar aristas del sub-grafo y almacenarlas en el set
            edge_x = []
            edge_y = []
            for u, v in G_sub.edges():
                if u in sub_nodes and v in sub_nodes:
                    u_idx = sub_nodes.index(u)
                    v_idx = sub_nodes.index(v)
                    # Extraer coordenadas de cada extremo (en proyección)
                    x0 = sub_positions[u_idx][1]
                    y0 = sub_positions[u_idx][0]
                    x1 = sub_positions[v_idx][1]
                    y1 = sub_positions[v_idx][0]
                    # Normalizar el segmento
                    norm_edge = normalize_edge(x0, y0, x1, y1)
                    classified_edges_set.add(norm_edge)
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
            edge_trace = go.Scattergl(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1.0, color=color),
                hoverinfo='none',
                name=f"Polígono {idx}-{sub_idx} ({poly_class})" if poly_class else f"Polígono {idx}-{sub_idx}"
            )
            x_sub = sub_positions[:,1]
            y_sub = sub_positions[:,0]
            node_trace = go.Scattergl(
                x=x_sub, y=y_sub,
                mode='markers',
                marker=dict(
                    size=1,
                    color=color,
                    opacity=0.9,
                    line=dict(width=0.4, color='black')
                ),
                hoverinfo='none',
                name=f"Polígono {idx}-{sub_idx} ({poly_class})" if poly_class else f"Polígono {idx}-{sub_idx}"
            )
            classification_traces.append(edge_trace)
            classification_traces.append(node_trace)

    # --------------------------------------------------------------------------------
    # C) CAPA BASE VECTORIAL: Filtrar segmentos que aparecen en la capa clasificada
    # --------------------------------------------------------------------------------
    print("Filtrando segmentos repetidos en la capa base...")
    base_edges_filtered = []
    base_edge_x = []
    base_edge_y = []
    for u, v in G_full.edges():
        if u in base_nodes and v in base_nodes:
            u_idx = base_nodes.index(u)
            v_idx = base_nodes.index(v)
            x0 = base_positions[u_idx][1]
            y0 = base_positions[u_idx][0]
            x1 = base_positions[v_idx][1]
            y1 = base_positions[v_idx][0]
            norm_edge = normalize_edge(x0, y0, x1, y1)
            if norm_edge not in classified_edges_set:
                # Este segmento no está en la capa clasificada, lo añadimos
                base_edge_x.extend([x0, x1, None])
                base_edge_y.extend([y0, y1, None])
                base_edges_filtered.append(norm_edge)
    
    base_edge_trace = go.Scattergl(
        x=base_edge_x, y=base_edge_y,
        mode='lines',
        line=dict(width=0.5, color='rgba(150,150,150,0.6)'),
        hoverinfo='none',
        name='BASE: Red completa'
    )

    # --------------------------------------------------------------------------------
    # D) Crear la figura final: Unir la capa base filtrada y las capas de clasificación
    # --------------------------------------------------------------------------------
    final_traces = [base_edge_trace] + classification_traces

    x_range = [global_x_min, global_x_max]
    y_range = [global_y_min, global_y_max]

    layout = go.Layout(
        title=f'Clasificación de Street Patterns - {place_name}',
        showlegend=True,
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
        paper_bgcolor="#F5F5F5",
        plot_bgcolor="#FFFFFF"
    )

    fig = go.Figure(data=final_traces, layout=layout)

    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
        'toImageButtonOptions': {
            'format': 'svg',
            'width': 1600,
            'height': 1600
        },
        'responsive': True
    }

    os.makedirs(output_folder, exist_ok=True)
    out_html = os.path.join(output_folder, f"StreetPatterns_{place_name}.html")
    fig.write_html(
        out_html, 
        config=config, 
        include_plotlyjs='cdn', 
        auto_open=False, 
        full_html=True,
        default_width='100%',
        default_height='100%'
        )

    print(f"Archivo HTML generado: {out_html}")
    print("Capa base filtrada (sin duplicados) + sub-polígonos clasificados superpuestos.")

# # ================== EJEMPLO DE USO ==================
# if __name__ == "__main__":
#     # Define las rutas a tus archivos
#     geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA.geojson"
#     stats_txt = "Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt"
#     stats_dict = load_polygon_stats_from_txt(stats_txt)
    

#     plot_street_patterns_classification(
#         geojson_path=geojson_file,
#         classify_func=classify_polygon,
#         stats_dict=stats_dict,
#         place_name="Medellin",
#         network_type="drive",
#         output_folder="Graphs_Cities",
#         simplify=False
#     )

def match_polygons_by_area(gdfA, gdfB, area_ratio_threshold=0.9, out_csv=None):
    """
    Empareja cada polígono de gdfA con el polígono de gdfB que más
    área de intersección tiene (area(A ∩ B) / area(A)).
    
    - Se asume que gdfA y gdfB tienen la misma proyección (CRS).
    - Para evitar colisiones con la geometría de B, clonamos la geometría
      de B en una columna 'geomB' antes del sjoin.
    - Al final, generamos un DataFrame con (indexA, indexB, area_ratio).
    - Si area_ratio < area_ratio_threshold no se considera match.
    
    Parámetros:
    -----------
    gdfA, gdfB : GeoDataFrames
        Con geometrías Polygons
    area_ratio_threshold : float
        Umbral mínimo para considerar un match.
    out_csv : str | None
        Ruta para guardar CSV, o None si no se desea.
    
    Retorna:
    --------
    DataFrame con columnas:
      - indexA : índice de polígono A
      - indexB : índice de polígono B
      - area_ratio : fracción del área de A que se superpone con B
    """

    # Copias locales (para no alterar los gdf originales)
    gdfA = gdfA.copy()
    gdfB = gdfB.copy()

    # Asegurar índices para rastreo
    gdfA["indexA"] = gdfA.index
    gdfB["indexB"] = gdfB.index

    # 1. Clonar la geometría de B en una columna normal 'geomB'
    #    para no depender de cómo sjoin renombra la geometry de B
    gdfB["geomB"] = gdfB.geometry

    # 2. Realizar sjoin con intersects
    #    - geometry principal en B sigue siendo 'geometry'
    #    - sjoin usará esa geometry para la intersección
    joined = gpd.sjoin(
        gdfA,
        gdfB.drop(columns="geomB"),  # geometry nativa para sjoin
        how="inner",
        predicate="intersects",
        lsuffix="_A",
        rsuffix="_B"
    )

    # 'joined' tiene 'geometry' = la de A
    # y las columnas de B, excepto geomB (porque la droppeamos).
    # Sin embargo, conservamos 'geomB' en gdfB para reasignarla tras el sjoin

    # 3. Unir la columna 'geomB' de B a joined manualmente:
    #    Cada fila de joined tiene un 'indexB' que indica qué polígono de B era.
    #    Podemos hacer un merge con gdfB[["indexB","geomB"]] (en memory).
    joined = joined.merge(
        gdfB[["indexB", "geomB"]],
        on="indexB",
        how="left"
    )

    # 4. Calcular area_ratio = area(A ∩ geomB) / area(A)
    def compute_area_ratio(row):
        geomA = row["geometry"]   # polígono de A
        geomB_ = row["geomB"]     # polígono original de B
        if geomA is None or geomB_ is None:
            return 0.0
        inter = geomA.intersection(geomB_)
        if inter.is_empty:
            return 0.0
        areaA = geomA.area
        if areaA == 0:
            return 0.0
        return inter.area / areaA

    joined["area_ratio"] = joined.apply(compute_area_ratio, axis=1)

    # 5. Para cada indexA, quedarnos con el polígono B de mayor area_ratio
    best_matches = joined.loc[
        joined.groupby("indexA")["area_ratio"].idxmax()
    ].copy()

    # 6. Filtrar por el umbral
    best_matches = best_matches[best_matches["area_ratio"] >= area_ratio_threshold]

    # 7. Extraer un DataFrame con indexA, indexB, area_ratio
    df_out = best_matches[["indexA", "indexB", "area_ratio"]].reset_index(drop=True)

    # 8. Guardar CSV si se desea
    if out_csv:
        df_out.to_csv(out_csv, index=False)
        print(f"Guardado {out_csv} con {len(df_out)} matches. Umbral={area_ratio_threshold}")

    return df_out

# # ==================== EJEMPLO DE USO ====================
# if __name__ == "__main__":

#     shpA = "Poligonos_Medellin/EOD_2017_SIT_only_AMVA.shp"
#     shpB = "Poligonos_Medellin/eod_gen_trips_mode.shp"

#     print("Leyendo shapefiles A y B...")
#     gdfA = gpd.read_file(shpA)
#     gdfB = gpd.read_file(shpB)

#     # Asegurar mismo CRS
#     if gdfA.crs != gdfB.crs:
#         gdfB = gdfB.to_crs(gdfA.crs)
#         print(f"Reproyectado B a {gdfA.crs}")

#     # Llamar función con threshold=0.9
#     print("Iniciando match_polygons_by_area con area_ratio_threshold=0.9")
#     df_matches = match_polygons_by_area(
#         gdfA,
#         gdfB,
#         area_ratio_threshold=0.9,
#         out_csv="Poligonos_Medellin/Resultados/Matchs_A_B/matches_by_area.csv"
#     )

#     print("Algunos matches:")
#     print(df_matches.head())
#     print(f"Total matches: {len(df_matches)}")

def process_mobility_data(area_type='urban_and_rural'):
    # Configurar rutas y opciones según el tipo de área
    if area_type == 'urban':
        a_path = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson"
        include_absolute = False
        output_xlsx = "Poligonos_Medellin/Resultados/Statics_Results/URBAN/Poligonos_Clasificados_Movilidad_URBANO.xlsx"
        output_image = "Poligonos_Medellin/Resultados/Statics_Results/URBAN/map_poligonosA_urbano_classified.png"
    else:
        a_path = "Poligonos_Medellin/EOD_2017_SIT_only_AMVA.shp"
        include_absolute = False
        output_xlsx = "Poligonos_Medellin/Resultados/Statics_Results/URBAN_AND_RURAL/Poligonos_Clasificados_Movilidad_Urban_and_Rural.xlsx"
        output_image = "Poligonos_Medellin/Resultados/Statics_Results/URBAN_AND_RURAL/map_poligonosA_classified_Urban_and_Rural.png"
    
    # Rutas comunes
    stats_txt = "Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt"
    matches_csv = "Poligonos_Medellin/Resultados/Matchs_A_B/matches_by_area.csv"
    shpB = "Poligonos_Medellin/eod_gen_trips_mode.shp"
    
    # 1) Cargar estadísticas
    stats_dict = load_polygon_stats_from_txt(stats_txt)
    print(f"Cargadas stats para {len(stats_dict)} polígonos (subpolígonos).")
    
    # 2) Cargar emparejamientos
    df_matches = pd.read_csv(matches_csv)
    print("Muestra df_matches:\n", df_matches.head(), "\n")
    
    # 3) Leer shapefile B (movilidad)
    gdfB = gpd.read_file(shpB)
    print("Columnas B:", gdfB.columns)
    
    # 4) Leer GeoDataFrame A (geometría)
    gdfA = gpd.read_file(a_path)
    print(f"Leídos {len(gdfA)} polígonos en {'GeoJSON URBANO' if area_type == 'urban' else 'SHP'}.")
    
    # 5) Construir DataFrame final
    final_rows = []
    for _, row in df_matches.iterrows():
        idxA = row["indexA"]
        idxB = row["indexB"]
        ratio = row["area_ratio"]
        
        # Obtener estadísticas y clasificar patrón
        key_stats = (idxA, 0)
        poly_stats = stats_dict.get(key_stats, {})
        pattern = classify_polygon(poly_stats)
        
        # Extraer datos de movilidad
        rowB = gdfB.loc[idxB]
        p_walk_ = rowB.get("p_walk", 0)
        p_tpc_ = rowB.get("p_tpc", 0)
        p_sitva_ = rowB.get("p_sitva", 0)
        p_auto_ = rowB.get("p_auto", 0)
        p_moto_ = rowB.get("p_moto", 0)
        p_taxi_ = rowB.get("p_taxi", 0)
        p_bike_ = rowB.get("p_bike", 0)
        
        # Datos base para la fila
        row_data = {
            "indexA": idxA,
            "indexB": idxB,
            "area_ratio": ratio,
            "street_pattern": pattern,
            "p_walk": p_walk_,
            "p_tpc": p_tpc_,
            "p_sitva": p_sitva_,
            "p_auto": p_auto_,
            "p_moto": p_moto_,
            "p_taxi": p_taxi_,
            "p_bike": p_bike_
        }
        
        # Agregar datos absolutos si corresponde
        if include_absolute:
            row_data.update({
                "Auto": rowB.get("Auto", 0),
                "Moto": rowB.get("Moto", 0),
                "Taxi": rowB.get("Taxi", 0)
            })
        
        final_rows.append(row_data)
    
    # Definir columnas del DataFrame final
    columns = ["indexA", "indexB", "area_ratio", "street_pattern"]
    if include_absolute:
        columns += ["Auto", "Moto", "Taxi"]
    columns += ["p_walk", "p_tpc", "p_sitva", "p_auto", "p_moto", "p_taxi", "p_bike"]
    
    df_final = pd.DataFrame(final_rows)[columns]
    
    # Guardar Excel
    df_final.to_excel(output_xlsx, index=False)
    print(f"Guardado Excel final en {output_xlsx} con {len(df_final)} filas.\n")
    
# 6) Graficar polígonos clasificados
    gdfA["pattern"] = gdfA.index.map(df_final.set_index("indexA")["street_pattern"])
    
    # Mapeo de patrones a colores específicos
    color_mapping = {
        'gridiron': 'Green',
        'cul_de_sac': 'Red',
        'hibrido': 'Blue',
        'organico': 'Yellow'
    }
    
    # Convertir a categorías ordenadas
    categories = ['gridiron', 'cul_de_sac', 'hibrido', 'organico']
    gdfA['pattern'] = pd.Categorical(gdfA['pattern'], categories=categories)
    
    # Crear colormap personalizado
    cmap = ListedColormap([color_mapping[cat] for cat in categories])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Graficar con bordes oscuros
    gdfA.plot(
        column="pattern",
        ax=ax,
        legend=True,
        cmap=cmap,
        edgecolor='black',  # Borde negro
        linewidth=1,      # Grosor del borde
    )

    
    # Personalizar leyenda
    legend = ax.get_legend()
    legend.set_title('Patrón de Calles')
    legend.set_bbox_to_anchor((1.05, 1))  # Mover leyenda a la derecha

    title = "Polígonos A clasificados" if area_type != 'urban' else "GeoJSON Urbano - Polígonos Clasificados"
    gdfA.plot(column="pattern", ax=ax, legend=True, cmap="Set2")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Mapa guardado en {output_image}")

# # Ejemplo de uso
# if __name__ == "__main__":
#     # Procesar datos urbanos y rurales
#     process_mobility_data(area_type='urban_and_rural')
    
#     # Procesar solo datos urbanos
#     process_mobility_data(area_type='urban')

def Statis_analisis(excel_file):
    # 1) Leer el Excel
    
    # Ajusta el nombre si tu archivo es diferente
    df = pd.read_excel(excel_file)
    # Esperamos cols: 
    # [indexA, indexB, area_ratio, street_pattern, p_walk, p_tpc, p_sitva, p_auto, p_moto, p_taxi, p_bike]
    print("Columnas df:", df.columns.tolist(), "\n")
    print(df.head(), "\n")

    # 2) Exploración inicial
    # 2a) Cuántos polígonos hay en cada Street Pattern:
    print("=== Conteo por Street Pattern ===")
    print(df["street_pattern"].value_counts())
    print()

    # 2b) Estadísticos descriptivos por street_pattern
    mobility_cols = ["p_walk","p_tpc","p_sitva","p_auto","p_moto","p_taxi","p_bike"]
    desc = df.groupby("street_pattern")[mobility_cols].describe().T
    print("=== Estadísticos descriptivos de las proporciones, por Street Pattern ===")
    print(desc)
    print()

    # 3) ANOVA / Kruskal-Wallis por variable
    #    Queremos ver si p_auto (por ej.) difiere segun street_pattern
    def anova_test(values_by_group):
        # values_by_group => lista de Series, una por cada StreetPattern
        return f_oneway(*values_by_group)

    def kruskal_test(values_by_group):
        return kruskal(*values_by_group)

    patterns = df["street_pattern"].unique()
    patterns_str = ", ".join(patterns)
    print(f"Street Patterns en el dataset: {patterns_str}\n")

    print("=== ANOVA / Kruskal para cada proporción de movilidad ===")
    for col in mobility_cols:
        groups = []
        for p in patterns:
            vals = df.loc[df["street_pattern"] == p, col].dropna()
            groups.append(vals)
        # ANOVA
        f_val, p_anova = anova_test(groups)
        # Kruskal
        h_val, p_kruskal = kruskal_test(groups)
        print(f"> {col.upper()} => ANOVA F={f_val:.3f} p={p_anova:.5g}, Kruskal H={h_val:.3f} p={p_kruskal:.5g}")
    print()

    # 4) Boxplots para visual
    for col in mobility_cols:
        sns.boxplot(x="street_pattern", y=col, data=df)
        plt.title(f"Distribución de {col} por Street Pattern")
        plt.tight_layout()
        plt.savefig(f"boxplot_{col}.png", dpi=150)
        plt.close()
    print("Boxplots guardados (boxplot_{variable}.png).")

    # 5) Correlación con One-Hot Encoding
    #    Creamos dummies => pattern_cul_de_sac, pattern_gridiron, etc.
    dummies = pd.get_dummies(df["street_pattern"], prefix="pattern")
    # Unimos con mobility_cols
    corr_df = pd.concat([df[mobility_cols], dummies], axis=1)
    # Spearman correlation
    corr_matrix = corr_df.corr(method="spearman")

    print("=== Matriz de correlación (Spearman) entre proporciones y dummies de Street Pattern ===")
    print(corr_matrix, "\n")

    # Heatmap
    plt.figure(figsize=(9,7))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu", vmin=-1, vmax=1)
    plt.title("Matriz de Correlación (Spearman)")
    plt.tight_layout()
    plt.savefig("heatmap_correlation.png", dpi=150)
    plt.close()
    print("Heatmap de correlaciones guardado en heatmap_correlation.png\n")

    # 6) REGRESIÓN LOGÍSTICA MULTINOMIAL
    #    Podríamos modelar street_pattern como variable dependiente,
    #    y p_walk, p_auto, etc. como predictores.
    #    Ejemplo con statsmodels MNLogit.
    #    Convertimos street_pattern a categorical codes 0,1,2,3
    df["pattern_code"] = df["street_pattern"].astype("category").cat.codes
    # Por ej. cul_de_sac=0, gridiron=1, organico=2, hibrido=3 (orden alfabético)

    # X = p_walk, p_auto, etc.
    X = df[mobility_cols]
    y = df["pattern_code"]

    # statsmodels MNLogit requiere agregarnos una constante
    X_ = sm.add_constant(X, prepend=True)
    # Ajuste
    mnlogit_model = sm.MNLogit(y, X_)
    mnlogit_result = mnlogit_model.fit(method='newton', maxiter=100, disp=0)

    print("=== Regresión Logística Multinomial (Street Pattern ~ proporciones) ===")
    print(mnlogit_result.summary())
    print("\nInterpretación: Coeficientes que comparan cada categoría vs la base (cul_de_sac si code=0).")
    print("p-values indican si la proporción p_auto, etc. es significativa para distinguir el pattern.\n")

# if __name__ == "__main__":
#     excel_file = "Poligonos_Medellin/Resultados/Statics_Results/URBAN_AND_RURAL/Poligonos_Clasificados_Movilidad_Urban_and_Rural.xlsx"
#     excel_file = "Poligonos_Medellin/Resultados/Statics_Results/URBAN/Poligonos_Clasificados_Movilidad_URBANO.xlsx"
#    Statis_analisis(excel_file)

def filter_periphery_polygons(in_geojson, out_geojson, area_threshold=5.0):
    """
    Lee un GeoJSON (in_geojson), elimina polígonos con área >= area_threshold (km²),
    y guarda un nuevo GeoJSON en out_geojson con los polígonos filtrados.
    Retorna un GeoDataFrame con el resultado.

    Parámetros:
    -----------
    in_geojson : ruta al archivo GeoJSON original.
    out_geojson: ruta donde se guardará el GeoJSON filtrado.
    area_threshold: float, umbral de área en km²; 
                    los polígonos con área >= threshold se considerarán "rurales" y se excluyen.

    Retorna:
    --------
    GeoDataFrame con los polígonos “urbanos” (área < area_threshold).
    """

    # 1. Cargar el GeoDataFrame
    gdf = gpd.read_file(in_geojson)
    print(f"Leído: {in_geojson} con {len(gdf)} polígonos totales.")

    # 2. Reproyectar a un sistema métrico para calcular área en km² (por ejemplo EPSG:3395 o 3857)
    #    EPSG:3395 (World Mercator) o 3857 (Pseudo Mercator). Ajusta según tu región si deseas mayor precisión.
    gdf_merc = gdf.to_crs(epsg=3395)

    # 3. Calcular área en km²
    gdf["area_km2"] = gdf_merc.geometry.area / 1e6

    # 4. Filtrar
    mask_urban = gdf["area_km2"] < area_threshold
    gdf_filtered = gdf[mask_urban].copy()
    print(f"Se excluyen {len(gdf) - len(gdf_filtered)} polígonos por ser >= {area_threshold} km².")

    # 5. Guardar como GeoJSON nuevo
    #    (si no deseas la columna "area_km2" en el resultado, la dropeas antes)
    gdf_filtered.drop(columns=["area_km2"], inplace=True)
    gdf_filtered.to_file(out_geojson, driver="GeoJSON")
    print(f"Archivo filtrado guardado en: {out_geojson} con {len(gdf_filtered)} polígonos.\n")

    return gdf_filtered


# gdf_filtrado = filter_periphery_polygons(
#     in_geojson="Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA.geojson",
#     out_geojson="Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson",
#     area_threshold=5.0  # Ajusta a tu criterio
# )


# # Graph of visualization of medellin with its respective polygons
# geojson_file_filtered = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson "
# plot_road_network_from_geojson(geojson_file_filtered, network_type='drive', simplify=True)






#  =============================================================================
# ------ CALCULO PARA MALLA SIMPLIFICADA DE AREA URBANA MEDELLIN ANTIOQUIA -----------
#  =============================================================================

# # Ejemplo de uso
# if __name__ == "__main__":
#     geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson"
#     stats_txt = "Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt"

#     # 1. Cargar stats desde .txt
#     stats_dict = load_polygon_stats_from_txt(stats_txt)

#     # 2. Generar PNG con cada fila = (idx, 0)
#     plot_polygons_classification_png(
#         geojson_path=geojson_file,
#         stats_dict=stats_dict,
#         classify_func=classify_polygon,
#         output_png="polygon_classification_URBAN_MESH.png"
#     )


# # ================== EJEMPLO DE USO ==================
# if __name__ == "__main__":
#     # Define las rutas a tus archivos
#     geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson"
#     stats_txt = "Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt"
#     stats_dict = load_polygon_stats_from_txt(stats_txt)
    

#     plot_street_patterns_classification(
#         geojson_path=geojson_file,
#         classify_func=classify_polygon,
#         stats_dict=stats_dict,
#         place_name="Medellin",
#         network_type="drive",
#         output_folder="Graphs_Cities",
#         simplify=False
#     )













def prepare_mobility_data(
    stats_txt="Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt", 
    matches_csv="Poligonos_Medellin/Resultados/Matchs_A_B/matches_by_area.csv", 
    shpB="Poligonos_Medellin/eod_gen_trips_mode.shp", 
    geojsonA="Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson"
 ):
    """
    Prepara un DataFrame de movilidad a partir de múltiples fuentes de datos.
    
    Parámetros:
    -----------
    stats_txt : str, ruta al archivo de estadísticas de polígonos
    matches_csv : str, ruta al CSV de emparejamientos A-B
    shpB : str, ruta al shapefile de movilidad
    geojsonA : str, ruta al GeoJSON de polígonos urbanos
    
    Retorna:
    --------
    DataFrame con datos de movilidad y características de polígonos
    """
    # 1) Cargar stats de polígonos
    stats_dict = load_polygon_stats_from_txt(stats_txt)
    print(f"Cargadas stats para {len(stats_dict)} polígonos (subpolígono).")

    # 2) Cargar CSV de emparejamientos A-B
    df_matches = pd.read_csv(matches_csv)  # [indexA, indexB, area_ratio]
    print("Muestra df_matches:\n", df_matches.head(), "\n")

    # 3) Leer shapefile/GeoDataFrame B (movilidad)
    gdfB = gpd.read_file(shpB)
    print("Columnas B:", gdfB.columns)

    # 4) Leer GeoJSON A "URBANO"
    gdfA = gpd.read_file(geojsonA)
    print(f"Leídos {len(gdfA)} polígonos en GeoJSON A URBANO.")

    # 5) Armar DataFrame final
    final_rows = []
    for i, row in df_matches.iterrows():
        idxA = row["indexA"]
        idxB = row["indexB"]
        ratio = row["area_ratio"]

        # Obtener estadísticas del polígono
        key_stats = (idxA, 0)
        poly_stats = stats_dict.get(key_stats, {})
        pattern = classify_polygon(poly_stats)

        # Extraer movilidad de B
        rowB = gdfB.loc[idxB]

        # Variables proporcionales de movilidad
        mobility_columns = [
            "p_walk", "p_tpc", "p_sitva", 
            "p_auto", "p_moto", "p_taxi", "p_bike"
        ]
        
        mobility_data = {col: rowB.get(col, 0) for col in mobility_columns}

        # Construir fila de datos
        row_data = {
            "indexA": idxA,
            "indexB": idxB,
            "area_ratio": ratio,
            "street_pattern": pattern,
            **mobility_data
        }

        final_rows.append(row_data)

    # Crear DataFrame final
    df_final = pd.DataFrame(final_rows)
    df_final = df_final[[
        "indexA", "indexB", "area_ratio", "street_pattern", 
        "p_walk", "p_tpc", "p_sitva", 
        "p_auto", "p_moto", "p_taxi", "p_bike"
    ]]

    # Guardar como CSV
    output_path = "Poligonos_Medellin/Resultados/mobility_data.csv"
    df_final.to_csv(output_path, index=False)
    print(f"Datos de movilidad guardados en {output_path}")

    return df_final

def enhanced_polygon_clustering_visualization(df_merged, geojson_file, mobility_data):
    """
    Realiza visualizaciones de clusters de polígonos utilizando match_polygons_by_area.
    """
    # Ruta base para guardar resultados
    clustering_dir = "Poligonos_Medellin/Resultados/disaggregated measures/clustering"
    os.makedirs(clustering_dir, exist_ok=True)
    
    # Métricas de movilidad
    mobility_metrics = [
        'p_walk', 'p_tpc', 'p_sitva', 'p_auto', 'p_moto', 'p_taxi', 'p_bike'
    ]
    
    # Cargar GeoJSON
    gdf = gpd.read_file(geojson_file)
    
    # Preparar diccionario para resultados
    clustering_geojsons = {}
    
    # Procesar cada métrica de movilidad
    for mobility_metric in mobility_metrics:
        # Columna de cluster
        cluster_column = f'cluster_{mobility_metric}'
        
        # Verificar que la columna exista
        if cluster_column not in df_merged.columns:
            print(f"ADVERTENCIA: No se encontró columna {cluster_column}")
            continue
        
        # Crear copia del GeoJSON
        gdf_clusters = gdf.copy()
        
        # Asegurar correspondencia correcta de polígonos
        cluster_map = {}
        for _, row in mobility_data.iterrows():
            indexA, indexB = row['indexA'], row['indexB']
            
            # Buscar el cluster correspondiente en df_merged
            cluster_value = df_merged[
                (df_merged['indexA'] == indexA) & 
                (df_merged[cluster_column].notna())
            ][cluster_column].values
            
            if len(cluster_value) > 0:
                cluster_map[indexB] = cluster_value[0]
        
        # Asignar clusters al GeoDataFrame
        gdf_clusters[cluster_column] = gdf_clusters.index.map(cluster_map)
        
        # Visualización de clusters
        plt.figure(figsize=(15, 10))
        
        # Plotear clusters
        gdf_clusters.plot(column=cluster_column, 
                           cmap='viridis', 
                           edgecolor='black', 
                           linewidth=0.5, 
                           legend=True, 
                           missing_kwds={'color': 'lightgrey'})
        
        plt.title(f'Clusters de Polígonos - {mobility_metric}')
        plt.axis('off')
        plt.tight_layout()
        
        # Guardar mapa de clusters
        plt.savefig(os.path.join(clustering_dir, f'polygon_clusters_map_{mobility_metric}.png'), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()
        
        # Almacenar en diccionario
        clustering_geojsons[mobility_metric] = gdf_clusters
    
    return clustering_geojsons

def polygon_detailed_statistical_analysis(polygon_stats_dict, mobility_data, geojson_file):
    """
    Realiza un análisis estadístico detallado de polígonos usando métricas originales.
    """
    # Ruta base para guardar resultados
    output_dir = "Poligonos_Medellin/Resultados/disaggregated measures"
    clustering_dir = os.path.join(output_dir, "clustering")
    os.makedirs(clustering_dir, exist_ok=True)

    # Preparar DataFrame con métricas de polígonos
    polygon_metrics = []
    for (poly_id, subpoly), stats_dict in polygon_stats_dict.items():
        poly_metrics = stats_dict.copy()
        poly_metrics['poly_id'] = poly_id
        poly_metrics['subpoly'] = subpoly
        polygon_metrics.append(poly_metrics)
    
    df_polygon_metrics = pd.DataFrame(polygon_metrics)
    
    # Combinar métricas de polígonos con datos de movilidad
    df_merged = pd.merge(df_polygon_metrics, mobility_data, left_on=['poly_id'], right_on=['indexA'])
    
    # Métricas de polígono para análisis
    structural_metrics = [
        'n', 'm', 'k_avg', 'edge_length_total', 'edge_length_avg', 
        'streets_per_node_avg', 'intersection_count', 'street_length_total', 
        'street_segment_count', 'street_length_avg', 'circuity_avg', 
        'intersection_density_km2', 'street_density_km2', 'area_km2'
    ]
    
    # Métricas de movilidad
    mobility_metrics = [
        'p_walk', 'p_tpc', 'p_sitva', 'p_auto', 'p_moto', 'p_taxi', 'p_bike'
    ]
    
    # Inicializar diccionario para almacenar resultados de clustering
    all_clustering_results = {}
        
    for mobility_metric in mobility_metrics:
        # Preparar características para clustering
        clustering_features = structural_metrics + [mobility_metric]
        
        # Escalar todas las características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_merged[clustering_features])
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42).fit(X_scaled)
        
        # Crear columna de cluster en df_merged
        cluster_column = f'cluster_{mobility_metric}'
        df_merged[cluster_column] = kmeans.labels_
        
        # Perfiles de movilidad por cluster
        cluster_mobility_profiles = df_merged.groupby(cluster_column)[mobility_metrics].mean()
        print(f"Perfiles de Movilidad por Cluster para {mobility_metric}:")
        print(cluster_mobility_profiles)
        
        # Visualización de clusters
        plt.figure(figsize=(10, 6))
        scatter_colors = ['blue', 'green', 'red', 'purple']
        for i in range(4):
            cluster_data = df_merged[df_merged[cluster_column] == i]
            plt.scatter(cluster_data['street_density_km2'], cluster_data[mobility_metric], 
                        label=f'Cluster {i}', color=scatter_colors[i], alpha=0.7)
        
        plt.xlabel('Densidad de Calles (km²)')
        plt.ylabel(f'Proporción de Viajes en {mobility_metric}')
        plt.title(f'Clusters de Polígonos - {mobility_metric}')
        plt.legend()
        plt.tight_layout()
        
        # Guardar imagen de clustering
        plt.savefig(os.path.join(clustering_dir, f'polygon_clusters_{mobility_metric}.png'))
        plt.close()
        
        # Guardar datos de clusters
        df_merged[['poly_id', cluster_column] + structural_metrics + mobility_metrics].to_csv(
            os.path.join(clustering_dir, f'polygon_cluster_data_{mobility_metric}.csv'), 
            index=False
        )
        
        # Guardar perfiles de movilidad por cluster
        cluster_mobility_profiles.to_csv(
            os.path.join(clustering_dir, f'cluster_mobility_profiles_{mobility_metric}.csv')
        )
        
        # Almacenar resultados
        all_clustering_results[mobility_metric] = {
            'cluster_mobility_profiles': cluster_mobility_profiles,
            'cluster_labels': df_merged[cluster_column]
        }
    
    # Visualización final de clusters usando GeoJSON
    gdf_clusters = enhanced_polygon_clustering_visualization(
        df_merged, 
        geojson_file, 
        mobility_data
    )
    
    return {
        'all_clustering_results': all_clustering_results,
        'merged_dataframe': df_merged
    }

# # Ejemplo de uso 
# geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson"
# stats_txt = "Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt"
# df_mobility = prepare_mobility_data()
# stats_dict = load_polygon_stats_from_txt(stats_txt)
# results = polygon_detailed_statistical_analysis(stats_dict, df_mobility,geojson_file)




def calculate_angle_features(G):
    """
    Calcula características basadas en ángulos de los segmentos viales
    """
    # Verificar si el grafo es válido y tiene nodos con coordenadas
    if G is None or G.number_of_nodes() == 0:
        return 0, 0, 0, 0
    
    # Lista para almacenar ángulos en intersecciones
    intersection_angles = []
    
    # Contador de segmentos ortogonales (aproximadamente 90°)
    ortho_segments = 0
    
    # Procesar cada nodo con más de 2 conexiones (intersecciones)
    for node, degree in G.degree():
        if degree > 2:  # Es una intersección
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 1:
                # Verificar que el nodo actual y vecinos tienen coordenadas
                valid_coords = True
                for n in [node] + neighbors:
                    if 'x' not in G.nodes[n] or 'y' not in G.nodes[n]:
                        valid_coords = False
                        break
                
                if not valid_coords:
                    continue
                
                # Calcular ángulos entre segmentos conectados a esta intersección
                node_angles = []
                for n1 in neighbors:
                    x1, y1 = G.nodes[node]['x'], G.nodes[node]['y']
                    x2, y2 = G.nodes[n1]['x'], G.nodes[n1]['y']
                    
                    # Calcular el ángulo del segmento respecto al eje horizontal
                    angle = degrees(atan2(y2 - y1, x2 - x1)) % 360
                    node_angles.append(angle)
                
                # Calcular ángulos entre pares de segmentos en esta intersección
                for i in range(len(node_angles)):
                    for j in range(i+1, len(node_angles)):
                        # Calcular la diferencia angular
                        angle_diff = abs(node_angles[i] - node_angles[j]) % 180
                        # Normalizar para obtener el ángulo menor
                        if angle_diff > 90:
                            angle_diff = 180 - angle_diff
                        
                        intersection_angles.append(angle_diff)
                        
                        # Contar si es una intersección aproximadamente ortogonal (90° ± 10°)
                        if 80 <= angle_diff <= 100:
                            ortho_segments += 1
    
    # Evitar división por cero
    if not intersection_angles:
        return 0, 0, 0, 0
    
    # Características basadas en ángulos
    mean_angle = np.mean(intersection_angles)
    std_angle = np.std(intersection_angles)
    
    # Proporción de intersecciones ortogonales
    ortho_proportion = ortho_segments / len(intersection_angles) if len(intersection_angles) > 0 else 0
    
    # Coeficiente de variación para ángulos (normaliza la desviación estándar)
    cv_angle = std_angle / mean_angle if mean_angle > 0 else 0
    
    return mean_angle, std_angle, ortho_proportion, cv_angle

def calculate_dead_end_features(G):
    """
    Calcula características relacionadas con calles sin salida y cul-de-sacs
    """
    # Verificar si el grafo es válido
    if G is None or G.number_of_nodes() == 0:
        return 0, 0
    
    # Contar nodos de grado 1 (calles sin salida)
    dead_ends = [node for node, degree in G.degree() if degree == 1]
    dead_end_count = len(dead_ends)
    
    # Total de nodos
    total_nodes = G.number_of_nodes()
    
    # Proporción de calles sin salida
    dead_end_ratio = dead_end_count / total_nodes if total_nodes > 0 else 0
    
    # Analizar las distancias entre calles sin salida
    distances = []
    if len(dead_ends) > 1:
        for i, d1 in enumerate(dead_ends):
            if 'x' not in G.nodes[d1] or 'y' not in G.nodes[d1]:
                continue
                
            x1, y1 = G.nodes[d1]['x'], G.nodes[d1]['y']
            for d2 in dead_ends[i+1:]:
                if 'x' not in G.nodes[d2] or 'y' not in G.nodes[d2]:
                    continue
                    
                x2, y2 = G.nodes[d2]['x'], G.nodes[d2]['y']
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                distances.append(dist)
    
    # Coeficiente de variación de distancias entre calles sin salida
    if distances:
        mean_dist = np.mean(distances)
        if mean_dist > 0:
            cv_distances = np.std(distances) / mean_dist
        else:
            cv_distances = 0
    else:
        cv_distances = 0
    
    return dead_end_ratio, cv_distances

def procesar_poligonos_y_generar_grafos(gdf):
    """
    Procesa polígonos y genera grafos de red vial para cada uno.
    
    Args:
        gdf: GeoDataFrame con polígonos a procesar
        
    Returns:
        graph_dict: Diccionario con los grafos generados (clave: poly_id, valor: grafo)
    """
    print("Procesando polígonos y generando grafos de red vial...")
    graph_dict = {}  # Diccionario para almacenar los grafos
    
    # Verificar versión de OSMnx para determinar los parámetros correctos
    print(f"Versión de OSMnx: {ox.__version__}")
    total_polygons = len(gdf)

    # Procesar cada polígono
    for idx, row in gdf.iterrows():
        

        try:
            poly_id = row['poly_id'] if 'poly_id' in gdf.columns else str(idx)
            geometry = row.geometry
            # print(f"Procesando polígono {idx+1}/{total_polygons} (ID: {poly_id})")
            # Asegurarnos de que el polígono es válido
            if geometry is None or not geometry.is_valid:
                print(f"Polígono {poly_id} no válido, omitiendo")
                continue
            
            # Intentar obtener la red vial del polígono
            try:
                # Obtener la red vial del polígono adaptándonos a la versión de OSMnx
                try:
                    # Primero intentamos con el parámetro clean_periphery
                    G = ox.graph_from_polygon(geometry, network_type='drive', simplify=True, clean_periphery=True)
                except TypeError:
                    # Si falla, probamos sin ese parámetro
                    G = ox.graph_from_polygon(geometry, network_type='drive', simplify=True)
                
                # Si el grafo está vacío, omitir este polígono
                if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                    print(f"Polígono {poly_id} generó un grafo vacío, omitiendo")
                    continue
                
                # Añadir proyección para asegurar que tenemos coordenadas x,y
                G = ox.project_graph(G)
                
                # Almacenar el grafo en el diccionario
                graph_dict[poly_id] = G
                
            except Exception as e:
                print(f"Error al obtener grafo para polígono {poly_id}: {e}")
                continue
                
        except Exception as e:
            print(f"Error general procesando polígono {idx}: {e}")
            continue

    print(f"Procesamiento completado. Se generaron {len(graph_dict)} grafos válidos.")
    
    return graph_dict

def preprocess_to_dimensionless(X_array, feature_names):
    """
    Transforma todas las variables a formas adimensionales sin usar valores de referencia fijos,
    sino utilizando estadísticas de los propios datos.
    
    Parámetros:
    - X_array: Array NumPy con los datos originales
    - feature_names: Lista con los nombres de las características
    
    Retorna:
    - X_preprocessed: Array NumPy con los datos pre-procesados completamente adimensionales
    """
    
    # Crear una copia para no modificar los datos originales
    X_preprocessed = X_array.copy()
    
    # Mapeo de cada característica a su tipo para pre-procesamiento específico
    feature_types = {
        # Densidades (con unidades km^-2)
        'edge_length_density': 'density',
        'street_density_km2': 'density',
        'node_density_km2': 'density',
        'edge_density_km2': 'density',
        'intersection_density_km2': 'density',
        'segment_density_km2': 'density',
        
        # Longitudes (con unidades de metros)
        'edge_length_avg': 'length',
        'street_length_avg': 'length',
        
        # Variables ya adimensionales
        'k_avg': 'dimensionless',
        'streets_per_node_avg': 'dimensionless',
        'network_connectivity_index': 'dimensionless',
        'circuity_avg': 'circuity',
        'orthogonal_proportion': 'dimensionless',
        'angle_coefficient_variation': 'dimensionless',
        'dead_end_ratio': 'dimensionless',
        'cv_dead_end_distances': 'dimensionless',
        
        # Ángulos (con unidades de grados)
        'mean_intersection_angle': 'angle',
        'std_intersection_angle': 'angle_deviation'
    }
    
    # Separar variables por tipo
    density_indices = [i for i, name in enumerate(feature_names) if feature_types.get(name) == 'density']
    length_indices = [i for i, name in enumerate(feature_names) if feature_types.get(name) == 'length']
    angle_indices = [i for i, name in enumerate(feature_names) if feature_types.get(name) == 'angle']
    deviation_indices = [i for i, name in enumerate(feature_names) if feature_types.get(name) == 'angle_deviation']
    circuity_indices = [i for i, name in enumerate(feature_names) if feature_types.get(name) == 'circuity']
    
    # 1. Transformar densidades: dividir por la densidad máxima o mediana de cada tipo
    if density_indices:
        for i in density_indices:
            # Usar percentil 95 como referencia para evitar outliers extremos
            reference = np.percentile(X_preprocessed[:, i][X_preprocessed[:, i] > 0], 95)
            if reference > 0:
                X_preprocessed[:, i] = X_preprocessed[:, i] / reference
    
    # 2. Transformar longitudes: dividir por la longitud máxima o mediana
    if length_indices:
        for i in length_indices:
            # Usar percentil 95 como referencia para evitar outliers extremos
            reference = np.percentile(X_preprocessed[:, i][X_preprocessed[:, i] > 0], 95)
            if reference > 0:
                X_preprocessed[:, i] = X_preprocessed[:, i] / reference
    
    # 3. Transformar ángulos: expresar como fracción de un círculo completo
    if angle_indices:
        for i in angle_indices:
            X_preprocessed[:, i] = X_preprocessed[:, i] / 360.0
    
    # 4. Transformar desviaciones de ángulos: expresar como fracción de un ángulo recto
    if deviation_indices:
        for i in deviation_indices:
            X_preprocessed[:, i] = X_preprocessed[:, i] / 90.0
    
    # 5. Transformar circuidad: restar 1.0 para que el ideal sea 0
    if circuity_indices:
        for i in circuity_indices:
            X_preprocessed[:, i] = X_preprocessed[:, i] - 1.0
    
    return X_preprocessed

def prepare_clustering_features_improved(stats_dict, graph_dict):  
    """
    Prepara características para clustering con normalización por área para medidas absolutas
    y conservación de métricas relativas, incluyendo características de ángulos y calles sin salida.
    
    Parámetros:
    - stats_dict: Diccionario con estadísticas por polígono
    - graph_dict: Diccionario con los grafos NetworkX por polígono (donde la clave es poly_id)
    """
      
    feature_names = [
        'edge_length_density',        # Longitud de enlaces por km² (normalizada)
        'street_density_km2',         # Longitud de calles por km² (ya normalizada)
        'node_density_km2',           # Densidad de nodos por km² (nueva)
        'edge_density_km2',           # Densidad de enlaces por km² (nueva)
        'k_avg',                      # Grado promedio (ya relativa)
        'edge_length_avg',            # Longitud promedio de enlaces (ya relativa)
        'streets_per_node_avg',       # Promedio de calles por nodo (ya relativa)
        'intersection_density_km2',   # Densidad de intersecciones por km² (ya normalizada)
        'segment_density_km2',        # Densidad de segmentos de calle por km² (nueva)
        'street_length_avg',          # Longitud promedio de calle (ya relativa)
        'circuity_avg',               # Circuidad promedio (ya relativa)
        'network_connectivity_index', # Índice de conectividad (ya relativa)
        'mean_intersection_angle',    # Ángulo promedio de intersección (nueva)
        'std_intersection_angle',     # Desviación estándar de ángulos (nueva)
        'orthogonal_proportion',      # Proporción de ángulos ortogonales (nueva) 
        'angle_coefficient_variation',# Coeficiente de variación de ángulos (nueva)
        'dead_end_ratio',             # Ratio de calles sin salida (nueva)
        'cv_dead_end_distances'       # Coef. de variación de distancias entre calles sin salida (nueva)
    ]

       # Crear un diccionario para mapear entre los diferentes formatos de ID
    id_map = {}
    for graph_id in graph_dict.keys():
        # Si es un string, intentar convertirlo a entero para comparar
        if isinstance(graph_id, str) and graph_id.isdigit():
            id_map[int(graph_id)] = graph_id
        # Añadir el ID original también
        id_map[graph_id] = graph_id
    
    X = []
    poly_ids = []
    
    
    
    # Extraer características
    for poly_id, stats in stats_dict.items():
        try:
            feature_vector = []
            
            # Verificar que tenemos área para normalizar
            area_km2 = stats.get('area_km2', 0)
            if area_km2 <= 0:
                print(f"Advertencia: área no válida para {poly_id}, omitiendo")
                continue
                
            # 1. edge_length_density (nueva: edge_length_total / area_km2)
            edge_length_total = stats.get('edge_length_total', 0)
            feature_vector.append(edge_length_total / area_km2)
            
            # 2. street_density_km2 (ya existe)
            feature_vector.append(stats.get('street_density_km2', 0))
            
            # 3. node_density_km2 (nueva: n / area_km2)
            n_nodes = stats.get('n', 0)
            feature_vector.append(n_nodes / area_km2)
            
            # 4. edge_density_km2 (nueva: m / area_km2)
            m_edges = stats.get('m', 0)
            feature_vector.append(m_edges / area_km2)
            
            # 5-7. Métricas relativas (se mantienen igual)
            feature_vector.append(stats.get('k_avg', 0))
            feature_vector.append(stats.get('edge_length_avg', 0))
            feature_vector.append(stats.get('streets_per_node_avg', 0))
            
            # 8. intersection_density_km2 (ya existe)
            feature_vector.append(stats.get('intersection_density_km2', 0))
            
            # 9. segment_density_km2 (nueva: street_segment_count / area_km2)
            segment_count = stats.get('street_segment_count', 0)
            feature_vector.append(segment_count / area_km2)
            
            # 10-11. Métricas relativas (se mantienen igual)
            feature_vector.append(stats.get('street_length_avg', 0))
            feature_vector.append(stats.get('circuity_avg', 0))
            
            # 12. network_connectivity_index (calcularlo según tu código original)
            # Calcular índice de conectividad a partir de streets_per_node_proportions o streets_per_node_counts
            connectivity_index = 0.0
            
            # Procesamiento de streets_per_node_proportions (mantener igual que en el código original)
            if 'streets_per_node_proportions' in stats:
                # Convertir a diccionario si es string
                if isinstance(stats['streets_per_node_proportions'], str):
                    try:
                        streets_prop = ast.literal_eval(stats['streets_per_node_proportions'])
                    except:
                        # Si falla la conversión, intentamos con streets_per_node_counts
                        if 'streets_per_node_counts' in stats and isinstance(stats['streets_per_node_counts'], str):
                            try:
                                streets_counts = ast.literal_eval(stats['streets_per_node_counts'])
                                # Convertir counts a proportions
                                total_nodes = sum(streets_counts.values())
                                streets_prop = {k: v/total_nodes for k, v in streets_counts.items()} if total_nodes else {1: 0.3, 3: 0.6, 4: 0.1}
                            except:
                                streets_prop = {1: 0.3, 3: 0.6, 4: 0.1}  # Valores predeterminados
                        else:
                            streets_prop = {1: 0.3, 3: 0.6, 4: 0.1}  # Valores predeterminados
                else:
                    streets_prop = stats['streets_per_node_proportions']
            
            # Alternativa: Si no tenemos proportions pero sí tenemos counts
            elif 'streets_per_node_counts' in stats:
                if isinstance(stats['streets_per_node_counts'], str):
                    try:
                        streets_counts = ast.literal_eval(stats['streets_per_node_counts'])
                        # Convertir counts a proportions
                        total_nodes = sum(streets_counts.values())
                        streets_prop = {k: v/total_nodes for k, v in streets_counts.items()} if total_nodes else {1: 0.3, 3: 0.6, 4: 0.1}
                    except:
                        streets_prop = {1: 0.3, 3: 0.6, 4: 0.1}  # Valores predeterminados
                else:
                    streets_counts = stats['streets_per_node_counts']
                    total_nodes = sum(streets_counts.values())
                    streets_prop = {k: v/total_nodes for k, v in streets_counts.items()} if total_nodes else {1: 0.3, 3: 0.6, 4: 0.1}
            else:
                # Si no tenemos ni proportions ni counts
                streets_prop = {1: 0.3, 3: 0.6, 4: 0.1}  # Valores predeterminados
            
            # Extraer proporciones por tipo de nodo
            dead_end_prop = streets_prop.get(1, 0.0)
            continuing_road_prop = streets_prop.get(2, 0.0)
            t_intersection_prop = streets_prop.get(3, 0.0)
            cross_intersection_prop = streets_prop.get(4, 0.0)
            
            # Fórmula para connectivity_index
            connectivity_index = (
                (1 * dead_end_prop) +
                (2 * continuing_road_prop) +
                (3 * t_intersection_prop) +
                (4 * cross_intersection_prop)
            ) / 4.0
            
            feature_vector.append(connectivity_index)
            
            # 13-18. Nuevas características de ángulos y calles sin salida
            # Obtener el grafo para este polígono
            G = None
            
            # Estrategia 1: Usar el ID directamente
            if poly_id in graph_dict:
                G = graph_dict[poly_id]
            
            # Estrategia 2: Si poly_id es una tupla, usar el primer elemento
            elif isinstance(poly_id, tuple) and len(poly_id) > 0:
                first_element = poly_id[0]
                if first_element in graph_dict:
                    G = graph_dict[first_element]
                elif str(first_element) in graph_dict:
                    G = graph_dict[str(first_element)]
                elif first_element in id_map:
                    G = graph_dict[id_map[first_element]]
            
            # Estrategia 3: Convertir a string
            elif str(poly_id) in graph_dict:
                G = graph_dict[str(poly_id)]
            
            if G is not None:
                # Calcular características de ángulos y calles sin salida
                mean_angle, std_angle, ortho_prop, cv_angle = calculate_angle_features(G)
                feature_vector.extend([mean_angle, std_angle, ortho_prop, cv_angle])
                
                dead_end_ratio, cv_dead_end = calculate_dead_end_features(G)
                feature_vector.extend([dead_end_ratio, cv_dead_end])
            else:
                # Si no tenemos grafo para este polígono, usar valores predeterminados
                print(f"Advertencia: No se encontró grafo para {poly_id}, usando valores predeterminados")
                feature_vector.extend([0, 0, 0, 0])  # Ángulos
                feature_vector.extend([0, 0])  
                        
            # Verificar valores atípicos en todo el vector
            if any(np.isnan(val) or np.isinf(val) for val in feature_vector):
                print(f"Advertencia: valores atípicos detectados para {poly_id}, omitiendo")
                continue
                
            # Si llegamos hasta aquí, añadimos el vector al conjunto de datos
            X.append(feature_vector)
            poly_ids.append(poly_id)
                
        except Exception as e:
            print(f"Error procesando {poly_id}: {e}")
            continue
    
    # Verificar que tenemos suficientes muestras
    if len(X) < 2:
        print(f"ADVERTENCIA: Solo se encontraron {len(X)} muestras válidas para clustering.")
    else:
        print(f"Se prepararon {len(X)} muestras válidas para clustering.")
    
    # Imprimir las características para verificación
    print("Características utilizadas:", feature_names)
    

    # Dentro de tu función, justo antes del return:
    X_array = np.array(X)

    # Aplicar pre-procesamiento para hacer todas las variables adimensionales
    X_preprocessed = preprocess_to_dimensionless(X_array, feature_names)
    
    # Luego aplicar StandardScaler para normalización estadística final
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_preprocessed)


    return X_normalized, poly_ids, feature_names

def find_optimal_k_improved(X_scaled, max_k=10, min_k=2):
    """
    Encuentra el número óptimo de clusters usando silhouette score,
    calinski-harabasz index y modularity score (para redes).
    """
   
    
    results = []
    
    for k in range(min_k, max_k + 1):
        # Usar KMeans++ para mejor inicialización
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
        labels = kmeans.fit_predict(X_scaled)
        
        # Calcular métricas
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        
        # Construir grafo basado en similitudes de clustering
        G = nx.Graph()
        for i in range(len(labels)):
            G.add_node(i, cluster=labels[i])
        
        # Agregar conexiones basadas en proximidad de clusters
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    G.add_edge(i, j)
        
        communities = {c: set(np.where(labels == c)[0]) for c in set(labels)}.values()
        mod_score = modularity(G, communities)
        
        results.append({
            'k': k,
            'silhouette': silhouette,
            'calinski': calinski,
            'modularity': mod_score,
            'inertia': kmeans.inertia_
        })
        
        print(f"K={k}, Silhouette={silhouette:.4f}, Calinski-Harabasz={calinski:.1f}, Modularity={mod_score:.4f}")
    
    # Visualizar resultados
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Silhouette score (mayor es mejor)
    axs[0, 0].plot([r['k'] for r in results], [r['silhouette'] for r in results], 'o-', color='blue')
    axs[0, 0].set_xlabel('Número de clusters')
    axs[0, 0].set_ylabel('Silhouette Score')
    axs[0, 0].set_title('Silhouette Score (mayor es mejor)')
    
    # Calinski-Harabasz (mayor es mejor)
    axs[0, 1].plot([r['k'] for r in results], [r['calinski'] for r in results], 'o-', color='green')
    axs[0, 1].set_xlabel('Número de clusters')
    axs[0, 1].set_ylabel('Calinski-Harabasz Index')
    axs[0, 1].set_title('Calinski-Harabasz Index (mayor es mejor)')
    
    # Modularity Score (mayor es mejor)
    axs[1, 0].plot([r['k'] for r in results], [r['modularity'] for r in results], 'o-', color='red')
    axs[1, 0].set_xlabel('Número de clusters')
    axs[1, 0].set_ylabel('Modularity Score')
    axs[1, 0].set_title('Modularity Score (mayor es mejor)')
    
    # Inertia (método del codo)
    axs[1, 1].plot([r['k'] for r in results], [r['inertia'] for r in results], 'o-', color='purple')
    axs[1, 1].set_xlabel('Número de clusters')
    axs[1, 1].set_ylabel('Inertia')
    axs[1, 1].set_title('Método del codo')
    
    plt.tight_layout()
    plt.savefig('Resultados/urbano_pattern_cluster/cluster_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Normalizar métricas para combinarlas
    sil_norm = [r['silhouette'] / max([r['silhouette'] for r in results]) for r in results]
    cal_norm = [r['calinski'] / max([r['calinski'] for r in results]) for r in results]
    mod_norm = [r['modularity'] / max([r['modularity'] for r in results]) for r in results]
    
    # Calcular score combinado
    combined_scores = [(s + c + m) / 3 for s, c, m in zip(sil_norm, cal_norm, mod_norm)]
    
    # Encontrar k óptimo por score combinado
    optimal_k_idx = np.argmax(combined_scores)
    optimal_k = results[optimal_k_idx]['k']
    
    print(f"\nK óptimo según score combinado: {optimal_k}")
    return optimal_k

# def optimal_clustering_improved(X, feature_names, n_clusters=None, use_pca=True, visualize=True):
#     """
#     Realiza clustering mejorado con KMeans y análisis de características importantes
#     """
    
#     # Eliminar filas con NaN o infinitos
#     valid_rows = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
#     X_clean = X[valid_rows]
#     if X_clean.shape[0] < X.shape[0]:
#         print(f"Eliminadas {X.shape[0] - X_clean.shape[0]} filas con valores no válidos")
    
#     # Normalizar características
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_clean)
    
#     # Reducción de dimensionalidad
#     if use_pca:
#         # Determinar número óptimo de componentes (varianza explicada > 0.95)
#         full_pca = PCA().fit(X_scaled)
#         cum_var = np.cumsum(full_pca.explained_variance_ratio_)
#         n_components = np.argmax(cum_var >= 0.95) + 1
#         n_components = max(2, min(n_components, X_scaled.shape[1]))
        
#         pca = PCA(n_components=n_components)
#         X_reduced = pca.fit_transform(X_scaled)
        
#         # Análisis de componentes principales
#         print(f"\nAnálisis PCA con {n_components} componentes:")
#         print(f"Varianza explicada: {pca.explained_variance_ratio_}")
#         print(f"Varianza total explicada: {sum(pca.explained_variance_ratio_):.4f}")
        
#         # Visualizar contribución de características a componentes
#         plt.figure(figsize=(12, 8))
#         components = pd.DataFrame(
#             pca.components_.T,
#             columns=[f'PC{i+1}' for i in range(n_components)],
#             index=feature_names
#         )
        
#         sns.heatmap(components, cmap='coolwarm', annot=True, fmt=".2f")
#         plt.title('Contribución de variables a componentes principales')
#         plt.tight_layout()
#         plt.savefig('Resultados/urbano_pattern_cluster/pca_components_contributions.png', dpi=300, bbox_inches='tight')
#         plt.close()
#     else:
#         X_reduced = X_scaled
    
#     # Encontrar número óptimo de clusters si no se proporciona
#     if n_clusters is None:
#         # Asumimos que la función find_optimal_k_improved está definida en otro lugar
#         n_clusters = find_optimal_k_improved(X_reduced, max_k=8, min_k=3)
    
#     print(f"\nRealizando clustering KMeans con {n_clusters} clusters")
    
#     # Usar KMeans con inicialización k-means++ y múltiples inicios
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, init='k-means++')
#     cluster_labels = kmeans.fit_predict(X_reduced)
    
#     # Analizar centros de clusters
#     if use_pca:
#         # Proyectar centros al espacio original
#         centers_pca = kmeans.cluster_centers_
#         centers_original = pca.inverse_transform(centers_pca)
#         centers_original = scaler.inverse_transform(centers_original)
#     else:
#         centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
#     # Crear DataFrame de centros
#     centers_df = pd.DataFrame(centers_original, columns=feature_names)
#     centers_df.index = [f'Cluster {i}' for i in range(n_clusters)]
    
#     print("\nCaracterísticas de los centros de clusters:")
#     print(centers_df)
    
#     # Analizar variables más discriminantes entre clusters
#     cluster_importance = {}
#     for feature in feature_names:
#         # Calcular varianza entre clusters para esta característica
#         values = centers_df[feature].values
#         variance = np.var(values)
#         max_diff = np.max(values) - np.min(values)
#         importance = variance * max_diff  # Ponderación por rango
#         cluster_importance[feature] = importance
    
#     sorted_features = sorted(cluster_importance.items(), key=lambda x: x[1], reverse=True)
    
#     print("\nCaracterísticas más importantes para diferenciar clusters:")
#     for feature, importance in sorted_features[:5]:
#         print(f"{feature}: {importance:.4f}")
    
#     # Visualizar clusters
#     if visualize:
#         # CORRECCIÓN: En lugar de usar t-SNE, que puede no preservar distancias globales,
#         # usar PCA para visualización si el número de características es alto
#         if X_reduced.shape[1] > 2:
#             # Para visualización, usamos PCA directamente desde los datos escalados
#             viz_pca = PCA(n_components=2)
#             X_viz = viz_pca.fit_transform(X_scaled)
            
#             plt.figure(figsize=(10, 8))
#             scatter = plt.scatter(X_viz[:, 0], X_viz[:, 1], c=cluster_labels, 
#                                  cmap='viridis', s=50, alpha=0.8)
            
#             # Transformar centros de clusters a 2D para visualización
#             if use_pca:
#                 # Primero a espacio escalado
#                 centers_scaled = scaler.transform(centers_original)
#                 # Luego proyectar a 2D con el mismo PCA de visualización
#                 centers_viz = viz_pca.transform(centers_scaled)
#             else:
#                 centers_viz = viz_pca.transform(kmeans.cluster_centers_)
            
#             # Mostrar centros en la visualización
#             plt.scatter(centers_viz[:, 0], centers_viz[:, 1], 
#                        c='red', s=200, alpha=0.8, marker='X')
            
#             # Añadir etiquetas de clusters
#             for i, (x, y) in enumerate(centers_viz):
#                 plt.annotate(f'Cluster {i}', (x, y), fontsize=12, 
#                              ha='center', va='center', color='white',
#                              bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
            
#             plt.colorbar(scatter, label='Cluster')
#             plt.title('Visualización de clusters usando PCA')
#             plt.xlabel('PC1')
#             plt.ylabel('PC2')
#             plt.tight_layout()
#             plt.savefig('Resultados/urbano_pattern_cluster/cluster_visualization_pca.png', dpi=300, bbox_inches='tight')
#             plt.close()
            
#             # Adicionalmente, podemos usar t-SNE como visualización complementaria
#             # pero con parámetros más adecuados
#             if X_clean.shape[0] > 5:  # Solo si hay suficientes datos
#                 perplexity = min(30, max(5, X_clean.shape[0] // 10))
#                 tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
#                            learning_rate='auto', init='pca')
#                 X_tsne = tsne.fit_transform(X_scaled)
                
#                 plt.figure(figsize=(10, 8))
#                 scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
#                                      c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
                
#                 plt.colorbar(scatter, label='Cluster')
#                 plt.title('Visualización de clusters usando t-SNE')
#                 plt.xlabel('t-SNE 1')
#                 plt.ylabel('t-SNE 2')
#                 plt.tight_layout()
#                 plt.savefig('Resultados/urbano_pattern_cluster/cluster_visualization_tsne_improved.png', dpi=300, bbox_inches='tight')
#                 plt.close()
#         else:
#             # Si ya tenemos 2 dimensiones, usar directamente
#             plt.figure(figsize=(10, 8))
#             scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
#                                  c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
            
#             plt.colorbar(scatter, label='Cluster')
#             plt.title('Visualización de clusters')
#             plt.xlabel('Dimensión 1')
#             plt.ylabel('Dimensión 2')
#             plt.tight_layout()
#             plt.savefig('Resultados/urbano_pattern_cluster/cluster_visualization_direct.png', dpi=300, bbox_inches='tight')
#             plt.close()
        
#         # Visualizar distribución de características más importantes por cluster
#         # Definir número de filas y columnas
        
#         number_of_graphs = len(feature_names)

#         # Calcular dinámicamente las filas y columnas
#         cols = 5  # Número fijo de columnas
#         rows = int(np.ceil(number_of_graphs / cols))  # Calcula cuántas filas son necesarias

#         # Crear la figura con el número adecuado de subgráficos
#         fig, axes = plt.subplots(rows, cols, figsize=(25, rows * 5))  # Altura ajustada dinámicamente
#         axes = axes.flatten()  # Aplanar en una lista 1D

#         # Crear DataFrame con datos originales y etiquetas de cluster
#         data_df = pd.DataFrame(X_clean, columns=feature_names)
#         data_df['cluster'] = cluster_labels

#         # Iterar sobre todas las características
#         for i, feature in enumerate(feature_names):
#             sns.boxplot(x='cluster', y=feature, data=data_df, ax=axes[i])
#             axes[i].set_title(f'Distribución de {feature}')
#             axes[i].set_xlabel('Cluster')
#             axes[i].set_ylabel(feature)

#         # Ocultar los ejes sobrantes si hay menos gráficos que subplots
#         for j in range(number_of_graphs, len(axes)):
#             fig.delaxes(axes[j])

#         # Ajustar el diseño
#         plt.tight_layout()
#         plt.savefig('Resultados/urbano_pattern_cluster/feature_distributions_by_cluster.png', dpi=300, bbox_inches='tight')
#         plt.close()

#     return n_clusters, cluster_labels, centers_df, sorted_features


def optimal_clustering_improved(X, feature_names, n_clusters=None, use_pca=True, 
                               pca_variance_threshold=0.95, max_pca_components=8, 
                               visualize=True, use_elbow_method=False):
    """
    Realiza clustering mejorado con KMeans y análisis de características importantes
    
    Parámetros:
    -----------
    X : array
        Datos de entrada
    feature_names : list
        Nombres de las características
    n_clusters : int, opcional
        Número de clusters. Si es None, se determina automáticamente
    use_pca : bool, por defecto True
        Si se debe usar PCA para reducción de dimensionalidad
    pca_variance_threshold : float, por defecto 0.95
        Umbral de varianza explicada acumulada para seleccionar componentes
    max_pca_components : int, por defecto 5
        Número máximo de componentes PCA a usar
    visualize : bool, por defecto True
        Si se deben generar visualizaciones
    use_elbow_method : bool, por defecto False
        Si se debe usar el método del codo para determinar componentes
    """
    
    # Eliminar filas con NaN o infinitos (tu código original)
    valid_rows = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    X_clean = X[valid_rows]
    if X_clean.shape[0] < X.shape[0]:
        print(f"Eliminadas {X.shape[0] - X_clean.shape[0]} filas con valores no válidos")
    
    # Normalizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Reducción de dimensionalidad
    if use_pca:
        # Determinar número óptimo de componentes
        full_pca = PCA().fit(X_scaled)
        cum_var = np.cumsum(full_pca.explained_variance_ratio_)
        
        # Visualizar la varianza explicada acumulada (scree plot)
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='-')
            plt.axhline(y=pca_variance_threshold, color='r', linestyle='--', 
                      label=f'Umbral ({pca_variance_threshold})')
            plt.title('Varianza explicada acumulada vs Número de componentes')
            plt.xlabel('Número de componentes')
            plt.ylabel('Varianza acumulada explicada')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig('Resultados/urbano_pattern_cluster/pca_variance_explained.png', dpi=300)
            plt.close()
        
        # Método del codo para PCA si se solicita
        if use_elbow_method:
            # Calcular "aceleración" de la curva de varianza
            npc = len(full_pca.explained_variance_ratio_)
            acceleration = np.diff(np.diff(cum_var)) + 0.001  # Evitar dividir por cero
            k_elbow = np.argmax(acceleration) + 1  # El punto donde la aceleración es máxima
            n_components = min(k_elbow + 1, max_pca_components)  # +1 porque los índices comienzan en 0
            
            if visualize:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, npc-1), acceleration, marker='o', linestyle='-')
                plt.axvline(x=k_elbow, color='r', linestyle='--', 
                          label=f'Punto de inflexión (k={k_elbow+1})')
                plt.title('Método del codo: Aceleración de la varianza explicada')
                plt.xlabel('Número de componentes')
                plt.ylabel('Aceleración de varianza')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig('Resultados/urbano_pattern_cluster/pca_elbow_method.png', dpi=300)
                plt.close()
        else:
            # Criterio basado en umbral de varianza
            n_components = np.argmax(cum_var >= pca_variance_threshold) + 1
        
        # Aplicar restricciones al número de componentes
        n_components = max(2, min(n_components, min(max_pca_components, X_scaled.shape[1])))
        
        print(f"\nSeleccionados {n_components} componentes PCA")
        print(f"Varianza explicada por estos componentes: {cum_var[n_components-1]:.4f}")
        
        # Realizar PCA con el número óptimo de componentes
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        
        # Análisis de componentes principales
        print(f"\nAnálisis PCA con {n_components} componentes:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {var:.4f} de varianza explicada")
        print(f"Varianza total explicada: {sum(pca.explained_variance_ratio_):.4f}")
   
        # Guardar información de los componentes PCA en un archivo de texto
        with open('Resultados/urbano_pattern_cluster/pca_analysis.txt', 'w') as f:
            f.write(f"ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)\n")
            f.write(f"======================================\n\n")
            f.write(f"Número de componentes seleccionados: {n_components}\n")
            f.write(f"Varianza total explicada: {sum(pca.explained_variance_ratio_):.4f}\n\n")
            
            f.write("VARIANZA EXPLICADA POR COMPONENTE:\n")
            for i, var in enumerate(pca.explained_variance_ratio_):
                f.write(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)\n")
            f.write("\n")
            
            f.write("CONTRIBUCIÓN DE VARIABLES A COMPONENTES:\n")
            # Crear un DataFrame con las contribuciones de las características a cada componente
            components_df = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=feature_names
            )
            
            # Para cada componente, listar las variables más influyentes
            for i in range(n_components):
                pc_name = f'PC{i+1}'
                f.write(f"\n{pc_name} - Explica {pca.explained_variance_ratio_[i]*100:.2f}% de la varianza:\n")
                
                # Ordenar características por su contribución absoluta a este componente
                component_contrib = components_df[pc_name].abs().sort_values(ascending=False)
                
                # Encontrar variables con mayor contribución (positiva y negativa)
                for feature, value in zip(components_df.index, components_df[pc_name]):
                    contribution = abs(value)
                    # Mostrar solo contribuciones significativas (ajustar umbral según necesidad)
                    if contribution > 0.2:  # Umbral arbitrario, ajustar según sea necesario
                        direction = "positiva" if value > 0 else "negativa"
                        f.write(f"  - {feature}: {value:.4f} (contribución {direction})\n")
            
            f.write("\n\nINTERPRETACIÓN DE COMPONENTES:\n")
            f.write("La interpretación de cada componente debe hacerse considerando las variables\n")
            f.write("con mayor contribución (positiva o negativa). Variables con contribuciones del\n")
            f.write("mismo signo están correlacionadas positivamente en ese componente, mientras que\n")
            f.write("variables con signos opuestos están correlacionadas negativamente.\n")
        
        # Visualizar contribución de características a componentes
        if visualize:
            plt.figure(figsize=(12, 8))
            components = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=feature_names
            )
            
            sns.heatmap(components, cmap='coolwarm', annot=True, fmt=".2f")
            plt.title('Contribución de variables a componentes principales')
            plt.tight_layout()
            plt.savefig('Resultados/urbano_pattern_cluster/pca_components_contributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    else:
        X_reduced = X_scaled
    
    # Encontrar número óptimo de clusters si no se proporciona
    if n_clusters is None:
        # Asumimos que la función find_optimal_k_improved está definida en otro lugar
        n_clusters = find_optimal_k_improved(X_reduced, max_k=8, min_k=3)
    
    print(f"\nRealizando clustering KMeans con {n_clusters} clusters")
    
    # Usar KMeans con inicialización k-means++ y múltiples inicios
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, init='k-means++')
    cluster_labels = kmeans.fit_predict(X_reduced)
    
    # Analizar centros de clusters
    if use_pca:
        # Proyectar centros al espacio original
        centers_pca = kmeans.cluster_centers_
        centers_original = pca.inverse_transform(centers_pca)
        centers_original = scaler.inverse_transform(centers_original)
    else:
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Crear DataFrame de centros
    centers_df = pd.DataFrame(centers_original, columns=feature_names)
    centers_df.index = [f'Cluster {i}' for i in range(n_clusters)]
    
    print("\nCaracterísticas de los centros de clusters:")
    print(centers_df)
    
    # Analizar variables más discriminantes entre clusters
    cluster_importance = {}
    for feature in feature_names:
        # Calcular varianza entre clusters para esta característica
        values = centers_df[feature].values
        variance = np.var(values)
        max_diff = np.max(values) - np.min(values)
        importance = variance * max_diff  # Ponderación por rango
        cluster_importance[feature] = importance
    
    sorted_features = sorted(cluster_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nCaracterísticas más importantes para diferenciar clusters:")
    for feature, importance in sorted_features[:5]:
        print(f"{feature}: {importance:.4f}")
    
    # Visualizar clusters
    if visualize:
        # CORRECCIÓN: En lugar de usar t-SNE, que puede no preservar distancias globales,
        # usar PCA para visualización si el número de características es alto
        if X_reduced.shape[1] > 2:
            # Para visualización, usamos PCA directamente desde los datos escalados
            viz_pca = PCA(n_components=2)
            X_viz = viz_pca.fit_transform(X_scaled)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_viz[:, 0], X_viz[:, 1], c=cluster_labels, 
                                 cmap='viridis', s=50, alpha=0.8)
            
            # Transformar centros de clusters a 2D para visualización
            if use_pca:
                # Primero a espacio escalado
                centers_scaled = scaler.transform(centers_original)
                # Luego proyectar a 2D con el mismo PCA de visualización
                centers_viz = viz_pca.transform(centers_scaled)
            else:
                centers_viz = viz_pca.transform(kmeans.cluster_centers_)
            
            # Mostrar centros en la visualización
            plt.scatter(centers_viz[:, 0], centers_viz[:, 1], 
                       c='red', s=200, alpha=0.8, marker='X')
            
            # Añadir etiquetas de clusters
            for i, (x, y) in enumerate(centers_viz):
                plt.annotate(f'Cluster {i}', (x, y), fontsize=12, 
                             ha='center', va='center', color='white',
                             bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
            
            plt.colorbar(scatter, label='Cluster')
            plt.title('Visualización de clusters usando PCA')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.tight_layout()
            plt.savefig('Resultados/urbano_pattern_cluster/cluster_visualization_pca.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Adicionalmente, podemos usar t-SNE como visualización complementaria
            # pero con parámetros más adecuados
            if X_clean.shape[0] > 5:  # Solo si hay suficientes datos
                perplexity = min(30, max(5, X_clean.shape[0] // 10))
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                           learning_rate='auto', init='pca')
                X_tsne = tsne.fit_transform(X_scaled)
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                     c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
                
                plt.colorbar(scatter, label='Cluster')
                plt.title('Visualización de clusters usando t-SNE')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.tight_layout()
                plt.savefig('Resultados/urbano_pattern_cluster/cluster_visualization_tsne_improved.png', dpi=300, bbox_inches='tight')
                plt.close()
        else:
            # Si ya tenemos 2 dimensiones, usar directamente
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                 c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
            
            plt.colorbar(scatter, label='Cluster')
            plt.title('Visualización de clusters')
            plt.xlabel('Dimensión 1')
            plt.ylabel('Dimensión 2')
            plt.tight_layout()
            plt.savefig('Resultados/urbano_pattern_cluster/cluster_visualization_direct.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Visualizar distribución de características más importantes por cluster
        # Definir número de filas y columnas
        
        number_of_graphs = len(feature_names)

        # Calcular dinámicamente las filas y columnas
        cols = 5  # Número fijo de columnas
        rows = int(np.ceil(number_of_graphs / cols))  # Calcula cuántas filas son necesarias

        # Crear la figura con el número adecuado de subgráficos
        fig, axes = plt.subplots(rows, cols, figsize=(25, rows * 5))  # Altura ajustada dinámicamente
        axes = axes.flatten()  # Aplanar en una lista 1D

        # Crear DataFrame con datos originales y etiquetas de cluster
        data_df = pd.DataFrame(X_clean, columns=feature_names)
        data_df['cluster'] = cluster_labels

        # Iterar sobre todas las características
        for i, feature in enumerate(feature_names):
            sns.boxplot(x='cluster', y=feature, data=data_df, ax=axes[i])
            axes[i].set_title(f'Distribución de {feature}')
            axes[i].set_xlabel('Cluster')
            axes[i].set_ylabel(feature)

        # Ocultar los ejes sobrantes si hay menos gráficos que subplots
        for j in range(number_of_graphs, len(axes)):
            fig.delaxes(axes[j])

        # Ajustar el diseño
        plt.tight_layout()
        plt.savefig('Resultados/urbano_pattern_cluster/feature_distributions_by_cluster.png', dpi=300, bbox_inches='tight')
        plt.close()

    return n_clusters, cluster_labels, centers_df, sorted_features


def urban_pattern_clustering(
    stats_dict, 
    graph_dict,
    classify_func, 
    geojson_file,
    n_clusters=None,
    output_dir="Resultados/urbano_pattern_cluster"
    ):
    """
    Versión mejorada para clustering de patrones urbanos
    """
    
    # Crear directorio para resultados
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar GeoDataFrame
    gdf = gpd.read_file(geojson_file)
    print(f"GeoDataFrame cargado con {len(gdf)} polígonos")
    
    # Preparar características mejoradas para clustering
    X, poly_ids, feature_names = prepare_clustering_features_improved(stats_dict, graph_dict)
    print(f"Características preparadas para {len(X)} polígonos con {len(feature_names)} variables")
    
    # Realizar clustering mejorado
    n_clusters, cluster_labels, centers_df, important_features = optimal_clustering_improved(
        X, feature_names, n_clusters=n_clusters
    )
    
    # Clasificación original de patrones
    original_patterns = []
    valid_poly_ids = []
    
    for poly_id in poly_ids:
        poly_stats = stats_dict[poly_id]
        category = classify_func(poly_stats)
        original_patterns.append(category)
        valid_poly_ids.append(poly_id)
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'poly_id': [pid[0] for pid in valid_poly_ids],
        'subpoly_id': [pid[1] for pid in valid_poly_ids],
        'original_pattern': original_patterns,
        'cluster': cluster_labels
    })
    
    # Análisis de relación entre patrones originales y clusters
    pattern_cluster_matrix = pd.crosstab(
        results_df['original_pattern'], 
        results_df['cluster'],
        normalize='columns'
    ) * 100
    
    print("\nDistribución de patrones por cluster (%):")
    print(pattern_cluster_matrix)
    
    # Visualizar matriz de patrones vs clusters como heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pattern_cluster_matrix, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title('Distribución de patrones urbanos por cluster (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pattern_cluster_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Determinar el patrón dominante para cada cluster
    cluster_dominant_pattern = {}
    for cluster in range(n_clusters):
        if cluster in pattern_cluster_matrix.columns:
            dominant_pattern = pattern_cluster_matrix[cluster].idxmax()
            percentage = pattern_cluster_matrix.loc[dominant_pattern, cluster]
            cluster_dominant_pattern[cluster] = (dominant_pattern, percentage)
    
    # Asignar nombres distintivos a los clusters
    cluster_names = {}
    named_patterns = set()
    
    for cluster, (pattern, percentage) in cluster_dominant_pattern.items():
        if pattern in named_patterns:
            # Si ya existe un cluster con este patrón, agregar un sufijo
            similar_clusters = [c for c, (p, _) in cluster_dominant_pattern.items() if p == pattern]
            suffix = len([c for c in similar_clusters if c < cluster]) + 1
            
            # Analizar diferencias con otros clusters del mismo patrón
            distinctive_features = []
            for feature, _ in important_features[:3]:
                feature_value = centers_df.loc[f'Cluster {cluster}', feature]
                avg_value = centers_df.loc[[f'Cluster {c}' for c in similar_clusters if c != cluster], feature].mean()
                if feature_value > avg_value * 1.2:
                    distinctive_features.append(f"alto_{feature.split('_')[0]}")
                elif feature_value < avg_value * 0.8:
                    distinctive_features.append(f"bajo_{feature.split('_')[0]}")
            
            if distinctive_features:
                cluster_names[cluster] = f"{pattern}_{distinctive_features[0]}"
            else:
                cluster_names[cluster] = f"{pattern}_{suffix}"
        else:
            cluster_names[cluster] = pattern
            named_patterns.add(pattern)
    
    # Filtrar y preparar GeoDataFrame para visualización
    poly_id_map = {pid[0]: i for i, pid in enumerate(valid_poly_ids)}
    valid_indices = [i for i in poly_id_map.keys() if i < len(gdf)]
    gdf_filtered = gdf.loc[valid_indices].copy()
    
    # Añadir resultados al GeoDataFrame
    gdf_filtered['original_pattern'] = gdf_filtered.index.map(
        lambda idx: results_df[results_df['poly_id'] == idx]['original_pattern'].values[0] 
        if idx in poly_id_map and any(results_df['poly_id'] == idx) else 'unknown'
    )
    
    gdf_filtered['cluster'] = gdf_filtered.index.map(
        lambda idx: results_df[results_df['poly_id'] == idx]['cluster'].values[0] 
        if idx in poly_id_map and any(results_df['poly_id'] == idx) else -1
    )
    
    gdf_filtered['cluster_name'] = gdf_filtered['cluster'].map(
        lambda c: cluster_names.get(c, f"cluster_{c}") if c != -1 else 'unknown'
    )
    
    # Definir esquema de colores mejorado para patrones urbanos
    # Usando colores distintivos y temáticos para cada tipo
    color_map = {
        'cul_de_sac': '#FF6B6B',   # Rojo para callejones sin salida
        'gridiron': '#006400',     # Verde oscuro para grid
        'organico': '#45B7D1',     # Azul para orgánico
        'hibrido': '#FDCB6E',      # Amarillo para híbrido
        'unknown': '#CCCCCC'       # Gris para desconocidos
    }
    
    # Añadir colores para nombres de cluster
    for cluster, name in cluster_names.items():
        pattern = name.split('_')[0]
        if pattern in color_map:
            base_color = np.array(mcolors.to_rgb(color_map[pattern]))
            # Ajustar color ligeramente para distinguir patrones similares
            if '_' in name and pattern in named_patterns:
                # Oscurecer o aclarar el color según el sufijo
                suffix = name.split('_')[1]
                if suffix.startswith('alto'):
                    # Más claro
                    adjusted_color = base_color + (1 - base_color) * 0.3
                elif suffix.startswith('bajo'):
                    # Más oscuro
                    adjusted_color = base_color * 0.7
                else:
                    # Alternar entre tonos
                    factor = int(suffix) * 0.15 if suffix.isdigit() else 0.2
                    adjusted_color = base_color + np.array([0, factor, -factor])
                
                # Recortar valores a rango válido
                adjusted_color = np.clip(adjusted_color, 0, 1)
                color_map[name] = mcolors.to_hex(adjusted_color)
            else:
                color_map[name] = color_map[pattern]
    
    # Visualización de comparación
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # Función para asignar colores
    def get_color(value, color_map):
        return color_map.get(value, '#CCCCCC')
    
    # Mapa de patrones originales
    gdf_filtered['color_pattern'] = gdf_filtered['original_pattern'].apply(
        lambda x: get_color(x, color_map)
    )
    
    gdf_filtered.plot(color=gdf_filtered['color_pattern'], 
                     ax=axes[0],
                     edgecolor='black', 
                     linewidth=0.5)
    
    # Crear leyenda para patrones
    pattern_legend_elements = [
        Patch(facecolor=color_map[pattern], edgecolor='black', label=pattern)
        for pattern in sorted(list(set(gdf_filtered['original_pattern'])))
        if pattern in color_map
    ]
    axes[0].legend(handles=pattern_legend_elements, loc='lower right', title="Patrones originales")
    axes[0].set_title('Patrones de calle teóricos', fontsize=14)
    axes[0].axis('off')
    
    # Mapa de clusters
    gdf_filtered['color_cluster'] = gdf_filtered['cluster_name'].apply(
        lambda x: get_color(x, color_map)
    )
    
    gdf_filtered.plot(color=gdf_filtered['color_cluster'], 
                     ax=axes[1],
                     edgecolor='black', 
                     linewidth=0.5)
    
    # Crear leyenda para clusters
    cluster_legend_elements = [
        Patch(facecolor=get_color(name, color_map), edgecolor='black', label=name)
        for name in sorted(list(set(gdf_filtered['cluster_name'])))
        if name != 'unknown'
    ]
    axes[1].legend(handles=cluster_legend_elements, loc='lower right', title="Clusters identificados")
    axes[1].set_title('Agrupación por características morfológicas', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'urban_pattern_comparison.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # Análisis de características por tipo de patrón urbano
    pattern_feature_summary = {}
    
    # Añadir valores normalizados de características al DataFrame
    feature_data = pd.DataFrame(X, columns=feature_names)
    feature_data['pattern'] = original_patterns
    feature_data['cluster'] = cluster_labels
    
    # Calcular estadísticas por patrón
    for pattern in set(original_patterns):
        pattern_data = feature_data[feature_data['pattern'] == pattern]
        pattern_feature_summary[pattern] = {
            feature: pattern_data[feature].mean() for feature in feature_names
        }
    
    # Crear DataFrame de resumen
    summary_df = pd.DataFrame(pattern_feature_summary).T
    
    # Guardar resultados en formato Excel
    with pd.ExcelWriter(os.path.join(output_dir, 'urban_pattern_analysis.xlsx')) as writer:
        # Hoja 1: Resumen de patrones y características
        summary_df.to_excel(writer, sheet_name='Pattern_Features')
        
        # Hoja 2: Matriz de confusión entre patrones y clusters
        pattern_cluster_matrix.to_excel(writer, sheet_name='Pattern_Cluster_Matrix')
        
        # Hoja 3: Características de centros de clusters
        centers_df.to_excel(writer, sheet_name='Cluster_Centers')
        
        # Hoja 4: Importancia de características
        pd.DataFrame(important_features, columns=['Feature', 'Importance']).to_excel(
            writer, sheet_name='Feature_Importance'
        )
    
    # Guardar GeoJSON con resultados
    gdf_filtered.to_file(
        os.path.join(output_dir, 'urban_patterns_clustered.geojson'),
        driver='GeoJSON'
    )
    
    return {
        'geodataframe': gdf_filtered,
        'cluster_centers': centers_df,
        'pattern_cluster_matrix': pattern_cluster_matrix,
        'cluster_names': cluster_names,
        'important_features': important_features,
        'n_clusters': n_clusters
    }




# # 1. Cargar el GeoJSON
# print("Cargando archivo GeoJSON...")
# geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson"
# gdf = gpd.read_file(geojson_file)
# print(f"GeoDataFrame cargado con {len(gdf)} polígonos")
# stats_txt = "Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt"
# graph_dict = procesar_poligonos_y_generar_grafos(gdf)


# stats_dict = load_polygon_stats_from_txt(stats_txt)
# resultados = urban_pattern_clustering(
#     stats_dict, 
#     graph_dict,
#     classify_polygon, 
#     geojson_file,
#     n_clusters= None # Automáticamente determinará el número óptimo
# )