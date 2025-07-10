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
plt.rcParams.update({'font.size': 14}) 
from shapely.geometry import LineString
import gc
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

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

def calculate_extended_cul_de_sac_features(G):
    """
    Calcula características básicas para la detección de patrones cul-de-sac.
    Versión con robustez reducida al 60%.
    
    Parámetros:
    -----------
    G : networkx.Graph
        Grafo de la red vial
        
    Retorna:
    --------
    dict : Diccionario con métricas muy básicas para patrones cul-de-sac
    """
   
    # Inicializar resultado con una sola métrica
    results = {
        'depth_avg': 0.0
    }
    
    # Verificación mínima del grafo
    if not G:
        return results
    
    # Identificar nodos relevantes con enfoque simplificado
    dead_end_nodes = []
    arterial_nodes = []
    
    # Obtener nodos de manera más simple
    for n, d in G.degree():
        if d == 1:
            dead_end_nodes.append(n)
            if len(dead_end_nodes) >= 3:  # Limitar a solo 3 nodos sin salida
                break
    
    for n, d in G.degree():
        if d >= 3:
            arterial_nodes.append(n)
            if len(arterial_nodes) >= 2:  # Limitar a solo 2 nodos arteriales
                break
    
    # Si no hay suficientes nodos, devolver valores iniciales
    if len(dead_end_nodes) == 0 or len(arterial_nodes) == 0:
        return results
    
    # Cálculo muy simplificado de profundidad
    depths = []
    for dead_end in dead_end_nodes[:3]:  # Usar máximo 3 nodos
        arterial = arterial_nodes[0]  # Usar solo el primer nodo arterial
        try:
            dist = nx.shortest_path_length(G, dead_end, arterial)
            depths.append(dist)
        except:
            depths.append(1)  # Valor predeterminado si falla
    
    # Cálculo promedio ultrasimplificado
    if depths:
        results['depth_avg'] = sum(depths) / len(depths) 
    
    return results

def classify_polygon(poly_stats, G=None):
    """
    Clasifica un polígono (o sub-polígono) en:
      'cul_de_sac', 'gridiron', 'organico' o 'hibrido'
    basado en la teoría de patrones urbanos y métricas morfológicas.
    
    Versión mejorada con características obligatorias de cálculo de ángulos y calles sin salida.
    Optimizada para mejor detección de patrones cul-de-sac.

    Parámetros:
    -----------
    poly_stats : dict con claves:
      - "streets_per_node_avg" (float): Promedio de calles por nodo
      - "streets_per_node_counts" (str o dict): Conteo de nodos por número de calles
      - "streets_per_node_proportions" (str o dict): Proporción de nodos por número de calles
      - "intersection_density_km2" (float): Densidad de intersecciones por km²
      - "circuity_avg" (float): Sinuosidad promedio de segmentos
      - "k_avg" (float): Grado promedio de nodos
      - "street_density_km2" (float): Densidad de calles por km²
      - "orientation_entropy" (float, opcional): Entropía de orientación de segmentos
      - "edge_length_avg" (float, opcional): Longitud promedio de aristas
      - "street_length_avg" (float, opcional): Longitud promedio de calles
      
    G : networkx.Graph, obligatorio
      Grafo de la red vial del polígono, necesario para calcular características adicionales
      como ángulos de intersección y métricas de calles sin salida.

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
    # Conectividad y estructura nodal
    streets_per_node = float(poly_stats.get("streets_per_node_avg", 0.0))
    k_avg = float(poly_stats.get("k_avg", 0.0))
    
    # Densidades
    intersection_density = float(poly_stats.get("intersection_density_km2", 0.0))
    street_density = float(poly_stats.get("street_density_km2", 0.0))
    
    # Geometría
    circuity = float(poly_stats.get("circuity_avg", 1.0))
    edge_length_avg = float(poly_stats.get("edge_length_avg", 0.0))
    street_length_avg = float(poly_stats.get("street_length_avg", 0.0))
    
    # Entropía de orientación (0=alineado, 1=diverso)
    orientation_entropy = float(poly_stats.get("orientation_entropy", 0.5))
    
    # Proporciones de nodos por grado
    prop_deg1 = float(spn_props.get('1', 0.0))  # callejones sin salida
    prop_deg3 = float(spn_props.get('3', 0.0))  # intersecciones en T
    prop_deg4 = float(spn_props.get('4', 0.0))  # intersecciones en cruz
    
    # -------------------------------------------------------------------
    # 3. Calcular obligatoriamente las métricas adicionales del grafo
    # -------------------------------------------------------------------
    # Valores predeterminados en caso de que el grafo sea None o no se puedan calcular
    mean_intersection_angle = 90.0  # Asumiendo ángulos en grados
    std_intersection_angle = 45.0
    orthogonal_proportion = 0.5
    angle_coefficient_variation = 0.5
    dead_end_ratio = prop_deg1  # Si no hay grafo, usar prop_deg1 como aproximación
    cv_dead_end_distances = 0.5
    
    # Características específicas para cul-de-sac mejoradas
    cul_de_sac_depth_avg = 0.0  # Profundidad promedio de calles sin salida
    cul_de_sac_branch_ratio = 0.0  # Proporción de ramificación en calles sin salida
    loop_ratio = 0.0  # Proporción de calles que forman bucles cerrados
    tree_like_structure_score = 0.0  # Puntuación de estructura arbórea
    
    # Calcular las características si el grafo está disponible
    if G is not None:
        try:
            from math import degrees, atan2
            import numpy as np
            
            # Calcular características de ángulos
            mean_angle, std_angle, ortho_prop, cv_angle = calculate_angle_features(G)
            mean_intersection_angle = mean_angle
            std_intersection_angle = std_angle
            orthogonal_proportion = ortho_prop
            angle_coefficient_variation = cv_angle
            
            # Calcular características de calles sin salida
            dead_end_r, cv_dead_end = calculate_dead_end_features(G)
            dead_end_ratio = dead_end_r  # Sobrescribir prop_deg1 con el cálculo directo
            cv_dead_end_distances = cv_dead_end
            
            # NUEVO: Características extendidas para cul-de-sac
            cul_de_sac_metrics = calculate_extended_cul_de_sac_features(G)
            cul_de_sac_depth_avg = cul_de_sac_metrics.get('depth_avg', 0.0)
            cul_de_sac_branch_ratio = cul_de_sac_metrics.get('branch_ratio', 0.0)
            loop_ratio = cul_de_sac_metrics.get('loop_ratio', 0.0)
            tree_like_structure_score = cul_de_sac_metrics.get('tree_structure_score', 0.0)
            
        except Exception as e:
            print(f"Error al calcular características del grafo: {e}")
            # Mantener los valores predeterminados en caso de error
    else:
        print("No se proporcionó un grafo válido. Usando valores aproximados para características adicionales.")
    
    # -------------------------------------------------------------------
    # 4. Sistema de puntuación para cada patrón
    # -------------------------------------------------------------------
    # Inicializamos los puntajes para cada categoría
    scores = {
        "cul_de_sac": 0,
        "gridiron": 0,
        "organico": 0,
        "hibrido": 0
    }
    
    # A. Puntuación para Cul-de-sac / Suburban
    # - SECCIÓN RECONSTRUIDA: Detección avanzada de patrones cul-de-sac
    # - Utilizando un enfoque multimétrica más sensible y detallado
    
    # 1. Característica principal: Calles sin salida (dead_end_ratio)
    # Enfoque más sensible con umbrales incrementales revisados
    # DESPUÉS:
    if dead_end_ratio > 0.29:  # era 0.25
        scores["cul_de_sac"] += 5
    elif dead_end_ratio > 0.23:  # era 0.20
        scores["cul_de_sac"] += 5
    elif dead_end_ratio > 0.17:  # era 0.15
        scores["cul_de_sac"] += 4
    elif dead_end_ratio > 0.12:  # era 0.10
        scores["cul_de_sac"] += 3
    elif dead_end_ratio > 0.08:  # era 0.07
        scores["cul_de_sac"] += 3
    elif dead_end_ratio > 0.06:  # era 0.05
        scores["cul_de_sac"] += 2
    
    # 2. NUEVO: Profundidad de calles sin salida
    # Evalúa qué tan profundas son las calles sin salida en relación a la red
    if cul_de_sac_depth_avg > 3.0:  # Calles sin salida muy profundas
        scores["cul_de_sac"] += 4  # Alta puntuación, indicador clave
    elif cul_de_sac_depth_avg > 2.5:  # Calles sin salida profundas
        scores["cul_de_sac"] += 3
    elif cul_de_sac_depth_avg > 2.0:  # Profundidad moderada
        scores["cul_de_sac"] += 2
    elif cul_de_sac_depth_avg > 1.5:  # Profundidad mínima significativa
        scores["cul_de_sac"] += 1

    # 3. NUEVO: Estructura arbórea de la red
    # Evalúa cuánto se asemeja la red a un árbol (típico en cul-de-sac)
    if tree_like_structure_score > 0.7:  # Estructura fuertemente arbórea
        scores["cul_de_sac"] += 5  # Indicador clave, alta puntuación
    elif tree_like_structure_score > 0.6:  # Estructura mayormente arbórea
        scores["cul_de_sac"] += 4
    elif tree_like_structure_score > 0.5:  # Estructura moderadamente arbórea
        scores["cul_de_sac"] += 3
    elif tree_like_structure_score > 0.4:  # Estructura ligeramente arbórea
        scores["cul_de_sac"] += 2
    elif tree_like_structure_score > 0.3:  # Estructura mínimamente arbórea
        scores["cul_de_sac"] += 1
    
    # 4. Estructura nodal: Proporción de nodos de grado 1
    # Complementario a dead_end_ratio, útil para casos límite
    if prop_deg1 >= 0.25:
        scores["cul_de_sac"] += 3
    elif prop_deg1 >= 0.18:
        scores["cul_de_sac"] += 2
    elif prop_deg1 >= 0.12:
        scores["cul_de_sac"] += 1
    
    # 5. Conectividad: Indicador clave secundario (streets_per_node)
    # Umbrales ajustados para mejor detección de cul-de-sac
    if streets_per_node < 2.0:  # era 2.1
        scores["cul_de_sac"] += 4
    elif streets_per_node < 2.2:  # era 2.3
        scores["cul_de_sac"] += 3
    elif streets_per_node < 2.3:  # era 2.5
        scores["cul_de_sac"] += 3
    elif streets_per_node < 2.5:  # era 2.7
        scores["cul_de_sac"] += 2
    elif streets_per_node < 2.7:  # era 2.9
        scores["cul_de_sac"] += 1

    
    # 6. NUEVO: Proporción de bucles cerrados (loop_ratio)
    # Los patrones cul-de-sac modernos a menudo incorporan bucles
    # Pero demasiados bucles indicarían otro patrón
    if loop_ratio > 0.0 and loop_ratio < 0.15:  # Presencia óptima de bucles
        scores["cul_de_sac"] += 3  # Indicador positivo para cul-de-sac modernos
    elif loop_ratio >= 0.15 and loop_ratio < 0.25:  # Cantidad moderada de bucles
        scores["cul_de_sac"] += 2  # Aún compatible con cul-de-sac
    elif loop_ratio >= 0.25 and loop_ratio < 0.35:  # Cantidad significativa de bucles
        scores["cul_de_sac"] += 1  # Menos típico pero posible en algunos cul-de-sac
    
    # 7. NUEVO: Ramificación de calles sin salida
    # Evalúa cómo las calles sin salida se ramifican desde arterias principales
    if cul_de_sac_branch_ratio > 0.6:  # Alta ramificación jerárquica
        scores["cul_de_sac"] += 4  # Fuerte indicador de patrón cul-de-sac planificado
    elif cul_de_sac_branch_ratio > 0.5:  # Ramificación moderada-alta
        scores["cul_de_sac"] += 3
    elif cul_de_sac_branch_ratio > 0.4:  # Ramificación moderada
        scores["cul_de_sac"] += 2
    elif cul_de_sac_branch_ratio > 0.3:  # Ramificación baja pero significativa
        scores["cul_de_sac"] += 1
    
    # 8. Densidad: Más flexible, reconociendo variantes más densas de cul-de-sac
    if intersection_density < 25:  # Extremadamente baja densidad (suburbios extensos)
        scores["cul_de_sac"] += 3
    elif intersection_density < 40:  # Muy baja densidad
        scores["cul_de_sac"] += 2
    elif intersection_density < 70:  # Baja-media densidad (permite cul-de-sac más urbanos)
        scores["cul_de_sac"] += 1
    
    # 9. Geometría y distribución
    # Variabilidad en distancias entre calles sin salida (indicador de diseño suburbano)
    if cv_dead_end_distances > 0.8:  # Alta variabilidad típica de desarrollos planificados
        scores["cul_de_sac"] += 3
    elif cv_dead_end_distances > 0.6:  # Variabilidad moderada-alta
        scores["cul_de_sac"] += 2
    elif cv_dead_end_distances > 0.4:  # Variabilidad moderada
        scores["cul_de_sac"] += 1
    
    # 10. Combinaciones específicas de métricas para patrones cul-de-sac
    # Estas combinaciones ayudan a detectar variantes específicas de cul-de-sac
    
    # 10.1 MEJORADO: Patrón cul-de-sac clásico
    # Combinación de dead-ends, baja conectividad y baja densidad
    if dead_end_ratio > 0.17 and streets_per_node < 2.3 and intersection_density < 50:  # más estricto
        scores["cul_de_sac"] += 3
        
    # 10.2 MEJORADO: Patrón cul-de-sac compacto
    # Calles sin salida pero densidad moderada y cierta regularidad
    if dead_end_ratio > 0.18 and intersection_density > 50 and intersection_density < 90:  # más estricto
        scores["cul_de_sac"] += 2

        
    # 10.3 MEJORADO: Patrón cul-de-sac con loops
    # Sinuosidad moderada con calles sin salida y presencia de bucles
    if dead_end_ratio > 0.12 and loop_ratio > 0.05 and loop_ratio < 0.20:  # más estricto
        scores["cul_de_sac"] += 2
        
    # 10.4 NUEVO: Cul-de-sac en zonas de transición
    # Áreas que mezclan cul-de-sac con elementos de retícula incompleta
    if dead_end_ratio > 0.15 and orthogonal_proportion > 0.4 and orthogonal_proportion < 0.7:  # más estricto
        scores["cul_de_sac"] += 1

        
    # 10.5 NUEVO: Cul-de-sac integrados
    # Áreas donde el patrón cul-de-sac está integrado con otras tipologías
    if dead_end_ratio > 0.10 and prop_deg3 > 0.30 and tree_like_structure_score > 0.35:
        scores["cul_de_sac"] += 2
    
    # 10.6 NUEVO: Cul-de-sac de nueva generación
    # Diseños contemporáneos que combinan calles sin salida con mejor conectividad
    if dead_end_ratio > 0.12 and cul_de_sac_branch_ratio > 0.4 and loop_ratio > 0.1:
        scores["cul_de_sac"] += 3
    
    # B. Puntuación para Gridiron / Reticular (manteniendo como estaba)
    if circuity < 1.02:  # Muy baja sinuosidad
        scores["gridiron"] += 2
    elif circuity < 1.04:
        scores["gridiron"] += 1
        
    if prop_deg4 >= 0.30:  # Alta proporción de cruces en X
        scores["gridiron"] += 3
    elif prop_deg4 >= 0.25:
        scores["gridiron"] += 2
    elif prop_deg4 >= 0.20:
        scores["gridiron"] += 1
        
    if orientation_entropy < 0.5:  # Orientaciones muy consistentes
        scores["gridiron"] += 2
    elif orientation_entropy < 0.65:
        scores["gridiron"] += 1
        
    if orthogonal_proportion > 0.7:  # Alto porcentaje de intersecciones ortogonales
        scores["gridiron"] += 2
    elif orthogonal_proportion > 0.5:
        scores["gridiron"] += 1
        
    if std_intersection_angle < 15:  # Baja desviación en ángulos de intersección
        scores["gridiron"] += 2
    elif std_intersection_angle < 30:
        scores["gridiron"] += 1
        
    if streets_per_node >= 3.0:  # Alta conectividad
        scores["gridiron"] += 2
    elif streets_per_node >= 2.7:
        scores["gridiron"] += 1
    
    # C. Puntuación para Orgánico / Irregular
    # - Más flexible en reconocer patrones orgánicos con variabilidad
    # - Menor penalización por pequeñas tendencias ortogonales
    
    # La sinuosidad (circuity) es una característica clave para patrones orgánicos
    if circuity > 1.2:  # Sinuosidad muy alta
        scores["organico"] += 3  
    elif circuity > 1.12:  # Sinuosidad alta
        scores["organico"] += 2  
    elif circuity > 1.07:  # Sinuosidad moderada
        scores["organico"] += 1  
    
    # Las intersecciones en T son típicas en patrones orgánicos
    if prop_deg3 > prop_deg4 * 1.7:  # Dominan las intersecciones en T
        scores["organico"] += 3
    elif prop_deg3 > prop_deg4 * 1.2:  
        scores["organico"] += 2
    elif prop_deg3 > prop_deg4 * 0.8:  
        scores["organico"] += 1
    
    # La variabilidad en orientaciones es característica de patrones orgánicos
    if orientation_entropy > 0.75:
        scores["organico"] += 3  
    elif orientation_entropy > 0.65:
        scores["organico"] += 2  
    elif orientation_entropy > 0.6:   
        scores["organico"] += 1
    
    # Coeficiente de variación de ángulos (irregularidad)
    if angle_coefficient_variation > 0.65:
        scores["organico"] += 2
    elif angle_coefficient_variation > 0.45:
        scores["organico"] += 1
    
    # Desviación estándar de ángulos de intersección
    if std_intersection_angle > 35:
        scores["organico"] += 2
    elif std_intersection_angle > 25:
        scores["organico"] += 1
    
    # Baja proporción de intersecciones ortogonales
    if orthogonal_proportion < 0.3:
        scores["organico"] += 2  
    elif orthogonal_proportion < 0.4:  # Nuevo umbral más inclusivo
        scores["organico"] += 1
    
    # Características combinadas para patrones orgánicos
    # 1. Patrones orgánicos tradicionales: alta sinuosidad con pocas calles sin salida
    if circuity > 1.1 and dead_end_ratio < 0.15:
        scores["organico"] += 2  # Mayor peso (era 1)
    
    # 2. Patrones orgánicos medievales: alta entropía con densidad moderada-alta
    if orientation_entropy > 0.7 and intersection_density > 50:
        scores["organico"] += 1  # Nueva característica
    
    # D. Híbrido - tiene características mezcladas o no cumple claramente con otro patrón
    # El híbrido no recibe puntos directamente, se determina por exclusión o combinación
    
    # -------------------------------------------------------------------
    # 5. Factores de penalización para patrones incompatibles (ajustados)
    # -------------------------------------------------------------------
    
    # REVISADO: Penalizaciones más selectivas para Cul-de-sac
    # Estas penalizaciones ahora son condicionales a múltiples factores
    
    # Penalización por alta conectividad solo si no hay evidencia clara de cul-de-sac
    # y no hay estructura arbórea significativa
    if streets_per_node > 3.2 and dead_end_ratio < 0.12 and tree_like_structure_score < 0.35:
        scores["cul_de_sac"] -= 2
    elif streets_per_node > 3.0 and dead_end_ratio < 0.10 and tree_like_structure_score < 0.30:
        scores["cul_de_sac"] -= 1
    
    # Penalización por densidad extremadamente alta, solo si no hay características
    # distintivas de cul-de-sac como profundidad o ramificación significativas
    if intersection_density > 120 and dead_end_ratio < 0.15 and cul_de_sac_depth_avg < 1.8:
        scores["cul_de_sac"] -= 2
    elif intersection_density > 100 and dead_end_ratio < 0.12 and cul_de_sac_depth_avg < 1.5:
        scores["cul_de_sac"] -= 1
    
    # Penalización por estructura demasiado regular o mallada
    # Solo penalizar si no hay características compensatorias de cul-de-sac
    if orthogonal_proportion > 0.85 and prop_deg4 > 0.4 and dead_end_ratio < 0.15 and tree_like_structure_score < 0.4:
        scores["cul_de_sac"] -= 2  # Patrón extremadamente regular no es cul-de-sac típico
    
    # Penalización por alto porcentaje de bucles, incompatible con cul-de-sac puros
    if loop_ratio > 0.5 and dead_end_ratio < 0.12:
        scores["cul_de_sac"] -= 2  # Demasiados bucles para ser cul-de-sac típico
    
    # Penalizar Gridiron si hay alta sinuosidad o irregularidad (sin cambios)
    if circuity > 1.12:
        scores["gridiron"] -= 2
    elif circuity > 1.08:
        scores["gridiron"] -= 1
        
    if prop_deg1 > 0.25:
        scores["gridiron"] -= 2
    elif prop_deg1 > 0.15:
        scores["gridiron"] -= 1
        
    if orientation_entropy > 0.75:
        scores["gridiron"] -= 1
    
    # Penalizar Orgánico si hay alta ortogonalidad o regularidad
    # Penalizaciones reducidas para flexibilidad
    if orthogonal_proportion > 0.7:  # Valor ajustado (era 0.6)
        scores["organico"] -= 2
    elif orthogonal_proportion > 0.5:  # Valor ajustado (era 0.4)
        scores["organico"] -= 1
        
    if circuity < 1.03:  # Valor ajustado (era 1.05)
        scores["organico"] -= 2
    elif circuity < 1.06:  # Valor ajustado (era 1.08)
        scores["organico"] -= 1
    
    # -------------------------------------------------------------------
    # 6. Combinar puntuaciones y determinar el patrón dominante
    # -------------------------------------------------------------------
    
    # Si hay empate entre patrones principales, favorecer híbrido
    max_score = max(scores.values())
    max_patterns = [k for k, v in scores.items() if v == max_score]
    
    # Detectar patrón híbrido explícitamente (múltiples patrones con puntuaciones cercanas)
    all_scores = sorted(scores.values(), reverse=True)
    
    # Consideramos híbrido si:
    # 1. El máximo score no es muy alto (menos de 4 puntos) o
    # 2. La diferencia entre los dos mejores scores es pequeña (1 punto o menos)
    if max_score < 4 or (len(all_scores) > 1 and (all_scores[0] - all_scores[1]) <= 1):
        scores["hibrido"] = max(scores["hibrido"], max_score - 1)  # Ajustamos score de híbrido
    
    # Obtener patrón con mayor puntuación (o "hibrido" en caso de empate)
    if len(max_patterns) > 1 and "hibrido" not in max_patterns:
        dominant_pattern = "hibrido"
    else:
        dominant_pattern = max(scores, key=scores.get)
    
    # Verificar umbral mínimo para clasificación confiable
    if max_score <= 2 and dominant_pattern != "hibrido":
        dominant_pattern = "hibrido"  # Si no hay un patrón claramente dominante, es híbrido
    
    # -------------------------------------------------------------------
    # 7. Post-procesamiento para ajustes finales - RECONSTRUIDO
    # -------------------------------------------------------------------
    
    # NUEVO: Sistema de verificación avanzada para patrones cul-de-sac
    # Prioriza la detección de cul-de-sac basándose en criterios jerárquicos
    
    # Regla 1: Evidencia directa muy fuerte - sobrescribe cualquier otra clasificación
    if dead_end_ratio > 0.20 and (tree_like_structure_score > 0.45 or cul_de_sac_depth_avg > 2.2):
        dominant_pattern = "cul_de_sac"
        
    # Regla 2: Combinación de múltiples indicadores fuertes de cul-de-sac
    elif dominant_pattern == "hibrido" or scores["cul_de_sac"] >= scores[dominant_pattern] - 1:
        # Verificamos combinaciones específicas que deberían garantizar clasificación como cul-de-sac
        
        # 2.1: Calles sin salida significativas + estructura arbórea
        if dead_end_ratio > 0.15 and tree_like_structure_score > 0.40:
            dominant_pattern = "cul_de_sac"
            
        # 2.2: Calles sin salida + baja conectividad + profundidad significativa
        elif dead_end_ratio > 0.12 and streets_per_node < 2.7 and cul_de_sac_depth_avg > 1.8:
            dominant_pattern = "cul_de_sac"
            
        # 2.3: Patrón de ramificación clara + calles sin salida moderadas
        elif dead_end_ratio > 0.10 and cul_de_sac_branch_ratio > 0.5:
            dominant_pattern = "cul_de_sac"
            
        # 2.4: Configuración de cul-de-sac modernos con loops
        elif dead_end_ratio > 0.08 and loop_ratio > 0.05 and loop_ratio < 0.25 and tree_like_structure_score > 0.35:
            dominant_pattern = "cul_de_sac"
    
    # Regla 3: Verificación adicional para evitar casos de falsos positivos
    # Si está clasificado como cul-de-sac pero tiene características contradictorias fuertes
    if dominant_pattern == "cul_de_sac":
        # Posible falso positivo - estructura demasiado ortogonal y densa para ser cul-de-sac
        if prop_deg4 > 0.4 and orthogonal_proportion > 0.8 and intersection_density > 100 and dead_end_ratio < 0.15:
            dominant_pattern = "hibrido"  # Reclasificar como híbrido
    
    
    
  
    return dominant_pattern

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
    output_png="polygons_classification.png",
    graph_dict=None
):
    """
    Lee un GeoDataFrame (geojson_path), asigna una 'clase' a cada polígono
    según las estadísticas en 'stats_dict', la 'classify_func', y los grafos
    en 'graph_dict', y dibuja en un PNG (Matplotlib) con colores distintos por clase.

    Parámetros:
    -----------
    geojson_path : str
        Ruta al archivo GeoJSON con los polígonos
    stats_dict : dict
        Diccionario con las estadísticas de cada polígono. Las claves son (idx, sub_idx)
    classify_func : function
        Función de clasificación que recibe estadísticas y un grafo como parámetros
    output_png : str, opcional
        Ruta donde guardar la imagen resultante
    graph_dict : dict, opcional
        Diccionario con los grafos de red vial para cada polígono.
        Las claves pueden tener formato diferente a las de stats_dict.

    Retorna:
    --------
    gdf : GeoDataFrame
        El GeoDataFrame con una columna adicional 'pattern' que contiene la clasificación
    """
 
    # Cargar el GeoDataFrame
    gdf = gpd.read_file(geojson_path)
    
    # Crear columna 'pattern' con la clase
    patterns = []
    
    for idx, row in gdf.iterrows():
        # Identificar el polígono
        poly_id = row['poly_id'] if 'poly_id' in gdf.columns else idx
        key = (idx, 0)  # Asumiendo que cada fila = sub-polígono 0
        
        if key in stats_dict:
            poly_stats = stats_dict[key]
            
            # Manejo mejorado de graph_dict
            G = None
            if graph_dict is not None:
                # 1. Extraer el ID principal del polígono
                if isinstance(poly_id, tuple) and len(poly_id) >= 1:
                    main_id = poly_id[0]
                else:
                    main_id = poly_id
                
                # 2. Generar posibles formatos de clave
                possible_keys = [main_id, str(main_id), idx, str(idx), key]
                
                # 3. Buscar el grafo usando las posibles claves
                for possible_key in possible_keys:
                    if possible_key in graph_dict:
                        G = graph_dict[possible_key]
                        break
                
                # 4. Verificar que G sea un objeto grafo válido
                if G is not None and not hasattr(G, 'number_of_nodes'):
                    print(f"Advertencia: El objeto para el polígono {poly_id} no es un grafo válido.")
                    G = None
            
            # Clasificar el polígono usando la función de clasificación
            # Pasando tanto las estadísticas como el grafo
            category = classify_func(poly_stats, G)
        else:
            print(f"Advertencia: No se encontraron estadísticas para el polígono {poly_id}")
            category = "unknown"
            
        patterns.append(category)

    # Añadir la columna de patrones al GeoDataFrame
    gdf["pattern"] = patterns

    # Contar cuántos polígonos hay de cada categoría
    pattern_counts = gdf["pattern"].value_counts()
    print("Conteo de patrones:")
    for pattern, count in pattern_counts.items():
        print(f"  - {pattern}: {count}")

    # Mapear cada clase a un color
    color_map = {
        'cul_de_sac': '#FF6B6B',   # Rojo para callejones sin salida
        'gridiron': '#006400',     # Verde oscuro para grid
        'organico': '#45B7D1',     # Azul para orgánico
        'hibrido': '#FDCB6E',      # Amarillo para híbrido
        'unknown': '#CCCCCC'       # Gris para desconocidos
    }

    # Función para obtener el color según la categoría
    def get_color(cat):
        return color_map.get(cat, "black")

    # Obtener colores para cada polígono
    plot_colors = [get_color(cat) for cat in gdf["pattern"]]

    # Graficar
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(
        ax=ax,
        color=plot_colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7  # Transparencia para mejor visualización
    )

    # Leyenda manual con conteo de cada categoría
    legend_patches = []
    for cat, col in color_map.items():
        count = pattern_counts.get(cat, 0)
        if count > 0:  # Solo mostrar en la leyenda las categorías presentes
            patch = mpatches.Patch(
                color=col, 
                label=f"{cat} ({count})"
            )
            legend_patches.append(patch)
    
    ax.legend(
        handles=legend_patches, 
        title="Tipos de tejido urbano",
        loc="upper right",
        frameon=True,
        framealpha=0.9
    )

    ax.set_title("Clasificación Morfológica de Tejidos Urbanos", fontsize=16)
    ax.set_axis_off()

    # Guardar la imagen
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Imagen guardada en: {output_png}")
    
    return gdf

def normalize_edge(x0, y0, x1, y1, tol=4):
    """
    Redondea las coordenadas a 'tol' decimales y retorna una tupla
    ordenada con los dos puntos. De esta forma, el segmento (A,B) es igual a (B,A).
    """
    p1 = (round(x0, tol), round(y0, tol))
    p2 = (round(x1, tol), round(y1, tol))
    return tuple(sorted([p1, p2]))

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

def find_optimal_k_improved(X_scaled, max_k=10, min_k=2, output_dir='filename'):
    """
    Encuentra el número óptimo de clusters usando silhouette score,
    calinski-harabasz index y modularity score (para redes).
    """
   
     # Crear directorio para resultados
    os.makedirs(output_dir, exist_ok=True)

    
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
    plt.savefig(os.path.join(output_dir,'cluster_metrics.png'), dpi=300, bbox_inches='tight')
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

def optimal_clustering_improved(X, feature_names, n_clusters=None, use_pca=True, 
                               pca_variance_threshold=0.95, max_pca_components=8, 
                               visualize=True, use_elbow_method=False,output_dir="Resultados/urbano_pattern_cluster"
    ):
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
    
     # Crear directorio para resultados
    os.makedirs(output_dir, exist_ok=True)

    # Eliminar filas con NaN o infinitos (tu código original)
    valid_rows = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    X_clean = X[valid_rows]
    if X_clean.shape[0] < 3:
        raise ValueError(f"El conjunto de datos limpio tiene solo {X_clean.shape[0]} filas. Se requieren al menos 3 para clustering.")

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
            plt.savefig(os.path.join(output_dir,'pca_variance_explained.png'), dpi=300)
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
                plt.savefig(os.path.join(output_dir,'pca_elbow_method.png'), dpi=300)
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

        # Add this code after line 78 (after the PCA components analysis section)


        # ENHANCED PCA VISUALIZATION FOR PAPER - SEPARATE PDF FILES
        if visualize:
            
            # GRAPH 1: Explained Variance Ratio (Bar chart)
            plt.figure(figsize=(10, 6))
            pc_labels = [f'PC{i+1}' for i in range(n_components)]
            bars = plt.bar(pc_labels, pca.explained_variance_ratio_, 
                          color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.8)
            plt.title('Explained Variance Ratio by Principal Component', fontsize=14, fontweight='bold')
            plt.xlabel('Principal Components', fontsize=12)
            plt.ylabel('Explained Variance Ratio', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, pca.explained_variance_ratio_):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pca_explained_variance_ratio.pdf'), 
                       format='pdf', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            
            # GRAPH 2: Cumulative Explained Variance
            plt.figure(figsize=(10, 6))
            cumulative_var = np.cumsum(pca.explained_variance_ratio_)
            plt.plot(range(1, n_components + 1), cumulative_var, 
                    marker='o', linewidth=3, markersize=8, color='darkred', markerfacecolor='red')
            plt.axhline(y=pca_variance_threshold, color='red', linestyle='--', 
                       linewidth=2, alpha=0.8, label=f'Threshold ({pca_variance_threshold})')
            plt.title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Components', fontsize=12)
            plt.ylabel('Cumulative Explained Variance', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            plt.ylim(0, 1.05)
            
            # Add percentage labels
            for i, val in enumerate(cumulative_var):
                plt.text(i+1, val + 0.02, f'{val:.1%}', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pca_cumulative_variance.pdf'), 
                       format='pdf', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            
            # GRAPH 4: Biplot (2D projection with feature vectors)
            if n_components >= 2:
                plt.figure(figsize=(12, 8))
                
                # Project data to first 2 PCs
                pc1_idx, pc2_idx = 0, 1
                scatter = plt.scatter(X_reduced[:, pc1_idx], X_reduced[:, pc2_idx], 
                                    alpha=0.6, s=40, c='steelblue', edgecolors='black', linewidth=0.5)
                
                # Add feature vectors (loadings)
                loadings = pca.components_[:2, :].T  # First 2 PCs
                max_loading = np.max(np.abs(loadings))
                scale_factor = 0.7 * (np.max(X_reduced[:, :2]) - np.min(X_reduced[:, :2])) / max_loading
                
                # Plot only the most important feature vectors
                loading_magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
                important_features = np.argsort(loading_magnitudes)[-8:]  # Top 8 features
                
                for i in important_features:
                    x_arrow = loadings[i, 0] * scale_factor
                    y_arrow = loadings[i, 1] * scale_factor
                    plt.arrow(0, 0, x_arrow, y_arrow, 
                             head_width=0.15, head_length=0.15, fc='red', ec='red', 
                             alpha=0.8, linewidth=2)
                    plt.text(x_arrow * 1.15, y_arrow * 1.15, feature_names[i], 
                            fontsize=9, ha='center', va='center', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                     alpha=0.8, edgecolor='gray'))
                
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=1)
                plt.axvline(x=0, color='k', linestyle='--', alpha=0.4, linewidth=1)
                plt.title('PCA Biplot (PC1 vs PC2)', fontsize=14, fontweight='bold')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'pca_biplot.pdf'), 
                           format='pdf', dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
            
            
            # GRAPH 6: Reconstruction Error Analysis
            plt.figure(figsize=(10, 6))
            reconstruction_errors = []
            components_range = range(1, min(10, X_scaled.shape[1]) + 1)
            
            for n_comp in components_range:
                temp_pca = PCA(n_components=n_comp)
                X_temp = temp_pca.fit_transform(X_scaled)
                X_reconstructed = temp_pca.inverse_transform(X_temp)
                mse = np.mean((X_scaled - X_reconstructed) ** 2)
                reconstruction_errors.append(mse)
            
            plt.plot(components_range, reconstruction_errors, 
                    marker='s', linewidth=3, markersize=8, color='purple', 
                    markerfacecolor='mediumorchid')
            plt.axvline(x=n_components, color='red', linestyle='--', alpha=0.8, linewidth=2,
                       label=f'Selected Components ({n_components})')
            plt.title('Reconstruction Error vs Number of Components', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Components', fontsize=12)
            plt.ylabel('Mean Squared Error', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pca_reconstruction_error.pdf'), 
                       format='pdf', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # GRAPH 5: Radar Plot - PCA Components Distribution (Improved)
            # GRAPH 5: Radar Plot - PCA Components Distribution (Improved)
            plt.figure(figsize=(12, 12))

            # Diccionario organizado por grupos temáticos
            feature_groups = {
                'connectivity': {
                    'network_connectivity_index': 'Connectivity Index',
                    'dead_end_ratio': 'Dead End Ratio',
                    'cv_dead_end_distances': 'Dead End CV'
                    
                },
                'network_geometry': {
                    'circuity_avg': 'Circuity',
                    'edge_length_avg': 'Avg Edge Length',
                    'street_length_avg': 'Avg Street Length'
                },
                'density': {
                    'edge_length_density': 'Edge Density',
                    'street_density_km2': 'Street Density',
                    'node_density_km2': 'Node Density',
                    'edge_density_km2': 'Edge/km²',
                    'intersection_density_km2': 'Intersection Density',
                    'segment_density_km2': 'Segment Density'
                },
                                  
                'angular_properties': {
                    'mean_intersection_angle': 'Mean Angle',
                    'std_intersection_angle': 'Angle Std',
                    'angle_coefficient_variation': 'Angle CV',
                    'orthogonal_proportion': 'Orthogonal %'
                },
                'nodal_structure': {
                    'streets_per_node_avg': 'Streets/Node',
                    'k_avg': 'Avg Degree'
                }
            }

            # Crear diccionario plano para el mapeo
            feature_name_mapping = {}
            for group_features in feature_groups.values():
                feature_name_mapping.update(group_features)

            # Prepare data for radar chart
            # Get the absolute loadings for better visualization
            radar_data = np.abs(pca.components_)

            # Usar TODAS las características organizadas por grupos temáticos
            ordered_features = []
            ordered_features_short = []

            # Agregar features en el orden de los grupos
            group_order = ['connectivity', 'network_geometry', 'density', 'angular_properties', 'nodal_structure']

            for group in group_order:
                for original_name, short_name in feature_groups[group].items():
                    if original_name in feature_names:  # Verificar que existe en nuestros datos
                        ordered_features.append(original_name)
                        ordered_features_short.append(short_name)

            # Obtener los índices de las features ordenadas
            ordered_features_idx = [feature_names.index(feat) for feat in ordered_features]
            radar_values = radar_data[:, ordered_features_idx]

            # Number of variables
            num_vars = len(ordered_features_short)

            # Compute angle for each axis
            angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
            angles += angles[:1]  # Complete the circle

            # Initialize the plot
            ax = plt.subplot(111, polar=True)

            # Colors for each PC (expandido para más componentes)
            colors = ['#FF0000', '#0000FF', '#00AA00', '#FF8800', '#AA00AA', 
                    '#8B4513', '#FF69B4', '#808080', '#00FFFF', '#FFD700',
                    '#FF1493', '#32CD32', '#FF4500', '#9370DB', '#20B2AA']

            # Plot each PC con diferentes estilos para evitar solapamiento
            line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.']

            for i in range(n_components):
                values = radar_values[i].tolist()
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=2.5, 
                    label=f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})', 
                    color=colors[i % len(colors)], 
                    markersize=6,
                    markerfacecolor='white',
                    markeredgewidth=2,
                    alpha=0.8)
                
                 # Solo fill para los primeros 3 componentes más importantes
                if i < 5:
                    ax.fill(angles, values, alpha=0.08, color=colors[i % len(colors)])

            # Configurar las etiquetas de las características organizadas por grupos
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(ordered_features_short, fontsize=9)

            # Añadir líneas divisorias entre grupos para mejor visualización
            group_boundaries = []
            current_position = 0
            for group in group_order:
                group_size = len(feature_groups[group])
                if current_position > 0:  # No agregar línea al inicio
                    boundary_angle = angles[current_position]
                    ax.axvline(x=boundary_angle, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                current_position += group_size
                group_boundaries.append(current_position)

            # Configurar los valores radiales
            ax.set_ylim(0, np.max(radar_values) * 1.1)
            ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
            ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=8)
            ax.grid(True, alpha=0.3)

            legend = ax.legend(
                loc='upper left',  # o 'center left', etc.
                bbox_to_anchor=(1.02, 1),  # apenas fuera del borde derecho
                fontsize=9,
                ncol=1,
                frameon=True,
                fancybox=True,
                shadow=True
            )
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)


            # Título más conciso
            plt.title('PCA Components: Feature Loading Distribution', 
                    fontsize=16, fontweight='bold', pad=30)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pca_radar_plot_improved.pdf'), 
                    format='pdf', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            plt.close()

            # Crear una tabla de referencia organizada por grupos
            print("🔍 FEATURE GROUPS MAPPING REFERENCE")
            print("=" * 80)

            group_emojis = {
                'connectivity': '🔗',
                'network_geometry': '📐', 
                'density': '🏘️',
                'angular_properties': '🧭',
                'nodal_structure': '🔴'
            }

            group_descriptions = {
                'connectivity': 'Network Connectivity & Flow',
                'network_geometry': 'Geometric Properties',
                'density': 'Spatial Density Measures', 
                'angular_properties': 'Angular & Directional Properties',
                'nodal_structure': 'Node Characteristics'
            }

            print(f"\n📊 RADAR PLOT ORGANIZATION (Clockwise from top):")
            print("-" * 60)

            for group_name in group_order:
                features = feature_groups[group_name]
                emoji = group_emojis.get(group_name, '📊')
                description = group_descriptions.get(group_name, group_name.title())
                
                print(f"\n{emoji} {description.upper()}:")
                
                for original, short in features.items():
                    if original in feature_names:  # Solo mostrar las que están en nuestros datos
                        print(f"  • {short}")

            print("\n" + "=" * 80)
            print(f"📈 Total Features: {len(ordered_features)}")
            print(f"🎯 Groups: {len(feature_groups)}")

            # Mostrar distribución por grupo
            print(f"\n📊 DISTRIBUTION BY GROUP:")
            for group_name in group_order:
                features = feature_groups[group_name]
                count = len([f for f in features.keys() if f in feature_names])
                emoji = group_emojis.get(group_name, '📊')
                print(f"  {emoji} {group_name.title()}: {count} features")

            print("=" * 80)
            


            
            # GRAPH 6: Lollipop Plot - PCA Components by Feature Groups
            plt.figure(figsize=(16, 12))

            # Configuración visual por grupo
            group_config = {
                'connectivity': {'emoji': '🔗', 'color': '#FF6B6B', 'description': 'Network Connectivity & Flow'},
                'network_geometry': {'emoji': '📐', 'color': '#4ECDC4', 'description': 'Geometric Properties'},
                'density': {'emoji': '🏘️', 'color': '#45B7D1', 'description': 'Spatial Density Measures'},
                'angular_properties': {'emoji': '🧭', 'color': '#96CEB4', 'description': 'Angular & Directional Properties'},
                'nodal_structure': {'emoji': '🔴', 'color': '#FECA57', 'description': 'Node Characteristics'}
            }

            # Preparar datos
            group_order = ['connectivity', 'network_geometry', 'density', 'angular_properties', 'nodal_structure']
            pca_loadings = np.abs(pca.components_)  # Valores absolutos para mejor visualización

            # Estructura para almacenar datos del plot
            plot_data = []
            y_positions = []
            group_separators = []
            current_y = 0

            # Procesar cada grupo
            for group_idx, group_name in enumerate(group_order):
                group_features = feature_groups[group_name]
                group_start_y = current_y
                
                # Agregar features del grupo
                for feat_original, feat_short in group_features.items():
                    if feat_original in feature_names:
                        feat_idx = feature_names.index(feat_original)
                        
                        # Para cada componente PCA
                        for pc_idx in range(min(5, pca.n_components_)):  # Mostrar solo los primeros 5 PC
                            plot_data.append({
                                'group': group_name,
                                'feature': feat_short,
                                'feature_original': feat_original,
                                'pc': f'PC{pc_idx+1}',
                                'loading': pca_loadings[pc_idx, feat_idx],
                                'y_pos': current_y,
                                'pc_idx': pc_idx,
                                'group_color': group_config[group_name]['color']
                            })
                        
                        current_y += 1
                
                # Guardar separador de grupo (línea entre grupos)
                if group_idx < len(group_order) - 1:  # No agregar línea después del último grupo
                    group_separators.append(current_y - 0.5)
                
                current_y += 0.5  # Espacio entre grupos

            # Convertir a DataFrame para facilitar el manejo
            import pandas as pd
            df_plot = pd.DataFrame(plot_data)

            # Crear el subplot
            fig, ax = plt.subplots(figsize=(16, 12))

            # Colores para cada PC
            pc_colors = ['#FF0000', '#0000FF', '#00AA00', '#FF8800', '#AA00AA']
            pc_markers = ['o', 's', '^', 'D', 'v']

            # Crear lollipops por cada PC
            for pc_idx in range(min(5, pca.n_components_)):
                pc_name = f'PC{pc_idx+1}'
                pc_data = df_plot[df_plot['pc'] == pc_name]
                
                # Offset horizontal para cada PC para evitar solapamiento
                x_offset = pc_idx * 0.15
                
                # Líneas verticales (stems)
                for _, row in pc_data.iterrows():
                    ax.plot([x_offset, row['loading'] + x_offset], 
                        [row['y_pos'], row['y_pos']], 
                        color=pc_colors[pc_idx], 
                        linewidth=2, 
                        alpha=0.7)
                
                # Puntos (lollipops)
                ax.scatter(pc_data['loading'] + x_offset, 
                        pc_data['y_pos'],
                        color=pc_colors[pc_idx],
                        marker=pc_markers[pc_idx],
                        s=80,
                        alpha=0.8,
                        edgecolors='white',
                        linewidth=1.5,
                        label=f'{pc_name} ({pca.explained_variance_ratio_[pc_idx]:.1%})',
                        zorder=5)

            # Configurar etiquetas del eje Y (features)
            unique_features = df_plot.drop_duplicates(['feature', 'y_pos']).sort_values('y_pos')
            ax.set_yticks(unique_features['y_pos'])
            ax.set_yticklabels(unique_features['feature'], fontsize=10)

            # Añadir separadores de grupo y etiquetas de grupo
            for separator in group_separators:
                ax.axhline(y=separator, color='gray', linestyle='--', alpha=0.5, linewidth=1)

            # Añadir etiquetas de grupo en el lado izquierdo
            for group_name in group_order:
                group_data = df_plot[df_plot['group'] == group_name]
                if not group_data.empty:
                    group_y_center = group_data['y_pos'].mean()
                    group_info = group_config[group_name]
                    
                    # Fondo colorido para el label del grupo
                    bbox_props = dict(boxstyle="round,pad=0.3", facecolor=group_info['color'], alpha=0.2)
                    ax.text(-0.15, group_y_center, 
                        f"{group_info['emoji']} {group_name.replace('_', ' ').title()}", 
                        fontsize=11, 
                        fontweight='bold',
                        ha='right', 
                        va='center',
                        transform=ax.get_yaxis_transform(),
                        bbox=bbox_props)

            # Configuración de ejes
            ax.set_xlabel('PCA Loading (Absolute Value)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Features by Group', fontsize=12, fontweight='bold')
            ax.set_title('PCA Feature Loadings by Thematic Groups\n(Lollipop Plot)', 
                        fontsize=16, fontweight='bold', pad=20)

            # Grid
            ax.grid(True, axis='x', alpha=0.3, linestyle='-')
            ax.grid(True, axis='y', alpha=0.1, linestyle=':')

            # Leyenda
            legend = ax.legend(loc='lower right', 
                            fontsize=10,
                            frameon=True,
                            fancybox=True,
                            shadow=True,
                            ncol=1)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)

            # Invertir eje Y para que el primer grupo esté arriba
            ax.invert_yaxis()

            # Ajustar límites
            ax.set_xlim(-0.05, max(df_plot['loading']) * 1.1)

            # Layout
            plt.tight_layout()

            # Guardar
            plt.savefig(os.path.join(output_dir, 'pca_lollipop_grouped.pdf'), 
                    format='pdf', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            plt.close()

            # Imprimir resumen
            print("🍭 LOLLIPOP PLOT - PCA LOADINGS BY GROUPS")
            print("=" * 70)
            print(f"📊 Features plotted: {len(unique_features)}")
            print(f"🎯 Groups: {len(group_order)}")
            print(f"📈 PCA Components shown: {min(5, pca.n_components_)}")

            print("\n📋 GROUP SUMMARY:")
            print("-" * 50)
            for group_name in group_order:
                group_info = group_config[group_name]
                group_data = df_plot[df_plot['group'] == group_name]
                feature_count = len(group_data['feature'].unique())
                print(f"{group_info['emoji']} {group_name.replace('_', ' ').title()}: {feature_count} features")

            print("\n🎨 VISUAL LEGEND:")
            print("-" * 30)
            for pc_idx in range(min(5, pca.n_components_)):
                variance_explained = pca.explained_variance_ratio_[pc_idx]
                print(f"  PC{pc_idx+1}: {pc_markers[pc_idx]} marker, explains {variance_explained:.1%} of variance")

            print("=" * 70)







            #############
            # DETAILED PCA ANALYSIS PRINTOUT
            print("\n" + "="*80)
            print("DETAILED PRINCIPAL COMPONENT ANALYSIS (PCA) SUMMARY")
            print("="*80)
            
            print(f"\nTotal Components Selected: {n_components}")
            print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.2f}%)")
            print(f"PCA Variance Threshold Used: {pca_variance_threshold}")
            
            for i in range(n_components):
                print(f"\n{'-'*50}")
                print(f"PRINCIPAL COMPONENT {i+1} (PC{i+1})")
                print(f"{'-'*50}")
                print(f"Explained Variance Ratio: {pca.explained_variance_ratio_[i]:.4f} ({pca.explained_variance_ratio_[i]*100:.2f}%)")
                print(f"Eigenvalue: {pca.explained_variance_[i]:.4f}")
                
                # Get component loadings
                component_loadings = pca.components_[i]
                
                # Find top positive and negative contributors
                positive_contributors = []
                negative_contributors = []
                
                for j, loading in enumerate(component_loadings):
                    if abs(loading) > 0.2:  # Threshold for significant contribution
                        if loading > 0:
                            positive_contributors.append((feature_names[j], loading))
                        else:
                            negative_contributors.append((feature_names[j], loading))
                
                # Sort by absolute value
                positive_contributors.sort(key=lambda x: abs(x[1]), reverse=True)
                negative_contributors.sort(key=lambda x: abs(x[1]), reverse=True)
                
                print(f"\nMost Important Features for PC{i+1}:")
                
                if positive_contributors:
                    print("  Positive Contributors (↑):")
                    for feature, loading in positive_contributors[:5]:  # Top 5
                        print(f"    • {feature}: {loading:.4f}")
                
                if negative_contributors:
                    print("  Negative Contributors (↓):")
                    for feature, loading in negative_contributors[:5]:  # Top 5
                        print(f"    • {feature}: {loading:.4f}")
                
                if not positive_contributors and not negative_contributors:
                    print("  No features with significant contributions (>0.2)")
                
                # Interpretation helper
                print(f"\nInterpretation Guide for PC{i+1}:")
                if positive_contributors and negative_contributors:
                    print("  This component contrasts features with positive loadings")
                    print("  against features with negative loadings.")
                elif positive_contributors:
                    print("  This component primarily represents the combined effect")
                    print("  of the positive contributing features.")
                elif negative_contributors:
                    print("  This component primarily represents the inverse effect")
                    print("  of the listed features.")
            
            print(f"\n{'-'*50}")
            print("PCA SUMMARY STATISTICS")
            print(f"{'-'*50}")
            print(f"Original Feature Space: {X_scaled.shape[1]} dimensions")
            print(f"Reduced Feature Space: {n_components} dimensions")
            print(f"Dimensionality Reduction: {((X_scaled.shape[1] - n_components) / X_scaled.shape[1] * 100):.1f}%")
            print(f"Information Retained: {sum(pca.explained_variance_ratio_)*100:.2f}%")
            print(f"Information Lost: {(1 - sum(pca.explained_variance_ratio_))*100:.2f}%")
            
            print(f"\n{'-'*50}")
            print("GENERATED VISUALIZATION FILES")
            print(f"{'-'*50}")
            print("Enhanced PCA visualizations saved as separate PDF files in:", output_dir)
            print("Generated files:")
            print("  - pca_explained_variance_ratio.pdf")
            print("  - pca_cumulative_variance.pdf") 
            print("  - pca_biplot.pdf")
            print("  - pca_reconstruction_error.pdf")
            print("  - pca_radar_plot.pdf")
            print("="*80)
   
        # Guardar información de los componentes PCA en un archivo de texto
        with open(os.path.join(output_dir,'pca_analysis.txt'), 'w') as f:
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
            plt.savefig(os.path.join(output_dir,'pca_components_contributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
    else:
        X_reduced = X_scaled
    
    # Encontrar número óptimo de clusters si no se proporciona
    if n_clusters is None:
        # Asumimos que la función find_optimal_k_improved está definida en otro lugar
        n_clusters = find_optimal_k_improved(X_reduced, max_k=8, min_k=3, output_dir=output_dir)
    
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
            plt.savefig(os.path.join(output_dir,'cluster_visualization_pca.png'), dpi=300, bbox_inches='tight')
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
                plt.savefig(os.path.join(output_dir,'cluster_visualization_tsne_improved.png'), dpi=300, bbox_inches='tight')
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
            plt.savefig(os.path.join(output_dir,'cluster_visualization_direct.png'), dpi=300, bbox_inches='tight')
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
        plt.savefig(os.path.join(output_dir,'feature_distributions_by_cluster.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return n_clusters, cluster_labels, centers_df, sorted_features




def plot_clustering_patterns_pdf_optimized(
    geojson_path,
    cluster_results,  # Resultados del clustering (dict con 'geodataframe', 'cluster_names', etc.)
    graph_dict,
    place_name="MyPlace",
    network_type="drive",
    output_folder="Graphs_Cities",
    simplify=False,
    filter_clusters=None,  # Lista de clusters específicos a incluir
    filter_poly_ids=None,  # Lista de IDs específicos
    max_polygons_per_cluster=None,  # Límite por cluster
    figsize=(15, 15),
    dpi=72,
    line_width=0.025,
    node_size=0,
    show_legend=True,
    background_color='white'
):
    """
    Genera PDFs vectoriales basados en los resultados del clustering:
    1) Versión completa con todos los polígonos clasificados por clustering
    2) Versión filtrada solo con los clusters/polígonos seleccionados
    
    Parameters:
    -----------
    cluster_results : dict
        Resultados del clustering que incluye:
        - 'geodataframe': GeoDataFrame con columnas 'cluster', 'cluster_name'
        - 'cluster_names': dict mapeo de cluster ID a nombre
        - Otros resultados del clustering
    """
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI
    plt.ioff()  # Sin modo interactivo

    # Configuración ultra-agresiva
    matplotlib.rcParams['path.simplify_threshold'] = 0.05  # Más agresivo
    matplotlib.rcParams['pdf.compression'] = 9
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['figure.max_open_warning'] = 0

    # Eliminar anti-aliasing globalmente
    matplotlib.rcParams['lines.antialiased'] = False
    matplotlib.rcParams['patch.antialiased'] = False
    
    # Backend optimizado para vectores masivos
    matplotlib.use('Agg')
    plt.ioff()
    print("Iniciando procesamiento optimizado para generar PDFs de clustering...")
    
    # Extraer datos del clustering
    gdf_clustered = cluster_results['geodataframe']
    cluster_names = cluster_results['cluster_names']
    
    # Leer GeoJSON original para obtener geometrías completas si es necesario
    print("Leyendo GeoJSON...")
    gdf_original = gpd.read_file(geojson_path)
    
    # Definir colores para clusters (basado en los nombres de cluster)
    cluster_colors = {}
    base_colors = {
        'cul_de_sac': "#F13F3F",   # Rojo para callejones sin salida
        'gridiron': "#0C850C",     # Verde oscuro para grid
        'organico': "#2399CF",     # Azul para orgánico
        'hibrido': "#E4CD4D"    # Amarillo para híbrido
    }
    
    # Asignar colores a cada cluster basado en su nombre
    import matplotlib.colors as mcolors
    import numpy as np
    


    for cluster_id, cluster_name in cluster_names.items():
        # Manejo especial para cul_de_sac (tiene dos guiones bajos)
        if cluster_name.startswith('cul_de_sac'):
            base_pattern = 'cul_de_sac'
            # Usar el mismo enfoque para obtener el sufijo
            parts = cluster_name.split('_')
            if len(parts) > 3:  # Si tiene sufijo (cul_de_sac_algo)
                suffix = parts[3]  # Tomar el cuarto elemento (índice 3)
                if base_pattern in base_colors:
                    if suffix.startswith('alto'):
                        # Color específico para cul_de_sac_alto (más oscuro) - 15% más notorio
                        base_color = np.array(mcolors.to_rgb(base_colors[base_pattern]))
                        adjusted_color = base_color * 0.5  # Era 0.7, ahora 0.55 (15% más oscuro)
                        adjusted_color = np.clip(adjusted_color, 0, 1)
                        cluster_colors[cluster_id] = mcolors.to_hex(adjusted_color)
                    elif suffix.startswith('bajo'):
                        # Color específico para cul_de_sac_bajo (más claro) - 15% más notorio
                        base_color = np.array(mcolors.to_rgb(base_colors[base_pattern]))
                        adjusted_color = base_color + (1 - base_color) * 0.6  # Era 0.4, ahora 0.55 (15% más claro)
                        adjusted_color = np.clip(adjusted_color, 0, 1)
                        cluster_colors[cluster_id] = mcolors.to_hex(adjusted_color)
                    else:
                        # Para otros sufijos de cul_de_sac, usar el color base
                        cluster_colors[cluster_id] = base_colors[base_pattern]
                else:
                    cluster_colors[cluster_id] = "#808080"
            else:
                # Si es solo 'cul_de_sac' sin sufijo
                if base_pattern in base_colors:
                    cluster_colors[cluster_id] = base_colors[base_pattern]
                else:
                    cluster_colors[cluster_id] = "#808080"
        else:
            # Para todos los demás patrones normales
            base_pattern = cluster_name.split('_')[0]
            
            if base_pattern in base_colors:
                base_color = np.array(mcolors.to_rgb(base_colors[base_pattern]))
                
                # Si tiene sufijo (variante), ajustar el color
                if '_' in cluster_name:
                    suffix = cluster_name.split('_')[1]  # Tomar el segundo elemento
                    if suffix.startswith('alto'):
                        # 15% más notorio: era 0.7, ahora 0.55
                        adjusted_color = base_color * 0.55
                    elif suffix.startswith('bajo'):
                        # 15% más notorio: era 0.3, ahora 0.45
                        adjusted_color = base_color + (1 - base_color) * 0.45
                    else:
                        # Alternar tonos
                        if suffix.isdigit():
                            factor = int(suffix) * 0.15
                        else:
                            factor = hash(suffix) % 3 * 0.15
                        adjusted_color = base_color + np.array([0, factor, -factor])
                    
                    adjusted_color = np.clip(adjusted_color, 0, 1)
                    cluster_colors[cluster_id] = mcolors.to_hex(adjusted_color)
                else:
                    cluster_colors[cluster_id] = base_colors[base_pattern]
            else:
                # Color por defecto para clusters sin patrón base reconocido
                cluster_colors[cluster_id] = "#808080"
    
    # Contadores para limitar polígonos por cluster
    cluster_counters = {cluster_id: 0 for cluster_id in cluster_names.keys()}
    
    # Estructuras de datos para almacenar información
    all_polygons_data = []
    
    # Variables para calcular el bounding box global
    global_x_min, global_x_max = float('inf'), float('-inf')
    global_y_min, global_y_max = float('inf'), float('-inf')
    
    print("Procesando polígonos con clasificación de clustering...")
    
    # Procesar cada polígono del GeoDataFrame con clustering
    for idx, row in gdf_clustered.iterrows():
        geom = row.geometry
        cluster_id = row.get('cluster', -1)
        cluster_name = row.get('cluster_name', 'unknown')
        
        if geom is None or geom.is_empty or cluster_id == -1:
            continue
            
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for sub_idx, poly in enumerate(polys):
            key = (idx, sub_idx)
            
            # Determinar si incluir en versión filtrada
            include_in_filtered = True
            
            # Filtro por clusters específicos
            if filter_clusters is not None and cluster_id not in filter_clusters:
                include_in_filtered = False
            
            # Filtro por ID específico
            if filter_poly_ids is not None:
                poly_id_options = [key, idx, f"{idx}-{sub_idx}", (idx, sub_idx), str(idx)]
                if not any(pid in filter_poly_ids for pid in poly_id_options):
                    include_in_filtered = False
            
            # Filtro por límite de polígonos por cluster
            if max_polygons_per_cluster is not None and include_in_filtered:
                if cluster_counters[cluster_id] >= max_polygons_per_cluster:
                    include_in_filtered = False
                else:
                    cluster_counters[cluster_id] += 1
            
            # Procesar el polígono y generar su grafo
            try:
                G_sub = ox.graph_from_polygon(poly, network_type=network_type, simplify=simplify)
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
                
                # Procesar edges para matplotlib
                edge_lines = []
                for u, v in G_sub.edges():
                    if u in sub_nodes and v in sub_nodes:
                        u_idx = sub_nodes.index(u)
                        v_idx = sub_nodes.index(v)
                        x0 = sub_positions[u_idx][1]
                        y0 = sub_positions[u_idx][0]
                        x1 = sub_positions[v_idx][1]
                        y1 = sub_positions[v_idx][0]
                        edge_lines.append([(x0, y0), (x1, y1)])
                
                # Almacenar datos del polígono
                polygon_data = {
                    'id': key,
                    'id_str': f"{idx}-{sub_idx}",
                    'polygon': poly,
                    'cluster_id': cluster_id,
                    'cluster_name': cluster_name,
                    'color': cluster_colors.get(cluster_id, "#808080"),
                    'nodes': sub_nodes,
                    'positions': sub_positions,
                    'edge_lines': edge_lines,
                    'include_in_filtered': include_in_filtered
                }
                
                all_polygons_data.append(polygon_data)
                
            except Exception as e:
                print(f"Error procesando polígono {idx}-{sub_idx}: {e}")
                continue
    
    print(f"Total de polígonos procesados: {len(all_polygons_data)}")
    
    # Crear directorio de salida
    os.makedirs(output_folder, exist_ok=True)
    
    # Función auxiliar para crear el plot
    # Función auxiliar para crear el plot
    
    def create_clustering_plot(polygons_data, filename):
        """Función auxiliar para crear un plot de Plotly con clustering ULTRA optimizado"""
        import plotly.graph_objects as go
        import plotly.io as pio
        import numpy as np
        import os
        from collections import defaultdict
        
        # OPTIMIZACIÓN CRÍTICA 1: Configurar Plotly para máximo rendimiento
        # Configuración básica sin acceso a kaleido.scope
        pio.templates.default = "plotly_white"  # Template limpio
        
        # Agrupar polígonos por cluster para la leyenda
        cluster_groups = {}
        for poly_data in polygons_data:
            cluster_name = poly_data['cluster_name']
            if cluster_name not in cluster_groups:
                cluster_groups[cluster_name] = []
            cluster_groups[cluster_name].append(poly_data)
        
        # OPTIMIZACIÓN CRÍTICA 2: Pre-procesar líneas para máxima eficiencia
        print("Agrupando líneas por color (clustering) - MODO ULTRA...")
        all_lines_by_color = {}
        cluster_counts = {}
        
        # OPTIMIZACIÓN: Convertir a arrays numpy una sola vez
        for cluster_name, poly_group in cluster_groups.items():
            if not poly_group:
                continue
                
            color = poly_group[0]['color']
            if color not in all_lines_by_color:
                all_lines_by_color[color] = []
            
            total_lines = 0
            for poly_data in poly_group:
                # Convertir líneas a numpy arrays de una vez (más eficiente)
                for line in poly_data['edge_lines']:
                    if len(line) >= 2:  # Solo líneas válidas
                        # Convertir a numpy array con dtype específico (más compacto)
                        line_array = np.array(line, dtype=np.float32)  # float32 vs float64
                        all_lines_by_color[color].append(line_array)
                        total_lines += 1
            
            cluster_counts[cluster_name] = len(poly_group)
            print(f"Cluster {cluster_name}: {total_lines} líneas procesadas")

        # OPTIMIZACIÓN CRÍTICA 3: Crear figura con configuración ultra eficiente
        print("Renderizando con configuración ULTRA optimizada...")
        fig = go.Figure()
        
        # Configurar layout ultra optimizado (equivalente a matplotlib)
        fig.update_layout(
            width=int(figsize[0] * 100),
            height=int(figsize[1] * 100),
            xaxis=dict(
                range=[global_x_min, global_x_max],
                showgrid=False,
                showticklabels=False,
                showline=False,
                zeroline=False,
                visible=False,
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                range=[global_y_min, global_y_max],
                showgrid=False,
                showticklabels=False,
                showline=False,
                zeroline=False,
                visible=False,
            ),
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=show_legend,
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(0,0,0,0.5)',
                borderwidth=0.5,
                font=dict(size=10)
            ) if show_legend else None,
            hovermode=False,  # Sin hover para mejor rendimiento
            dragmode=False,   # Sin interactividad
        )

        # OPTIMIZACIÓN CRÍTICA 4: Renderizado ultra eficiente
        legend_handles = []
        legend_added = set()

        for cluster_name, poly_group in cluster_groups.items():
            if not poly_group:
                continue
                
            color = poly_group[0]['color']
            if color in all_lines_by_color and all_lines_by_color[color]:
                lines = all_lines_by_color[color]
                print(f"Dibujando {len(lines)} líneas de {cluster_name} - ULTRA MODO")
                
                # CONFIGURACIÓN ULTRA AGRESIVA para PDFs ligeros:
                # OPTIMIZACIÓN EXTREMA: Simplificar líneas antes de renderizar
                x_coords = []
                y_coords = []
                
                for line in lines:
                    if len(line) >= 2:
                        # CRÍTICO: Simplificar líneas muy densas (Douglas-Peucker simplificado)
                        if len(line) > 10:  # Solo simplificar líneas complejas
                            # Tomar cada N-ésimo punto para líneas muy densas
                            step = max(1, len(line) // 8)  # Máximo 8 puntos por línea
                            simplified_line = line[::step]
                            # Asegurar que incluimos el último punto
                            if not np.array_equal(simplified_line[-1], line[-1]):
                                simplified_line = np.vstack([simplified_line, line[-1]])
                        else:
                            simplified_line = line
                        
                        x_coords.extend(simplified_line[:, 0])
                        y_coords.extend(simplified_line[:, 1])
                        # Agregar None para separar líneas (estándar Plotly)
                        x_coords.append(None)
                        y_coords.append(None)
                
                # Remover el último None
                if x_coords and x_coords[-1] is None:
                    x_coords.pop()
                    y_coords.pop()
                    
                print(f"Puntos simplificados para {cluster_name}: {len(x_coords)} puntos")
                
                # Crear traza única (equivalente a LineCollection de matplotlib)
                show_in_legend = cluster_name not in legend_added
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=color,
                        width=0.02,  # AÚN MÁS delgado (de 0.03 a 0.02)
                        simplify=True,  # CRÍTICO: Simplificación automática
                    ),
                    name=f'{cluster_name} ({cluster_counts.get(cluster_name, 0)})',
                    showlegend=show_in_legend and show_legend,
                    hoverinfo='skip',  # Sin hover
                    connectgaps=False,  # No conectar a través de None
                    # OPTIMIZACIONES ADICIONALES PARA VECTORIAL
                    fill=None,
                    opacity=1.0,
                ))
                
                if show_in_legend:
                    legend_added.add(cluster_name)
                
                # Limpiar memoria inmediatamente
                del all_lines_by_color[color]
        
        # OPTIMIZACIÓN CRÍTICA 5: Guardado ULTRA comprimido
        pdf_path = os.path.join(output_folder, filename)
        
        # Configurar guardado optimizado
        try:
            fig.write_image(
                pdf_path,
                format='pdf',
                width=int(figsize[0] * 80),  # Reducido de 100 a 80
                height=int(figsize[1] * 80),
                scale=0.8,  # Reducido de 1 a 0.8 para menor resolución
            )
        except Exception as e:
            print(f"Error con kaleido, intentando con orca: {e}")
            try:
                fig.write_image(
                    pdf_path,
                    format='pdf',
                    width=int(figsize[0] * 80),
                    height=int(figsize[1] * 80),
                    scale=0.8,
                    engine='orca'
                )
            except Exception as e2:
                print(f"Error con orca también, guardando como HTML: {e2}")
                # Fallback: guardar como HTML y convertir
                html_path = pdf_path.replace('.pdf', '.html')
                fig.write_html(html_path)
                print(f"Guardado como HTML: {html_path}")
                return html_path
        
        # OPTIMIZACIÓN CRÍTICA 6: Post-procesamiento del PDF (igual que matplotlib)
        try:
            # Verificar tamaño original
            original_size = os.path.getsize(pdf_path)
            print(f"Tamaño PDF original: {original_size/1024:.1f} KB")
            
            # Si el archivo es muy grande, intentar compresión adicional
            if original_size > 200000:  # Bajado de 500KB a 200KB para ser más agresivo
                temp_path = pdf_path.replace('.pdf', '_temp.pdf')
                
                # Usar ghostscript si está disponible (compresión máxima)
                import subprocess
                gs_commands = [
                    ['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4', 
                    '-dPDFSETTINGS=/prepress', '-dNOPAUSE', '-dQUIET', '-dBATCH',
                    f'-sOutputFile={temp_path}', pdf_path],
                    ['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
                    '-dPDFSETTINGS=/ebook', '-dNOPAUSE', '-dQUIET', '-dBATCH',
                    f'-sOutputFile={temp_path}', pdf_path]
                ]
                
                for cmd in gs_commands:
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        if os.path.exists(temp_path):
                            compressed_size = os.path.getsize(temp_path)
                            if compressed_size < original_size * 0.7:  # Si reduce >30% (más agresivo)
                                os.replace(temp_path, pdf_path)
                                print(f"PDF comprimido: {compressed_size/1024:.1f} KB (-{100*(1-compressed_size/original_size):.1f}%)")
                                break
                            else:
                                os.remove(temp_path)
                    except:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        continue
                        
        except Exception as e:
            print(f"Compresión adicional no disponible: {e}")
        
        # Liberar memoria AGRESIVAMENTE
        fig.data = []
        del fig
        
        # Forzar garbage collection
        import gc
        gc.collect()
        
        return pdf_path

    def create_clustering_plot_png(polygons_data, filename):
        """Función para crear un plot de clustering en PNG usando matplotlib con DPI configurable y colores completamente sólidos"""
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import numpy as np
        import os
        import gc
        from collections import defaultdict
        target_dpi=400
        print(f"Configurando PNG con matplotlib - {target_dpi} DPI...")
        
        # CONFIGURACIÓN MATPLOTLIB PARA COLORES SÓLIDOS
        plt.rcParams['figure.dpi'] = target_dpi
        plt.rcParams['savefig.dpi'] = target_dpi
        
        # Ajustar configuraciones según DPI
        if target_dpi <= 100:
            font_size = 14
            line_width = 2.0
            axes_line_width = 1.5
        elif target_dpi <= 200:
            font_size = 12
            line_width = 1.2
            axes_line_width = 1.0
        else:  # DPI alto (300+)
            font_size = 8
            line_width = 0.6
            axes_line_width = 0.5
        
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.linewidth'] = axes_line_width
        plt.rcParams['lines.linewidth'] = line_width
        
        # ANTIALIASING PARA CALIDAD
        plt.rcParams['lines.antialiased'] = True
        plt.rcParams['patch.antialiased'] = True
        plt.rcParams['text.antialiased'] = True
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize, facecolor=background_color, dpi=target_dpi)
        ax.set_facecolor(background_color)
        
        # Configurar aspecto y límites
        ax.set_aspect('equal')
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)
        
        # Remover ejes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # AGRUPAR DATOS POR CLUSTER
        print("Agrupando datos por cluster...")
        cluster_groups = {}
        for poly_data in polygons_data:
            cluster_name = poly_data['cluster_name']
            if cluster_name not in cluster_groups:
                cluster_groups[cluster_name] = {
                    'lines': [],
                    'color': poly_data['color'],
                    'count': 0
                }
            
            # Procesar líneas del polígono
            for line in poly_data['edge_lines']:
                if len(line) >= 2:
                    # Simplificación adaptativa según DPI
                    if target_dpi <= 100 and len(line) > 8:
                        # DPI bajo: simplificación agresiva
                        step = max(1, len(line) // 4)
                        simplified = line[::step]
                        if not np.array_equal(simplified[-1], line[-1]):
                            simplified = np.vstack([simplified, [line[-1]]])
                    elif target_dpi <= 200 and len(line) > 15:
                        # DPI medio: simplificación moderada
                        step = max(1, len(line) // 8)
                        simplified = line[::step]
                        if not np.array_equal(simplified[-1], line[-1]):
                            simplified = np.vstack([simplified, [line[-1]]])
                    elif target_dpi > 200 and len(line) > 25:
                        # DPI alto: simplificación mínima
                        step = max(1, len(line) // 15)
                        simplified = line[::step]
                        if not np.array_equal(simplified[-1], line[-1]):
                            simplified = np.vstack([simplified, [line[-1]]])
                    else:
                        simplified = line
                    
                    # Solo agregar líneas con longitud mínima
                    if len(simplified) >= 2:
                        start, end = simplified[0], simplified[-1]
                        length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        min_length = 0.5 if target_dpi <= 100 else 0.3 if target_dpi <= 200 else 0.15
                        
                        if length > min_length:
                            cluster_groups[cluster_name]['lines'].append(simplified)
            
            cluster_groups[cluster_name]['count'] += 1
        
        # RENDERIZADO CON MATPLOTLIB - COLORES GARANTIZADOS SÓLIDOS
        print("Renderizando con matplotlib - colores sólidos garantizados...")
        legend_handles = []
        
        for cluster_name, data in cluster_groups.items():
            lines = data['lines']
            color = data['color']
            count = data['count']
            
            if not lines:
                continue
            
            print(f"Renderizando cluster {cluster_name}: {len(lines)} líneas en color {color}")
            
            # CREAR LINECOLLECTION SIN TRANSPARENCIA
            lc = LineCollection(
                    lines,
                    colors=[color],
                    linewidths=1,         # Grosor normal para 40 DPI
                    alpha=None,             # ELIMINAMOS alpha para evitar transparencia
                    antialiased=True,
                    capstyle='butt',        # Extremos cuadrados (más sólidos)
                    joinstyle='miter',      # Uniones en ángulo (más definidas)
                    rasterized=True,        # Rasterizado para colores más sólidos
                    picker=False
                )
                
            # AÑADIR AL PLOT SIN MODIFICAR COLOR
            ax.add_collection(lc)
            
            # Crear handle para leyenda CON COLOR SÓLIDO
            import matplotlib.colors as mcolors

            # Convertir a HSV y aumentar saturación
            hsv = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
            hsv[1] = min(1.0, hsv[1] * 1.3)  # Aumentar saturación 30%
            enhanced_color = mcolors.hsv_to_rgb(hsv)

            legend_handles.append(
                plt.Line2D(
                    [0], [0], 
                    color=enhanced_color,
                    linewidth=max(4, line_width * 3),
                    label=f'{cluster_name} ({count})',
                    alpha=1.0,
                    solid_capstyle='round'
                )
            )
        



        # LEYENDA CON COLORES SÓLIDOS
        if show_legend and legend_handles:
            print("Agregando leyenda con colores sólidos...")
            legend = ax.legend(
                handles=legend_handles,
                loc='upper right',
                frameon=True,
                fontsize=font_size,
                bbox_to_anchor=(0.98, 0.98),
                borderaxespad=0,
                handlelength=2.0,
                handletextpad=0.6,
                columnspacing=1.0,
                framealpha=1.0,             # MARCO COMPLETAMENTE OPACO
                facecolor='white',          # FONDO BLANCO SÓLIDO
                edgecolor='black'           # BORDE NEGRO SÓLIDO
            )
            
            # ASEGURAR FRAME SÓLIDO
            legend.get_frame().set_linewidth(1.0)
            legend.get_frame().set_alpha(1.0)  # COMPLETAMENTE OPACO
        
        # GUARDADO PNG CON CONFIGURACIÓN PARA COLORES SÓLIDOS
        print(f"Guardando PNG {target_dpi} DPI con colores sólidos...")
        
        png_filename = filename.replace('.pdf', '.png').replace('.svg', '.png')
        if not png_filename.endswith('.png'):
            png_filename = f"{filename}.png"
        png_path = os.path.join(output_folder, png_filename)
        
        # CONFIGURACIÓN DE GUARDADO PARA MÁXIMA SOLIDEZ DE COLORES
        plt.savefig(
            png_path, 
            format='png',
            dpi=300,                     # DPI reducido
            bbox_inches='tight',
            pad_inches=0.05,            # Padding ligeramente mayor
            facecolor=background_color,
            edgecolor='none',
            transparent=False,          # CRUCIAL: Sin transparencia
            # CONFIGURACIONES PARA COLORES SÓLIDOS:
            pil_kwargs={
                'compress_level': 1,     # Menos compresión para colores más puros
                'optimize': False        # Sin optimización que pueda afectar colores
            }
        )
        
        # VERIFICACIÓN DEL ARCHIVO
        try:
            if os.path.exists(png_path):
                file_size = os.path.getsize(png_path)
                print(f"PNG {target_dpi} DPI generado - Tamaño: {file_size/1024:.1f} KB")
                
                # Verificación de tamaño según DPI
                expected_min = target_dpi * 5   # Mínimo esperado
                expected_max = target_dpi * 2000  # Máximo esperado
                
                if file_size < expected_min:
                    print(f"ADVERTENCIA: Archivo muy pequeño para {target_dpi} DPI")
                elif file_size > expected_max:
                    print(f"INFO: Archivo grande para {target_dpi} DPI (normal si hay muchos datos)")
                else:
                    print(f"Tamaño de archivo apropiado para {target_dpi} DPI")
                    
            else:
                print("ERROR: No se pudo crear el archivo PNG")
                
        except Exception as e:
            print(f"Error verificando archivo: {e}")
        
        # LIMPIEZA COMPLETA DE MEMORIA
        ax.clear()
        plt.close(fig)
        plt.clf()
        plt.cla()
        
        # Limpiar configuración matplotlib
        plt.rcdefaults()
        
        # Garbage collection agresivo
        gc.collect()
        
        print(f"PNG con colores sólidos guardado: {png_path}")
        return png_path
    
    
    # Generar PDF completo (todos los polígonos con clustering)
    print("Generando PDF COMPLETO con clasificación de clustering...")
    full_filename = f"ClusteringPatterns_{place_name}_Completo.pdf"
    full_pdf = create_clustering_plot_png(all_polygons_data, full_filename)
    print(f"PDF completo generado: {full_pdf}")
    
    # Generar PDF filtrado
    print("Generando PDF FILTRADO con clustering...")
    filtered_polygons = [p for p in all_polygons_data if p['include_in_filtered']]
    
    if not filtered_polygons:
        print("No hay polígonos que cumplan con los criterios de filtrado.")
        return {
            'all_polygons': all_polygons_data,
            'filtered_polygons': [],
            'full_pdf': full_pdf,
            'filtered_pdf': None
        }
    
    # Resumen de polígonos filtrados
    print(f"Polígonos incluidos en el PDF filtrado: {len(filtered_polygons)}")
    filtered_by_cluster = {}
    for p in filtered_polygons:
        cluster_name = p['cluster_name']
        if cluster_name not in filtered_by_cluster:
            filtered_by_cluster[cluster_name] = []
        filtered_by_cluster[cluster_name].append(p['id_str'])
    
    for cluster_name, ids in filtered_by_cluster.items():
        print(f"- {cluster_name}: {len(ids)} polígonos")
    
    # Crear nombre de archivo para versión filtrada
    filename = f"ClusteringPatterns_{place_name}_Filtrado"
    if filter_clusters:
        filename += f"_clusters_{'_'.join(map(str, filter_clusters))}"
    if filter_poly_ids:
        filename += "_specific_polys"
    if max_polygons_per_cluster:
        filename += f"_max{max_polygons_per_cluster}"
    filename += ".pdf"
    
    filtered_pdf = create_clustering_plot(filtered_polygons, filename)
    
    print(f"PDF filtrado generado: {filtered_pdf}")
    print("Proceso completado. Se generaron 2 archivos PDF vectoriales:")
    print(f"1. {full_pdf} - Versión completa con todos los clusters")
    print(f"2. {filtered_pdf} - Versión filtrada")
    
    return {
        'all_polygons': all_polygons_data,
        'filtered_polygons': filtered_polygons,
        'full_pdf': full_pdf,
        'filtered_pdf': filtered_pdf,
        'cluster_summary': filtered_by_cluster
    }


# Función auxiliar para integrar con tu código existente
def add_clustering_pdf_to_main(cluster_results, geojson_file, graph_dict, ciudad):
    """
    Función para añadir la generación de PDFs de clustering a tu flujo principal
    """
    output_folder = f"Polygons_analysis/{ciudad}/clustering_pdfs"
    
    print(f"\nGenerando PDFs vectoriales para clustering de {ciudad}...")
    
    pdf_results = plot_clustering_patterns_pdf_optimized(
        geojson_path=geojson_file,
        cluster_results=cluster_results,
        graph_dict=graph_dict,
        place_name=ciudad,
        output_folder=output_folder,
        show_legend=True,
        # Opcional: filtros para versión filtrada
        # filter_clusters=[0, 1, 2],  # Solo clusters específicos
        # max_polygons_per_cluster=50,  # Máximo 50 polígonos por cluster
    )
    
    return pdf_results




def urban_pattern_clustering(
    place_name,
    stats_dict, 
    graph_dict,
    classify_func, 
    geojson_file,
    n_clusters=None,
    output_dir="Resultados/urbano_pattern_cluster"):
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
        X, feature_names, n_clusters=n_clusters, output_dir=output_dir
    )
    
    # Clasificación original de patrones
    original_patterns = []
    valid_poly_ids = []
    
    # Essential block for running the analysis. Avoid making changes here.
    # Block necesary to extract the G = graph for each polygon with te Poly_id and main_id
    

    for poly_id in poly_ids:
        poly_stats = stats_dict[poly_id]
        
        # PROBLEMA: graph_dict podría no tener la misma estructura que poly_id
        # Solución: implementar una verificación más robusta
        
        # 1. Extraer el ID principal del polígono
        if isinstance(poly_id, tuple) and len(poly_id) >= 1:
            main_id = poly_id[0]
        else:
            main_id = poly_id
        
        # 2. Convertir a posibles formatos (como string) para ser compatible con graph_dict
        possible_keys = [main_id, str(main_id)]
        
        # 3. Buscar el grafo usando las posibles claves
        G = None
        for key in possible_keys:
            if key in graph_dict:
                G = graph_dict[key]
                break
        
        # 4. Verificar explícitamente que G sea un objeto grafo antes de pasarlo
        if G is not None:
            if hasattr(G, 'number_of_nodes'):  # Verificar que es un grafo válido
                category = classify_func(poly_stats, G)
            else:
                print(f"Advertencia: El objeto para {poly_id} no es un grafo válido.")
                category = classify_func(poly_stats, None)
        else:
            # No se encontró grafo, pasar None
            category = classify_func(poly_stats, None)
        
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
    
    # Añadir después de crear cluster_names (línea ~96)
    # Crear un archivo de texto explicativo de las características de cada cluster
    with open(os.path.join(output_dir, 'cluster_explanations.txt'), 'w') as f:
        f.write("EXPLICACIÓN DETALLADA DE CLUSTERS\n")
        f.write("================================\n\n")
        
        for cluster in range(n_clusters):
            if cluster in cluster_dominant_pattern:
                pattern, percentage = cluster_dominant_pattern[cluster]
                cluster_name = cluster_names[cluster]
                
                f.write(f"Cluster {cluster}: {cluster_name}\n")
                f.write(f"Patrón dominante: {pattern} ({percentage:.1f}%)\n")
                
                # Obtener el centro de este cluster
                cluster_center = centers_df.loc[f'Cluster {cluster}']
                
                # Comparar con otros clusters del mismo patrón base
                similar_clusters = [c for c, (p, _) in cluster_dominant_pattern.items() 
                                if p == pattern and c != cluster]
                
                if similar_clusters:
                    f.write(f"Comparación con otros clusters de tipo '{pattern}':\n")
                    
                    # Calcular estadísticas para todos los clusters de este patrón
                    pattern_centers = centers_df.loc[[f'Cluster {c}' for c 
                                                in similar_clusters + [cluster]]]
                    pattern_avg = pattern_centers.mean()
                    pattern_std = pattern_centers.std()
                    
                    # Encontrar las características más distintivas
                    differences = []
                    for feature in feature_names:
                        # Valor de este cluster
                        value = cluster_center[feature]
                        # Promedio para este patrón
                        avg = pattern_avg[feature]
                        # Desviación estándar 
                        std = pattern_std[feature]
                        
                        # Normalizar la diferencia
                        if avg != 0 and std != 0:
                            z_score = (value - avg) / std
                            percent_diff = ((value / avg) - 1) * 100
                            
                            if abs(z_score) > 0.5:  # Diferencia significativa
                                label = "mayor" if z_score > 0 else "menor"
                                differences.append({
                                    'feature': feature,
                                    'value': value,
                                    'avg': avg,
                                    'z_score': z_score,
                                    'percent_diff': percent_diff,
                                    'label': label
                                })
                    
                    # Ordenar por magnitud de diferencia (absoluta)
                    differences.sort(key=lambda x: abs(x['z_score']), reverse=True)
                    
                    # Mostrar las 5 características más distintivas
                    for i, diff in enumerate(differences[:5]):
                        f.write(f"  - {diff['feature']}: {diff['value']:.3f} ")
                        f.write(f"({diff['label']} que el promedio de {diff['avg']:.3f} ")
                        f.write(f"por {abs(diff['percent_diff']):.1f}%, ")
                        f.write(f"z-score: {diff['z_score']:.2f})\n")
                        
                    # Explicar específicamente el sufijo
                    if '_' in cluster_name and cluster_name != pattern:
                        suffix = cluster_name.split('_')[1]
                        # Verificar que el sufijo tiene el formato esperado (alto_X o bajo_X)
                        if suffix.startswith('alto') or suffix.startswith('bajo'):
                            # Verificar que el sufijo tiene un guion bajo y al menos dos partes
                            suffix_parts = suffix.split('_')
                            if len(suffix_parts) > 1:
                                feature_part = suffix_parts[1]
                                matching_features = [d for d in differences 
                                                if feature_part in d['feature']]
                                
                                if matching_features:
                                    f.write(f"\nEXPLICACIÓN DEL NOMBRE '{cluster_name}':\n")
                                    f.write(f"  Este cluster tiene un valor {'superior' if suffix.startswith('alto') else 'inferior'} ")
                                    f.write(f"al promedio en variables relacionadas con '{feature_part}'.\n")
                                    f.write(f"  Específicamente:\n")
                                    
                                    for feat in matching_features[:2]:
                                        f.write(f"    - {feat['feature']}: {feat['value']:.3f} vs. promedio de {feat['avg']:.3f} ")
                                        f.write(f"({abs(feat['percent_diff']):.1f}% de diferencia)\n")
                            else:
                                # Si el sufijo no tiene el formato esperado (alto_X o bajo_X donde X es la característica)
                                f.write(f"\nEXPLICACIÓN DEL NOMBRE '{cluster_name}':\n")
                                f.write(f"  Este cluster representa una variante del patrón base '{pattern}' ")
                                f.write(f"con la característica distintiva '{suffix}'.\n")
                else:
                    f.write("  Este es el único cluster con este patrón base.\n")
                    
                    # Características más destacadas de este cluster
                    top_features = centers_df.loc[f'Cluster {cluster}'].sort_values(ascending=False)
                    f.write("\nCaracterísticas principales:\n")
                    for feature, value in top_features[:5].items():
                        f.write(f"  - {feature}: {value:.3f}\n")
                
                f.write("\n" + "-"*50 + "\n\n")
            
        # Añadir una tabla de resumen de todos los clusters
        f.write("\nRESUMEN DE TODOS LOS CLUSTERS\n")
        f.write("=========================\n\n")
        f.write("Cluster | Nombre | Patrón Base | Característica Distintiva\n")
        f.write("-" * 60 + "\n")
        
        for cluster in range(n_clusters):
            if cluster in cluster_dominant_pattern:
                pattern, percentage = cluster_dominant_pattern[cluster]
                name = cluster_names[cluster]
                
                # Extraer la característica distintiva de manera segura
                distinctive = "N/A"
                if '_' in name and name != pattern:
                    suffix = name.split('_')[1]
                    if '_' in suffix:
                        distinctive = suffix.split('_')[1]
                    else:
                        distinctive = suffix
                
                f.write(f"{cluster} | {name} | {pattern} | {distinctive}\n")

        # Añadir explicación sobre los sufijos
        f.write("\n\nNOTA SOBRE NOMENCLATURA:\n")
        f.write("- 'alto_[característica]': Indica que este cluster tiene valores significativamente mayores\n")
        f.write("  que el promedio de otros clusters del mismo patrón base para esta característica.\n")
        f.write("- 'bajo_[característica]': Indica que este cluster tiene valores significativamente menores\n")
        f.write("  que el promedio de otros clusters del mismo patrón base para esta característica.\n")

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
         'cul_de_sac': "#F13F3F",   # Rojo para callejones sin salida
        'gridiron': "#0C850C",     # Verde oscuro para grid
        'organico': "#2399CF",
        'hibrido': "#E4CD4D" ,    # Amarillo para híbrido
        'unknown': '#CCCCCC'       # Gris para desconocidos
    }
    
    
    # Añadir colores para nombres de cluster
    for cluster, name in cluster_names.items():
    # Manejo especial para cul_de_sac (tiene dos guiones bajos)
        if name.startswith('cul_de_sac'):
            pattern = 'cul_de_sac'
            # Usar el mismo enfoque para obtener el sufijo
            parts = name.split('_')
            if len(parts) > 3:  # Si tiene sufijo (cul_de_sac_algo)
                suffix = parts[3]  # Tomar el cuarto elemento (índice 3)
                if suffix.startswith('alto'):
                    color_map[name] = "#610f0f"  # Color específico para cul_de_sac_alto
                elif suffix.startswith('bajo'):
                    color_map[name] = '#ff9797'  # Color específico para cul_de_sac_bajo
                else:
                    # Para otros sufijos de cul_de_sac, usar el color base
                    color_map[name] = color_map[pattern]
            else:
                # Si es solo 'cul_de_sac' sin sufijo
                color_map[name] = color_map[pattern]
        else:
            # Para todos los demás patrones normales
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

    # ---- NUEVAS ADICIONES: ANÁLISIS DE EXACTITUD ----
    
    # Crear un mapeo de cluster a patrón dominante para cada polígono
    results_df['cluster_pattern'] = results_df['cluster'].map(
        lambda c: cluster_dominant_pattern.get(c, ('unknown', 0))[0]
    )
    
    # Calcular coincidencias entre clasificación original y clustering
    results_df['match'] = results_df['original_pattern'] == results_df['cluster_pattern']
    
    # Estadísticas de exactitud
    accuracy = results_df['match'].mean() * 100
    print(f"\nExactitud global: {accuracy:.2f}%")
    
    # Análisis de exactitud por patrón original
    pattern_accuracy = {}
    for pattern in set(results_df['original_pattern']):
        pattern_df = results_df[results_df['original_pattern'] == pattern]
        pattern_accuracy[pattern] = pattern_df['match'].mean() * 100
        print(f"Exactitud para {pattern}: {pattern_accuracy[pattern]:.2f}%")
    
    # Crear DataFrame de exactitud para visualización
    accuracy_df = pd.DataFrame({
        'Patrón': list(pattern_accuracy.keys()) + ['Global'],
        'Exactitud (%)': list(pattern_accuracy.values()) + [accuracy]
    })
    
    # Visualización de comparación con tres elementos:
    # 1. Patrones originales
    # 2. Clusters
    # 3. Gráfico de exactitud
    fig = plt.figure(figsize=(16, 9))
    import matplotlib.gridspec as gridspec

    # Definir cuadrícula: 2 filas, 3 columnas con diferentes tamaños
    gs = gridspec.GridSpec(1, 3, height_ratios=[1])
    
    # Mapa de patrones originales
    ax1 = plt.subplot(gs[0, 0])
    
    # Función para asignar colores
    def get_color(value, color_map):
        return color_map.get(value, '#CCCCCC')
    
    gdf_filtered['color_pattern'] = gdf_filtered['original_pattern'].apply(
        lambda x: get_color(x, color_map)
    )
    
    gdf_filtered.plot(color=gdf_filtered['color_pattern'], 
                     ax=ax1,
                     edgecolor='black', 
                     linewidth=0.5)
    
    # Crear leyenda para patrones
    pattern_legend_elements = [
        Patch(facecolor=color_map[pattern], edgecolor='black', label=pattern)
        for pattern in sorted(list(set(gdf_filtered['original_pattern'])))
        if pattern in color_map
    ]
    ax1.legend(handles=pattern_legend_elements, loc='lower right', title="Patrones originales")
    ax1.set_title('Patrones de calle teóricos', fontsize=16)
    ax1.axis('off')
    
    # Mapa de clusters
    ax2 = plt.subplot(gs[0, 1])
    gdf_filtered['color_cluster'] = gdf_filtered['cluster_name'].apply(
        lambda x: get_color(x, color_map)
    )
    
    gdf_filtered.plot(color=gdf_filtered['color_cluster'], 
                     ax=ax2,
                     edgecolor='black', 
                     linewidth=0.5)
    
    # Crear leyenda para clusters
    cluster_legend_elements = [
        Patch(facecolor=get_color(name, color_map), edgecolor='black', label=name)
        for name in sorted(list(set(gdf_filtered['cluster_name'])))
        if name != 'unknown'
    ]
    ax2.legend(handles=cluster_legend_elements, loc='lower right', title="Clusters identificados")
    ax2.set_title('Agrupación por características morfológicas', fontsize=16)
    ax2.axis('off')
    
    # Gráfico de barras de exactitud
    ax3 = plt.subplot(gs[0, 2])
    
    # Ordenar por exactitud para mejor visualización
    accuracy_df = accuracy_df.sort_values('Exactitud (%)', ascending=False)
    
    # Definir colores para las barras
    bar_colors = [color_map.get(pattern, '#333333') for pattern in accuracy_df['Patrón']]
    bar_colors[-1] = '#000000'  # Color negro para la exactitud global
    
    # Crear gráfico de barras
    bars = ax3.bar(accuracy_df['Patrón'], accuracy_df['Exactitud (%)'], color=bar_colors)
    ax3.set_ylabel('Porcentaje de Coincidencia (%)')
    ax3.set_ylim(0, 100)
    
    # Añadir etiquetas de valores
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.1f%%')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'urban_pattern_comparison_accuracy.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
       
    # Visualización original de comparación (mantenida para compatibilidad)
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # Mapa de patrones originales
    gdf_filtered.plot(color=gdf_filtered['color_pattern'], 
                     ax=axes[0],
                     edgecolor='black', 
                     linewidth=0.5)
    
    axes[0].legend(handles=pattern_legend_elements, loc='lower right', title="Patrones originales")
    axes[0].set_title('Patrones de calle teóricos', fontsize=16)
    axes[0].axis('off')
    
    # Mapa de clusters
    gdf_filtered.plot(color=gdf_filtered['color_cluster'], 
                     ax=axes[1],
                     edgecolor='black', 
                     linewidth=0.5)
    
    axes[1].legend(handles=cluster_legend_elements, loc='lower right', title="Clusters identificados")
    axes[1].set_title('Agrupación por características morfológicas', fontsize=16)
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
    
    # Crear DataFrames detallados para las nuevas hojas de Excel
    # 1. Hoja de polígonos con clasificación original
    polygons_original_df = results_df[['poly_id', 'subpoly_id', 'original_pattern']].copy()
    
    # 2. Hoja de polígonos con clustering
    polygons_cluster_df = results_df[['poly_id', 'subpoly_id', 'cluster', 'cluster_pattern']].copy()
    polygons_cluster_df['cluster_name'] = polygons_cluster_df['cluster'].map(
        lambda c: cluster_names.get(c, f"cluster_{c}")
    )
    
    # 3. Hoja de comparación de exactitud
    comparison_df = results_df[['poly_id', 'subpoly_id', 'original_pattern', 'cluster_pattern', 'match']].copy()
    comparison_df['match_text'] = comparison_df['match'].map({True: 'Coincide', False: 'No coincide'})
    
    # Guardar resultados en formato Excel
    with pd.ExcelWriter(os.path.join(output_dir, 'urban_pattern_analysis.xlsx')) as writer:
        # Hojas originales
        summary_df.to_excel(writer, sheet_name='Pattern_Features')
        pattern_cluster_matrix.to_excel(writer, sheet_name='Pattern_Cluster_Matrix')
        centers_df.to_excel(writer, sheet_name='Cluster_Centers')
        
        pd.DataFrame(important_features, columns=['Feature', 'Importance']).to_excel(
            writer, sheet_name='Feature_Importance'
        )
        
        # Nuevas hojas
         # Nuevas hojas
        polygons_original_df.to_excel(writer, sheet_name='Polygons_Original', index=False)
        polygons_cluster_df.to_excel(writer, sheet_name='Polygons_Cluster', index=False)
        comparison_df.to_excel(writer, sheet_name='Classification_Compare', index=False)
        accuracy_df.to_excel(writer, sheet_name='Accuracy_Summary', index=False)
    
    # Guardar GeoJSON con resultados
    gdf_filtered.to_file(
        os.path.join(output_dir, 'urban_patterns_clustered.geojson'),
        driver='GeoJSON'
    )
    
    print("\nGenerando PDFs vectoriales de clustering...")
    try:
        
        
        pdf_results = plot_clustering_patterns_pdf_optimized(
            geojson_path=geojson_file,
            simplify=False,
            cluster_results={
                'geodataframe': gdf_filtered,
                'cluster_names': cluster_names,
                'cluster_centers': centers_df,
                'pattern_cluster_matrix': pattern_cluster_matrix,
                'important_features': important_features,
                'n_clusters': n_clusters
            },
            graph_dict=graph_dict,
            place_name=place_name if 'ciudad' in locals() else "Unknown",
            output_folder=os.path.join(output_dir, "clustering_pdfs"),
            show_legend=False
        )
        
        print(f"PDFs generados exitosamente:")
        print(f"- Completo: {pdf_results['full_pdf']}")
        if pdf_results['filtered_pdf']:
            print(f"- Filtrado: {pdf_results['filtered_pdf']}")
            
    except Exception as e:
        print(f"Error generando PDFs: {e}")

    

    return {
        'geodataframe': gdf_filtered,
        'cluster_centers': centers_df,
        'pattern_cluster_matrix': pattern_cluster_matrix,
        'cluster_names': cluster_names,
        'important_features': important_features,
        'n_clusters': n_clusters,
        'accuracy': accuracy,
        'pattern_accuracy': pattern_accuracy
    }

# Lista de ciudades a procesar
ciudades = [
    # "Moscow_ID",
    # "Peachtree_GA",
    # "Philadelphia_PA",
    "Boston_MA",
    "Chandler_AZ",
    "Salt_Lake_UT",
    # "Santa_Fe_NM",
    # "Medellin_ANT",
    # "Charleston_SC",
    # "Cary_Town_NC",
    # "Fort_Collins_CO"
]

def main_clustering():
    # Procesamos cada ciudad
    for ciudad in ciudades:
        print(f"\n{'='*80}")
        print(f"Procesando ciudad: {ciudad}")
        print(f"{'='*80}")
        
        # Convertir nombre de ciudad a formato para rutas de archivos
        ciudad_lower = ciudad.lower().replace("_", "_")
        
        # Construir rutas de archivos
        geojson_file = f"GeoJSON_Export/{ciudad_lower}/tracts/{ciudad_lower}_tracts.geojson"
        stats_txt = f"Polygons_analysis/{ciudad}/stats/Polygon_Analisys_{ciudad}_sorted.txt"

        # Crear directorio de salida
        output_dir = f"Polygons_analysis/{ciudad}/clustering_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Archivos de entrada:")
        print(f"  - GeoJSON: {geojson_file}")
        print(f"  - Stats: {stats_txt}")
        print(f"Directorio de salida: {output_dir}")
        
        try:
            # Cargar el GeoJSON
            print("\nCargando archivo GeoJSON...")


            gdf = gpd.read_file(geojson_file)
            print(f"GeoDataFrame cargado con {len(gdf)} polígonos")
            
            # Procesar polígonos y generar grafos
            print("Procesando polígonos y generando grafos...")
            graph_dict = procesar_poligonos_y_generar_grafos(gdf)

            # Cargar estadísticas
            print("Cargando estadísticas de polígonos...")
            stats_dict = load_polygon_stats_from_txt(stats_txt)
            
            # Ejecutar el análisis de clustering
            print(f"Ejecutando urban_pattern_clustering para {ciudad}...")

            place_name=ciudad_lower
            urban_pattern_clustering(
                place_name,
                stats_dict, 
                graph_dict,
                classify_polygon, 
                geojson_file,
                n_clusters=None,  # Automáticamente determinará el número óptimo
                output_dir=output_dir  # Directorio específico para esta ciudad
            )

           
            
            print(f"\nAnálisis completado para {ciudad}.")
            print(f"Resultados guardados en: {output_dir}")
            
        except Exception as e:
            print(f"\nERROR al procesar {ciudad}: {str(e)}")
            print("Continuando con la siguiente ciudad...")
    
    print("\nProcesamiento de todas las ciudades completado.")

# if __name__ == "__main__":
#     main_clustering()














# # 1. Cargar los polígonos
# geojson_path = "GeoJSON_Export/medellin_ant/tracts/medellin_ant_tracts.geojson"
# gdf = gpd.read_file(geojson_path)

# # 2. Procesar polígonos y generar grafos
# graph_dict = procesar_poligonos_y_generar_grafos(gdf)
# print("Diagnóstico de graph_dict:")
# print(f"Tipo de graph_dict: {type(graph_dict)}")
# if graph_dict:
#     # Mostrar algunas claves para diagnóstico
#     sample_keys = list(graph_dict.keys())[:3]
#     print(f"Ejemplos de claves en graph_dict: {sample_keys}")
#     for key in sample_keys:
#         value = graph_dict[key]
#         print(f"Tipo de valor para clave {key}: {type(value)}")
#         if hasattr(value, 'number_of_nodes'):
#             print(f"  Es un grafo válido con {value.number_of_nodes()} nodos")
#         else:
#             print(f"  NO es un grafo válido")


# stats_txt = "Polygons_analysis/Medellin_ANT/stats/Polygon_Analisys_Medellin_ANT_sorted.txt"

# stats_dict = load_polygon_stats_from_txt(stats_txt)



# # 4. Clasificar y visualizar
# gdf_with_patterns = plot_polygons_classification_png(
#     geojson_path,
#     stats_dict,
#     classify_polygon,
#     output_png="clasificacion_morfologica.png",
#     graph_dict=graph_dict
# )











def plot_street_patterns_optimized(
    geojson_path,
    classify_func,
    graph_dict,
    stats_dict,
    place_name="MyPlace",
    network_type="drive",
    output_folder="Graphs_Cities",
    simplify=True,
    filter_patterns=None,  # Lista de patrones a incluir en la versión filtrada
    filter_poly_ids=None,  # Lista de IDs específicos para la versión filtrada
    max_polygons_per_pattern=None  # Límite por patrón para la versión filtrada
):
    """
    Genera DOS archivos HTML interactivos en una sola pasada:
    1) Versión completa con todos los polígonos (igual que la función original)
    2) Versión filtrada solo con los polígonos seleccionados
    
    Esto evita duplicar cálculos costosos y reduce significativamente el tiempo total.
    """
    print("Iniciando procesamiento optimizado para generar dos visualizaciones...")
    
    # Leer y preparar datos (solo una vez)
    print("Leyendo GeoJSON y construyendo la red completa...")
    gdf = gpd.read_file(geojson_path)
    try:
        poly_union = gdf.union_all()
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
    # Primer paso: Procesar TODOS los polígonos y almacenar sus datos
    # --------------------------------------------------------------------------------
    print("Procesando todos los polígonos y clasificándolos (una sola vez)...")
    
    pattern_colors = {
        'cul_de_sac': '#FF6B6B',   # Rojo para callejones sin salida
        'gridiron': '#006400',     # Verde oscuro para grid
        'organico': '#0F2F76',     # Azul para orgánico
        'hibrido': '#FDCB6E'     # Amarillo para híbrido
    }
    default_color = "gray"
    
    # Contadores para limitar polígonos por patrón en la versión filtrada
    pattern_counters = {pattern: 0 for pattern in pattern_colors.keys()}
    
    # Estructuras de datos para almacenar toda la información
    all_polygons_data = []  # Todos los polígonos procesados
    classified_edges_all = set()  # Todos los segmentos de bordes clasificados
    
    # Para cada polígono, procesarlo y almacenar sus datos
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
            
            # Clasificar el polígono (igual que antes, acceso robusto al grafo)
            poly_class = None
            if key in stats_dict:
                poly_stats = stats_dict[key]
                
                # Método robusto para acceder al grafo
                if isinstance(key, tuple) and len(key) >= 1:
                    main_id = key[0]
                else:
                    main_id = key
                
                possible_keys = [main_id, str(main_id), key, str(key)]
                G = None
                for possible_key in possible_keys:
                    if possible_key in graph_dict:
                        G = graph_dict[possible_key]
                        break
                
                if G is not None and hasattr(G, 'number_of_nodes'):
                    poly_class = classify_func(poly_stats, G)
                else:
                    print(f"Advertencia: No se encontró grafo válido para el polígono {key}, usando None.")
                    poly_class = classify_func(poly_stats, None)
            
            # Determinar si este polígono debe incluirse en la versión filtrada
            include_in_filtered = True
            
            # Filtro por patrón
            if filter_patterns is not None and poly_class not in filter_patterns:
                include_in_filtered = False
            
            # Filtro por ID específico
            if filter_poly_ids is not None:
                poly_id_options = [key, idx, f"{idx}-{sub_idx}", (idx, sub_idx), str(idx)]
                if not any(pid in filter_poly_ids for pid in poly_id_options):
                    include_in_filtered = False
            
            # Filtro por límite de polígonos por patrón
            if max_polygons_per_pattern is not None and poly_class is not None and include_in_filtered:
                if pattern_counters[poly_class] >= max_polygons_per_pattern:
                    include_in_filtered = False
                else:
                    pattern_counters[poly_class] += 1
            
            # Procesar el polígono y generar su grafo
            try:
                G_sub = ox.graph_from_polygon(poly, network_type=network_type, simplify=simplify)
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
                
                # Procesar bordes
                edge_data = []
                for u, v in G_sub.edges():
                    if u in sub_nodes and v in sub_nodes:
                        u_idx = sub_nodes.index(u)
                        v_idx = sub_nodes.index(v)
                        x0 = sub_positions[u_idx][1]
                        y0 = sub_positions[u_idx][0]
                        x1 = sub_positions[v_idx][1]
                        y1 = sub_positions[v_idx][0]
                        norm_edge = normalize_edge(x0, y0, x1, y1)
                        classified_edges_all.add(norm_edge)
                        edge_data.append((x0, y0, x1, y1, norm_edge))
                
                # Almacenar todos los datos del polígono para usar después
                polygon_data = {
                    'id': key,
                    'id_str': f"{idx}-{sub_idx}",
                    'polygon': poly,
                    'class': poly_class,
                    'color': pattern_colors.get(poly_class, default_color),
                    'nodes': sub_nodes,
                    'positions': sub_positions,
                    'edges': edge_data,
                    'include_in_filtered': include_in_filtered
                }
                
                all_polygons_data.append(polygon_data)
                
            except Exception as e:
                print(f"Error procesando polígono {idx}-{sub_idx}: {e}")
                continue
    
    print(f"Total de polígonos procesados: {len(all_polygons_data)}")
    
    # --------------------------------------------------------------------------------
    # Generar la visualización completa (todos los polígonos)
    # --------------------------------------------------------------------------------
    print("Generando visualización COMPLETA con todos los polígonos...")
    
    # Preparar trazas para la versión completa
    all_traces = []
    
    # Añadir todas las trazas de polígonos
    for poly_data in all_polygons_data:
        # Extraer datos del polígono
        edge_x = []
        edge_y = []
        for x0, y0, x1, y1, _ in poly_data['edges']:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scattergl(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1.0, color=poly_data['color']),
            hoverinfo='none',
            name=f"Polígono {poly_data['id_str']} ({poly_data['class']})" if poly_data['class'] else f"Polígono {poly_data['id_str']}"
        )
        
        x_sub = poly_data['positions'][:,1]
        y_sub = poly_data['positions'][:,0]
        node_trace = go.Scattergl(
            x=x_sub, y=y_sub,
            mode='markers',
            marker=dict(
                size=1,
                color=poly_data['color'],
                opacity=0.9,
                line=dict(width=0.4, color='black')
            ),
            hoverinfo='none',
            name=f"Polígono {poly_data['id_str']} ({poly_data['class']})" if poly_data['class'] else f"Polígono {poly_data['id_str']}"
        )
        
        all_traces.append(edge_trace)
        all_traces.append(node_trace)
    
    # Añadir capa base (por consistencia con la función original)
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
            if norm_edge not in classified_edges_all:
                base_edge_x.extend([x0, x1, None])
                base_edge_y.extend([y0, y1, None])
    
    base_edge_trace = go.Scattergl(
        x=base_edge_x, y=base_edge_y,
        mode='lines',
        line=dict(width=0.5, color='rgba(150,150,150,0.6)'),
        hoverinfo='none',
        name='BASE: Red completa'
    )
    
    # Insertar la capa base al principio
    all_traces.insert(0, base_edge_trace)
    
    # Crear la figura completa
    layout_full = go.Layout(
        title=f'Clasificación de Street Patterns - {place_name} (Completo)',
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(
            range=[global_x_min, global_x_max],
            autorange=False,
            scaleanchor="y",
            constrain='domain',
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[global_y_min, global_y_max],
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
    
    fig_full = go.Figure(data=all_traces, layout=layout_full)
    
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
    
    # Guardar visualización completa
    os.makedirs(output_folder, exist_ok=True)
    full_html = os.path.join(output_folder, f"StreetPatterns_{place_name}_Completo.html")
    fig_full.write_html(
        full_html, 
        config=config, 
        include_plotlyjs='cdn', 
        auto_open=False, 
        full_html=True,
        default_width='100%',
        default_height='100%'
    )
    
    print(f"Archivo HTML completo generado: {full_html}")
    
    # --------------------------------------------------------------------------------
    # Generar la visualización filtrada (solo polígonos seleccionados)
    # --------------------------------------------------------------------------------
    print("Generando visualización FILTRADA solo con los polígonos seleccionados...")
    
    # Contar cuántos polígonos filtrados hay por categoría para el resumen
    filtered_polygons = [p for p in all_polygons_data if p['include_in_filtered']]
    
    if not filtered_polygons:
        print("No hay polígonos que cumplan con los criterios de filtrado. No se generará la visualización filtrada.")
        return
    
    # Resumen de polígonos filtrados
    print(f"Polígonos incluidos en la visualización filtrada: {len(filtered_polygons)}")
    filtered_by_class = {}
    for p in filtered_polygons:
        pattern = p['class'] or 'sin_clasificar'
        if pattern not in filtered_by_class:
            filtered_by_class[pattern] = []
        filtered_by_class[pattern].append(p['id_str'])
    
    for pattern, ids in filtered_by_class.items():
        print(f"- {pattern}: {len(ids)} polígonos")
        if len(ids) <= 10:
            print(f"  IDs: {', '.join(ids)}")
        else:
            print(f"  Primeros 5 IDs: {', '.join(ids[:5])}...")
    
    # Preparar trazas para la versión filtrada
    filtered_traces = []
    classified_edges_filtered = set()
    
    # Añadir solo las trazas de polígonos filtrados
    for poly_data in filtered_polygons:
        # Extraer datos del polígono
        edge_x = []
        edge_y = []
        for x0, y0, x1, y1, edge in poly_data['edges']:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            classified_edges_filtered.add(edge)
        
        edge_trace = go.Scattergl(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1.0, color=poly_data['color']),
            hoverinfo='none',
            name=f"Polígono {poly_data['id_str']} ({poly_data['class']})" if poly_data['class'] else f"Polígono {poly_data['id_str']}"
        )
        
        x_sub = poly_data['positions'][:,1]
        y_sub = poly_data['positions'][:,0]
        node_trace = go.Scattergl(
            x=x_sub, y=y_sub,
            mode='markers',
            marker=dict(
                size=1,
                color=poly_data['color'],
                opacity=0.9,
                line=dict(width=0.4, color='black')
            ),
            hoverinfo='none',
            name=f"Polígono {poly_data['id_str']} ({poly_data['class']})" if poly_data['class'] else f"Polígono {poly_data['id_str']}"
        )
        
        filtered_traces.append(edge_trace)
        filtered_traces.append(node_trace)
    
    # Añadir capa base para la versión filtrada
    filtered_base_edge_x = []
    filtered_base_edge_y = []
    for u, v in G_full.edges():
        if u in base_nodes and v in base_nodes:
            u_idx = base_nodes.index(u)
            v_idx = base_nodes.index(v)
            x0 = base_positions[u_idx][1]
            y0 = base_positions[u_idx][0]
            x1 = base_positions[v_idx][1]
            y1 = base_positions[v_idx][0]
            norm_edge = normalize_edge(x0, y0, x1, y1)
            if norm_edge not in classified_edges_filtered:
                filtered_base_edge_x.extend([x0, x1, None])
                filtered_base_edge_y.extend([y0, y1, None])
    
    filtered_base_edge_trace = go.Scattergl(
        x=filtered_base_edge_x, y=filtered_base_edge_y,
        mode='lines',
        line=dict(width=0.5, color='rgba(150,150,150,0.6)'),
        hoverinfo='none',
        name='BASE: Red completa'
    )
    
    # Insertar la capa base al principio
    filtered_traces.insert(0, filtered_base_edge_trace)
    
    # Crear título para versión filtrada
    filtered_title = f'Clasificación de Street Patterns - {place_name} (Filtrado)'
    if filter_patterns:
        filtered_title += f" - Patrones: {', '.join(filter_patterns)}"
    if filter_poly_ids:
        filtered_title += f" - Polígonos específicos"
    if max_polygons_per_pattern:
        filtered_title += f" (máx {max_polygons_per_pattern} por patrón)"
    
    # Crear la figura filtrada
    layout_filtered = go.Layout(
        title=filtered_title,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(
            range=[global_x_min, global_x_max],
            autorange=False,
            scaleanchor="y",
            constrain='domain',
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[global_y_min, global_y_max],
            autorange=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        template='plotly_white',
        paper_bgcolor="#F5F5F5",
        plot_bgcolor="#FFFFFF"
    )
    
    fig_filtered = go.Figure(data=filtered_traces, layout=layout_filtered)
    

    
    # Crear nombre de archivo para versión filtrada
    filename = f"StreetPatterns_{place_name}_Filtrado"
    if filter_patterns:
        filename += f"_patterns_{'_'.join(filter_patterns)}"
    if filter_poly_ids:
        filename += "_specific_polys"
    if max_polygons_per_pattern:
        filename += f"_max{max_polygons_per_pattern}"
    
    # Guardar visualización filtrada
    filtered_html = os.path.join(output_folder, f"{filename}.html")
    fig_filtered.write_html(
        filtered_html, 
        config=config, 
        include_plotlyjs='cdn', 
        auto_open=False, 
        full_html=True
     
    )
    
    print(f"Archivo HTML filtrado generado: {filtered_html}")
    print("Proceso completado. Se generaron 2 archivos HTML:")
    print(f"1. {full_html} - Versión completa con todos los polígonos")
    print(f"2. {filtered_html} - Versión filtrada solo con los polígonos seleccionados")
    
    return {
        'all_polygons': all_polygons_data,
        'filtered_polygons': filtered_polygons,
        'full_html': full_html,
        'filtered_html': filtered_html
    }


# # ##================== EJEMPLO DE USO ==================
# if __name__ == "__main__":
#     # Define las rutas a tus archivos
#     geojson_file = "GeoJSON_Export/cary_town_nc/tracts/cary_town_nc_tracts.geojson"
#     stats_txt = "Polygons_analysis/Cary_Town_NC/stats/Polygon_Analisys_Cary_Town_NC_sorted.txt"
#     stats_dict = load_polygon_stats_from_txt(stats_txt)
#     gdf = gpd.read_file(geojson_file)
#     graph_dict = procesar_poligonos_y_generar_grafos(gdf)
    

#    # También se pueden filtrar polígonos específicos por ID
#     resultados = plot_street_patterns_optimized(
#     geojson_path=geojson_file,
#     classify_func=classify_polygon,
#     stats_dict=stats_dict,
#     graph_dict=graph_dict,
#     place_name="test1",
#     network_type="drive",
#     simplify=False,
#     filter_poly_ids= [   ]

# )

# # Para ver información detallada sobre los polígonos procesados
# # (útil para análisis adicionales)
# if resultados:
#     print("\nDetalles de los polígonos filtrados:")
#     for idx, poly in enumerate(resultados['filtered_polygons']):
#         print(f"{idx+1}. ID: {poly['id_str']}, Patrón: {poly['class']}, Nodos: {len(poly['nodes'])}")














# import geopandas as gpd
# import os
# import json

# def add_polygon_ids_to_geojson(
#     input_geojson_path,
#     output_geojson_path=None,
#     id_field_name="Py_poly_id"
# ):
#     gdf = gpd.read_file(input_geojson_path)
#     gdf[id_field_name] = None
#     gdf["is_multipolygon"] = False
#     gdf["num_sub_polygons"] = 1

#     id_mapping = {}

#     for idx, row in gdf.iterrows():
#         geom = row.geometry

#         if geom is None or geom.is_empty:
#             gdf.at[idx, id_field_name] = f"empty_{idx}"
#             continue

#         if geom.geom_type == "Polygon":
#             poly_id = f"{idx}-0"
#             gdf.at[idx, id_field_name] = poly_id
#             id_mapping[poly_id] = {
#                 "original_idx": idx,
#                 "sub_idx": 0,
#                 "area": geom.area,
#                 "perimeter": geom.length,
#                 "tuple_id": f"({idx}, 0)"
#             }

#         elif geom.geom_type == "MultiPolygon":
#             subpolys = list(geom.geoms)
#             gdf.at[idx, "is_multipolygon"] = True
#             gdf.at[idx, "num_sub_polygons"] = len(subpolys)

#             id_list = []
#             for sub_idx, _ in enumerate(subpolys):
#                 poly_id = f"{idx}-{sub_idx}"
#                 id_list.append(poly_id)
#                 id_mapping[poly_id] = {
#                     "original_idx": idx,
#                     "sub_idx": sub_idx,
#                     "tuple_id": f"({idx}, {sub_idx})"
#                 }

#             gdf.at[idx, id_field_name] = "|".join(id_list)

#     if output_geojson_path is None:
#         base, ext = os.path.splitext(input_geojson_path)
#         output_geojson_path = f"{base}_with_poly_ids{ext}"

#     gdf.to_file(output_geojson_path, driver="GeoJSON")
#     return output_geojson_path

# # Ejecutar solo esto
# if __name__ == "__main__":
#     geojson_file = "GeoJSON_Export/medellin_ant/tracts/medellin_ant_tracts.geojson"
#     add_polygon_ids_to_geojson(input_geojson_path=geojsson_file, id_field_name="Py_poly_id")
#     print("GeoJson file with poly_id included processed")













def plot_street_patterns_pdf_optimized(
    geojson_path,
    classify_func,
    graph_dict,
    stats_dict,
    place_name="MyPlace",
    network_type="drive",
    output_folder="Graphs_Cities",
    simplify=True,
    filter_patterns=None,  # Lista de patrones a incluir en la versión filtrada
    filter_poly_ids=None,  # Lista de IDs específicos para la versión filtrada
    max_polygons_per_pattern=None,  # Límite por patrón para la versión filtrada
    figsize=(20, 20),  # Tamaño de figura en pulgadas
    target_dpi=350,  # DPI para rasterización de elementos si es necesario
    line_width=0.08,  # Grosor de líneas
    node_size=0,  # Tamaño de nodos
    show_legend=True,  # Mostrar leyenda
    background_color='white'  # Color de fondo
):
    """
    Genera DOS archivos PDF vectoriales en una sola pasada:
    1) Versión completa con todos los polígonos clasificados (SIN capa base)
    2) Versión filtrada solo con los polígonos seleccionados (SIN capa base)
    
    Solo muestra la clasificación pura de street patterns sin la red base.
    """
    # Al inicio de create_plot:
    # Al inicio de create_plot, ANTES de crear la figura:
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI
    plt.ioff()  # Sin modo interactivo

    # Configuración ultra-agresiva
    matplotlib.rcParams['path.simplify'] = True
    matplotlib.rcParams['path.simplify_threshold'] = 0.05  # Más agresivo
    matplotlib.rcParams['pdf.compression'] = 9
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['figure.max_open_warning'] = 0

    # Eliminar anti-aliasing globalmente
    matplotlib.rcParams['lines.antialiased'] = False
    matplotlib.rcParams['patch.antialiased'] = False
    
    # ANTIALIASING PARA CALIDAD
    plt.rcParams['lines.antialiased'] = True
    plt.rcParams['patch.antialiased'] = True
    plt.rcParams['text.antialiased'] = True
    # CONFIGURACIÓN MATPLOTLIB PARA COLORES SÓLIDOS
    plt.rcParams['figure.dpi'] = target_dpi
    plt.rcParams['savefig.dpi'] = target_dpi


    # Backend optimizado para vectores masivos
    matplotlib.use('Agg')
    plt.ioff()  # Desactivar interactividad
    print("Iniciando procesamiento optimizado para generar PDFs vectoriales...")
    
    # Leer y preparar datos (solo una vez)
    print("Leyendo GeoJSON...")
    gdf = gpd.read_file(geojson_path)
    try:
        poly_union = gdf.union_all()
    except AttributeError:
        poly_union = gdf.unary_union

    # Definir colores y estilos para cada patrón
    pattern_colors = {
        'cul_de_sac': "#F13F3F",   # Rojo para callejones sin salida
        'gridiron': "#0C850C",     # Verde oscuro para grid
        'organico': "#2399CF",     # Azul para orgánico
        'hibrido': "#E4CD4D"       # Amarillo para híbrido
    }
    default_color = "#808080"  # Gris para sin clasificar
    
    # Contadores para limitar polígonos por patrón en la versión filtrada
    pattern_counters = {pattern: 0 for pattern in pattern_colors.keys()}
    
    # Estructuras de datos para almacenar toda la información
    all_polygons_data = []  # Todos los polígonos procesados
    
    # Variables para calcular el bounding box global
    global_x_min, global_x_max = float('inf'), float('-inf')
    global_y_min, global_y_max = float('inf'), float('-inf')
    
    print("Procesando todos los polígonos y clasificándolos...")
    
    # Procesar cada polígono
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
            
            # Clasificar el polígono
            poly_class = None
            if key in stats_dict:
                poly_stats = stats_dict[key]
                
                # Método robusto para acceder al grafo
                if isinstance(key, tuple) and len(key) >= 1:
                    main_id = key[0]
                else:
                    main_id = key
                
                possible_keys = [main_id, str(main_id), key, str(key)]
                G = None
                for possible_key in possible_keys:
                    if possible_key in graph_dict:
                        G = graph_dict[possible_key]
                        break
                
                if G is not None and hasattr(G, 'number_of_nodes'):
                    poly_class = classify_func(poly_stats, G)
                else:
                    print(f"Advertencia: No se encontró grafo válido para el polígono {key}")
                    poly_class = classify_func(poly_stats, None)
            
            # Determinar si este polígono debe incluirse en la versión filtrada
            include_in_filtered = True
            
            # Filtro por patrón
            if filter_patterns is not None and poly_class not in filter_patterns:
                include_in_filtered = False
            
            # Filtro por ID específico
            if filter_poly_ids is not None:
                poly_id_options = [key, idx, f"{idx}-{sub_idx}", (idx, sub_idx), str(idx)]
                if not any(pid in filter_poly_ids for pid in poly_id_options):
                    include_in_filtered = False
            
            # Filtro por límite de polígonos por patrón
            if max_polygons_per_pattern is not None and poly_class is not None and include_in_filtered:
                if pattern_counters[poly_class] >= max_polygons_per_pattern:
                    include_in_filtered = False
                else:
                    pattern_counters[poly_class] += 1
            
            # Procesar el polígono y generar su grafo
            try:
                G_sub = ox.graph_from_polygon(poly, network_type=network_type, simplify=simplify)
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
                
                # Procesar edges para matplotlib
                edge_lines = []
                for u, v in G_sub.edges():
                    if u in sub_nodes and v in sub_nodes:
                        u_idx = sub_nodes.index(u)
                        v_idx = sub_nodes.index(v)
                        x0 = sub_positions[u_idx][1]
                        y0 = sub_positions[u_idx][0]
                        x1 = sub_positions[v_idx][1]
                        y1 = sub_positions[v_idx][0]
                        edge_lines.append([(x0, y0), (x1, y1)])
                
                # Almacenar datos del polígono
                polygon_data = {
                    'id': key,
                    'id_str': f"{idx}-{sub_idx}",
                    'polygon': poly,
                    'class': poly_class,
                    'color': pattern_colors.get(poly_class, default_color),
                    'nodes': sub_nodes,
                    'positions': sub_positions,
                    'edge_lines': edge_lines,
                    'include_in_filtered': include_in_filtered
                }
                
                all_polygons_data.append(polygon_data)
                
            except Exception as e:
                print(f"Error procesando polígono {idx}-{sub_idx}: {e}")
                continue
    
    print(f"Total de polígonos procesados: {len(all_polygons_data)}")
    
    # Crear directorio de salida
    os.makedirs(output_folder, exist_ok=True)
    
    # --------------------------------------------------------------------------------
    # Función auxiliar para crear el plot
    # --------------------------------------------------------------------------------
    def create_plot(polygons_data, filename):
        """Función ultra-optimizada para crear plots vectoriales eficientes"""
        
        # OPTIMIZACIÓN 1: Configuración optimizada de matplotlib para vectores
        import matplotlib
        matplotlib.use('Agg')  # Backend sin GUI más rápido
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
        ax.set_facecolor(background_color)
        
        # Configurar aspecto igual y límites
        ax.set_aspect('equal')
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)
        
        # Remover ejes y ticks (optimizado)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # OPTIMIZACIÓN 2: Pre-procesamiento y filtrado inteligente
        print("Pre-procesando y optimizando datos geométricos...")
        pattern_groups = {}
        total_original_lines = 0
        tolerance = 0.2  # Tolerancia para simplificación geométrica
        
        for poly_data in polygons_data:
            pattern = poly_data['class'] or 'sin_clasificar'
            if pattern not in pattern_groups:
                pattern_groups[pattern] = {'lines': [], 'count': 0}
            
            # Procesar y optimizar líneas del polígono
            lines = poly_data['edge_lines']
            total_original_lines += len(lines)
            optimized_lines = []
            
            for line in lines:
                if len(line) < 2:
                    continue
                    
                # OPTIMIZACIÓN 3: Simplificación geométrica básica (Douglas-Peucker simplificado)
                if len(line) > 2:
                    simplified = [line[0]]  # Mantener primer punto
                    for i in range(1, len(line)-1):
                        prev_point = simplified[-1]
                        curr_point = line[i]
                        # Solo agregar punto si está suficientemente lejos
                        dist = ((curr_point[0] - prev_point[0])**2 + 
                            (curr_point[1] - prev_point[1])**2)**0.5
                        if dist > tolerance:
                            simplified.append(curr_point)
                    simplified.append(line[-1])  # Mantener último punto
                    
                    # Verificar que la línea simplificada tenga longitud mínima significativa
                    if len(simplified) >= 2:
                        start, end = simplified[0], simplified[-1]
                        total_length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                        if total_length > 0.3:  # Solo líneas visualmente significativas
                            optimized_lines.append(simplified)
                else:
                    # Para líneas simples, verificar longitud mínima
                    start, end = line[0], line[1]
                    length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                    if length > 0.3:
                        optimized_lines.append(line)
            
            pattern_groups[pattern]['lines'].extend(optimized_lines)
            pattern_groups[pattern]['count'] += 1
        
        print(f"Líneas originales: {total_original_lines}")
        total_optimized = sum(len(data['lines']) for data in pattern_groups.values())
        print(f"Líneas optimizadas: {total_optimized} ({100*(total_original_lines-total_optimized)/total_original_lines:.1f}% reducción)")
        
        # OPTIMIZACIÓN 4: Consolidación masiva por color con LineCollection ultra-optimizada
        from matplotlib.collections import LineCollection
        print("Consolidando y renderizando por color...")
        
        legend_handles = []
        color_line_counts = {}
        
        for pattern, data in pattern_groups.items():
            color = pattern_colors.get(pattern, default_color)
            lines = data['lines']
            count = data['count']
            
            if not lines:
                continue
                
            print(f"Patrón {pattern}: {len(lines)} líneas optimizadas en {count} polígonos")
            
            # OPTIMIZACIÓN 5: Agrupar líneas por color para renderizado masivo
            if color not in color_line_counts:
                color_line_counts[color] = {'lines': [], 'patterns': []}
            
            color_line_counts[color]['lines'].extend(lines)
            color_line_counts[color]['patterns'].append(f'{pattern} ({count})')
        
        # OPTIMIZACIÓN 6: Renderizado masivo por color con configuración ultra-optimizada
        for color, data in color_line_counts.items():
            lines = data['lines']
            patterns_info = data['patterns']
            
            if lines:
                print(f"Renderizando {len(lines)} líneas de color {color} en UNA operación masiva")
                
                # Configuración extremadamente optimizada para vectores ligeros
                lc = LineCollection(
                    lines,
                    colors=[color],         # Un solo color para toda la colección
                    linewidths=0.05,        # Ultra-delgado para mínimo peso vectorial
                    alpha=1.0,
                    antialiased=False,      # CRÍTICO: sin antialiasing = vectores más simples
                    capstyle='butt',        # Más eficiente que 'round'
                    joinstyle='miter',      # Más eficiente que 'round'
                    rasterized=False,       # Mantener vectorial
                    # Configuraciones adicionales para máxima eficiencia
                    picker=False,           # Sin interactividad
                    urls=None,              # Sin metadatos de URL
                    gid=None,               # Sin IDs de grupo
                    label=None              # Sin labels individuales
                )
                
                ax.add_collection(lc)
                
                # OPTIMIZACIÓN 7: Leyenda consolidada por color
                pattern_label = ' + '.join(patterns_info)
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, linewidth=2, label=pattern_label)
                )
        
        # OPTIMIZACIÓN 8: Leyenda ultra-simplificada
        if show_legend and legend_handles:
            print("Agregando leyenda optimizada...")
            legend = ax.legend(
                handles=legend_handles, 
                loc='upper right',
                frameon=True,
                fancybox=False,         # Sin efectos fancy = menos elementos vectoriales
                shadow=False,           # Sin sombra = menos elementos
                fontsize=9,             # Fuente compacta
                bbox_to_anchor=(0.98, 0.98),
                borderaxespad=0,        # Sin padding extra
                handlelength=1.2,       # Líneas de muestra más cortas
                handletextpad=0.4,      # Menos espacio entre línea y texto
                columnspacing=0.8,      # Espaciado ultra-compacto
                framealpha=0.9          # Ligera transparencia para menos peso visual
            )
            # Marco de leyenda simplificado
            legend.get_frame().set_linewidth(0.4)
            legend.get_frame().set_edgecolor('gray')
        
        # OPTIMIZACIÓN 9: Guardado PDF ultra-optimizado
        print("Guardando PDF ultra-optimizado...")
        pdf_path = os.path.join(output_folder, filename)
        
        plt.savefig(
            pdf_path, 
            format='pdf',
            dpi=60,                     # DPI ultra-bajo para vectores (máxima reducción de tamaño)
            bbox_inches='tight',
            pad_inches=0.01,            # Padding ultra-mínimo
            facecolor=background_color,
            edgecolor='none',
            transparent=False,
            # CONFIGURACIONES CRÍTICAS PARA MÍNIMO TAMAÑO DE ARCHIVO:
            metadata={},                # Sin metadatos absolutamente
            # Configuraciones adicionales para matplotlib/PDF
            orientation='portrait'      # Orientación fija para consistencia
        )
        
        # OPTIMIZACIÓN 10: Limpieza agresiva de memoria
        ax.clear()
        plt.close(fig)
        plt.clf()
        plt.cla()
        
        # Limpieza profunda del garbage collector
        gc.collect()
        
        print(f"PDF ultra-optimizado guardado: {pdf_path}")
        total_final_elements = sum(len(data['lines']) for data in color_line_counts.values())
        print(f"Elementos vectoriales finales: {total_final_elements}")
        
        return pdf_path
    

    def create_plot_png_hq(polygons_data, filename):
        """Función optimizada para crear PNG con 40 DPI y colores sólidos"""
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import gc
        
        # CONFIGURACIÓN PARA PNG 40 DPI CON COLORES SÓLIDOS
        plt.rcParams['figure.dpi'] = 300         # DPI base reducido
        plt.rcParams['savefig.dpi'] = 300        # DPI de guardado reducido
      
                
        # Figura con DPI reducido
        fig, ax = plt.subplots(figsize=figsize, facecolor=background_color, dpi=300)
        ax.set_facecolor(background_color)
        
        # Configurar aspecto y límites
        ax.set_aspect('equal')
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)
        
        # Remover ejes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # PROCESAMIENTO OPTIMIZADO CON TOLERANCIA RELAJADA
        pattern_groups = {}
        total_original_lines = 0
        tolerance = 0.5  # Tolerancia más relajada para 40 DPI
        
        for poly_data in polygons_data:
            pattern = poly_data['class'] or 'sin_clasificar'
            if pattern not in pattern_groups:
                pattern_groups[pattern] = {'lines': [], 'count': 0}
            
            lines = poly_data['edge_lines']
            total_original_lines += len(lines)
            optimized_lines = []
            
            for line in lines:
                if len(line) < 2:
                    continue
                    
                # Simplificación más agresiva para 40 DPI
                if len(line) > 2:
                    simplified = [line[0]]
                    for i in range(1, len(line)-1):
                        prev_point = simplified[-1]
                        curr_point = line[i]
                        dist = ((curr_point[0] - prev_point[0])**2 + 
                            (curr_point[1] - prev_point[1])**2)**0.5
                        if dist > tolerance:  # Tolerancia más relajada
                            simplified.append(curr_point)
                    simplified.append(line[-1])
                    
                    if len(simplified) >= 2:
                        start, end = simplified[0], simplified[-1]
                        total_length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                        if total_length > 0.3:  # Umbral más alto para 40 DPI
                            optimized_lines.append(simplified)
                else:
                    start, end = line[0], line[1]
                    length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                    if length > 0.3:
                        optimized_lines.append(line)
            
            pattern_groups[pattern]['lines'].extend(optimized_lines)
            pattern_groups[pattern]['count'] += 1
        
        print(f"Líneas para PNG 40 DPI: {sum(len(data['lines']) for data in pattern_groups.values())}")
             
        legend_handles = []
        color_line_counts = {}
        
        for pattern, data in pattern_groups.items():
            color = pattern_colors.get(pattern, default_color)
            lines = data['lines']
            count = data['count']
            
            if not lines:
                continue
                
            if color not in color_line_counts:
                color_line_counts[color] = {'lines': [], 'patterns': []}
            
            color_line_counts[color]['lines'].extend(lines)
            color_line_counts[color]['patterns'].append(f'{pattern} ({count})')
        
        # RENDERIZADO CON COLORES COMPLETAMENTE OPACOS
        for color, data in color_line_counts.items():
            lines = data['lines']
            patterns_info = data['patterns']
            
            if lines:
                print(f"Renderizando {len(lines)} líneas de color {color} con opacidad total")
                
                lc = LineCollection(
                    lines,
                    colors=[color],
                    linewidths=1.2,         # Grosor normal para 40 DPI
                    capstyle='butt',        # Extremos cuadrados (más sólidos)
                    joinstyle='miter',      # Uniones en ángulo (más definidas)
                    rasterized=True,        # Rasterizado para colores más sólidos
                    picker=False
                )
                
                ax.add_collection(lc)
                
                pattern_label = ' + '.join(patterns_info)
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, linewidth=3, label=pattern_label, alpha=None)
                )
        
        # LEYENDA OPTIMIZADA PARA 40 DPI
        if show_legend and legend_handles:
            print("Agregando leyenda optimizada para 40 DPI...")
            legend = ax.legend(
                handles=legend_handles, 
                loc='upper right',
                frameon=True,
                fancybox=False,         # Sin bordes fancy para más solidez
                shadow=False,           # Sin sombra para evitar transparencias
                bbox_to_anchor=(0.98, 0.98),
                borderaxespad=0,
                handlelength=2.0,
                handletextpad=0.6,
                columnspacing=1.0,
                framealpha=1.0          # Marco completamente opaco
            )
            legend.get_frame().set_linewidth(1.0)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_facecolor('white')  # Fondo blanco sólido
        
        # GUARDADO PNG CON CONFIGURACIÓN SÓLIDA
        print("Guardando PNG 40 DPI con colores sólidos...")
        
        png_filename = filename.replace('.pdf', '.png').replace('.svg', '.png')
        if not png_filename.endswith('.png'):
            png_filename = f"{filename}.png"
        png_path = os.path.join(output_folder, png_filename)
        
        plt.savefig(
            png_path, 
            format='png',
            dpi=300,                     # DPI reducido
            bbox_inches='tight',
            pad_inches=0.05,            # Padding ligeramente mayor
            facecolor=background_color,
            edgecolor='none',
            pil_kwargs={
                'compress_level': 1,     # Menos compresión para colores más puros
                'optimize': False        # Sin optimización que pueda afectar colores
            }
        )
        
        # Limpieza de memoria
        ax.clear()
        plt.close(fig)
        plt.clf()
        plt.cla()
        gc.collect()
        return png_path




    # --------------------------------------------------------------------------------
    # Generar PDF completo (todos los polígonos)
    # --------------------------------------------------------------------------------
    print("Generando PDF COMPLETO con todos los polígonos...")
    
    full_filename = f"StreetPatterns_{place_name}_Completo.pdf"
    full_pdf = create_plot_png_hq(all_polygons_data,  full_filename)
    
    print(f"PDF completo generado: {full_pdf}")
    
    # --------------------------------------------------------------------------------
    # Generar PDF filtrado (solo polígonos seleccionados)
    # --------------------------------------------------------------------------------
    print("Generando PDF FILTRADO solo con los polígonos seleccionados...")
    
    # Filtrar polígonos
    filtered_polygons = [p for p in all_polygons_data if p['include_in_filtered']]
    
    if not filtered_polygons:
        print("No hay polígonos que cumplan con los criterios de filtrado. No se generará el PDF filtrado.")
        return {
            'all_polygons': all_polygons_data,
            'filtered_polygons': [],
            'full_pdf': full_pdf,
            'filtered_pdf': None
        }
    
    # Resumen de polígonos filtrados
    print(f"Polígonos incluidos en el PDF filtrado: {len(filtered_polygons)}")
    filtered_by_class = {}
    for p in filtered_polygons:
        pattern = p['class'] or 'sin_clasificar'
        if pattern not in filtered_by_class:
            filtered_by_class[pattern] = []
        filtered_by_class[pattern].append(p['id_str'])
    
    for pattern, ids in filtered_by_class.items():
        print(f"- {pattern}: {len(ids)} polígonos")
    
    # Crear título para versión filtrada
    
    if filter_patterns:
        filtered_title += f" - Patrones: {', '.join(filter_patterns)}"
    if filter_poly_ids:
        filtered_title += f" - Polígonos específicos"  
    if max_polygons_per_pattern:
        filtered_title += f" (máx {max_polygons_per_pattern} por patrón)"
    
    # Crear nombre de archivo para versión filtrada
    filename = f"StreetPatterns_{place_name}_Filtrado"
    if filter_patterns:
        filename += f"_patterns_{'_'.join(filter_patterns)}"
    if filter_poly_ids:
        filename += "_specific_polys"
    if max_polygons_per_pattern:
        filename += f"_max{max_polygons_per_pattern}"
    filename += ".pdf"
    
    filtered_pdf = create_plot(filtered_polygons,  filename)
    
    print(f"PDF filtrado generado: {filtered_pdf}")
    print("Proceso completado. Se generaron 2 archivos PDF vectoriales:")
    print(f"1. {full_pdf} - Versión completa con todos los polígonos")
    print(f"2. {filtered_pdf} - Versión filtrada solo con los polígonos seleccionados")
    
    return {
        'all_polygons': all_polygons_data,
        'filtered_polygons': filtered_polygons,
        'full_pdf': full_pdf,
        'filtered_pdf': filtered_pdf
    }


def plot_street_patterns_pdf_single(
    geojson_path,
    classify_func,
    graph_dict,
    stats_dict,
    place_name="MyPlace",
    network_type="drive",
    output_folder="Graphs_Cities",
    simplify=False,
    filter_patterns=None,
    filter_poly_ids=None,
    max_polygons_per_pattern=None,
    figsize=(20, 20),
    target_dpi=300,
    line_width=0.8,
    node_size=0,
    show_legend=False,
    background_color='white',
    version='complete'  # 'complete' o 'filtered'
):
    """
    Versión simplificada que genera UN SOLO archivo PDF.
    
    Args:
        version: 'complete' para todos los polígonos, 'filtered' para solo los filtrados
    """
    print(f"Generando PDF de street patterns ({version})...")
    
    # Usar la función optimizada pero solo retornar el archivo deseado
    result = plot_street_patterns_pdf_optimized(
        geojson_path=geojson_path,
        classify_func=classify_func,
        graph_dict=graph_dict,
        stats_dict=stats_dict,
        place_name=place_name,
        network_type=network_type,
        output_folder=output_folder,
        simplify=simplify,
        filter_patterns=filter_patterns,
        filter_poly_ids=filter_poly_ids,
        max_polygons_per_pattern=max_polygons_per_pattern,
        figsize=figsize,
        target_dpi=target_dpi,
        line_width=line_width,
        node_size=node_size,
        show_legend=show_legend,
        background_color=background_color
    )
    
    if version == 'complete':
        return result['full_pdf']
    else:
        return result['filtered_pdf']
    

ciudades = [
    # "Peachtree_GA", 
    # "Philadelphia_PA", 
    "Boston_MA", 
    "Chandler_AZ", 
    "Salt_Lake_UT", 
    # "Santa_Fe_NM", 
    # "Charleston_SC", 
    # "Cary_Town_NC", 
    # "Fort_Collins_CO"
]

# Procesar cada ciudad
for ciudad in ciudades:
    try:
        print(f"Procesando {ciudad}...")
        
        # Construir las rutas siguiendo el patrón
        ciudad_lower = ciudad.lower()  # Para las rutas que usan minúsculas
        ciudad_title = ciudad.replace("_", " ").title().replace(" ", "_")  # Para el archivo de stats
        
        # Ruta del archivo GeoJSON
        geojson_file = f"GeoJSON_Export/{ciudad_lower}/tracts/{ciudad_lower}_tracts.geojson"
        
        # Leer el archivo GeoJSON
        gdf = gpd.read_file(geojson_file)
        
        # Procesar polígonos y generar grafos
        graph_dict = procesar_poligonos_y_generar_grafos(gdf)
        
        # Ruta del archivo de estadísticas
        stats_txt = f"Polygons_analysis/{ciudad_title}/stats/Polygon_Analisys_{ciudad_title}_sorted.txt"
        
        # Cargar estadísticas
        stats_dict = load_polygon_stats_from_txt(stats_txt)
        
        # Generar el PDF
        pdf_path = plot_street_patterns_pdf_single(
            geojson_path=geojson_file,
            classify_func=classify_polygon,
            simplify=False,
            target_dpi=400,
            stats_dict=stats_dict,
            graph_dict=graph_dict, 
            place_name=ciudad_title,
            version='complete'  # o 'filtered'
        )
        
        print(f"✓ {ciudad} procesada exitosamente. PDF: {pdf_path}")
        
    except Exception as e:
        print(f"✗ Error procesando {ciudad}: {str(e)}")
        continue

print("Procesamiento completado para todas las ciudades.")





# ## Generar ambos PDFs (igual que tu función original)
# result = plot_street_patterns_pdf_optimized(
#     geojson_path="GeoJSON_Export/cary_town_nc/tracts/cary_town_nc_tracts.geojson",
#     classify_func=classify_polygon,
#     stats_dict=stats_dict,
#     graph_dict=graph_dict,
#     place_name="TEST1"
# )

