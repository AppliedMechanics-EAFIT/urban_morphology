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
import ast
import numpy as np
import seaborn as sns
from scipy.stats import f_oneway, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf


def convert_shapefile_to_geojson(shapefile_paths, output_directory="Poligonos_Medellin/Json_files"):
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
        "cul_de_sac":  "red",
        "gridiron":    "green",
        "organico":    "blue",
        "hibrido":     "orange",
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
        }
    }

    os.makedirs(output_folder, exist_ok=True)
    out_html = os.path.join(output_folder, f"StreetPatterns_{place_name}.html")
    fig.write_html(out_html, config=config, include_plotlyjs='cdn', auto_open=False)

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
#     import sys

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







def Excel_for_rural_and_urban_movility_data():
    # Rutas
    stats_txt = "Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt"
    matches_csv = "Poligonos_Medellin/Resultados/Matchs_A_B/matches_by_area.csv"
    shpB = "Poligonos_Medellin/eod_gen_trips_mode.shp"
    shpA = "Poligonos_Medellin/EOD_2017_SIT_only_AMVA.shp"

    # 1) Cargar stats
    stats_dict = load_polygon_stats_from_txt(stats_txt)
    print(f"Cargadas stats para {len(stats_dict)} polígonos (subpolígonos).")

    # 2) Cargar matches CSV
    df_matches = pd.read_csv(matches_csv)  # [indexA, indexB, area_ratio]
    print("Muestra df_matches:\n", df_matches.head(), "\n")

    # 3) Leer shapefile B (movilidad)
    gdfB = gpd.read_file(shpB)
    print("Columnas B:", gdfB.columns)

    # 4) Leer shapefile A (para graficar)
    gdfA = gpd.read_file(shpA)

    # 5) Armar DataFrame final
    final_rows = []
    for i, row in df_matches.iterrows():
        idxA = row["indexA"]
        idxB = row["indexB"]
        ratio = row["area_ratio"]

        # stats => (idxA,0)
        key_stats = (idxA, 0)
        poly_stats = stats_dict.get(key_stats, {})
        pattern = classify_polygon(poly_stats)

        # Extraer movilidad de B
        rowB = gdfB.loc[idxB]  # asumiendo gdfB.index coincide con indexB

        # Columnas absolutas
        auto_ = rowB.get("Auto", 0)
        moto_ = rowB.get("Moto", 0)
        taxi_ = rowB.get("Taxi", 0)
        # Columnas proporcionales
        p_walk_  = rowB.get("p_walk",  0)
        p_tpc_   = rowB.get("p_tpc",   0)
        p_sitva_ = rowB.get("p_sitva", 0)
        p_auto_  = rowB.get("p_auto",  0)
        p_moto_  = rowB.get("p_moto",  0)
        p_taxi_  = rowB.get("p_taxi",  0)
        p_bike_  = rowB.get("p_bike",  0)

        final_rows.append({
            "indexA": idxA,
            "indexB": idxB,
            "area_ratio": ratio,
            "street_pattern": pattern,
            "Auto": auto_,
            "Moto": moto_,
            "Taxi": taxi_,
            "p_walk":  p_walk_,
            "p_tpc":   p_tpc_,
            "p_sitva": p_sitva_,
            "p_auto":  p_auto_,
            "p_moto":  p_moto_,
            "p_taxi":  p_taxi_,
            "p_bike":  p_bike_
        })

    df_final = pd.DataFrame(final_rows)
    df_final = df_final[[
        "indexA", "indexB", "area_ratio", "street_pattern", 
        "Auto", "Moto", "Taxi",  # absolutos
        "p_walk", "p_tpc", "p_sitva", "p_auto", "p_moto", "p_taxi", "p_bike"
    ]]

    output_xlsx = "Poligonos_Clasificados_Movilidad.xlsx"
    df_final.to_excel(output_xlsx, index=False)
    print(f"Guardado Excel final en {output_xlsx} con {len(df_final)} filas.\n")

    # 6) Graficar
    # Asignar pattern a gdfA
    gdfA["pattern"] = None
    indexA_to_pattern = {}
    for i, rowF in df_final.iterrows():
        indexA_to_pattern[rowF["indexA"]] = rowF["street_pattern"]

    for i in gdfA.index:
        pat = indexA_to_pattern.get(i, None)
        gdfA.loc[i,"pattern"] = pat

    fig, ax = plt.subplots(figsize=(8,8))
    gdfA.plot(column="pattern", ax=ax, legend=True, cmap="Set2")
    ax.set_title("Polígonos A clasificados")
    plt.tight_layout()
    plt.savefig("map_poligonosA_classified.png", dpi=300)
    print("Mapa guardado en map_poligonosA_classified.png")


# # ==================== EJEMPLO DE USO ====================
# if __name__ == "__main__":
#     Excel_for_rural_and_urban_movility_data()





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
#     Statis_analisis(excel_file)











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
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================

# ------ CALCULO PARA MALLA SIMPLIFICADA DE AREA URBANA MEDELLIN ANTIOQUIA -----------
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
#  =============================================================================
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

# def Excel_for_urban_movility_data():
#     # 1) Rutas ajustadas
#     stats_txt = "Poligonos_Medellin/Resultados/poligonos_stats_ordenado.txt"
#     matches_csv = "Poligonos_Medellin/Resultados/Matchs_A_B/matches_by_area.csv"
#     shpB = "Poligonos_Medellin/eod_gen_trips_mode.shp"

#     # Importante: en vez del shapefile original,
#     # usamos el GeoJSON filtrado “_URBANO.geojson”.
#     geojsonA = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson"

#     # 2) Cargar stats
#     stats_dict = load_polygon_stats_from_txt(stats_txt)
#     print(f"Cargadas stats para {len(stats_dict)} polígonos (subpolígono).")

#     # 3) Cargar CSV de emparejamientos A-B
#     df_matches = pd.read_csv(matches_csv)  # [indexA, indexB, area_ratio]
#     print("Muestra df_matches:\n", df_matches.head(), "\n")

#     # 4) Leer shapefile/GeoDataFrame B (movilidad)
#     gdfB = gpd.read_file(shpB)
#     print("Columnas B:", gdfB.columns)

#     # 5) Leer AHORA el GeoJSON A “URBANO”
#     gdfA = gpd.read_file(geojsonA)
#     print(f"Leídos {len(gdfA)} polígonos en GeoJSON A URBANO.")

#     # 6) Armar DataFrame final (como antes)
#     final_rows = []
#     for i, row in df_matches.iterrows():
#         idxA = row["indexA"]
#         idxB = row["indexB"]
#         ratio = row["area_ratio"]

#         # stats => (idxA,0)
#         key_stats = (idxA, 0)
#         poly_stats = stats_dict.get(key_stats, {})
#         pattern = classify_polygon(poly_stats)

#         # Extraer movilidad de B (asumiendo gdfB.index == indexB)
#         rowB = gdfB.loc[idxB]

          

#         # Variables proporcionales
#         p_walk_  = rowB.get("p_walk",  0)
#         p_tpc_   = rowB.get("p_tpc",   0)
#         p_sitva_ = rowB.get("p_sitva", 0)
#         p_auto_  = rowB.get("p_auto",  0)
#         p_moto_  = rowB.get("p_moto",  0)
#         p_taxi_  = rowB.get("p_taxi",  0)
#         p_bike_  = rowB.get("p_bike",  0)

#         final_rows.append({
#             "indexA": idxA,
#             "indexB": idxB,
#             "area_ratio": ratio,
#             "street_pattern": pattern,
#             "p_walk":  p_walk_,
#             "p_tpc":   p_tpc_,
#             "p_sitva": p_sitva_,
#             "p_auto":  p_auto_,
#             "p_moto":  p_moto_,
#             "p_taxi":  p_taxi_,
#             "p_bike":  p_bike_
#         })

#     df_final = pd.DataFrame(final_rows)
#     df_final = df_final[[
#         "indexA", "indexB", "area_ratio", "street_pattern", 
#         "p_walk", "p_tpc", "p_sitva", "p_auto", "p_moto", "p_taxi", "p_bike"
#     ]]

#     output_xlsx = "Poligonos_Medellin/Resultados/Statics_Results/RURAL/Poligonos_Clasificados_Movilidad_URBANO.xlsx"
#     df_final.to_excel(output_xlsx, index=False)
#     print(f"Guardado Excel final en {output_xlsx} con {len(df_final)} filas.\n")

#     # 7) Graficar
#     #   => asignar pattern a gdfA
#     gdfA["pattern"] = None
#     indexA_to_pattern = {}
#     for i, rowF in df_final.iterrows():
#         indexA_to_pattern[rowF["indexA"]] = rowF["street_pattern"]

#     # Mapear pattern
#     for i in gdfA.index:
#         gdfA.loc[i,"pattern"] = indexA_to_pattern.get(i, None)

#     fig, ax = plt.subplots(figsize=(8,8))
#     gdfA.plot(column="pattern", ax=ax, legend=True, cmap="Set2")
#     ax.set_title("GeoJSON Urbano - Polígonos Clasificados")
#     plt.tight_layout()
#     plt.savefig("map_poligonosA_urbano_classified.png", dpi=300)
#     print("Mapa guardado en map_poligonosA_urbano_classified.png")

# if __name__ == "__main__":
#     Excel_for_urban_movility_data()

# if __name__ == "__main__":
#     excel_file = "Poligonos_Medellin/Resultados/Statics_Results/RURAL/Poligonos_Clasificados_Movilidad_URBANO.xlsx"
#     Statis_analisis(excel_file)