import geopandas as gpd
import os
import osmnx as ox
import json
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


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














import re







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


# Ejemplo de uso
if __name__ == "__main__":
    geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA.geojson"
    get_street_network_metrics_per_polygon(
        geojson_path=geojson_file,
        network_type='drive',
        output_txt='Resultados/poligonos_stats.txt'
    )


# def plot_classified_network(geojson_path, network_type='all'):
#     """
#     Loads a GeoJSON, computes street network metrics, and visualizes it with classification.
#     """
#     # Get street network statistics
#     network_stats = get_street_network_metrics(geojson_path, network_type)

#     # Load the polygon and network
#     gdf = gpd.read_file(geojson_path)
#     polygon = gdf.geometry.union_all()
#     G = ox.graph_from_polygon(polygon, network_type=network_type, simplify=True)

#     # Assign a classification based on computed stats
#     classification = classify_from_metrics(network_stats)

#     # Assign colors based on classification
#     color_map = {
#         "Gridiron": "blue",
#         "Cul-de-sac": "red",
#         "Hybrid": "purple",
#         "Organic": "green"
#     }

#     fig, ax = plt.subplots(figsize=(12, 12))
#     gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1, linestyle="--")

#     # Plot the road network
#     ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.4, edge_color=color_map[classification])


# def classify_from_metrics(stats):
#     """
#     Classifies the street network based on calculated metrics.
#     """
#     intersection_density = stats["intersection_density_km2"]
#     avg_node_degree = stats["streets_per_node_avg"]
#     circuity = stats["circuity_avg"]

#     if intersection_density > 200 and avg_node_degree >= 3 and circuity < 1.2:
#         return "Gridiron"
#     elif intersection_density < 50 and avg_node_degree < 2.5 and circuity > 1.3:
#         return "Cul-de-sac"
#     elif 50 <= intersection_density <= 150 and 2.5 <= avg_node_degree < 3 and 1.2 <= circuity <= 1.4:
#         return "Hybrid"
#     else:
#         return "Organic"

# # Example usage
# geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA.geojson"
# plot_classified_network(geojson_file, network_type='drive')
