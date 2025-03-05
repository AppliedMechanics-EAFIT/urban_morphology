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
































def load_geojson(geojson_path, to_crs=None):
    """
    Carga un GeoJSON y (opcionalmente) lo reproyecta a 'to_crs'.
    Retorna un GeoDataFrame con la geometría en el CRS deseado (por defecto, se queda como venga).
    """
    gdf = gpd.read_file(geojson_path)

    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)
    return gdf



def classify_polygon(polygon, network_type="drive", epsg_utm=32618):
    """
    Recibe un polígono en EPSG:4326.
    1) Lo reproyecta a UTM para calcular área (en metros) y, si lo deseas, buffer.
    2) Vuelve a EPSG:4326 para usar OSMnx.
    3) Calcula estadísticas de la subred.
    4) Retorna la categoría y algunas métricas útiles.
    """

    try:
        # Envolver el polígono en una GeoSeries con CRS 4326
        poly_gs = gpd.GeoSeries([polygon], crs="EPSG:4326")

        # Reproyectar a UTM para mediciones en metros
        poly_utm = poly_gs.to_crs(epsg=epsg_utm).geometry[0]

        # (Opcional) Hacer un pequeño buffer si deseas una “zona de influencia”
        # poly_utm_buffer = poly_utm.buffer(10)  # 10m o 15m, lo que desees
        # En este ejemplo no hacemos buffer; si lo deseas, reemplaza `poly_utm` por `poly_utm_buffer`.
        poly_utm_buffer = poly_utm

        # Área en km^2 para densidad
        area_km2 = poly_utm_buffer.area / 1e6 if poly_utm_buffer.area > 0 else 0

        # Reproyectar de vuelta a WGS84 para OSMnx
        poly_4326_buffer = gpd.GeoSeries([poly_utm_buffer], crs=f"EPSG:{epsg_utm}").to_crs(epsg=4326).geometry[0]

        # Extraer la red desde OSM
        G = ox.graph_from_polygon(poly_4326_buffer, network_type=network_type, simplify=True)

        # Calcular métricas generales
        stats = ox.stats.basic_stats(G)

        # Intersecciones, grado medio
        intersection_count = stats["intersection_count"]
        intersection_density = intersection_count / area_km2 if area_km2 > 0 else 0
        avg_street_per_node = stats["streets_per_node_avg"]

        # Definir tu propia lógica de clasificación
        if intersection_density > 120 and avg_street_per_node > 3.0:
            category = "gridiron"
        elif intersection_density < 60 and avg_street_per_node < 2.5:
            category = "cul-de-sac"
        elif 60 <= intersection_density <= 120 and 2.5 <= avg_street_per_node <= 3.0:
            category = "hibrido"
        else:
            category = "organico"

        return category, intersection_density, avg_street_per_node, intersection_count, area_km2

    except Exception as e:
        print(f"[ERROR] Polígono no pudo clasificarse: {e}")
        return "indeterminado", 0, 0, 0, 0
    


def classify_all_polygons(gdf, network_type="drive", epsg_utm=32618):
    """
    Aplica 'classify_polygon' a cada polígono del GeoDataFrame (asumido en EPSG:4326).
    Agrega columnas de resultado: clasificacion, densidad, grado, intersec_count, area_km2, etc.
    Retorna el mismo gdf con las columnas extras.
    """

    # Crear listas para cada métrica
    categories = []
    intersection_densities = []
    avg_degs = []
    intersection_counts = []
    areas_km2 = []

    for geom in gdf.geometry:
        cat, i_dens, avg_deg, i_count, a_km2 = classify_polygon(
            geom,
            network_type=network_type,
            epsg_utm=epsg_utm
        )
        categories.append(cat)
        intersection_densities.append(i_dens)
        avg_degs.append(avg_deg)
        intersection_counts.append(i_count)
        areas_km2.append(a_km2)

    # Insertar columnas en el DataFrame
    gdf["clasificacion"] = categories
    gdf["densidad_intersecciones"] = intersection_densities
    gdf["grado_medio"] = avg_degs
    gdf["num_intersecciones"] = intersection_counts
    gdf["area_km2"] = areas_km2

    return gdf




def plot_classified_polygons(gdf_classified):
    """
    Dibuja todos los polígonos del gdf con un color distinto según 'clasificacion'.
    """
    # Definir colores para cada categoría
    color_map = {
        "gridiron": "lightblue",
        "cul-de-sac": "red",
        "hibrido": "purple",
        "organico": "green",
        "indeterminado": "gray"
    }

    # Crear la figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Graficar cada categoría con su color
    for cat, color in color_map.items():
        subset = gdf_classified[gdf_classified["clasificacion"] == cat]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.5, label=cat)

    # Ajustar título, leyenda, etc.
    ax.set_title("Clasificación Morfológica por Polígono")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    # Manejo de leyenda sin duplicados
    handles, labels = ax.get_legend_handles_labels()
    unique = list(dict(zip(labels, handles)).items())  # elimina duplicados
    ax.legend([u[1] for u in unique], [u[0] for u in unique], loc="best", title="Tipo de Patrón")

    plt.tight_layout()
    plt.show()



def full_workflow_classification(
    geojson_path,
    network_type="drive",
    epsg_utm=32618,
    plot_result=True
):
    """
    1) Carga un GeoJSON.
    2) Clasifica cada polígono en 'cul-de-sac', 'gridiron', 'orgánico', etc. según sus métricas de la red OSM.
    3) (Opcional) Grafica el resultado.
    4) Retorna un GeoDataFrame con columnas de clasificación y métricas.
    """

    # 1. Cargar
    gdf = load_geojson(geojson_path, to_crs="EPSG:4326")

    # 2. Clasificar
    gdf_classified = classify_all_polygons(gdf, network_type=network_type, epsg_utm=epsg_utm)

    # 3. Graficar resultado
    if plot_result:
        plot_classified_polygons(gdf_classified)

    return gdf_classified


if __name__ == "__main__":
    geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA.geojson"
    gdf_final = full_workflow_classification(geojson_file)

    # Mostrar en consola las primeras filas con sus métricas
    print(gdf_final[[
        "clasificacion",
        "densidad_intersecciones",
        "grado_medio",
        "num_intersecciones",
        "area_km2"
    ]].head())

