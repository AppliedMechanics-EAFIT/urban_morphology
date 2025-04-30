## GitHub repository
## https://github.com/AppliedMechanics-EAFIT/urban_morphology
## Main file to executethe different programs and functions
from Lecture_and_Cleaning_ABC import CiudadesABC, read_nodes_from_excel,clean_and_filter_data
from network_Indicators import plot_centrality, Numeric_coefficient_centrality,compute_edge_betweenness_data, plot_edge_centrality, plot_geo_centrality_heatmap
from Polygon_analisis import graph_from_geojson, process_selected_layers,get_street_network_metrics_per_polygon,parallel_process_geojsons,procesar_geojson_files
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
import os
import plotly.graph_objects as go
from matplotlib.colors import Normalize, rgb2hex
from osmnx import convert
import geopandas as gpd
from shapely.ops import unary_union, linemerge
from shapely.geometry import LineString, MultiLineString, GeometryCollection
# from Poligonos_Medellin import plot_road_network_from_geojson, convert_shapefile_to_geojson
import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from matplotlib import cm
import fiona


# cities = [
#     "Medellin, Colombia",
#     "Pasto, Colombia",
# ]

# # Available metrics
metrics = [ "eigenvector", "closeness", "pagerank", "betweenness", "degree", "slc", "lsc"]


# # Multi city analysis
# for city in cities:
#     place_name = city

#     try:
#         print(f"\n{'='*40}\nProcessing: {place_name}\n{'='*40}")

#         # 1. Get graph
#         graph = ox.graph_from_place(place_name, network_type='drive')

#         # 2. Process node metrics
#         for metric in metrics:
#             print(f"\nNode metric: {metric.upper()}")

#             # calculate and save centrality DATA
#             Numeric_coefficient_centrality(graph, metric, place_name)

#             # Use the current metric and city in the functions
#             plot_geo_centrality_heatmap(
#                 graph=graph,
#                 metric=metric,
#                 place_name=place_name,
#                 weight='length',
#                 cmap='inferno',
#                 resolution=2080,  # High resolution for more detail
#                 log_scale=True,
#                 road_opacity=0.25,
#                 buffer_ratio=0.005,
#                 smoothing=1.5
#             )

#             plot_centrality(
#                 graph=graph,
#                 metric=metric,
#                 place_name=place_name,
#                 weight='length'
#             )

#         # 3. Calculate edge centrality after node metrics
#         print("\nCalculating edge centrality (Edge Betweenness)")
#         edge_data = compute_edge_betweenness_data(graph, metric="betweenness", weight='length')
#         plot_edge_centrality(edge_data, place_name)

    # except Exception as e:
    #     print(f"Error processing {place_name}: {e}")
    #     continue

geojson_files = {
        # "GeoJSON_Export/moscow_id/tracts/moscow_id_tracts.geojson": "Moscow, ID",
        # "GeoJSON_Export/santa_fe_nm/tracts/santa_fe_nm_tracts.geojson": "Santa Fe, NM",
        # "GeoJSON_Export/peachtree_ga/tracts/peachtree_ga_tracts.geojson": "Peachtree, GA",
        # "GeoJSON_Export/chandler_az/tracts/chandler_az_tracts.geojson": "Chandler, AZ",
        # "GeoJSON_Export/salt_lake_ut/tracts/salt_lake_ut_tracts.geojson": "Salt Lake, UT",
        # "GeoJSON_Export/boston_ma/tracts/boston_ma_tracts.geojson": "Boston, MA",
        "GeoJSON_Export/philadelphia_pa/tracts/philadelphia_pa_tracts.geojson": "Philadelphia, PA"
    }



for geojson_path, pretty_name in geojson_files.items():
    try:
        print(f"\n{'='*40}\nProcessing: {pretty_name}\n{'='*40}")
        
        # 1. Get graph from GeoJSON
        graph, _ = graph_from_geojson(geojson_path)  # ya no necesitas place_name aquí

        # 2. Process node metrics
        for metric in metrics:
            print(f"\nNode metric: {metric.upper()}")
            
            #calculate and save centrality DATA
            Numeric_coefficient_centrality(graph, metric, pretty_name)
            # Usa el nombre bonito aquí
            # plot_geo_centrality_heatmap(
            #     graph=graph,
            #     metric=metric,
            #     place_name=pretty_name,
            #     weight='length',
            #     cmap='inferno',
            #     resolution=1080,
            #     log_scale=True,
            #     road_opacity=0.25,
            #     buffer_ratio=0.005,
            #     smoothing=1.5
            # )
            # plot_centrality(
            #     graph=graph,
            #     metric=metric,
            #     place_name=pretty_name,
            #     weight='length'
            # )
            
    except Exception as e:
        print(f"Error processing {pretty_name}: {e}")
        continue










# # Data excel with centrality values for each node
# archivo_resultado = coefficient_centrality(graph, "all", place_name)

# # The ABC of movility DATA filtered
# output_file_ABC="Data_ABC/Data_The_ABC_cleaned.xlsx"
# report_file_ABC="Data_ABC/Removed_Duplicates.xlsx"
# filename_ABC= "Data_ABC/DATOS_THE_ABC.xlsx"
# Raw_data_the_abc = read_nodes_from_excel(filename_ABC, "DATA")
# clean_and_filter_data(filename_ABC, output_file_ABC, report_file_ABC)

# # Lecture of polygons for Medellin with modal share
# shapefiles = [
#     "Poligonos_Medellin/EOD_2017_SIT_only_AMVA.shp",
#     "Poligonos_Medellin/eod_gen_trips_mode.shp"
# ]

# # Convert data to geojson for visualization
# geojson_results = convert_shapefile_to_geojson(shapefiles)


# # Graph of visualization of medellin with its respective polygons
# geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA.geojson"
# plot_road_network_from_geojson(geojson_file, network_type='drive', simplify=True)





# Procesamiento para todos los GDB disponibles
gdbs = [
    "boston_ma.gdb",
    "chandler_az.gdb",
    "moscow_id.gdb",
    "peachtree_ga.gdb",
    "philadelphia_pa.gdb",
    "salt_lake_ut.gdb",
    "santa_fe_nm.gdb"
]

# Carpeta base donde están todos los GDBs
base_folder = "MLowry_files"

# for gdb_file in gdbs:
#     # Construir la ruta completa al archivo GDB
#     gdb_path = os.path.join(base_folder, gdb_file)
    
#     print(f"\n\n{'#'*60}")
#     print(f"PROCESANDO GDB: {gdb_file}")
#     print(f"Ruta completa: {gdb_path}")
#     print(f"{'#'*60}")
    
#     # Procesar los primeros 2 layers de cada GDB
#     # Ajusta los índices según necesites o usa todos los layers
#     try:
#         layer_data = process_selected_layers(gdb_path, layer_indices=[1, 2])
#     except Exception as e:
#         print(f"Error al procesar {gdb_file}: {str(e)}")
#         continue

# # Ejemplo de uso
# if __name__ == '__main__':
#     # Diccionario de archivos GeoJSON
#     geojson_files = {
#         "GeoJSON_Export/moscow_id/tracts/moscow_id_tracts.geojson": "Moscow, ID",
#         "GeoJSON_Export/santa_fe_nm/tracts/santa_fe_nm_tracts.geojson": "Santa Fe, NM",
#         "GeoJSON_Export/peachtree_ga/tracts/peachtree_ga_tracts.geojson": "Peachtree, GA",
#         "GeoJSON_Export/chandler_az/tracts/chandler_az_tracts.geojson": "Chandler, AZ",
#         "GeoJSON_Export/salt_lake_ut/tracts/salt_lake_ut_tracts.geojson": "Salt Lake, UT",
#         "GeoJSON_Export/boston_ma/tracts/boston_ma_tracts.geojson": "Boston, MA",
#         "GeoJSON_Export/philadelphia_pa/tracts/philadelphia_pa_tracts.geojson": "Philadelphia, PA"
#     }

#     # Procesar en paralelo (ajusta los parámetros según tu CPU)
#     output_files = parallel_process_geojsons(
#         geojson_files, 
#         network_type='drive', 
#         max_city_workers=None,  # No se usa en esta implementación
#         max_polygon_workers=None  # Usar número máximo de procesos disponibles para polígonos
#     )

#     # Imprimir resultados
#     for path in output_files:
#         print(f"Archivo generado: {path}")

# if __name__ == "__main__":
#     # Definir diccionario con archivos a procesar
#     geojson_files2 = {
#     "Polygons_analysis/Moscow_ID/stats/Polygon_Stats_for_Moscow_ID.txt": "Moscow, ID",
#     "Polygons_analysis/Santa_Fe_NM/stats/Polygon_Stats_for_Santa_Fe_NM.txt": "Santa Fe, NM",
#     "Polygons_analysis/Peachtree_GA/stats/Polygon_Stats_for_Peachtree_GA.txt": "Peachtree, GA",
#     "Polygons_analysis/Chandler_AZ/stats/Polygon_Stats_for_Chandler_AZ.txt": "Chandler, AZ",
#     "Polygons_analysis/Salt_Lake_UT/stats/Polygon_Stats_for_Salt_Lake_UT.txt": "Salt Lake, UT",
#     "Polygons_analysis/Boston_MA/stats/Polygon_Stats_for_Boston_MA.txt": "Boston, MA",
#     "Polygons_analysis/Philadelphia_PA/stats/Polygon_Stats_for_Philadelphia_PA.txt": "Philadelphia, PA"

#     }

#     output_folder = None
    
#     # Procesar los archivos
#     procesar_geojson_files(geojson_files2, output_folder)






