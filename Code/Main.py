## GitHub repository
## https://github.com/AppliedMechanics-EAFIT/urban_morphology
## Main file to executethe different programs and functions
from Lecture_and_Cleaning_ABC import CiudadesABC, read_nodes_from_excel,clean_and_filter_data
from network_Indicators import plot_centrality, Numeric_coefficient_centrality,compute_edge_betweenness_data, plot_edge_centrality, plot_geo_centrality_heatmap
from Polygon_analisis import graph_from_geojson, process_selected_layers,get_street_network_metrics_per_polygon,parallel_process_geojsons,post_processing_stats,process_city_geojsons
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
import pandas as pd
from typing import Union, List, Dict, Any                               
    
# cities = [
#     # "Medellin, Colombia",
#     # "Pasto, Colombia",
# ]

# # # # # Available metrics
# # metrics = [ "eigenvector", "closeness", "pagerank", "betweenness", "degree", "slc", "lsc"]


# # Multi city analysis
# for city in cities:
#     place_name = city

#     try:
#         print(f"\n{'='*40}\nProcessing: {place_name}\n{'='*40}")

#         # 1. Get graph
#         graph = ox.graph_from_place(place_name, network_type='drive')
#         ox.plot_graph(graph, node_size=1)

#         # 2. Process node metrics
#         for metric in metrics:
#             print(f"\nNode metric: {metric.upper()}")

#             # calculate and save centrality DATA
#             Numeric_coefficient_centrality(graph, metric, place_name, weight='length')
#             print(place_name)
#             # Use the current metric and city in the functions
#             plot_geo_centrality_heatmap(
#                 graph=graph,
#                 metric=metric,
#                 place_name=place_name,
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
#                 place_name=place_name
#             )

#         # 3. Calculate edge centrality after node metrics
#         print("\nCalculating edge centrality (Edge Betweenness)")
#         edge_data = compute_edge_betweenness_data(graph, metric="betweenness", weight='length')
#         plot_edge_centrality(edge_data, place_name)

    # except Exception as e:
    #     print(f"Error processing {place_name}: {e}")
    #     continue








# geojson_files = {
#         "GeoJSON_Export/moscow_id/tracts/moscow_id_tracts.geojson": "Moscow, ID",
#         "GeoJSON_Export/santa_fe_nm/tracts/santa_fe_nm_tracts.geojson": "Santa Fe, NM",
#         "GeoJSON_Export/peachtree_ga/tracts/peachtree_ga_tracts.geojson": "Peachtree, GA",
#         "GeoJSON_Export/chandler_az/tracts/chandler_az_tracts.geojson": "Chandler, AZ",
#         "GeoJSON_Export/salt_lake_ut/tracts/salt_lake_ut_tracts.geojson": "Salt Lake, UT",
#         "GeoJSON_Export/boston_ma/tracts/boston_ma_tracts.geojson": "Boston, MA",
#         "GeoJSON_Export/philadelphia_pa/tracts/philadelphia_pa_tracts.geojson": "Philadelphia, PA"
#     }



# for geojson_path, pretty_name in geojson_files.items():
#     try:
#         print(f"\n{'='*40}\nProcessing: {pretty_name}\n{'='*40}")
        
#         # 1. Get graph from GeoJSON
#         graph, _ = graph_from_geojson(geojson_path)  

#         # 2. Process node metrics
#         for metric in metrics:
#             print(f"\nNode metric: {metric.upper()}")
            
#             #calculate and save centrality DATA
#             Numeric_coefficient_centrality(graph, metric, pretty_name)

           
#             plot_geo_centrality_heatmap(
#                 graph=graph,
#                 metric=metric,
#                 place_name=pretty_name,
#                 cmap='inferno',
#                 resolution=1080,
#                 log_scale=True,
#                 road_opacity=0.25,
#                 buffer_ratio=0.005,
#                 smoothing=1.5
#             )
#             plot_centrality(
#                 graph=graph,
#                 metric=metric,
#                 place_name=pretty_name
#             )
        
#     except Exception as e:
#         print(f"Error processing {pretty_name}: {e}")
#         continue








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





# # Procesamiento para todos los GDB disponibles
# gdbs = [
#     "boston_ma.gdb",
#     "chandler_az.gdb",
#     "moscow_id.gdb",
#     "peachtree_ga.gdb",
#     "philadelphia_pa.gdb",
#     "salt_lake_ut.gdb",
#     "santa_fe_nm.gdb",
#     "charleston_sc.gdb",
#     "cary_town_nc.gdb",
#     "fort_collins_co.gdb"
# ]

# # Carpeta base donde están todos los GDBs
# base_folder = "MLowry_files"

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
#         layer_data = process_selected_layers(
#         gdb_path, 
#         layer_names=["tracts", "streets"],  # Cambia estos nombres por los reales
#         save_geojson=True,
#         visualize=False  # Cambia a True si quieres ver los mapas
#     )
#     except Exception as e:
#         print(f"Error al procesar {gdb_file}: {str(e)}")
#         continue






# if __name__ == '__main__':
#     # Dictionary of GeoJSON files
#     geojson_files = {
#         "GeoJSON_Export/moscow_id/tracts/moscow_id_tracts.geojson": "Moscow, ID",
#         "GeoJSON_Export/santa_fe_nm/tracts/santa_fe_nm_tracts.geojson": "Santa Fe, NM",
#         "GeoJSON_Export/peachtree_ga/tracts/peachtree_ga_tracts.geojson": "Peachtree, GA",
#         "GeoJSON_Export/chandler_az/tracts/chandler_az_tracts.geojson": "Chandler, AZ",
#         "GeoJSON_Export/salt_lake_ut/tracts/salt_lake_ut_tracts.geojson": "Salt Lake, UT",
#         "GeoJSON_Export/boston_ma/tracts/boston_ma_tracts.geojson": "Boston, MA",
#         "GeoJSON_Export/philadelphia_pa/tracts/philadelphia_pa_tracts.geojson": "Philadelphia, PA",
#         "GeoJSON_Export/charleston_sc/tracts/charleston_sc_tracts.geojson": "Charleston, SC",
#         "GeoJSON_Export/cary_town_nc/tracts/cary_town_nc_tracts.geojson": "Cary Town, NC",
#         "GeoJSON_Export/fort_collins_co/tracts/fort_collins_co_tracts.geojson": "Fort Collins, CO"
#     }

#     # Dictionary of Stats txt files without sorting and cleaning 
#     Stats_preprocessing = {
#         "Polygons_analysis/Moscow_ID/stats/Polygon_Stats_for_Moscow_ID.txt": "Moscow, ID",
#         "Polygons_analysis/Santa_Fe_NM/stats/Polygon_Stats_for_Santa_Fe_NM.txt": "Santa Fe, NM",
#         "Polygons_analysis/Peachtree_GA/stats/Polygon_Stats_for_Peachtree_GA.txt": "Peachtree, GA",
#         "Polygons_analysis/Chandler_AZ/stats/Polygon_Stats_for_Chandler_AZ.txt": "Chandler, AZ",
#         "Polygons_analysis/Salt_Lake_UT/stats/Polygon_Stats_for_Salt_Lake_UT.txt": "Salt Lake, UT",
#         "Polygons_analysis/Boston_MA/stats/Polygon_Stats_for_Boston_MA.txt": "Boston, MA",
#         "Polygons_analysis/Philadelphia_PA/stats/Polygon_Stats_for_Philadelphia_PA.txt": "Philadelphia, PA",
#         "Polygons_analysis/Charleston_SC/stats/Polygon_Stats_for_Charleston_SC.txt": "Charleston, SC",
#         "Polygons_analysis/Cary_Town_NC/stats/Polygon_Stats_for_Cary_Town_NC.txt": "Cary Town, NC",
#         "Polygons_analysis/Fort_Collins_CO/stats/Polygon_Stats_for_Fort_Collins_CO.txt": "Fort Collins, CO"
#     }

#     # Calculate polygon stats with OSMX
#     output_files = parallel_process_geojsons(
#         geojson_files, 
#         network_type='drive', 
#         max_city_workers=None,  
#         max_polygon_workers=None  
#     )

#     # Print results
#     for path in output_files:
#         print(f"Generated file: {path}")

#     # Main folder for cities
#     output_folder = None
    
#     # After the calculation this function sorts and cleans the data
#     post_processing_stats(Stats_preprocessing, output_folder)

#     # Extract the available Mobility data for each polygon
#     mobility_data = process_city_geojsons(geojson_files)

#     # Summary of mobility information processing
#     print("\nSummary of mobility information processing:")
#     for city, df in mobility_data.items():
#         print(f"{city}: {len(df)} records, {df.shape[1]} columns")

    

    




















# import osmnx as ox
# import geopandas as gpd
# import folium
# import webbrowser
# import os

# # Cargar GeoJSON con geopandas
# geojson_path = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson"
# gdf = gpd.read_file(geojson_path)

# # Centrar el mapa en Medellín
# centroide = gdf.unary_union.centroid
# mapa = folium.Map(location=[centroide.y, centroide.x], zoom_start=12, tiles='cartodbpositron')

# # Agregar los polígonos al mapa solo con bordes azules (sin relleno)
# folium.GeoJson(
#     gdf,
#     style_function=lambda x: {
#         'color': 'black',      # Color del borde
#         'weight': 2,          # Grosor del borde
#         'fillOpacity': 0      # Transparencia del relleno (0 = completamente transparente)
#     }
# ).add_to(mapa)

# # Guardar y abrir en el navegador
# mapa.save("mapa_eod.html")
# webbrowser.open("file://" + os.path.abspath("mapa_eod.html"))