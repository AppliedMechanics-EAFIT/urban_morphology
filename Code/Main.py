## GitHub repository
## https://github.com/AppliedMechanics-EAFIT/urban_morphology
## Main file to executethe different programs and functions
from Lecture_and_Cleaning_ABC import CiudadesABC, read_nodes_from_excel,clean_and_filter_data
from network_Indicators import plot_centrality, Numeric_coefficient_centrality,compute_edge_betweenness_data, plot_edge_centrality, plot_geo_centrality_heatmap
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

cities = [
    "Medellin, Colombia",
    "Pasto, Colombia",
]

# Available metrics
metrics = [ "eigenvector", "closeness", "pagerank", "betweenness", "degree", "slc", "lsc"]


# Multi city analysis
for city in cities:
    place_name = city

    try:
        print(f"\n{'='*40}\nProcessing: {place_name}\n{'='*40}")

        # 1. Get graph
        graph = ox.graph_from_place(place_name, network_type='drive')

        # 2. Process node metrics
        for metric in metrics:
            print(f"\nNode metric: {metric.upper()}")

            # calculate and save centrality DATA
            # Numeric_coefficient_centrality(graph, metric, place_name)
            # Use the current metric and city in the functions
            plot_geo_centrality_heatmap(
                graph=graph,
                metric=metric,
                place_name=place_name,
                weight='length',
                cmap='inferno',
                resolution=1080,  # High resolution for more detail
                log_scale=True,
                road_opacity=0.25,
                buffer_ratio=0.005,
                smoothing=1.5
            )

            # plot_centrality(
            #     graph=graph,
            #     metric=metric,
            #     place_name=place_name,
            #     weight='length'
            # )

        # # 3. Calculate edge centrality after node metrics
        # print("\nCalculating edge centrality (Edge Betweenness)")
        # edge_data = compute_edge_betweenness_data(graph, metric="betweenness", weight='length')
        # plot_edge_centrality(edge_data, place_name)

    except Exception as e:
        print(f"Error processing {place_name}: {e}")
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

