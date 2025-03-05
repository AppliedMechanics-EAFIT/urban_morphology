## GitHub repository
## https://github.com/AppliedMechanics-EAFIT/urban_morphology
## Main file to executethe different programs and functions
from Lecture import CiudadesABC,read_nodes_from_excel
from network_Indicators import plot_centrality, coefficient_centrality
from cleaning_DATA_ABC import clean_and_filter_data
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
from Poligonos_Medellin import plot_road_network_from_geojson, convert_shapefile_to_geojson
import json
import plotly.graph_objects as go
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


# # Define a place or city to compile
# place_name = "Pasto, Colombia"

# # Graph 
# graph = ox.graph_from_place(place_name, network_type='drive')
# ox.plot_graph(graph)

# # Avaliable metrics
# metrics = [ "eigenvector", "closeness","pagerank", "betweenness", "degree","slc" , "lsc"]

# # Generate a graph for each metric in metrics
# for metric in metrics:
#     plot_centrality(graph, metric, place_name)


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


# Graph of visualization of medellin with its respective polygons
geojson_file = "Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA.geojson"
plot_road_network_from_geojson(geojson_file, network_type='drive', simplify=True)

