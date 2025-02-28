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

# # Define a place or city to compile
place_name = "Pasto, Colombia"

# Graph 
graph = ox.graph_from_place(place_name, network_type='drive')
ox.plot_graph(graph)

# # Métricas disponibles
# metrics = [ "eigenvector", "closeness","pagerank", "betweenness", "degree","slc" , "lsc"]

# # # Generar gráficos para cada métrica
# for metric in metrics:
#     plot_centrality(graph, metric, place_name)


# # Ejemplo de uso
# archivo_resultado = coefficient_centrality(graph, "all", place_name)

output_file_ABC="Data_ABC/Data_The_ABC_cleaned.xlsx"
report_file_ABC="Data_ABC/Removed_Duplicates.xlsx"
filename_ABC= "Data_ABC/DATOS_THE_ABC.xlsx"
Raw_data_the_abc = read_nodes_from_excel(filename_ABC, "DATA")

# Uso de la función
clean_and_filter_data(filename_ABC, output_file_ABC, report_file_ABC)


