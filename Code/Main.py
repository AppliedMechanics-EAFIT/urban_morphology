## Main file to executethe different programs and functions
from Lecture import Node,read_nodes_from_excel
from network_Indicators import plot_centrality, coefficient_centrality
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

# Define a place or city to compile
place_name = "Pasto, Colombia"

# Graph 
graph = ox.graph_from_place(place_name, network_type='drive')

# Representation of the road mesh
ox.plot_graph(ox.project_graph(graph))

# Métricas disponibles
metrics = [ "eigenvector", "closeness","pagerank", "betweenness", "degree"]

# Generar gráficos para cada métrica
for metric in metrics:
    plot_centrality(graph, metric, place_name)


# Ejemplo de uso
archivo_resultado = coefficient_centrality(graph, "all", place_name)

#
filename= "tableConvert.com_8cmcfq.xlsx.xlsx"
nodes = read_nodes_from_excel(filename, "DATA")
