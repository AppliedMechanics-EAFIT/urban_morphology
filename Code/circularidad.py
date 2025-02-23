# Coordenadas del origen (latitud y longitud)
orig_lat = 1.2120 
orig_lon = -77.2867 

# Coordenadas del destino (latitud y longitud)
dest_lat =  1.2120
dest_lon = -77.2934


# Descargar el gráfico de la red de carreteras para Nueva York

# Obtener el nodo más cercano al origen y destino usando las coordenadas
orig_node = ox.distance.nearest_nodes(graph, X=orig_lon, Y=orig_lat)
dest_node = ox.distance.nearest_nodes(graph, X=dest_lon, Y=dest_lat)

# Encontrar la ruta más corta entre el origen y el destino
route = ox.shortest_path(graph, orig_node, dest_node, weight='length')
