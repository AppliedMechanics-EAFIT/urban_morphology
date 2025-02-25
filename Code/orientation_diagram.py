graph = convert.to_undirected(graph)
graph = ox.add_edge_bearings(graph)
# Graficar el diagrama de orientación de calles
fig, ax = ox.plot_orientation(graph, figsize=(10, 8))

# Agregar título
ax.set_title("Pasto", fontsize=15, fontweight="bold")

# Mostrar la figura
plt.show()