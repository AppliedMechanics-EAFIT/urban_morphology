import geopandas as gpd
import os
import osmnx as ox
import json
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import re
import shapely
from shapely.geometry import Polygon, MultiPolygon
import plotly.graph_objects as go
import numpy as np
import matplotlib.patches as mpatches
import sys
import ast
import seaborn as sns
from scipy.stats import f_oneway, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report , calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import colors as mcolors
from shapely.wkt import loads
from shapely.geometry import shape
import scipy.stats as stats
from matplotlib.colors import ListedColormap
from networkx.algorithms.community import modularity
from matplotlib.patches import Patch
from math import degrees, atan2
import fiona
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
plt.rcParams.update({'font.size': 14}) 
from shapely.geometry import LineString
import gc
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from Polygon_clustering import normalize_edge, classify_polygon, procesar_poligonos_y_generar_grafos, load_polygon_stats_from_txt




def plot_street_patterns_optimized(
    geojson_path,
    classify_func,
    graph_dict,
    stats_dict,
    place_name="MyPlace",
    network_type="drive",
    output_folder="Graphs_Cities",
    simplify=True,
    filter_patterns=None,  # Lista de patrones a incluir en la versión filtrada
    filter_poly_ids=None,  # Lista de IDs específicos para la versión filtrada
    max_polygons_per_pattern=None  # Límite por patrón para la versión filtrada
):
    """
    Genera DOS archivos HTML interactivos en una sola pasada:
    1) Versión completa con todos los polígonos (igual que la función original)
    2) Versión filtrada solo con los polígonos seleccionados
    
    Esto evita duplicar cálculos costosos y reduce significativamente el tiempo total.
    """
    print("Iniciando procesamiento optimizado para generar dos visualizaciones...")
    
    # Leer y preparar datos (solo una vez)
    print("Leyendo GeoJSON y construyendo la red completa...")
    gdf = gpd.read_file(geojson_path)
    try:
        poly_union = gdf.union_all()
    except AttributeError:
        poly_union = gdf.unary_union

    try:
        G_full = ox.graph_from_polygon(poly_union, network_type=network_type, simplify=simplify)
    except Exception as e:
        print(f"Error al crear la red base: {e}")
        return

    if len(G_full.edges()) == 0:
        print("La red base (unión) está vacía. Revisa tu GeoJSON y tipo de red.")
        return

    G_full = ox.project_graph(G_full)
    base_nodes = list(G_full.nodes())
    base_positions = np.array([(G_full.nodes[n]['y'], G_full.nodes[n]['x']) for n in base_nodes])
    x_vals = base_positions[:, 1]
    y_vals = base_positions[:, 0]
    global_x_min, global_x_max = x_vals.min(), x_vals.max()
    global_y_min, global_y_max = y_vals.min(), y_vals.max()

    # --------------------------------------------------------------------------------
    # Primer paso: Procesar TODOS los polígonos y almacenar sus datos
    # --------------------------------------------------------------------------------
    print("Procesando todos los polígonos y clasificándolos (una sola vez)...")
    
    pattern_colors = {
        'cul_de_sac': '#FF6B6B',   # Rojo para callejones sin salida
        'gridiron': '#006400',     # Verde oscuro para grid
        'organico': '#0F2F76',     # Azul para orgánico
        'hibrido': '#FDCB6E'     # Amarillo para híbrido
    }
    default_color = "gray"
    
    # Contadores para limitar polígonos por patrón en la versión filtrada
    pattern_counters = {pattern: 0 for pattern in pattern_colors.keys()}
    
    # Estructuras de datos para almacenar toda la información
    all_polygons_data = []  # Todos los polígonos procesados
    classified_edges_all = set()  # Todos los segmentos de bordes clasificados
    
    # Para cada polígono, procesarlo y almacenar sus datos
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for sub_idx, poly in enumerate(polys):
            key = (idx, sub_idx)
            
            # Clasificar el polígono (igual que antes, acceso robusto al grafo)
            poly_class = None
            if key in stats_dict:
                poly_stats = stats_dict[key]
                
                # Método robusto para acceder al grafo
                if isinstance(key, tuple) and len(key) >= 1:
                    main_id = key[0]
                else:
                    main_id = key
                
                possible_keys = [main_id, str(main_id), key, str(key)]
                G = None
                for possible_key in possible_keys:
                    if possible_key in graph_dict:
                        G = graph_dict[possible_key]
                        break
                
                if G is not None and hasattr(G, 'number_of_nodes'):
                    poly_class = classify_func(poly_stats, G)
                else:
                    print(f"Advertencia: No se encontró grafo válido para el polígono {key}, usando None.")
                    poly_class = classify_func(poly_stats, None)
            
            # Determinar si este polígono debe incluirse en la versión filtrada
            include_in_filtered = True
            
            # Filtro por patrón
            if filter_patterns is not None and poly_class not in filter_patterns:
                include_in_filtered = False
            
            # Filtro por ID específico
            if filter_poly_ids is not None:
                poly_id_options = [key, idx, f"{idx}-{sub_idx}", (idx, sub_idx), str(idx)]
                if not any(pid in filter_poly_ids for pid in poly_id_options):
                    include_in_filtered = False
            
            # Filtro por límite de polígonos por patrón
            if max_polygons_per_pattern is not None and poly_class is not None and include_in_filtered:
                if pattern_counters[poly_class] >= max_polygons_per_pattern:
                    include_in_filtered = False
                else:
                    pattern_counters[poly_class] += 1
            
            # Procesar el polígono y generar su grafo
            try:
                G_sub = ox.graph_from_polygon(poly, network_type=network_type, simplify=simplify)
                if len(G_sub.edges()) == 0:
                    continue
                
                G_sub = ox.project_graph(G_sub)
                sub_nodes = list(G_sub.nodes())
                sub_positions = np.array([(G_sub.nodes[n]['y'], G_sub.nodes[n]['x']) for n in sub_nodes])
                
                if sub_positions.size == 0:
                    continue
                
                # Actualizar bounding box global
                lx_min, lx_max = sub_positions[:,1].min(), sub_positions[:,1].max()
                ly_min, ly_max = sub_positions[:,0].min(), sub_positions[:,0].max()
                global_x_min = min(global_x_min, lx_min)
                global_x_max = max(global_x_max, lx_max)
                global_y_min = min(global_y_min, ly_min)
                global_y_max = max(global_y_max, ly_max)
                
                # Procesar bordes
                edge_data = []
                for u, v in G_sub.edges():
                    if u in sub_nodes and v in sub_nodes:
                        u_idx = sub_nodes.index(u)
                        v_idx = sub_nodes.index(v)
                        x0 = sub_positions[u_idx][1]
                        y0 = sub_positions[u_idx][0]
                        x1 = sub_positions[v_idx][1]
                        y1 = sub_positions[v_idx][0]
                        norm_edge = normalize_edge(x0, y0, x1, y1)
                        classified_edges_all.add(norm_edge)
                        edge_data.append((x0, y0, x1, y1, norm_edge))
                
                # Almacenar todos los datos del polígono para usar después
                polygon_data = {
                    'id': key,
                    'id_str': f"{idx}-{sub_idx}",
                    'polygon': poly,
                    'class': poly_class,
                    'color': pattern_colors.get(poly_class, default_color),
                    'nodes': sub_nodes,
                    'positions': sub_positions,
                    'edges': edge_data,
                    'include_in_filtered': include_in_filtered
                }
                
                all_polygons_data.append(polygon_data)
                
            except Exception as e:
                print(f"Error procesando polígono {idx}-{sub_idx}: {e}")
                continue
    
    print(f"Total de polígonos procesados: {len(all_polygons_data)}")
    
    # --------------------------------------------------------------------------------
    # Generar la visualización completa (todos los polígonos)
    # --------------------------------------------------------------------------------
    print("Generando visualización COMPLETA con todos los polígonos...")
    
    # Preparar trazas para la versión completa
    all_traces = []
    
    # Añadir todas las trazas de polígonos
    for poly_data in all_polygons_data:
        # Extraer datos del polígono
        edge_x = []
        edge_y = []
        for x0, y0, x1, y1, _ in poly_data['edges']:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scattergl(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1.0, color=poly_data['color']),
            hoverinfo='none',
            name=f"Polígono {poly_data['id_str']} ({poly_data['class']})" if poly_data['class'] else f"Polígono {poly_data['id_str']}"
        )
        
        x_sub = poly_data['positions'][:,1]
        y_sub = poly_data['positions'][:,0]
        node_trace = go.Scattergl(
            x=x_sub, y=y_sub,
            mode='markers',
            marker=dict(
                size=1,
                color=poly_data['color'],
                opacity=0.9,
                line=dict(width=0.4, color='black')
            ),
            hoverinfo='none',
            name=f"Polígono {poly_data['id_str']} ({poly_data['class']})" if poly_data['class'] else f"Polígono {poly_data['id_str']}"
        )
        
        all_traces.append(edge_trace)
        all_traces.append(node_trace)
    
    # Añadir capa base (por consistencia con la función original)
    base_edge_x = []
    base_edge_y = []
    for u, v in G_full.edges():
        if u in base_nodes and v in base_nodes:
            u_idx = base_nodes.index(u)
            v_idx = base_nodes.index(v)
            x0 = base_positions[u_idx][1]
            y0 = base_positions[u_idx][0]
            x1 = base_positions[v_idx][1]
            y1 = base_positions[v_idx][0]
            norm_edge = normalize_edge(x0, y0, x1, y1)
            if norm_edge not in classified_edges_all:
                base_edge_x.extend([x0, x1, None])
                base_edge_y.extend([y0, y1, None])
    
    base_edge_trace = go.Scattergl(
        x=base_edge_x, y=base_edge_y,
        mode='lines',
        line=dict(width=0.5, color='rgba(150,150,150,0.6)'),
        hoverinfo='none',
        name='BASE: Red completa'
    )
    
    # Insertar la capa base al principio
    all_traces.insert(0, base_edge_trace)
    
    # Crear la figura completa
    layout_full = go.Layout(
        title=f'Clasificación de Street Patterns - {place_name} (Completo)',
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(
            range=[global_x_min, global_x_max],
            autorange=False,
            scaleanchor="y",
            constrain='domain',
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[global_y_min, global_y_max],
            autorange=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        template='plotly_white',
        height=800,
        paper_bgcolor="#F5F5F5",
        plot_bgcolor="#FFFFFF"
    )
    
    fig_full = go.Figure(data=all_traces, layout=layout_full)
    
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
        'toImageButtonOptions': {
            'format': 'svg',
            'width': 1600,
            'height': 1600
        },
        'responsive': True
    }
    
    # Guardar visualización completa
    os.makedirs(output_folder, exist_ok=True)
    full_html = os.path.join(output_folder, f"StreetPatterns_{place_name}_Completo.html")
    fig_full.write_html(
        full_html, 
        config=config, 
        include_plotlyjs='cdn', 
        auto_open=False, 
        full_html=True,
        default_width='100%',
        default_height='100%'
    )
    
    print(f"Archivo HTML completo generado: {full_html}")
    
    # --------------------------------------------------------------------------------
    # Generar la visualización filtrada (solo polígonos seleccionados)
    # --------------------------------------------------------------------------------
    print("Generando visualización FILTRADA solo con los polígonos seleccionados...")
    
    # Contar cuántos polígonos filtrados hay por categoría para el resumen
    filtered_polygons = [p for p in all_polygons_data if p['include_in_filtered']]
    
    if not filtered_polygons:
        print("No hay polígonos que cumplan con los criterios de filtrado. No se generará la visualización filtrada.")
        return
    
    # Resumen de polígonos filtrados
    print(f"Polígonos incluidos en la visualización filtrada: {len(filtered_polygons)}")
    filtered_by_class = {}
    for p in filtered_polygons:
        pattern = p['class'] or 'sin_clasificar'
        if pattern not in filtered_by_class:
            filtered_by_class[pattern] = []
        filtered_by_class[pattern].append(p['id_str'])
    
    for pattern, ids in filtered_by_class.items():
        print(f"- {pattern}: {len(ids)} polígonos")
        if len(ids) <= 10:
            print(f"  IDs: {', '.join(ids)}")
        else:
            print(f"  Primeros 5 IDs: {', '.join(ids[:5])}...")
    
    # Preparar trazas para la versión filtrada
    filtered_traces = []
    classified_edges_filtered = set()
    
    # Añadir solo las trazas de polígonos filtrados
    for poly_data in filtered_polygons:
        # Extraer datos del polígono
        edge_x = []
        edge_y = []
        for x0, y0, x1, y1, edge in poly_data['edges']:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            classified_edges_filtered.add(edge)
        
        edge_trace = go.Scattergl(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1.0, color=poly_data['color']),
            hoverinfo='none',
            name=f"Polígono {poly_data['id_str']} ({poly_data['class']})" if poly_data['class'] else f"Polígono {poly_data['id_str']}"
        )
        
        x_sub = poly_data['positions'][:,1]
        y_sub = poly_data['positions'][:,0]
        node_trace = go.Scattergl(
            x=x_sub, y=y_sub,
            mode='markers',
            marker=dict(
                size=1,
                color=poly_data['color'],
                opacity=0.9,
                line=dict(width=0.4, color='black')
            ),
            hoverinfo='none',
            name=f"Polígono {poly_data['id_str']} ({poly_data['class']})" if poly_data['class'] else f"Polígono {poly_data['id_str']}"
        )
        
        filtered_traces.append(edge_trace)
        filtered_traces.append(node_trace)
    
    # Añadir capa base para la versión filtrada
    filtered_base_edge_x = []
    filtered_base_edge_y = []
    for u, v in G_full.edges():
        if u in base_nodes and v in base_nodes:
            u_idx = base_nodes.index(u)
            v_idx = base_nodes.index(v)
            x0 = base_positions[u_idx][1]
            y0 = base_positions[u_idx][0]
            x1 = base_positions[v_idx][1]
            y1 = base_positions[v_idx][0]
            norm_edge = normalize_edge(x0, y0, x1, y1)
            if norm_edge not in classified_edges_filtered:
                filtered_base_edge_x.extend([x0, x1, None])
                filtered_base_edge_y.extend([y0, y1, None])
    
    filtered_base_edge_trace = go.Scattergl(
        x=filtered_base_edge_x, y=filtered_base_edge_y,
        mode='lines',
        line=dict(width=0.5, color='rgba(150,150,150,0.6)'),
        hoverinfo='none',
        name='BASE: Red completa'
    )
    
    # Insertar la capa base al principio
    filtered_traces.insert(0, filtered_base_edge_trace)
    
    # Crear título para versión filtrada
    filtered_title = f'Clasificación de Street Patterns - {place_name} (Filtrado)'
    if filter_patterns:
        filtered_title += f" - Patrones: {', '.join(filter_patterns)}"
    if filter_poly_ids:
        filtered_title += f" - Polígonos específicos"
    if max_polygons_per_pattern:
        filtered_title += f" (máx {max_polygons_per_pattern} por patrón)"
    
    # Crear la figura filtrada
    layout_filtered = go.Layout(
        title=filtered_title,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(
            range=[global_x_min, global_x_max],
            autorange=False,
            scaleanchor="y",
            constrain='domain',
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[global_y_min, global_y_max],
            autorange=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        template='plotly_white',
        paper_bgcolor="#F5F5F5",
        plot_bgcolor="#FFFFFF"
    )
    
    fig_filtered = go.Figure(data=filtered_traces, layout=layout_filtered)
    

    
    # Crear nombre de archivo para versión filtrada
    filename = f"StreetPatterns_{place_name}_Filtrado"
    if filter_patterns:
        filename += f"_patterns_{'_'.join(filter_patterns)}"
    if filter_poly_ids:
        filename += "_specific_polys"
    if max_polygons_per_pattern:
        filename += f"_max{max_polygons_per_pattern}"
    
    # Guardar visualización filtrada
    filtered_html = os.path.join(output_folder, f"{filename}.html")
    fig_filtered.write_html(
        filtered_html, 
        config=config, 
        include_plotlyjs='cdn', 
        auto_open=False, 
        full_html=True
     
    )
    
    print(f"Archivo HTML filtrado generado: {filtered_html}")
    print("Proceso completado. Se generaron 2 archivos HTML:")
    print(f"1. {full_html} - Versión completa con todos los polígonos")
    print(f"2. {filtered_html} - Versión filtrada solo con los polígonos seleccionados")
    
    return {
        'all_polygons': all_polygons_data,
        'filtered_polygons': filtered_polygons,
        'full_html': full_html,
        'filtered_html': filtered_html
    }


###====================================

# Define las rutas a tus archivos
geojson_file = "GeoJSON_Export/cary_town_nc/tracts/cary_town_nc_tracts.geojson"
stats_txt = "Polygons_analysis/Cary_Town_NC/stats/Polygon_Analisys_Cary_Town_NC_sorted.txt"
stats_dict = load_polygon_stats_from_txt(stats_txt)
gdf = gpd.read_file(geojson_file)
graph_dict = procesar_poligonos_y_generar_grafos(gdf)


# También se pueden filtrar polígonos específicos por ID
resultados = plot_street_patterns_optimized(
geojson_path=geojson_file,
classify_func=classify_polygon,
stats_dict=stats_dict,
graph_dict=graph_dict,
place_name="test1",
network_type="drive",
simplify=False,
filter_poly_ids= [   ]

)

def plot_street_patterns_pdf_optimized(
    geojson_path,
    classify_func,
    graph_dict,
    stats_dict,
    place_name="MyPlace",
    network_type="drive",
    output_folder="Graphs_Cities",
    simplify=True,
    filter_patterns=None,  # Lista de patrones a incluir en la versión filtrada
    filter_poly_ids=None,  # Lista de IDs específicos para la versión filtrada
    max_polygons_per_pattern=None,  # Límite por patrón para la versión filtrada
    figsize=(20, 20),  # Tamaño de figura en pulgadas
    target_dpi=350,  # DPI para rasterización de elementos si es necesario
    line_width=0.08,  # Grosor de líneas
    node_size=0,  # Tamaño de nodos
    show_legend=True,  # Mostrar leyenda
    background_color='white'  # Color de fondo
):
    """
    Genera DOS archivos PDF vectoriales en una sola pasada:
    1) Versión completa con todos los polígonos clasificados (SIN capa base)
    2) Versión filtrada solo con los polígonos seleccionados (SIN capa base)
    
    Solo muestra la clasificación pura de street patterns sin la red base.
    """
    # Al inicio de create_plot:
    # Al inicio de create_plot, ANTES de crear la figura:
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI
    plt.ioff()  # Sin modo interactivo

    # Configuración ultra-agresiva
    matplotlib.rcParams['path.simplify'] = True
    matplotlib.rcParams['path.simplify_threshold'] = 0.05  # Más agresivo
    matplotlib.rcParams['pdf.compression'] = 9
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['figure.max_open_warning'] = 0

    # Eliminar anti-aliasing globalmente
    matplotlib.rcParams['lines.antialiased'] = False
    matplotlib.rcParams['patch.antialiased'] = False
    
    # ANTIALIASING PARA CALIDAD
    plt.rcParams['lines.antialiased'] = True
    plt.rcParams['patch.antialiased'] = True
    plt.rcParams['text.antialiased'] = True
    # CONFIGURACIÓN MATPLOTLIB PARA COLORES SÓLIDOS
    plt.rcParams['figure.dpi'] = target_dpi
    plt.rcParams['savefig.dpi'] = target_dpi


    # Backend optimizado para vectores masivos
    matplotlib.use('Agg')
    plt.ioff()  # Desactivar interactividad
    print("Iniciando procesamiento optimizado para generar PDFs vectoriales...")
    
    # Leer y preparar datos (solo una vez)
    print("Leyendo GeoJSON...")
    gdf = gpd.read_file(geojson_path)
    try:
        poly_union = gdf.union_all()
    except AttributeError:
        poly_union = gdf.unary_union

    # Definir colores y estilos para cada patrón
    pattern_colors = {
        'cul_de_sac': "#F13F3F",   # Rojo para callejones sin salida
        'gridiron': "#0C850C",     # Verde oscuro para grid
        'organico': "#2399CF",     # Azul para orgánico
        'hibrido': "#E4CD4D"       # Amarillo para híbrido
    }
    default_color = "#808080"  # Gris para sin clasificar
    
    # Contadores para limitar polígonos por patrón en la versión filtrada
    pattern_counters = {pattern: 0 for pattern in pattern_colors.keys()}
    
    # Estructuras de datos para almacenar toda la información
    all_polygons_data = []  # Todos los polígonos procesados
    
    # Variables para calcular el bounding box global
    global_x_min, global_x_max = float('inf'), float('-inf')
    global_y_min, global_y_max = float('inf'), float('-inf')
    
    print("Procesando todos los polígonos y clasificándolos...")
    
    # Procesar cada polígono
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for sub_idx, poly in enumerate(polys):
            key = (idx, sub_idx)
            
            # Clasificar el polígono
            poly_class = None
            if key in stats_dict:
                poly_stats = stats_dict[key]
                
                # Método robusto para acceder al grafo
                if isinstance(key, tuple) and len(key) >= 1:
                    main_id = key[0]
                else:
                    main_id = key
                
                possible_keys = [main_id, str(main_id), key, str(key)]
                G = None
                for possible_key in possible_keys:
                    if possible_key in graph_dict:
                        G = graph_dict[possible_key]
                        break
                
                if G is not None and hasattr(G, 'number_of_nodes'):
                    poly_class = classify_func(poly_stats, G)
                else:
                    print(f"Advertencia: No se encontró grafo válido para el polígono {key}")
                    poly_class = classify_func(poly_stats, None)
            
            # Determinar si este polígono debe incluirse en la versión filtrada
            include_in_filtered = True
            
            # Filtro por patrón
            if filter_patterns is not None and poly_class not in filter_patterns:
                include_in_filtered = False
            
            # Filtro por ID específico
            if filter_poly_ids is not None:
                poly_id_options = [key, idx, f"{idx}-{sub_idx}", (idx, sub_idx), str(idx)]
                if not any(pid in filter_poly_ids for pid in poly_id_options):
                    include_in_filtered = False
            
            # Filtro por límite de polígonos por patrón
            if max_polygons_per_pattern is not None and poly_class is not None and include_in_filtered:
                if pattern_counters[poly_class] >= max_polygons_per_pattern:
                    include_in_filtered = False
                else:
                    pattern_counters[poly_class] += 1
            
            # Procesar el polígono y generar su grafo
            try:
                G_sub = ox.graph_from_polygon(poly, network_type=network_type, simplify=simplify)
                if len(G_sub.edges()) == 0:
                    continue
                
                G_sub = ox.project_graph(G_sub)
                sub_nodes = list(G_sub.nodes())
                sub_positions = np.array([(G_sub.nodes[n]['y'], G_sub.nodes[n]['x']) for n in sub_nodes])
                
                if sub_positions.size == 0:
                    continue
                
                # Actualizar bounding box global
                lx_min, lx_max = sub_positions[:,1].min(), sub_positions[:,1].max()
                ly_min, ly_max = sub_positions[:,0].min(), sub_positions[:,0].max()
                global_x_min = min(global_x_min, lx_min)
                global_x_max = max(global_x_max, lx_max)
                global_y_min = min(global_y_min, ly_min)
                global_y_max = max(global_y_max, ly_max)
                
                # Procesar edges para matplotlib
                edge_lines = []
                for u, v in G_sub.edges():
                    if u in sub_nodes and v in sub_nodes:
                        u_idx = sub_nodes.index(u)
                        v_idx = sub_nodes.index(v)
                        x0 = sub_positions[u_idx][1]
                        y0 = sub_positions[u_idx][0]
                        x1 = sub_positions[v_idx][1]
                        y1 = sub_positions[v_idx][0]
                        edge_lines.append([(x0, y0), (x1, y1)])
                
                # Almacenar datos del polígono
                polygon_data = {
                    'id': key,
                    'id_str': f"{idx}-{sub_idx}",
                    'polygon': poly,
                    'class': poly_class,
                    'color': pattern_colors.get(poly_class, default_color),
                    'nodes': sub_nodes,
                    'positions': sub_positions,
                    'edge_lines': edge_lines,
                    'include_in_filtered': include_in_filtered
                }
                
                all_polygons_data.append(polygon_data)
                
            except Exception as e:
                print(f"Error procesando polígono {idx}-{sub_idx}: {e}")
                continue
    
    print(f"Total de polígonos procesados: {len(all_polygons_data)}")
    
    # Crear directorio de salida
    os.makedirs(output_folder, exist_ok=True)
    
    # --------------------------------------------------------------------------------
    # Función auxiliar para crear el plot
    # --------------------------------------------------------------------------------
    def create_plot(polygons_data, filename):
        """Función ultra-optimizada para crear plots vectoriales eficientes"""
        
        # OPTIMIZACIÓN 1: Configuración optimizada de matplotlib para vectores
        import matplotlib
        matplotlib.use('Agg')  # Backend sin GUI más rápido
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
        ax.set_facecolor(background_color)
        
        # Configurar aspecto igual y límites
        ax.set_aspect('equal')
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)
        
        # Remover ejes y ticks (optimizado)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # OPTIMIZACIÓN 2: Pre-procesamiento y filtrado inteligente
        print("Pre-procesando y optimizando datos geométricos...")
        pattern_groups = {}
        total_original_lines = 0
        tolerance = 0.2  # Tolerancia para simplificación geométrica
        
        for poly_data in polygons_data:
            pattern = poly_data['class'] or 'sin_clasificar'
            if pattern not in pattern_groups:
                pattern_groups[pattern] = {'lines': [], 'count': 0}
            
            # Procesar y optimizar líneas del polígono
            lines = poly_data['edge_lines']
            total_original_lines += len(lines)
            optimized_lines = []
            
            for line in lines:
                if len(line) < 2:
                    continue
                    
                # OPTIMIZACIÓN 3: Simplificación geométrica básica (Douglas-Peucker simplificado)
                if len(line) > 2:
                    simplified = [line[0]]  # Mantener primer punto
                    for i in range(1, len(line)-1):
                        prev_point = simplified[-1]
                        curr_point = line[i]
                        # Solo agregar punto si está suficientemente lejos
                        dist = ((curr_point[0] - prev_point[0])**2 + 
                            (curr_point[1] - prev_point[1])**2)**0.5
                        if dist > tolerance:
                            simplified.append(curr_point)
                    simplified.append(line[-1])  # Mantener último punto
                    
                    # Verificar que la línea simplificada tenga longitud mínima significativa
                    if len(simplified) >= 2:
                        start, end = simplified[0], simplified[-1]
                        total_length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                        if total_length > 0.3:  # Solo líneas visualmente significativas
                            optimized_lines.append(simplified)
                else:
                    # Para líneas simples, verificar longitud mínima
                    start, end = line[0], line[1]
                    length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                    if length > 0.3:
                        optimized_lines.append(line)
            
            pattern_groups[pattern]['lines'].extend(optimized_lines)
            pattern_groups[pattern]['count'] += 1
        
        print(f"Líneas originales: {total_original_lines}")
        total_optimized = sum(len(data['lines']) for data in pattern_groups.values())
        print(f"Líneas optimizadas: {total_optimized} ({100*(total_original_lines-total_optimized)/total_original_lines:.1f}% reducción)")
        
        # OPTIMIZACIÓN 4: Consolidación masiva por color con LineCollection ultra-optimizada
        from matplotlib.collections import LineCollection
        print("Consolidando y renderizando por color...")
        
        legend_handles = []
        color_line_counts = {}
        
        for pattern, data in pattern_groups.items():
            color = pattern_colors.get(pattern, default_color)
            lines = data['lines']
            count = data['count']
            
            if not lines:
                continue
                
            print(f"Patrón {pattern}: {len(lines)} líneas optimizadas en {count} polígonos")
            
            # OPTIMIZACIÓN 5: Agrupar líneas por color para renderizado masivo
            if color not in color_line_counts:
                color_line_counts[color] = {'lines': [], 'patterns': []}
            
            color_line_counts[color]['lines'].extend(lines)
            color_line_counts[color]['patterns'].append(f'{pattern} ({count})')
        
        # OPTIMIZACIÓN 6: Renderizado masivo por color con configuración ultra-optimizada
        for color, data in color_line_counts.items():
            lines = data['lines']
            patterns_info = data['patterns']
            
            if lines:
                print(f"Renderizando {len(lines)} líneas de color {color} en UNA operación masiva")
                
                # Configuración extremadamente optimizada para vectores ligeros
                lc = LineCollection(
                    lines,
                    colors=[color],         # Un solo color para toda la colección
                    linewidths=0.05,        # Ultra-delgado para mínimo peso vectorial
                    alpha=1.0,
                    antialiased=False,      # CRÍTICO: sin antialiasing = vectores más simples
                    capstyle='butt',        # Más eficiente que 'round'
                    joinstyle='miter',      # Más eficiente que 'round'
                    rasterized=False,       # Mantener vectorial
                    # Configuraciones adicionales para máxima eficiencia
                    picker=False,           # Sin interactividad
                    urls=None,              # Sin metadatos de URL
                    gid=None,               # Sin IDs de grupo
                    label=None              # Sin labels individuales
                )
                
                ax.add_collection(lc)
                
                # OPTIMIZACIÓN 7: Leyenda consolidada por color
                pattern_label = ' + '.join(patterns_info)
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, linewidth=2, label=pattern_label)
                )
        
        # OPTIMIZACIÓN 8: Leyenda ultra-simplificada
        if show_legend and legend_handles:
            print("Agregando leyenda optimizada...")
            legend = ax.legend(
                handles=legend_handles, 
                loc='upper right',
                frameon=True,
                fancybox=False,         # Sin efectos fancy = menos elementos vectoriales
                shadow=False,           # Sin sombra = menos elementos
                fontsize=9,             # Fuente compacta
                bbox_to_anchor=(0.98, 0.98),
                borderaxespad=0,        # Sin padding extra
                handlelength=1.2,       # Líneas de muestra más cortas
                handletextpad=0.4,      # Menos espacio entre línea y texto
                columnspacing=0.8,      # Espaciado ultra-compacto
                framealpha=0.9          # Ligera transparencia para menos peso visual
            )
            # Marco de leyenda simplificado
            legend.get_frame().set_linewidth(0.4)
            legend.get_frame().set_edgecolor('gray')
        
        # OPTIMIZACIÓN 9: Guardado PDF ultra-optimizado
        print("Guardando PDF ultra-optimizado...")
        pdf_path = os.path.join(output_folder, filename)
        
        plt.savefig(
            pdf_path, 
            format='pdf',
            dpi=60,                     # DPI ultra-bajo para vectores (máxima reducción de tamaño)
            bbox_inches='tight',
            pad_inches=0.01,            # Padding ultra-mínimo
            facecolor=background_color,
            edgecolor='none',
            transparent=False,
            # CONFIGURACIONES CRÍTICAS PARA MÍNIMO TAMAÑO DE ARCHIVO:
            metadata={},                # Sin metadatos absolutamente
            # Configuraciones adicionales para matplotlib/PDF
            orientation='portrait'      # Orientación fija para consistencia
        )
        
        # OPTIMIZACIÓN 10: Limpieza agresiva de memoria
        ax.clear()
        plt.close(fig)
        plt.clf()
        plt.cla()
        
        # Limpieza profunda del garbage collector
        gc.collect()
        
        print(f"PDF ultra-optimizado guardado: {pdf_path}")
        total_final_elements = sum(len(data['lines']) for data in color_line_counts.values())
        print(f"Elementos vectoriales finales: {total_final_elements}")
        
        return pdf_path
    

    def create_plot_png_hq(polygons_data, filename):
        """Función optimizada para crear PNG con 40 DPI y colores sólidos"""
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import gc
        
        # CONFIGURACIÓN PARA PNG 40 DPI CON COLORES SÓLIDOS
        plt.rcParams['figure.dpi'] = 300         # DPI base reducido
        plt.rcParams['savefig.dpi'] = 300        # DPI de guardado reducido
      
                
        # Figura con DPI reducido
        fig, ax = plt.subplots(figsize=figsize, facecolor=background_color, dpi=300)
        ax.set_facecolor(background_color)
        
        # Configurar aspecto y límites
        ax.set_aspect('equal')
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)
        
        # Remover ejes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # PROCESAMIENTO OPTIMIZADO CON TOLERANCIA RELAJADA
        pattern_groups = {}
        total_original_lines = 0
        tolerance = 0.5  # Tolerancia más relajada para 40 DPI
        
        for poly_data in polygons_data:
            pattern = poly_data['class'] or 'sin_clasificar'
            if pattern not in pattern_groups:
                pattern_groups[pattern] = {'lines': [], 'count': 0}
            
            lines = poly_data['edge_lines']
            total_original_lines += len(lines)
            optimized_lines = []
            
            for line in lines:
                if len(line) < 2:
                    continue
                    
                # Simplificación más agresiva para 40 DPI
                if len(line) > 2:
                    simplified = [line[0]]
                    for i in range(1, len(line)-1):
                        prev_point = simplified[-1]
                        curr_point = line[i]
                        dist = ((curr_point[0] - prev_point[0])**2 + 
                            (curr_point[1] - prev_point[1])**2)**0.5
                        if dist > tolerance:  # Tolerancia más relajada
                            simplified.append(curr_point)
                    simplified.append(line[-1])
                    
                    if len(simplified) >= 2:
                        start, end = simplified[0], simplified[-1]
                        total_length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                        if total_length > 0.3:  # Umbral más alto para 40 DPI
                            optimized_lines.append(simplified)
                else:
                    start, end = line[0], line[1]
                    length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                    if length > 0.3:
                        optimized_lines.append(line)
            
            pattern_groups[pattern]['lines'].extend(optimized_lines)
            pattern_groups[pattern]['count'] += 1
        
        print(f"Líneas para PNG 40 DPI: {sum(len(data['lines']) for data in pattern_groups.values())}")
             
        legend_handles = []
        color_line_counts = {}
        
        for pattern, data in pattern_groups.items():
            color = pattern_colors.get(pattern, default_color)
            lines = data['lines']
            count = data['count']
            
            if not lines:
                continue
                
            if color not in color_line_counts:
                color_line_counts[color] = {'lines': [], 'patterns': []}
            
            color_line_counts[color]['lines'].extend(lines)
            color_line_counts[color]['patterns'].append(f'{pattern} ({count})')
        
        # RENDERIZADO CON COLORES COMPLETAMENTE OPACOS
        for color, data in color_line_counts.items():
            lines = data['lines']
            patterns_info = data['patterns']
            
            if lines:
                print(f"Renderizando {len(lines)} líneas de color {color} con opacidad total")
                
                lc = LineCollection(
                    lines,
                    colors=[color],
                    linewidths=1.2,         # Grosor normal para 40 DPI
                    capstyle='butt',        # Extremos cuadrados (más sólidos)
                    joinstyle='miter',      # Uniones en ángulo (más definidas)
                    rasterized=True,        # Rasterizado para colores más sólidos
                    picker=False
                )
                
                ax.add_collection(lc)
                
                pattern_label = ' + '.join(patterns_info)
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, linewidth=3, label=pattern_label, alpha=None)
                )
        
        # LEYENDA OPTIMIZADA PARA 40 DPI
        if show_legend and legend_handles:
            print("Agregando leyenda optimizada para 40 DPI...")
            legend = ax.legend(
                handles=legend_handles, 
                loc='upper right',
                frameon=True,
                fancybox=False,         # Sin bordes fancy para más solidez
                shadow=False,           # Sin sombra para evitar transparencias
                bbox_to_anchor=(0.98, 0.98),
                borderaxespad=0,
                handlelength=2.0,
                handletextpad=0.6,
                columnspacing=1.0,
                framealpha=1.0          # Marco completamente opaco
            )
            legend.get_frame().set_linewidth(1.0)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_facecolor('white')  # Fondo blanco sólido
        
        # GUARDADO PNG CON CONFIGURACIÓN SÓLIDA
        print("Guardando PNG 40 DPI con colores sólidos...")
        
        png_filename = filename.replace('.pdf', '.png').replace('.svg', '.png')
        if not png_filename.endswith('.png'):
            png_filename = f"{filename}.png"
        png_path = os.path.join(output_folder, png_filename)
        
        plt.savefig(
            png_path, 
            format='png',
            dpi=300,                     # DPI reducido
            bbox_inches='tight',
            pad_inches=0.05,            # Padding ligeramente mayor
            facecolor=background_color,
            edgecolor='none',
            pil_kwargs={
                'compress_level': 1,     # Menos compresión para colores más puros
                'optimize': False        # Sin optimización que pueda afectar colores
            }
        )
        
        # Limpieza de memoria
        ax.clear()
        plt.close(fig)
        plt.clf()
        plt.cla()
        gc.collect()
        return png_path




    # --------------------------------------------------------------------------------
    # Generar PDF completo (todos los polígonos)
    # --------------------------------------------------------------------------------
    print("Generando PDF COMPLETO con todos los polígonos...")
    
    full_filename = f"StreetPatterns_{place_name}_Completo.pdf"
    full_pdf = create_plot_png_hq(all_polygons_data,  full_filename)
    
    print(f"PDF completo generado: {full_pdf}")
    
    # --------------------------------------------------------------------------------
    # Generar PDF filtrado (solo polígonos seleccionados)
    # --------------------------------------------------------------------------------
    print("Generando PDF FILTRADO solo con los polígonos seleccionados...")
    
    # Filtrar polígonos
    filtered_polygons = [p for p in all_polygons_data if p['include_in_filtered']]
    
    if not filtered_polygons:
        print("No hay polígonos que cumplan con los criterios de filtrado. No se generará el PDF filtrado.")
        return {
            'all_polygons': all_polygons_data,
            'filtered_polygons': [],
            'full_pdf': full_pdf,
            'filtered_pdf': None
        }
    
    # Resumen de polígonos filtrados
    print(f"Polígonos incluidos en el PDF filtrado: {len(filtered_polygons)}")
    filtered_by_class = {}
    for p in filtered_polygons:
        pattern = p['class'] or 'sin_clasificar'
        if pattern not in filtered_by_class:
            filtered_by_class[pattern] = []
        filtered_by_class[pattern].append(p['id_str'])
    
    for pattern, ids in filtered_by_class.items():
        print(f"- {pattern}: {len(ids)} polígonos")
    
    # Crear título para versión filtrada
    
    if filter_patterns:
        filtered_title += f" - Patrones: {', '.join(filter_patterns)}"
    if filter_poly_ids:
        filtered_title += f" - Polígonos específicos"  
    if max_polygons_per_pattern:
        filtered_title += f" (máx {max_polygons_per_pattern} por patrón)"
    
    # Crear nombre de archivo para versión filtrada
    filename = f"StreetPatterns_{place_name}_Filtrado"
    if filter_patterns:
        filename += f"_patterns_{'_'.join(filter_patterns)}"
    if filter_poly_ids:
        filename += "_specific_polys"
    if max_polygons_per_pattern:
        filename += f"_max{max_polygons_per_pattern}"
    filename += ".pdf"
    
    filtered_pdf = create_plot(filtered_polygons,  filename)
    
    print(f"PDF filtrado generado: {filtered_pdf}")
    print("Proceso completado. Se generaron 2 archivos PDF vectoriales:")
    print(f"1. {full_pdf} - Versión completa con todos los polígonos")
    print(f"2. {filtered_pdf} - Versión filtrada solo con los polígonos seleccionados")
    
    return {
        'all_polygons': all_polygons_data,
        'filtered_polygons': filtered_polygons,
        'full_pdf': full_pdf,
        'filtered_pdf': filtered_pdf
    }


def plot_street_patterns_pdf_single(
    geojson_path,
    classify_func,
    graph_dict,
    stats_dict,
    place_name="MyPlace",
    network_type="drive",
    output_folder="Graphs_Cities",
    simplify=False,
    filter_patterns=None,
    filter_poly_ids=None,
    max_polygons_per_pattern=None,
    figsize=(20, 20),
    target_dpi=300,
    line_width=0.8,
    node_size=0,
    show_legend=False,
    background_color='white',
    version='complete'  # 'complete' o 'filtered'
):
    """
    Versión simplificada que genera UN SOLO archivo PDF.
    
    Args:
        version: 'complete' para todos los polígonos, 'filtered' para solo los filtrados
    """
    print(f"Generando PDF de street patterns ({version})...")
    
    # Usar la función optimizada pero solo retornar el archivo deseado
    result = plot_street_patterns_pdf_optimized(
        geojson_path=geojson_path,
        classify_func=classify_func,
        graph_dict=graph_dict,
        stats_dict=stats_dict,
        place_name=place_name,
        network_type=network_type,
        output_folder=output_folder,
        simplify=simplify,
        filter_patterns=filter_patterns,
        filter_poly_ids=filter_poly_ids,
        max_polygons_per_pattern=max_polygons_per_pattern,
        figsize=figsize,
        target_dpi=target_dpi,
        line_width=line_width,
        node_size=node_size,
        show_legend=show_legend,
        background_color=background_color
    )
    
    if version == 'complete':
        return result['full_pdf']
    else:
        return result['filtered_pdf']
    

ciudades = [
    "Moscow_ID",
    "Peachtree_GA",
    "Philadelphia_PA",
    "Boston_MA",
    "Chandler_AZ",
    "Salt_Lake_UT",
    "Santa_Fe_NM",
    "Medellin_ANT",
    "Charleston_SC",
    "Cary_Town_NC",
    "Fort_Collins_CO"
]

# Procesar cada ciudad
for ciudad in ciudades:
    try:
        print(f"Procesando {ciudad}...")
        
        # Construir las rutas siguiendo el patrón
        ciudad_lower = ciudad.lower()  # Para las rutas que usan minúsculas
        ciudad_title = ciudad.replace("_", " ").title().replace(" ", "_")  # Para el archivo de stats
        
        # Ruta del archivo GeoJSON
        geojson_file = f"GeoJSON_Export/{ciudad_lower}/tracts/{ciudad_lower}_tracts.geojson"
        
        # Leer el archivo GeoJSON
        gdf = gpd.read_file(geojson_file)
        
        # Procesar polígonos y generar grafos
        graph_dict = procesar_poligonos_y_generar_grafos(gdf)
        
        # Ruta del archivo de estadísticas
        stats_txt = f"Polygons_analysis/{ciudad_title}/stats/Polygon_Analisys_{ciudad_title}_sorted.txt"
        
        # Cargar estadísticas
        stats_dict = load_polygon_stats_from_txt(stats_txt)
        
        # Generar el PDF
        pdf_path = plot_street_patterns_pdf_single(
            geojson_path=geojson_file,
            classify_func=classify_polygon,
            simplify=False,
            target_dpi=400,
            stats_dict=stats_dict,
            graph_dict=graph_dict, 
            place_name=ciudad_title,
            version='complete'  # o 'filtered'
        )
        
        print(f"✓ {ciudad} procesada exitosamente. PDF: {pdf_path}")
        
    except Exception as e:
        print(f"✗ Error procesando {ciudad}: {str(e)}")
        continue

print("Procesamiento completado para todas las ciudades.")


