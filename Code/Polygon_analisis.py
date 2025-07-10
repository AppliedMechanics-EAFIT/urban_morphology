import os
import fiona
import matplotlib as plt
import json
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box
import re
import multiprocessing
import time
from filelock import FileLock  
import pandas as pd
from typing import Union, List, Dict, Any

def process_selected_layers(gdb_path, layer_indices=None, layer_names=None, save_geojson=True, visualize=True):
    """
    Process selected layers from a geodatabase and convert to EPSG:4326 with coordinate validation.
    Saves GeoJSON files in a directory structure based on the GDB name and layer.
    """
    # Verificar que el archivo GDB existe
    if not os.path.exists(gdb_path):
        print(f"Error: No se encuentra el archivo GDB en {gdb_path}")
        return {}
    
    # Extract GDB folder name for organizing outputs
    gdb_name = os.path.basename(gdb_path).replace(".gdb", "")
    
    # Create base output folder
    base_output_dir = "GeoJSON_Export"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create folder specific to this GDB
    gdb_output_dir = os.path.join(base_output_dir, gdb_name)
    if save_geojson:
        os.makedirs(gdb_output_dir, exist_ok=True)
    
    # List available layers
    print(f"\nSearching for layers in the GDB: {gdb_path}")
    try:
        all_layers = fiona.listlayers(gdb_path)
    except Exception as e:
        print(f"Error al leer las capas del GDB: {str(e)}")
        return {}
        
    print(f"Layers found ({len(all_layers)}):")
    for i, layer in enumerate(all_layers):
        print(f"{i+1}. {layer}")
    
    # Determine which layers to process
    selected_layers = []
    if layer_indices:
        selected_layers.extend([all_layers[i-1] for i in layer_indices if 1 <= i <= len(all_layers)])
    if layer_names:
        selected_layers.extend([layer for layer in layer_names if layer in all_layers])
    
    selected_layers = list(dict.fromkeys(selected_layers))
    
    if not selected_layers:
        print("No valid layers selected.")
        print("Capas disponibles:", all_layers)
        return {}
    
    # Process selected layers
    print(f"\nProcessing {len(selected_layers)} selected layers:")
    results = {}
    
    for layer in selected_layers:
        print(f"\nReading layer: {layer}")
        try:
            gdf = gpd.read_file(gdb_path, layer=layer)
            
            if gdf.empty:
                print(f"Warning: Layer {layer} está vacía")
                continue
            
            # Check original CRS and save for reference
            original_crs = gdf.crs
            print(f"Original CRS: {original_crs}")
            
            # Convert to EPSG:4326 if needed
            if gdf.crs != "EPSG:4326":
                print(f"Converting from {original_crs} to EPSG:4326")
                gdf = gdf.to_crs(epsg=4326)
            
            # Validate geometry coordinates
            invalid_geoms = []
            valid_rows = []
            
            for idx, row in gdf.iterrows():
                try:
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        print(f"Warning: Null/empty geometry at index {idx}")
                        invalid_geoms.append(idx)
                        continue
                        
                    # Check if coordinates are valid for long/lat
                    coords_valid = True
                    
                    # Function to check if a coordinate is valid
                    def check_coords(x, y):
                        return -180 <= x <= 180 and -90 <= y <= 90
                    
                    # Handle different geometry types
                    if geom.geom_type == 'Point':
                        if not check_coords(geom.x, geom.y):
                            coords_valid = False
                    elif geom.geom_type in ['LineString', 'MultiPoint']:
                        for x, y in geom.coords:
                            if not check_coords(x, y):
                                coords_valid = False
                                break
                    elif geom.geom_type == 'Polygon':
                        for x, y in geom.exterior.coords:
                            if not check_coords(x, y):
                                coords_valid = False
                                break
                    elif geom.geom_type == 'MultiLineString':
                        for line in geom.geoms:
                            for x, y in line.coords:
                                if not check_coords(x, y):
                                    coords_valid = False
                                    break
                            if not coords_valid:
                                break
                    elif geom.geom_type == 'MultiPolygon':
                        for polygon in geom.geoms:
                            # Corección aquí: desempaquetar correctamente las coordenadas
                            for x, y in polygon.exterior.coords:
                                if not check_coords(x, y):
                                    coords_valid = False
                                    break
                            if not coords_valid:
                                break
                    
                    if coords_valid:
                        valid_rows.append(idx)
                    else:
                        print(f"Warning: Invalid coordinates at index {idx}")
                        invalid_geoms.append(idx)
                        
                except Exception as e:
                    print(f"Error processing geometry at index {idx}: {str(e)}")
                    invalid_geoms.append(idx)
            
            if invalid_geoms:
                print(f"Removing {len(invalid_geoms)} invalid geometries")
                gdf = gdf.iloc[valid_rows]
            
            # Check if we still have data after filtering
            if len(gdf) == 0:
                print("No valid geometries remaining after coordinate validation")
                continue
                
            results[layer] = gdf
            
            # Show properties
            print("Columns:", gdf.columns.tolist())
            print("Total valid geometries:", len(gdf))
            print(f"Current CRS: {gdf.crs}")
            print(gdf.head())
            
            # Visualize if requested
            if visualize:
                try:
                    gdf.plot(figsize=(8, 6), edgecolor="black", cmap="Set2")
                    plt.title(f"{gdb_name} - Layer: {layer}")
                    plt.xlabel("Longitude")
                    plt.ylabel("Latitude")
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Error en visualización: {str(e)}")
            
            # Save as GeoJSON if requested
            if save_geojson:
                try:
                    # Create directory for this specific layer
                    layer_dir = os.path.join(gdb_output_dir, layer)
                    os.makedirs(layer_dir, exist_ok=True)
                    
                    # Define output path with layer name
                    output_path = os.path.join(layer_dir, f"{gdb_name}_{layer}.geojson")
                    
                    # Use to_crs again to ensure we have the right projection
                    gdf_final = gdf.to_crs(epsg=4326)
                    
                    # Use to_file with explicit driver
                    gdf_final.to_file(output_path, driver="GeoJSON")
                    print(f"Saved as GeoJSON: {output_path}")
                    
                    # Optional: Validate the saved file
                    try:
                        with open(output_path, 'r') as f:
                            geojson_data = json.load(f)
                        print(f"GeoJSON file validated successfully")
                    except Exception as e:
                        print(f"Warning: Generated GeoJSON file may have issues: {str(e)}")
                        
                except Exception as e:
                    print(f"Error saving GeoJSON: {str(e)}")
                
        except Exception as e:
            print(f"Error processing layer {layer}: {str(e)}")
            import traceback
            traceback.print_exc()  # Esto te dará más detalles del error
    
    return results

def graph_from_geojson(geojson_path):
    """
    Carga un grafo de OSM usando los límites de un archivo GeoJSON
    """
    # Cargar el GeoJSON
    gdf = gpd.read_file(geojson_path)
    
    # Asegurarse que esté en EPSG:4326
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    
    # Obtener los límites (bbox) del GeoJSON
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Crear geometría de polígono para los límites
    bbox = box(*bounds)
    
    # Extraer el nombre del archivo sin extensión
    place_name = os.path.splitext(os.path.basename(geojson_path))[0]
    
    # Descargar el grafo usando el polígono
    G = ox.graph_from_polygon(bbox, network_type='drive')
    
    return G, place_name

# Global variable to count progress
_progress_counter = 0
_progress_lock = multiprocessing.Lock()

def process_polygon(args):
    """
    Procesa un solo polígono para extraer estadísticas de la red vial.
    
    Parámetros:
    -----------
    args: tupla que contiene (polygon_tuple, gdf_crs, network_type, output_txt, total_polygons)
        - polygon_tuple: tupla con (idx, sub_idx, poly)
        - gdf_crs: sistema de coordenadas del GeoDataFrame original
        - network_type: tipo de red a recuperar
        - output_txt: ruta del archivo de salida
        - total_polygons: número total de polígonos para mostrar progreso
    
    Retorna:
    --------
    tupla con (idx, sub_idx, result_text, status)
        - status puede ser: "processed" (procesado), "empty" (grafo vacío), "error" (hubo un error)
    """
    global _progress_counter, _progress_lock
    
    # Desempaquetar argumentos
    polygon_tuple, gdf_crs, network_type, output_txt, total_polygons = args
    idx, sub_idx, poly = polygon_tuple
    
    try:
        # Procesar el polígono
        G = ox.graph_from_polygon(
            poly,
            network_type=network_type,
            simplify=True
        )
        
        # Preparar resultado
        result_text = f"\n=== Polígono {idx} - SubPolígono {sub_idx} ===\n"
        
        if len(G.edges()) == 0:
            result_text += "Grafo vacío (sin vías)\n"
            status = "empty"
        else:
            # Calcular estadísticas
            stats = ox.stats.basic_stats(G)
            
            # Calcular área de este sub-polígono en km²
            area_km2 = (
                gpd.GeoDataFrame(geometry=[poly], crs=gdf_crs)
                .to_crs(epsg=3395)
                .geometry.area.sum() / 1e6
            )
            
            intersection_count = stats.get("intersection_count", 0)
            street_length_total = stats.get("street_length_total", 0.0)
            
            if area_km2 > 0:
                intersection_density_km2 = intersection_count / area_km2
                street_density_km2 = street_length_total / area_km2
            else:
                intersection_density_km2 = 0
                street_density_km2 = 0
            
            stats["intersection_density_km2"] = intersection_density_km2
            stats["street_density_km2"] = street_density_km2
            stats["area_km2"] = area_km2
            
            for k, v in stats.items():
                result_text += f"{k}: {v}\n"
            
            status = "processed"
        
        # Escribir resultados con un lock de archivo para evitar problemas de concurrencia
        with FileLock(f"{output_txt}.lock"):
            with open(output_txt, 'a', encoding='utf-8') as f:
                f.write(result_text)
                f.flush()  # Asegurar que se escriba inmediatamente
        
        # Actualizar y mostrar progreso
        with _progress_lock:
            _progress_counter += 1
            if _progress_counter % 5 == 0 or _progress_counter == total_polygons:
                print(f"Progreso: {_progress_counter}/{total_polygons} polígonos procesados ({(_progress_counter/total_polygons*100):.1f}%)")
        
        return (idx, sub_idx, result_text, status)
            
    except Exception as e:
        # Preparar mensaje de error
        error_text = f"\n--- Polígono {idx}-{sub_idx}: ERROR al crear la red ---\n{str(e)}\n"
        
        # Escribir el error al archivo
        with FileLock(f"{output_txt}.lock"):
            with open(output_txt, 'a', encoding='utf-8') as f:
                f.write(error_text)
                f.flush()
        
        # Actualizar contador incluso si hay error
        with _progress_lock:
            _progress_counter += 1
            if _progress_counter % 5 == 0 or _progress_counter == total_polygons:
                print(f"Progreso: {_progress_counter}/{total_polygons} polígonos procesados ({(_progress_counter/total_polygons*100):.1f}%)")
        
        return (idx, sub_idx, error_text, "error")

def get_street_network_metrics_per_polygon(
    geojson_path,
    network_type='drive',
    geojson_files_dict=None,
    max_polygon_workers=None
    ):
    """
    Lee un archivo GeoJSON que contiene uno o varios polígonos (o multipolígonos)
    y, para cada polígono/sub-polígono, calcula estadísticas de la red vial usando OSMnx.
    Luego, almacena los resultados en un archivo de texto, con un índice que
    identifica cada polígono procesado.

    Parámetros:
    -----------
    geojson_path : str
        Ruta al archivo GeoJSON.
    network_type : str
        Tipo de vías a recuperar ('all', 'drive', 'walk', etc.). Por defecto 'drive'.
    geojson_files_dict : dict
        Diccionario donde las claves son las rutas GeoJSON y los valores son los nombres descriptivos.
    max_polygon_workers : int, opcional
        Número máximo de procesos paralelos para procesar polígonos dentro de cada ciudad.

    Retorna:
    --------
    str: Ruta del archivo de salida generado.
    """
    
    # Obtener nombre descriptivo o usar el nombre base del archivo
    if geojson_files_dict and geojson_path in geojson_files_dict:
        pretty_name = geojson_files_dict[geojson_path]
    else:
        # Fallback si no se encuentra en el diccionario
        pretty_name = os.path.splitext(os.path.basename(geojson_path))[0]
    
    # Crear estructura de directorios
    # 1. Carpeta principal
    main_folder = "Polygons_analysis"
    
    # 2. Subcarpeta con el pretty name (limpiando posibles caracteres problemáticos)
    pretty_folder_name = pretty_name.replace(", ", "_").replace(" ", "_")
    pretty_folder_path = os.path.join(main_folder, pretty_folder_name)
    
    # 3. Carpeta stats dentro de la subcarpeta del pretty name
    stats_folder_path = os.path.join(pretty_folder_path, "stats")
    
    # Asegurar que existan todas las carpetas necesarias
    os.makedirs(stats_folder_path, exist_ok=True)
    
    # 4. Nombre del archivo final
    output_filename = f"Polygon_Stats_for_{pretty_folder_name}.txt"
    output_txt = os.path.join(stats_folder_path, output_filename)
    
    # ---------------------------------------------------------------------
    # 0. Si ya existe el .txt, leer su contenido previo
    #    y detectar qué Polígono/SubPolígono se han calculado.
    # ---------------------------------------------------------------------
    old_lines = []
    processed_pairs = set()  # para almacenar (idx, sub_idx)

    if os.path.exists(output_txt):
        with open(output_txt, 'r', encoding='utf-8') as old_file:
            old_lines = old_file.readlines()

        # Buscar líneas con el patrón: "=== Polígono X - SubPolígono Y ==="
        pattern = r"=== Polígono (\d+) - SubPolígono (\d+) ==="
        for line in old_lines:
            match = re.search(pattern, line)
            if match:
                i_str, s_str = match.groups()
                processed_pairs.add((int(i_str), int(s_str)))
    
    # ---------------------------------------------------------------------
    # 1. Cargar el GeoJSON como un GeoDataFrame
    # ---------------------------------------------------------------------
    gdf = gpd.read_file(geojson_path)
    
    # ---------------------------------------------------------------------
    # 2. Reescribir el archivo con el contenido previo
    # ---------------------------------------------------------------------
    with open(output_txt, 'w', encoding='utf-8') as f:
        # Si el archivo está vacío, agregar un encabezado
        if not old_lines:
            f.write(f"# Estadísticas para {pretty_name}\n")
            f.write(f"# Generado inicialmente: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        else:
            # Reescribir el contenido previo
            for line in old_lines:
                f.write(line)
    
    # ---------------------------------------------------------------------
    # 3. Preparar la lista de polígonos pendientes de procesar
    # ---------------------------------------------------------------------
    pending_polygons = []
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        
        if geometry is None or geometry.is_empty:
            continue
            
        if geometry.geom_type == 'Polygon':
            polygons_list = [geometry]
        elif geometry.geom_type == 'MultiPolygon':
            polygons_list = list(geometry.geoms)
        else:
            continue
            
        for sub_idx, poly in enumerate(polygons_list):
            # Verificar si ya está procesado
            if (idx, sub_idx) not in processed_pairs:
                pending_polygons.append((idx, sub_idx, poly))
    
    total_polygons = len(pending_polygons)
    if total_polygons == 0:
        print(f"¡Todos los polígonos ya están procesados para {pretty_name}!")
        return output_txt
    
    print(f"Procesando {total_polygons} polígonos pendientes para {pretty_name}...")
    
    # Determinar número de workers para polígonos
    if max_polygon_workers is None:
        max_polygon_workers = max(1, multiprocessing.cpu_count() - 1)
    max_polygon_workers = min(max_polygon_workers, total_polygons)
    
    # Reiniciar el contador de progreso global
    global _progress_counter
    _progress_counter = 0
    
    # Preparar los argumentos para process_polygon
    args_list = [(polygon_tuple, gdf.crs, network_type, output_txt, total_polygons) for polygon_tuple in pending_polygons]
    
    # Procesar en paralelo si hay suficientes polígonos
    results = []
    if total_polygons > 1 and max_polygon_workers > 1:
        # Usar multiprocessing.Pool para paralelizar
        with multiprocessing.Pool(processes=max_polygon_workers) as pool:
            results = pool.map(process_polygon, args_list)
            print(f"\nProcesamiento paralelo completado con {max_polygon_workers} workers")
    else:
        # Procesar secuencialmente si no vale la pena paralelizar
        print("Procesando secuencialmente...")
        results = [process_polygon(args) for args in args_list]
    
    # Contar resultados por tipo
    processed_count = sum(1 for _, _, _, status in results if status == "processed")
    empty_count = sum(1 for _, _, _, status in results if status == "empty")
    error_count = sum(1 for _, _, _, status in results if status == "error")
    
    print(f"Resumen del procesamiento:")
    print(f"  - Polígonos con red vial: {processed_count}")
    print(f"  - Polígonos sin vías (vacíos): {empty_count}")
    print(f"  - Polígonos con errores: {error_count}")
    print(f"  - Total procesados en esta ejecución: {total_polygons}")
    print(f"Resultados guardados en: {output_txt}")
    
    return output_txt

def process_single_geojson(geojson_path, geojson_files_dict=None, network_type='drive', max_polygon_workers=None):
    """Función para procesar un solo archivo GeoJSON"""
    return get_street_network_metrics_per_polygon(
        geojson_path=geojson_path,
        network_type=network_type,
        geojson_files_dict=geojson_files_dict,
        max_polygon_workers=max_polygon_workers
    )

def parallel_process_geojsons(geojson_files_dict, network_type='drive', max_city_workers=None, max_polygon_workers=None):
    """
    Procesa múltiples archivos GeoJSON de forma secuencial pero paralelizando el procesamiento
    de polígonos dentro de cada archivo.

    Parámetros:
    -----------
    geojson_files_dict : dict
        Diccionario donde las claves son rutas GeoJSON y los valores son nombres descriptivos.
    network_type : str
        Tipo de red a recuperar.
    max_city_workers : int, opcional
        Número máximo de procesos paralelos para ciudades. No se usa en esta implementación.
    max_polygon_workers : int, opcional
        Número máximo de procesos paralelos para polígonos dentro de cada ciudad.

    Retorna:
    --------
    list: Lista de rutas de archivos de salida generados.
    """
    # Usar todos los núcleos disponibles para polígonos si no se especifica
    if max_polygon_workers is None:
        max_polygon_workers = max(1, multiprocessing.cpu_count() - 1)
        
    results = []
    # Procesar ciudades de forma secuencial
    for geojson_path in geojson_files_dict.keys():
        city_name = geojson_files_dict[geojson_path]
        print(f"\nProcesando ciudad: {city_name}...")
        
        # Procesar cada ciudad con paralelismo a nivel de polígonos
        output_path = get_street_network_metrics_per_polygon(
            geojson_path=geojson_path,
            network_type=network_type,
            geojson_files_dict=geojson_files_dict,
            max_polygon_workers=max_polygon_workers
        )
        results.append(output_path)
        
    return results

def ordenar_y_limpiar_txt(input_txt, output_txt):
    """
    Lee un archivo .txt con bloques relacionados a "Polígono X - SubPolígono Y",
    o líneas de red vacía y errores, los ordena, elimina duplicados
    y genera un nuevo archivo .txt.

    Parámetros:
    -----------
    input_txt : str
        Ruta al archivo original desordenado.
    output_txt : str
        Ruta al archivo de salida ya ordenado y sin duplicados.

    Ejemplo de bloques reconocidos:
      "=== Polígono 181 - SubPolígono 0 ==="
      "--- Polígono 24-0: Grafo vacío (sin vías) ---"
      "=== Polígono 12: GEOMETRÍA VACÍA ==="
    """

    # 1. Leer todas las líneas
    if not os.path.exists(input_txt):
        print(f"Archivo {input_txt} no existe.")
        return

    with open(input_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. Expresiones regulares para reconocer "cabeceras"
    #    Bloques que inician con "=== Polígono X - SubPolígono Y ==="
    #    o "=== Polígono X: ..." (geom vacía) o "--- Polígono X-Y: Grafo vacío..."
    # ------------------------------------------------------------------
    #    Vamos a capturar:
    #      pol_idx -> \d+     (grupo 1)
    #      sub_idx -> \d+     (grupo 2), opcional
    #      type/block -> para distinguir "estadísticas", "vacío", "error", etc.
    #    
    #    Para simplificar, usaremos varios patrones y vemos cuál coincide.
    # ------------------------------------------------------------------

    # Caso A: "=== Polígono 181 - SubPolígono 0 ==="
    pattern_stats = re.compile(r'^=== Polígono\s+(\d+)\s*-\s*SubPolígono\s+(\d+)\s*===\s*$')
    # Caso B: "--- Polígono 24-0: Grafo vacío (sin vías) ---"
    pattern_empty = re.compile(r'^--- Polígono\s+(\d+)-(\d+):\s*(.+)---$')
    # Caso C: "=== Polígono 181: GEOMETRÍA VACÍA ==="
    pattern_geom = re.compile(r'^=== Polígono\s+(\d+):\s*(.+)===\s*$')

    # Vamos a recorrer las líneas y construir "bloques" de texto.
    # Cada vez que detectamos una cabecera, creamos una nueva entrada.
    data_dict = {}  
    # clave -> (pol_idx, sub_idx, tipo) 
    # valor -> list of lines (contenido)

    current_key = None
    current_block = []

    def save_block(key, block):
        """Guarda el bloque en data_dict, sobrescribiendo si la clave ya existía."""
        if key is None or not block:
            return
        # Sobrescribimos => la "última" versión de un polígono es la que prevalece
        data_dict[key] = block

    for line in lines:
        line_stripped = line.strip('\n')

        # ¿Coincide con pattern_stats?
        match_stats = pattern_stats.match(line_stripped)
        if match_stats:
            # Antes de iniciar nuevo bloque, guardamos el anterior
            save_block(current_key, current_block)
            current_block = [line]  # empezamos un nuevo bloque con esta línea
            pol = int(match_stats.group(1))
            sub = int(match_stats.group(2))
            current_key = (pol, sub, "stats") 
            continue

        # ¿Coincide con pattern_empty? (ej: Grafo vacío)
        match_empty = pattern_empty.match(line_stripped)
        if match_empty:
            save_block(current_key, current_block)
            current_block = [line]
            pol = int(match_empty.group(1))
            sub = int(match_empty.group(2))
            info = match_empty.group(3).strip()  # "Grafo vacío (sin vías)"
            current_key = (pol, sub, "empty:" + info)
            continue

        # ¿Coincide con pattern_geom? (ej: GEOMETRÍA VACÍA)
        match_geom = pattern_geom.match(line_stripped)
        if match_geom:
            save_block(current_key, current_block)
            current_block = [line]
            pol = int(match_geom.group(1))
            info = match_geom.group(2).strip()  # "GEOMETRÍA VACÍA" u otro
            # sub_idx = None => usaremos -1 para indicar "sin subpolígono"
            current_key = (pol, -1, "geom:" + info)
            continue

        # Si no coincide, es una línea dentro del bloque actual
        if current_key is not None:
            current_block.append(line)
        else:
            # No estamos dentro de un bloque (puede ser texto suelto),
            # lo ignoramos o guardamos si quieres.
            pass

    # Al final, guardar el último bloque (si existiera)
    save_block(current_key, current_block)

    # 3. Ahora data_dict tiene las entradas clave -> [lines_de_bloque].
    #    Queremos ordenarlas por polígono, subpolígono.
    #    Nota: sub_idx podría ser -1 en caso de "GEOMETRÍA VACÍA" (sin subpolígono).
    sorted_keys = sorted(data_dict.keys(), key=lambda x: (x[0], x[1]))

    # 4. Escribir el resultado ordenado y sin duplicados en output_txt
    with open(output_txt, 'w', encoding='utf-8') as out:
        for key in sorted_keys:
            block_lines = data_dict[key]
            for bl in block_lines:
                out.write(bl)
        out.write("\n")  # Salto de línea final

    print(f"Archivo ordenado y limpio guardado en: {output_txt}")

def post_processing_stats(geojson_files, output_folder=None):
    """
    Procesa múltiples archivos GeoJSON, los ordena, elimina duplicados
    y genera nuevos archivos en la carpeta stats de cada ciudad.

    Parámetros:
    -----------
    geojson_files : dict
        Diccionario con formato {ruta_archivo: nombre_bonito}
    output_folder : str, opcional
        No se utiliza en esta versión de la función, mantenido por compatibilidad.
    """
    # Procesar cada archivo GeoJSON
    for input_file, pretty_name in geojson_files.items():
        if not os.path.exists(input_file):
            print(f"Archivo {input_file} no existe. Saltando...")
            continue
        
        # Obtener el directorio donde está el archivo de entrada
        input_dir = os.path.dirname(input_file)
        
        # Determinar nombre del archivo de salida
        filename = os.path.basename(input_file)
        base_name = os.path.splitext(filename)[0]
        
        # Si hay pretty_name, úsalo en el nombre del archivo
        if pretty_name:
            # Eliminar caracteres especiales del pretty_name para usarlo en nombre de archivo
            safe_pretty_name = re.sub(r'[^\w\s-]', '', pretty_name).strip().replace(' ', '_')
            output_filename = f"Polygon_Analisys_{safe_pretty_name}_sorted.txt"
        else:
            output_filename = f"Polygon_Analisys_{base_name}_sorted.txt"
        
        # El archivo de salida va en la misma carpeta stats donde está el archivo original
        output_filepath = os.path.join(input_dir, output_filename)
        
        # Llamar a la función para procesar el archivo
        ordenar_y_limpiar_txt(input_file, output_filepath)
      
def geojson_to_excel(
    geojson_path: str, 
    output_path: str = None, 
    exclude_columns: List[str] = None,
    include_columns: List[str] = None,
    add_poly_id: bool = True,
    id_field: str = None,
    id_prefix: str = ""
) -> pd.DataFrame:
    """
    Extrae las propiedades de un archivo GeoJSON y las exporta a un archivo Excel.
    
    Args:
        geojson_path (str): Ruta al archivo GeoJSON.
        output_path (str, optional): Ruta donde guardar el archivo Excel. 
                                     Si es None, no se guarda el archivo.
        exclude_columns (List[str], optional): Lista de nombres de columnas a excluir.
        include_columns (List[str], optional): Lista de nombres de columnas a incluir.
                                               Si se especifica, solo se incluirán estas columnas.
        add_poly_id (bool): Si es True, agrega un identificador único para cada polígono.
        id_field (str, optional): Campo a usar como identificador si existe. Si es None,
                                 se generará un ID secuencial.
        id_prefix (str, optional): Prefijo para los IDs generados (ej: "CITY_").
    
    Returns:
        pd.DataFrame: DataFrame con las propiedades extraídas.
    """
    # Verificar si el archivo existe
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"El archivo {geojson_path} no existe")
    
    # Leer el archivo GeoJSON
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    # Extraer las propiedades de cada feature
    properties_list = []
    
    # Determinar el tipo de estructura del GeoJSON
    if "type" in geojson_data:
        if geojson_data["type"] == "FeatureCollection":
            features = geojson_data.get("features", [])
        elif geojson_data["type"] == "Feature":
            features = [geojson_data]
        else:
            # Asumimos que es una geometría individual con propiedades
            features = [{"type": "Feature", "geometry": geojson_data, "properties": geojson_data.get("properties", {})}]
    else:
        # Si no tiene tipo, asumimos que es un array de features
        features = geojson_data if isinstance(geojson_data, list) else [geojson_data]
    
    # Extraer propiedades de cada feature
    for i, feature in enumerate(features):
        if "properties" in feature and feature["properties"]:
            properties = feature["properties"].copy()
            
            # Agregar poly_id si se solicita
            if add_poly_id:
                if id_field and id_field in properties:
                    # Usar un campo existente como ID
                    poly_id = properties[id_field]
                    # Asegurarse de que sea una cadena
                    poly_id = f"{id_prefix}{poly_id}" if id_prefix else str(poly_id)
                else:
                    # Generar un ID secuencial
                    poly_id = f"{id_prefix}{i+1}" if id_prefix else f"poly_{i+1}"
                
                # Agregar el poly_id al principio de las propiedades para que aparezca primero en el Excel
                properties = {"poly_id": poly_id, **properties}
            
            properties_list.append(properties)
    
    if not properties_list:
        raise ValueError("No se encontraron propiedades en el archivo GeoJSON")
    
    # Crear DataFrame con todas las propiedades
    df = pd.DataFrame(properties_list)
    
    # Filtrar columnas si es necesario
    if include_columns:
        # Solo incluir las columnas especificadas
        valid_columns = [col for col in include_columns if col in df.columns]
        df = df[valid_columns]
    elif exclude_columns:
        # Excluir las columnas especificadas
        df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
    
    # Guardar en Excel si se especificó una ruta de salida
    if output_path:
        # Asegurarse de que exista el directorio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_excel(output_path, index=False)
        print(f"Archivo Excel guardado en: {output_path}")
    
    return df

def process_city_geojsons(
    geojson_files: Dict[str, str],
    base_output_dir: str = "Polygons_analysis",
    exclude_columns: List[str] = None,
    include_columns: List[str] = None,
    add_poly_id: bool = True,
    id_field: str = None,
    use_city_prefix: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Procesa múltiples archivos GeoJSON de diferentes ciudades y los guarda en una estructura
    de carpetas organizada.
    
    Args:
        geojson_files (Dict[str, str]): Diccionario con las rutas a los archivos GeoJSON como claves
                                       y los nombres legibles de las ciudades como valores.
        base_output_dir (str): Directorio base donde se crearán las carpetas de las ciudades.
        exclude_columns (List[str], optional): Lista de columnas a excluir.
        include_columns (List[str], optional): Lista de columnas a incluir.
        add_poly_id (bool): Si es True, agrega un identificador único para cada polígono.
        id_field (str, optional): Campo a usar como identificador si existe en el GeoJSON.
        use_city_prefix (bool): Si es True, usa el código de la ciudad como prefijo para los poly_id.
        
    Returns:
        Dict[str, pd.DataFrame]: Diccionario con los nombres de las ciudades como claves
                               y los DataFrames correspondientes como valores.
    """
    results = {}
    
    for geojson_path, city_name in geojson_files.items():
        # Formatear el nombre para carpetas y archivos
        formatted_name = city_name.replace(", ", "_").replace(" ", "_")
        
        # Crear la estructura de directorios
        city_dir = os.path.join(base_output_dir, formatted_name)
        mobility_dir = os.path.join(city_dir, "Mobility_Data")
        os.makedirs(mobility_dir, exist_ok=True)
        
        # Definir la ruta de salida del Excel
        excel_filename = f"polygon_mobility_data_{formatted_name}.xlsx"
        excel_path = os.path.join(mobility_dir, excel_filename)
        
        try:
            # Determinar el prefijo para los IDs de polígonos basados en el nombre de la ciudad
            city_code = ""
            if use_city_prefix:
                # Extraer código de la ciudad (ej: "Boston, MA" -> "BOS")
                parts = city_name.split(", ")
                if len(parts) == 2:
                    city_name_part = parts[0]
                    state_part = parts[1]
                    # Usar las primeras 3 letras de la ciudad en mayúsculas
                    city_code = city_name_part[:3].upper() + "_"
                else:
                    # Si no tiene formato "Ciudad, Estado", usar las primeras letras
                    city_code = formatted_name[:3].upper() + "_"
            
            # Procesar el archivo GeoJSON
            print(f"Procesando datos de {city_name}...")
            df = geojson_to_excel(
                geojson_path=geojson_path,
                output_path=excel_path,
                exclude_columns=exclude_columns,
                include_columns=include_columns,
                add_poly_id=add_poly_id,
                id_field=id_field,
                id_prefix=city_code
            )
            
            results[city_name] = df
            print(f"✓ Datos de {city_name} procesados correctamente.")
            
        except Exception as e:
            print(f"✗ Error al procesar {city_name}: {str(e)}")
    
    return results
