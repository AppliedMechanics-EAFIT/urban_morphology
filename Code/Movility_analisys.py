# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# import glob
# import sys
# import re
# import pathlib

# # Definir la ruta principal
# base_path = "Polygons_analysis"

# # Crear carpeta para resultados
# results_dir = "Resultados_Analisis"
# os.makedirs(results_dir, exist_ok=True)
# cities_results_dir = os.path.join(results_dir, "Ciudades")
# os.makedirs(cities_results_dir, exist_ok=True)

# # Lista para almacenar los dataframes combinados de todas las ciudades
# all_cities_data = []
# city_dataframes = {}  # Diccionario para guardar los dataframes por ciudad

# # Verificar que la ruta base exista
# if not os.path.exists(base_path):
#     print(f"Error: La carpeta {base_path} no existe.")
#     print(f"Directorio actual: {os.getcwd()}")
#     print(f"Carpetas disponibles: {os.listdir('.')}")
#     sys.exit(1)

# # Recorrer todas las carpetas de ciudades
# city_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
# print(f"Carpetas de ciudades encontradas: {city_folders}")

# # Función para extraer número de poly_id
# def extract_id_number(poly_id):
#     if isinstance(poly_id, str):
#         # Extraer el número después del prefijo (por ejemplo, 'BOS_1' → '1')
#         match = re.search(r'_(\d+)$', poly_id)
#         if match:
#             return int(match.group(1)) - 1  # Restamos 1 porque parece que los índices en patterns empiezan desde 0
#     return poly_id

# # Función para generar análisis por ciudad
# def analyze_city_data(city_data, city_name, output_dir):
#     print(f"\n=== Generando análisis para {city_name} ===")
    
#     # Crear carpeta específica para esta ciudad
#     city_dir = os.path.join(output_dir, city_name)
#     os.makedirs(city_dir, exist_ok=True)
    
#     # Verificar que las columnas necesarias estén presentes
#     required_cols = ['pattern', 'a', 'b', 'c']
#     missing_cols = [col for col in required_cols if col not in city_data.columns]
    
#     if missing_cols:
#         print(f"Faltan columnas necesarias para el análisis de {city_name}: {missing_cols}")
#         return False
    
#     # Estadísticas descriptivas
#     mobility_cols = ['a', 'b', 'c']
#     stats = city_data[mobility_cols].describe()
#     print(f"\nEstadísticas de movilidad para {city_name}:")
#     print(stats)
    
#     # Guardar estadísticas a CSV
#     stats.to_csv(os.path.join(city_dir, f"{city_name}_estadisticas_movilidad.csv"))
    
#     # Codificar patrones usando LabelEncoder
#     le = LabelEncoder()
#     city_data['pattern_code'] = le.fit_transform(city_data['pattern'])
    
#     # Análisis de correlación
#     corr_data = city_data[['pattern_code'] + mobility_cols].copy()
#     correlation_matrix = corr_data.corr()
#     print(f"\nCorrelación para {city_name}:")
#     print(correlation_matrix)
    
#     # Guardar matriz de correlación
#     correlation_matrix.to_csv(os.path.join(city_dir, f"{city_name}_correlacion.csv"))
    
#     # ANOVA
#     print(f"\nANOVA para {city_name}:")
#     anova_results = {}
#     for mobility_type in mobility_cols:
#         try:
#             model = ols(f'{mobility_type} ~ C(pattern)', data=city_data).fit()
#             anova_table = sm.stats.anova_lm(model, typ=2)
#             print(f"ANOVA para {mobility_type}:")
#             print(anova_table)
#             anova_results[mobility_type] = anova_table
#         except Exception as e:
#             print(f"Error en ANOVA para {mobility_type}: {e}")
    
#     # Guardar resultados de ANOVA
#     with open(os.path.join(city_dir, f"{city_name}_anova_resultados.txt"), 'w') as f:
#         for mobility_type, anova_table in anova_results.items():
#             f.write(f"ANOVA para {mobility_type}:\n")
#             f.write(anova_table.to_string())
#             f.write("\n\n")
    
#     # Visualizaciones
#     try:
#         # 1. Boxplots de movilidad por patrón
#         plt.figure(figsize=(15, 12))
        
#         plt.subplot(3, 1, 1)
#         sns.boxplot(x='pattern', y='a', data=city_data)
#         plt.title(f'{city_name}: Movilidad Activa por Patrón')
#         plt.xticks(rotation=45)
        
#         plt.subplot(3, 1, 2)
#         sns.boxplot(x='pattern', y='b', data=city_data)
#         plt.title(f'{city_name}: Movilidad Pública por Patrón')
#         plt.xticks(rotation=45)
        
#         plt.subplot(3, 1, 3)
#         sns.boxplot(x='pattern', y='c', data=city_data)
#         plt.title(f'{city_name}: Movilidad Privada por Patrón')
#         plt.xticks(rotation=45)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(city_dir, f"{city_name}_boxplots.png"))
#         plt.close()
        
#         # 2. Heatmap de correlación
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
#         plt.title(f'{city_name}: Correlación entre Patrones y Movilidad')
#         plt.savefig(os.path.join(city_dir, f"{city_name}_heatmap.png"))
#         plt.close()
        
#         # 3. Gráfico de barras de medias de movilidad por patrón
#         pattern_means = city_data.groupby('pattern')[mobility_cols].mean().reset_index()
        
#         # Reorganizar para el gráfico
#         pattern_means_melted = pd.melt(pattern_means, id_vars=['pattern'], 
#                                      value_vars=mobility_cols,
#                                      var_name='Tipo de Movilidad', 
#                                      value_name='Porcentaje')
        
#         # Nombres descriptivos
#         mobility_type_names = {
#             'a': 'Activa (caminar, bicicleta)',
#             'b': 'Pública (bus, tren, metro)',
#             'c': 'Privada (auto, moto)'
#         }
#         pattern_means_melted['Tipo de Movilidad'] = pattern_means_melted['Tipo de Movilidad'].map(mobility_type_names)
        
#         plt.figure(figsize=(14, 10))
#         sns.barplot(x='pattern', y='Porcentaje', hue='Tipo de Movilidad', data=pattern_means_melted)
#         plt.title(f'{city_name}: Porcentaje de Tipos de Movilidad por Patrón')
#         plt.xlabel('Patrón de Calle')
#         plt.ylabel('Porcentaje Medio (%)')
#         plt.xticks(rotation=45)
#         plt.legend(title='Tipo de Movilidad')
#         plt.tight_layout()
#         plt.savefig(os.path.join(city_dir, f"{city_name}_barras_movilidad.png"))
#         plt.close()
        
#         # 4. Si hay subpatrones, análisis de subpatrones
#         if 'cluster_name' in city_data.columns:
#             # Top subpatrones frecuentes
#             plt.figure(figsize=(12, 8))
#             subpattern_counts = city_data['cluster_name'].value_counts().head(10)
#             subpattern_counts.plot(kind='bar', color='skyblue')
#             plt.title(f'{city_name}: Top 10 Subpatrones más Frecuentes')
#             plt.xlabel('Subpatrón')
#             plt.ylabel('Número de Polígonos')
#             plt.xticks(rotation=45, ha='right')
#             plt.tight_layout()
#             plt.savefig(os.path.join(city_dir, f"{city_name}_top_subpatrones.png"))
#             plt.close()
            
#             # Movilidad por subpatrón
#             top_subpatterns = city_data['cluster_name'].value_counts().head(8).index.tolist()
#             if len(top_subpatterns) > 0:
#                 subpattern_data = city_data[city_data['cluster_name'].isin(top_subpatterns)]
                
#                 plt.figure(figsize=(14, 12))
                
#                 plt.subplot(3, 1, 1)
#                 sns.boxplot(x='cluster_name', y='a', data=subpattern_data, order=top_subpatterns)
#                 plt.title(f'{city_name}: Movilidad Activa por Subpatrón')
#                 plt.xticks(rotation=45, ha='right')
                
#                 plt.subplot(3, 1, 2)
#                 sns.boxplot(x='cluster_name', y='b', data=subpattern_data, order=top_subpatterns)
#                 plt.title(f'{city_name}: Movilidad Pública por Subpatrón')
#                 plt.xticks(rotation=45, ha='right')
                
#                 plt.subplot(3, 1, 3)
#                 sns.boxplot(x='cluster_name', y='c', data=subpattern_data, order=top_subpatterns)
#                 plt.title(f'{city_name}: Movilidad Privada por Subpatrón')
#                 plt.xticks(rotation=45, ha='right')
                
#                 plt.tight_layout()
#                 plt.subplots_adjust(hspace=0.4)
#                 plt.savefig(os.path.join(city_dir, f"{city_name}_subpatrones_movilidad.png"))
#                 plt.close()
        
#         # 5. Guardar datos procesados
#         city_data.to_excel(os.path.join(city_dir, f"{city_name}_datos_procesados.xlsx"), index=False)
#         pattern_means.to_csv(os.path.join(city_dir, f"{city_name}_medias_por_patron.csv"), index=False)
        
#         print(f"Análisis completado para {city_name}")
#         return True
        
#     except Exception as e:
#         print(f"Error en visualizaciones para {city_name}: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# # Función para análisis global de todas las ciudades
# def analyze_global_data(combined_data, output_dir):
#     print("\n=== Generando análisis global de todas las ciudades ===")
    
#     # Crear directorio para análisis global
#     global_dir = os.path.join(output_dir, "Analisis_Global")
#     os.makedirs(global_dir, exist_ok=True)
    
#     # Verificar columnas necesarias
#     required_cols = ['pattern', 'a', 'b', 'c', 'city']
#     missing_cols = [col for col in required_cols if col not in combined_data.columns]
    
#     if missing_cols:
#         print(f"Faltan columnas necesarias para el análisis global: {missing_cols}")
#         return False
    
#     # Estadísticas descriptivas generales
#     mobility_cols = ['a', 'b', 'c']
#     stats_global = combined_data[mobility_cols].describe()
#     print("\nEstadísticas globales de movilidad:")
#     print(stats_global)
    
#     # Estadísticas por ciudad
#     stats_by_city = combined_data.groupby('city')[mobility_cols].describe()
#     print("\nEstadísticas de movilidad por ciudad:")
#     print(stats_by_city)
    
#     # Estadísticas por patrón
#     stats_by_pattern = combined_data.groupby('pattern')[mobility_cols].describe()
#     print("\nEstadísticas de movilidad por patrón:")
#     print(stats_by_pattern)
    
#     # Codificar patrones
#     le = LabelEncoder()
#     combined_data['pattern_code'] = le.fit_transform(combined_data['pattern'])
#     pattern_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
#     # Análisis de correlación global
#     corr_data = combined_data[['pattern_code'] + mobility_cols].copy()
#     correlation_matrix = corr_data.corr()
#     print("\nCorrelación global:")
#     print(correlation_matrix)
    
#     # ANOVA global
#     print("\nANOVA global:")
#     anova_results_global = {}
#     for mobility_type in mobility_cols:
#         try:
#             model = ols(f'{mobility_type} ~ C(pattern)', data=combined_data).fit()
#             anova_table = sm.stats.anova_lm(model, typ=2)
#             print(f"ANOVA global para {mobility_type}:")
#             print(anova_table)
#             anova_results_global[mobility_type] = anova_table
#         except Exception as e:
#             print(f"Error en ANOVA global para {mobility_type}: {e}")
    
#     # Guardar estadísticas y resultados
#     stats_global.to_csv(os.path.join(global_dir, "estadisticas_globales.csv"))
#     stats_by_city.to_csv(os.path.join(global_dir, "estadisticas_por_ciudad.csv"))
#     stats_by_pattern.to_csv(os.path.join(global_dir, "estadisticas_por_patron.csv"))
#     correlation_matrix.to_csv(os.path.join(global_dir, "correlacion_global.csv"))
    
#     with open(os.path.join(global_dir, "anova_resultados_globales.txt"), 'w') as f:
#         for mobility_type, anova_table in anova_results_global.items():
#             f.write(f"ANOVA para {mobility_type}:\n")
#             f.write(anova_table.to_string())
#             f.write("\n\n")
    
#     # Visualizaciones globales
#     try:
#         # 1. Boxplots globales por patrón
#         plt.figure(figsize=(16, 12))
        
#         plt.subplot(3, 1, 1)
#         sns.boxplot(x='pattern', y='a', data=combined_data)
#         plt.title('Movilidad Activa por Patrón (Todas las ciudades)')
#         plt.xticks(rotation=45)
        
#         plt.subplot(3, 1, 2)
#         sns.boxplot(x='pattern', y='b', data=combined_data)
#         plt.title('Movilidad Pública por Patrón (Todas las ciudades)')
#         plt.xticks(rotation=45)
        
#         plt.subplot(3, 1, 3)
#         sns.boxplot(x='pattern', y='c', data=combined_data)
#         plt.title('Movilidad Privada por Patrón (Todas las ciudades)')
#         plt.xticks(rotation=45)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(global_dir, "global_boxplots_patron.png"))
#         plt.close()
        
#         # 2. Boxplots por ciudad
#         plt.figure(figsize=(16, 12))
        
#         plt.subplot(3, 1, 1)
#         sns.boxplot(x='city', y='a', data=combined_data)
#         plt.title('Movilidad Activa por Ciudad')
#         plt.xticks(rotation=45)
        
#         plt.subplot(3, 1, 2)
#         sns.boxplot(x='city', y='b', data=combined_data)
#         plt.title('Movilidad Pública por Ciudad')
#         plt.xticks(rotation=45)
        
#         plt.subplot(3, 1, 3)
#         sns.boxplot(x='city', y='c', data=combined_data)
#         plt.title('Movilidad Privada por Ciudad')
#         plt.xticks(rotation=45)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(global_dir, "global_boxplots_ciudad.png"))
#         plt.close()
        
#         # 3. Heatmap de correlación global
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
#         plt.title('Correlación Global entre Patrones y Movilidad')
#         plt.savefig(os.path.join(global_dir, "global_heatmap.png"))
#         plt.close()
        
#         # 4. Gráfico de barras global por patrón
#         global_pattern_means = combined_data.groupby('pattern')[mobility_cols].mean().reset_index()
#         global_means_melted = pd.melt(global_pattern_means, id_vars=['pattern'], 
#                                      value_vars=mobility_cols,
#                                      var_name='Tipo de Movilidad', 
#                                      value_name='Porcentaje')
        
#         mobility_type_names = {
#             'a': 'Activa (caminar, bicicleta)',
#             'b': 'Pública (bus, tren, metro)',
#             'c': 'Privada (auto, moto)'
#         }
#         global_means_melted['Tipo de Movilidad'] = global_means_melted['Tipo de Movilidad'].map(mobility_type_names)
        
#         plt.figure(figsize=(14, 10))
#         sns.barplot(x='pattern', y='Porcentaje', hue='Tipo de Movilidad', data=global_means_melted)
#         plt.title('Porcentaje Global de Tipos de Movilidad por Patrón')
#         plt.xlabel('Patrón de Calle')
#         plt.ylabel('Porcentaje Medio (%)')
#         plt.xticks(rotation=45)
#         plt.legend(title='Tipo de Movilidad')
#         plt.tight_layout()
#         plt.savefig(os.path.join(global_dir, "global_barras_patron.png"))
#         plt.close()
        
#         # 5. Gráfico de barras por ciudad
#         city_means = combined_data.groupby('city')[mobility_cols].mean().reset_index()
#         city_means_melted = pd.melt(city_means, id_vars=['city'], 
#                                   value_vars=mobility_cols,
#                                   var_name='Tipo de Movilidad', 
#                                   value_name='Porcentaje')
        
#         city_means_melted['Tipo de Movilidad'] = city_means_melted['Tipo de Movilidad'].map(mobility_type_names)
        
#         plt.figure(figsize=(14, 8))
#         sns.barplot(x='city', y='Porcentaje', hue='Tipo de Movilidad', data=city_means_melted)
#         plt.title('Tipos de Movilidad por Ciudad')
#         plt.xlabel('Ciudad')
#         plt.ylabel('Porcentaje Medio (%)')
#         plt.xticks(rotation=45)
#         plt.legend(title='Tipo de Movilidad')
#         plt.tight_layout()
#         plt.savefig(os.path.join(global_dir, "global_barras_ciudad.png"))
#         plt.close()
        
#         # 6. Distribución de patrones por ciudad
#         pattern_counts = pd.crosstab(combined_data['city'], combined_data['pattern'])
#         pattern_counts_pct = pattern_counts.div(pattern_counts.sum(axis=1), axis=0) * 100
        
#         plt.figure(figsize=(16, 10))
#         pattern_counts_pct.plot(kind='bar', stacked=True, colormap='viridis')
#         plt.title('Distribución de Patrones por Ciudad')
#         plt.xlabel('Ciudad')
#         plt.ylabel('Porcentaje (%)')
#         plt.legend(title='Patrón', bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.tight_layout()
#         plt.savefig(os.path.join(global_dir, "distribucion_patrones_por_ciudad.png"))
#         plt.close()
        
#         # 7. Análisis de relación entre patrones y movilidad por ciudad
#         city_pattern_mobility = combined_data.groupby(['city', 'pattern'])[mobility_cols].mean().reset_index()
#         city_pattern_mobility.to_csv(os.path.join(global_dir, "ciudad_patron_movilidad.csv"), index=False)
        
#         # Guardar dataframe combinado
#         combined_data.to_excel(os.path.join(global_dir, "datos_combinados.xlsx"), index=False)
        
#         print("Análisis global completado")
#         return True
        
#     except Exception as e:
#         print(f"Error en visualizaciones globales: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# # Principal: Procesar las ciudades
# for city in city_folders:
#     print(f"\n--- Procesando ciudad: {city} ---")
    
#     # Cargar datos de movilidad
#     mobility_path = os.path.join(base_path, city, "Mobility_data")
    
#     if not os.path.exists(mobility_path):
#         print(f"La carpeta {mobility_path} no existe")
#         continue
        
#     mobility_files = glob.glob(os.path.join(mobility_path, "*.xlsx"))
    
#     if not mobility_files:
#         print(f"No se encontraron archivos Excel de movilidad en {mobility_path}")
#         continue
    
#     print(f"Archivo de movilidad encontrado: {mobility_files[0]}")
#     try:
#         mobility_df = pd.read_excel(mobility_files[0])
#         print(f"Columnas en datos de movilidad: {mobility_df.columns.tolist()}")
#         print(f"Número de registros en mobility_df: {len(mobility_df)}")
        
#         # Verificar que las columnas necesarias estén presentes
#         required_cols = ['poly_id', 'a', 'b', 'c']
#         missing_cols = [col for col in required_cols if col not in mobility_df.columns]
#         if missing_cols:
#             print(f"Faltan columnas en el archivo de movilidad: {missing_cols}")
#             continue
        
#         # Añadir columna para mapeo con el patrón urbano
#         mobility_df['numeric_id'] = mobility_df['poly_id'].apply(extract_id_number)
#         print(f"Muestra de IDs convertidos: {list(zip(mobility_df['poly_id'].head(), mobility_df['numeric_id'].head()))}")
        
#     except Exception as e:
#         print(f"Error al leer el archivo de movilidad: {e}")
#         continue
    
#     # Cargar datos de patrones urbanos
#     pattern_path = os.path.join(base_path, city, "clustering_analysis")
#     pattern_file = os.path.join(pattern_path, "urban_pattern_analysis.xlsx")
    
#     if not os.path.exists(pattern_file):
#         print(f"No se encontró el archivo de patrones urbanos en {pattern_path}")
#         continue
    
#     # Cargar hoja 5 (patrones principales)
#     try:
#         patterns_main_df = pd.read_excel(pattern_file, sheet_name=4)  # 0-indexed, hoja 5
#         # Cargar hoja 6 (subpatrones)
#         patterns_sub_df = pd.read_excel(pattern_file, sheet_name=5)  # 0-indexed, hoja 6
        
#         print(f"Columnas en hoja 5: {patterns_main_df.columns.tolist()}")
#         print(f"Columnas en hoja 6: {patterns_sub_df.columns.tolist()}")
#         print(f"Número de registros en patterns_main_df: {len(patterns_main_df)}")
#         print(f"Número de registros en patterns_sub_df: {len(patterns_sub_df)}")
        
#         # Asegurar que los campos numéricos sean consistentes
#         if 'poly_id' in patterns_main_df.columns:
#             if patterns_main_df['poly_id'].dtype != mobility_df['numeric_id'].dtype:
#                 patterns_main_df['poly_id'] = patterns_main_df['poly_id'].astype(int)
#                 patterns_sub_df['poly_id'] = patterns_sub_df['poly_id'].astype(int)
#                 mobility_df['numeric_id'] = mobility_df['numeric_id'].astype(int)
        
#         # Unir los dataframes usando el ID numérico
#         try:
#             print("Intentando unir dataframes por numeric_id...")
#             city_data = pd.merge(mobility_df, patterns_main_df, 
#                                 left_on='numeric_id', right_on='poly_id', 
#                                 how='inner', suffixes=('', '_pattern'))
            
#             city_data = pd.merge(city_data, patterns_sub_df,
#                                 left_on='numeric_id', right_on='poly_id',
#                                 how='inner', suffixes=('', '_sub'))
            
#             matches = len(city_data)
#             print(f"Coincidencias encontradas después del merge: {matches}")
            
#             if matches == 0:
#                 print("No se encontraron coincidencias. Intentando un enfoque alternativo...")
                
#                 # Crear índice numérico basado en el orden de los datos si los tamaños son similares
#                 if abs(len(mobility_df) - len(patterns_main_df)) <= 5:  # Tolerancia para pequeñas diferencias
#                     print("Los tamaños de los dataframes son similares. Intentando alinear por índice...")
#                     mobility_df = mobility_df.reset_index(drop=True)
#                     patterns_main_df = patterns_main_df.reset_index(drop=True)
#                     patterns_sub_df = patterns_sub_df.reset_index(drop=True)
                    
#                     # Crear un dataframe combinado asumiendo que están en el mismo orden
#                     city_data = mobility_df.copy()
#                     if 'original_pattern' in patterns_main_df.columns:
#                         city_data['pattern'] = patterns_main_df['original_pattern'].values
#                     if 'cluster_name' in patterns_sub_df.columns:
#                         city_data['cluster_name'] = patterns_sub_df['cluster_name'].values
                    
#                     print(f"Alineado por índice completado con {len(city_data)} registros")
#                 else:
#                     print("Los tamaños de los dataframes difieren significativamente. No se puede alinear.")
#                     continue
#             else:
#                 # Renombrar 'original_pattern' a 'pattern' para consistencia
#                 if 'original_pattern' in city_data.columns:
#                     city_data = city_data.rename(columns={'original_pattern': 'pattern'})
            
#             # Añadir columna de ciudad
#             city_data['city'] = city
            
#             # Guardar dataframe para análisis individual
#             city_dataframes[city] = city_data
            
#             # Añadir a la lista de todas las ciudades
#             if len(city_data) > 0:
#                 all_cities_data.append(city_data)
#                 print(f"Ciudad {city} procesada correctamente con {len(city_data)} registros.")
#             else:
#                 print(f"No se encontraron coincidencias para la ciudad {city}")
            
#         except Exception as e:
#             print(f"Error durante el merge de datos: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
            
#     except Exception as e:
#         print(f"Error al procesar {city}: {e}")
#         import traceback
#         traceback.print_exc()

# # Análisis por ciudad
# print("\n===== INICIANDO ANÁLISIS POR CIUDAD =====")
# for city, city_df in city_dataframes.items():
#     analyze_city_data(city_df, city, cities_results_dir)

# # Análisis global combinando todas las ciudades
# if all_cities_data:
#     print("\n===== INICIANDO ANÁLISIS GLOBAL =====")
#     combined_data = pd.concat(all_cities_data, ignore_index=True)
#     print(f"Total de registros combinados: {len(combined_data)}")
    
#     if len(combined_data) > 0:
#         analyze_global_data(combined_data, results_dir)
#     else:
#         print("No hay datos combinados para realizar análisis global.")
# else:
#     print("No se pudieron cargar datos de ninguna ciudad.")
#     sys.exit(1)

# print("\n===== ANÁLISIS COMPLETADO =====")
# print(f"Los resultados se han guardado en la carpeta: {os.path.abspath(results_dir)}")





import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr, spearmanr
import glob
import re
import pathlib
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analisis_patrones_movilidad.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class StreetPatternMobilityAnalyzer:
    """
    Clase para analizar la relación entre patrones de calles y movilidad urbana.
    """
    def __init__(self, base_path="Polygons_analysis", results_dir="Resultados_Analisis"):
        """
        Inicializa el analizador.
        
        Args:
            base_path: Ruta base donde se encuentran los datos de las ciudades
            results_dir: Directorio donde se guardarán los resultados
        """
        self.base_path = base_path
        self.results_dir = results_dir
        self.cities_results_dir = os.path.join(results_dir, "Ciudades")
        self.global_dir = os.path.join(results_dir, "Analisis_Global")
        
        # Crear directorios para resultados
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cities_results_dir, exist_ok=True)
        os.makedirs(self.global_dir, exist_ok=True)
        
        # Almacenamiento para datos procesados
        self.city_dataframes = {}
        self.all_cities_data = []
        
        # Definir las columnas de movilidad y sus nombres descriptivos
        self.mobility_columns = {
            'a': 'Movilidad Activa',
            'b': 'Movilidad Pública',
            'c': 'Movilidad Privada',
            'car_mode': 'Uso de Automóvil',
            'transit_mode': 'Uso de Transporte Público',
            'bicycle_mode': 'Uso de Bicicleta',
            'walked_mode': 'Desplazamiento a Pie',
            'car_share': 'Porcentaje Automóvil',
            'transit_share': 'Porcentaje Transporte Público',
            'bicycle_share': 'Porcentaje Bicicleta',
            'walked_share': 'Porcentaje a Pie'
        }
        
        # Columnas que necesitan normalización por población
        self.normalize_by_population = ['car_mode', 'transit_mode', 'bicycle_mode', 'walked_mode']
        
        # Columnas que ya están en porcentaje
        self.percentage_columns = ['a', 'b', 'c', 'car_share', 'transit_share', 'bicycle_share', 'walked_share']
        
        # Columnas agrupadas por tipo de movilidad
        self.mobility_groups = {
            'Activa': ['a', 'bicycle_mode', 'walked_mode', 'bicycle_share', 'walked_share'],
            'Pública': ['b', 'transit_mode', 'transit_share'],
            'Privada': ['c', 'car_mode', 'car_share']
        }
    
    def extract_id_number(self, poly_id):
        """
        Extrae el número identificador del polígono.
        
        Args:
            poly_id: Identificador del polígono (ej: 'BOS_1')
            
        Returns:
            El número identificador extraído
        """
        if isinstance(poly_id, str):
            match = re.search(r'_(\d+)$', poly_id)
            if match:
                return int(match.group(1)) - 1
        return poly_id
    
    def load_mobility_data(self, city):
        """
        Carga los datos de movilidad para una ciudad específica.
        
        Args:
            city: Nombre de la ciudad a procesar
            
        Returns:
            DataFrame con los datos de movilidad o None si hay error
        """
        mobility_path = os.path.join(self.base_path, city, "Mobility_Data")
        
        if not os.path.exists(mobility_path):
            logger.error(f"La carpeta {mobility_path} no existe")
            return None
            
        mobility_files = glob.glob(os.path.join(mobility_path, "*.xlsx"))
        
        if not mobility_files:
            logger.error(f"No se encontraron archivos Excel de movilidad en {mobility_path}")
            return None
        
        logger.info(f"Archivo de movilidad encontrado: {mobility_files[0]}")
        try:
            mobility_df = pd.read_excel(mobility_files[0])
            logger.info(f"Columnas en datos de movilidad: {mobility_df.columns.tolist()}")
            logger.info(f"Número de registros: {len(mobility_df)}")
            
            # Verificar columnas necesarias
            required_cols = ['poly_id', 'total_pop']
            missing_cols = [col for col in required_cols if col not in mobility_df.columns]
            
            if missing_cols:
                logger.error(f"Faltan columnas necesarias en el archivo de movilidad: {missing_cols}")
                return None
            
            # Añadir identificador numérico para mapeo
            mobility_df['numeric_id'] = mobility_df['poly_id'].apply(self.extract_id_number)
            
            # Normalizar columnas por población si es necesario
            for col in self.normalize_by_population:
                if col in mobility_df.columns:
                    if 'total_pop' in mobility_df.columns and (mobility_df['total_pop'] > 0).all():
                        mobility_df[f'{col}_norm'] = mobility_df[col] / mobility_df['total_pop'] * 100
                        logger.info(f"Columna {col} normalizada por población")
                    else:
                        logger.warning(f"No se pudo normalizar {col} por población")
            
            return mobility_df
            
        except Exception as e:
            logger.error(f"Error al leer el archivo de movilidad: {e}", exc_info=True)
            return None
    
    def load_patterns_data(self, city):
        """
        Carga los datos de patrones urbanos para una ciudad específica.
        
        Args:
            city: Nombre de la ciudad a procesar
            
        Returns:
            Tupla con (patterns_main_df, patterns_sub_df) o (None, None) si hay error
        """
        pattern_path = os.path.join(self.base_path, city, "clustering_analysis")
        pattern_file = os.path.join(pattern_path, "urban_pattern_analysis.xlsx")
        
        if not os.path.exists(pattern_file):
            logger.error(f"No se encontró el archivo de patrones urbanos en {pattern_path}")
            return None, None
        
        try:
            # Cargar datos de patrones principales (hoja 5, índice 4)
            patterns_main_df = pd.read_excel(pattern_file, sheet_name=4)
            
            # Cargar datos de subpatrones (hoja 6, índice 5)
            patterns_sub_df = pd.read_excel(pattern_file, sheet_name=5)
            
            logger.info(f"Datos de patrones cargados: {len(patterns_main_df)} patrones principales, {len(patterns_sub_df)} subpatrones")
            
            # Asegurar que los IDs son del tipo correcto
            if 'poly_id' in patterns_main_df.columns:
                patterns_main_df['poly_id'] = patterns_main_df['poly_id'].astype(int)
            if 'poly_id' in patterns_sub_df.columns:
                patterns_sub_df['poly_id'] = patterns_sub_df['poly_id'].astype(int)
                
            return patterns_main_df, patterns_sub_df
            
        except Exception as e:
            logger.error(f"Error al cargar datos de patrones: {e}", exc_info=True)
            return None, None
    
    def merge_city_data(self, mobility_df, patterns_main_df, patterns_sub_df, city):
        """
        Une los datos de movilidad y patrones para una ciudad.
        
        Args:
            mobility_df: DataFrame con datos de movilidad
            patterns_main_df: DataFrame con patrones principales
            patterns_sub_df: DataFrame con subpatrones
            city: Nombre de la ciudad
            
        Returns:
            DataFrame combinado o None si hay error
        """
        if mobility_df is None or patterns_main_df is None:
            return None
            
        try:
            # Intentar unir por ID numérico
            city_data = pd.merge(mobility_df, patterns_main_df, 
                               left_on='numeric_id', right_on='poly_id', 
                               how='inner', suffixes=('', '_pattern'))
            
            if patterns_sub_df is not None:
                city_data = pd.merge(city_data, patterns_sub_df,
                                   left_on='numeric_id', right_on='poly_id',
                                   how='inner', suffixes=('', '_sub'))
            
            matches = len(city_data)
            logger.info(f"Coincidencias encontradas después del merge: {matches}")
            
            if matches == 0:
                logger.warning("No se encontraron coincidencias. Intentando un enfoque alternativo...")
                
                # Alternativa: alinear por índice si los tamaños son similares
                if abs(len(mobility_df) - len(patterns_main_df)) <= 5:
                    logger.info("Los tamaños de los dataframes son similares. Alineando por índice...")
                    mobility_df = mobility_df.reset_index(drop=True)
                    patterns_main_df = patterns_main_df.reset_index(drop=True)
                    
                    city_data = mobility_df.copy()
                    
                    if 'original_pattern' in patterns_main_df.columns:
                        city_data['pattern'] = patterns_main_df['original_pattern'].values
                    elif 'pattern' in patterns_main_df.columns:
                        city_data['pattern'] = patterns_main_df['pattern'].values
                    
                    if patterns_sub_df is not None:
                        patterns_sub_df = patterns_sub_df.reset_index(drop=True)
                        if 'cluster_name' in patterns_sub_df.columns:
                            city_data['cluster_name'] = patterns_sub_df['cluster_name'].values
                    
                    logger.info(f"Alineado por índice completado con {len(city_data)} registros")
                else:
                    logger.error("Los tamaños de los dataframes difieren significativamente. No se puede alinear.")
                    return None
            else:
                # Renombrar 'original_pattern' a 'pattern' para consistencia
                if 'original_pattern' in city_data.columns and 'pattern' not in city_data.columns:
                    city_data = city_data.rename(columns={'original_pattern': 'pattern'})
            
            # Añadir columna de ciudad
            city_data['city'] = city
            
            return city_data
            
        except Exception as e:
            logger.error(f"Error durante el merge de datos: {e}", exc_info=True)
            return None
    
    def process_cities(self):
        """
        Procesa todas las ciudades disponibles en la ruta base.
        
        Returns:
            True si se procesó al menos una ciudad con éxito, False en caso contrario
        """
        # Obtener lista de carpetas de ciudades
        city_folders = [f for f in os.listdir(self.base_path) 
                      if os.path.isdir(os.path.join(self.base_path, f))]
        
        logger.info(f"Carpetas de ciudades encontradas: {city_folders}")
        
        if not city_folders:
            logger.error("No se encontraron carpetas de ciudades")
            return False
        
        # Procesar cada ciudad
        success = False
        for city in city_folders:
            logger.info(f"\n--- Procesando ciudad: {city} ---")
            
            # Cargar datos
            mobility_df = self.load_mobility_data(city)
            patterns_main_df, patterns_sub_df = self.load_patterns_data(city)
            
            if mobility_df is None or patterns_main_df is None:
                logger.warning(f"No se pudieron cargar datos para {city}")
                continue
                
            # Unir datos
            city_data = self.merge_city_data(mobility_df, patterns_main_df, patterns_sub_df, city)
            
            if city_data is not None and len(city_data) > 0:
                # Guardar dataframe para análisis individual
                self.city_dataframes[city] = city_data
                
                # Añadir a la lista de todas las ciudades
                self.all_cities_data.append(city_data)
                logger.info(f"Ciudad {city} procesada correctamente con {len(city_data)} registros")
                success = True
            else:
                logger.warning(f"No se pudieron combinar datos para {city}")
        
        return success
    
    def analyze_city(self, city_data, city_name):
        """
        Realiza análisis estadístico y visualizaciones para una ciudad.
        
        Args:
            city_data: DataFrame con datos combinados de la ciudad
            city_name: Nombre de la ciudad
            
        Returns:
            True si el análisis fue exitoso, False en caso contrario
        """
        logger.info(f"\n=== Generando análisis para {city_name} ===")
        
        # Crear carpeta específica para esta ciudad
        city_dir = os.path.join(self.cities_results_dir, city_name)
        os.makedirs(city_dir, exist_ok=True)
        
        # Verificar columnas disponibles de movilidad
        available_mobility_cols = [col for col in self.mobility_columns.keys() 
                                 if col in city_data.columns]
        
        if not available_mobility_cols:
            logger.error(f"No hay columnas de movilidad disponibles para {city_name}")
            return False
            
        if 'pattern' not in city_data.columns:
            logger.error(f"No hay columna de patrón disponible para {city_name}")
            return False
        
        try:
            # 1. Estadísticas descriptivas por patrón
            stats_by_pattern = city_data.groupby('pattern')[available_mobility_cols].describe()
            stats_by_pattern.to_csv(os.path.join(city_dir, f"{city_name}_estadisticas_por_patron.csv"))
            
            # 2. Análisis de correlación
            # Codificar patrones
            le = LabelEncoder()
            city_data['pattern_code'] = le.fit_transform(city_data['pattern'])
            
            # Correlación entre patrones y movilidad
            correlation_vars = ['pattern_code'] + available_mobility_cols
            correlation_matrix = city_data[correlation_vars].corr()
            correlation_matrix.to_csv(os.path.join(city_dir, f"{city_name}_correlacion.csv"))
            
            # 3. ANOVA: diferencias significativas entre patrones
            anova_results = {}
            for mobility_type in available_mobility_cols:
                try:
                    model = ols(f'{mobility_type} ~ C(pattern)', data=city_data).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    anova_results[mobility_type] = anova_table
                    logger.info(f"ANOVA para {mobility_type} en {city_name}")
                except Exception as e:
                    logger.error(f"Error en ANOVA para {mobility_type}: {e}")
            
            # Guardar resultados de ANOVA
            with open(os.path.join(city_dir, f"{city_name}_anova_resultados.txt"), 'w') as f:
                for mobility_type, anova_table in anova_results.items():
                    f.write(f"ANOVA para {mobility_type}:\n")
                    f.write(anova_table.to_string())
                    f.write("\n\n")
            
            # 4. Visualizaciones avanzadas
            self._generate_city_visualizations(city_data, city_name, city_dir, available_mobility_cols)
            
            # 5. Análisis de subpatrones si están disponibles
            if 'cluster_name' in city_data.columns:
                self._analyze_subpatterns(city_data, city_name, city_dir, available_mobility_cols)
            
            # 6. Guardar datos procesados
            city_data.to_excel(os.path.join(city_dir, f"{city_name}_datos_procesados.xlsx"), index=False)
            
            logger.info(f"Análisis completado para {city_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error en análisis para {city_name}: {e}", exc_info=True)
            return False
    
    def _generate_city_visualizations(self, city_data, city_name, city_dir, mobility_cols):
        """
        Genera visualizaciones para el análisis de una ciudad.
        
        Args:
            city_data: DataFrame con datos de la ciudad
            city_name: Nombre de la ciudad
            city_dir: Directorio donde guardar las visualizaciones
            mobility_cols: Lista de columnas de movilidad disponibles
        """
        try:
            # 1. Boxplots de movilidad por patrón
            for i, mobility_type in enumerate(mobility_cols):
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='pattern', y=mobility_type, data=city_data)
                plt.title(f'{city_name}: {self.mobility_columns[mobility_type]} por Patrón')
                plt.xlabel('Patrón de Calle')
                plt.ylabel(self.mobility_columns[mobility_type])
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(city_dir, f"{city_name}_boxplot_{mobility_type}.png"))
                plt.close()
            
            # 2. Heatmap de correlación
            correlation_vars = ['pattern_code'] + mobility_cols
            correlation_matrix = city_data[correlation_vars].corr()
            
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                      linewidths=.5, mask=mask, vmin=-1, vmax=1)
            plt.title(f'{city_name}: Correlación entre Patrones y Movilidad')
            plt.tight_layout()
            plt.savefig(os.path.join(city_dir, f"{city_name}_heatmap.png"))
            plt.close()
            
            # 3. Gráfico de barras de medias de movilidad por patrón
            pattern_means = city_data.groupby('pattern')[mobility_cols].mean().reset_index()
            pattern_means_melted = pd.melt(pattern_means, id_vars=['pattern'], 
                                         value_vars=mobility_cols,
                                         var_name='Tipo de Movilidad', 
                                         value_name='Valor')
            
            pattern_means_melted['Tipo de Movilidad'] = pattern_means_melted['Tipo de Movilidad'].map(self.mobility_columns)
            
            plt.figure(figsize=(14, 10))
            sns.barplot(x='pattern', y='Valor', hue='Tipo de Movilidad', data=pattern_means_melted)
            plt.title(f'{city_name}: Valores de Movilidad por Patrón')
            plt.xlabel('Patrón de Calle')
            plt.ylabel('Valor Medio')
            plt.xticks(rotation=45)
            plt.legend(title='Tipo de Movilidad', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(city_dir, f"{city_name}_barras_movilidad.png"))
            plt.close()
            
            # 4. Análisis de componentes principales (PCA) para movilidad
            if len(mobility_cols) >= 3:
                # Escalar datos para PCA
                scaler = StandardScaler()
                mobility_scaled = scaler.fit_transform(city_data[mobility_cols])
                
                # Aplicar PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(mobility_scaled)
                
                # Crear DataFrame con resultados
                pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
                pca_df['pattern'] = city_data['pattern'].values
                
                # Graficar PCA
                plt.figure(figsize=(12, 10))
                sns.scatterplot(x='PC1', y='PC2', hue='pattern', data=pca_df, palette='viridis', s=100)
                plt.title(f'{city_name}: PCA de Movilidad por Patrón')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(title='Patrón', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(city_dir, f"{city_name}_pca_movilidad.png"))
                plt.close()
                
                # Guardar coeficientes de componentes principales
                pca_components = pd.DataFrame(
                    data=pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                    index=mobility_cols
                )
                pca_components.to_csv(os.path.join(city_dir, f"{city_name}_pca_componentes.csv"))
            
        except Exception as e:
            logger.error(f"Error generando visualizaciones para {city_name}: {e}", exc_info=True)
    
    def _analyze_subpatterns(self, city_data, city_name, city_dir, mobility_cols):
        """
        Realiza análisis específico de subpatrones para una ciudad.
        
        Args:
            city_data: DataFrame con datos de la ciudad
            city_name: Nombre de la ciudad
            city_dir: Directorio donde guardar los resultados
            mobility_cols: Lista de columnas de movilidad disponibles
        """
        try:
            # Crear subcarpeta para análisis de subpatrones
            subpatterns_dir = os.path.join(city_dir, "Subpatrones")
            os.makedirs(subpatterns_dir, exist_ok=True)
            
            # Contar frecuencia de subpatrones
            subpattern_counts = city_data['cluster_name'].value_counts()
            subpattern_counts.to_csv(os.path.join(subpatterns_dir, f"{city_name}_frecuencia_subpatrones.csv"))
            
            # Seleccionar los 10 subpatrones más frecuentes
            top_subpatterns = subpattern_counts.head(10).index.tolist()
            subpattern_data = city_data[city_data['cluster_name'].isin(top_subpatterns)]
            
            # Estadísticas por subpatrón
            stats_by_subpattern = subpattern_data.groupby('cluster_name')[mobility_cols].describe()
            stats_by_subpattern.to_csv(os.path.join(subpatterns_dir, f"{city_name}_estadisticas_por_subpatron.csv"))
            
            # Visualizaciones para subpatrones
            for mobility_type in mobility_cols:
                plt.figure(figsize=(14, 8))
                sns.boxplot(x='cluster_name', y=mobility_type, data=subpattern_data, order=top_subpatterns)
                plt.title(f'{city_name}: {self.mobility_columns[mobility_type]} por Subpatrón')
                plt.xlabel('Subpatrón')
                plt.ylabel(self.mobility_columns[mobility_type])
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(subpatterns_dir, f"{city_name}_boxplot_subpatron_{mobility_type}.png"))
                plt.close()
            
            # Análisis de correlación entre subpatrones y movilidad
            # Codificar subpatrones
            le_sub = LabelEncoder()
            subpattern_data['subpattern_code'] = le_sub.fit_transform(subpattern_data['cluster_name'])
            
            # Calcular correlación
            corr_vars = ['subpattern_code'] + mobility_cols
            corr_matrix_sub = subpattern_data[corr_vars].corr()
            corr_matrix_sub.to_csv(os.path.join(subpatterns_dir, f"{city_name}_correlacion_subpatrones.csv"))
            
            # Heatmap de correlación
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix_sub, dtype=bool))
            sns.heatmap(corr_matrix_sub, annot=True, cmap='coolwarm', 
                      linewidths=.5, mask=mask, vmin=-1, vmax=1)
            plt.title(f'{city_name}: Correlación entre Subpatrones y Movilidad')
            plt.tight_layout()
            plt.savefig(os.path.join(subpatterns_dir, f"{city_name}_heatmap_subpatrones.png"))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error en análisis de subpatrones para {city_name}: {e}", exc_info=True)
    
    def analyze_global_data(self):
        """
        Realiza análisis global con los datos combinados de todas las ciudades.
        
        Returns:
            True si el análisis fue exitoso, False en caso contrario
        """
        logger.info("\n=== Generando análisis global de todas las ciudades ===")
        
        if not self.all_cities_data:
            logger.error("No hay datos disponibles para análisis global")
            return False
        
        try:
            # Combinar todos los datos
            combined_data = pd.concat(self.all_cities_data, ignore_index=True)
            logger.info(f"Total de registros combinados: {len(combined_data)}")
            
            if len(combined_data) == 0:
                logger.error("DataFrame combinado está vacío")
                return False
            
            # Verificar columnas disponibles
            available_mobility_cols = [col for col in self.mobility_columns.keys() 
                                     if col in combined_data.columns]
            
            # 1. Estadísticas descriptivas globales
            stats_global = combined_data[available_mobility_cols].describe()
            stats_global.to_csv(os.path.join(self.global_dir, "estadisticas_globales.csv"))
            
            # 2. Estadísticas por ciudad
            stats_by_city = combined_data.groupby('city')[available_mobility_cols].describe()
            stats_by_city.to_csv(os.path.join(self.global_dir, "estadisticas_por_ciudad.csv"))
            
            # 3. Estadísticas por patrón
            stats_by_pattern = combined_data.groupby('pattern')[available_mobility_cols].describe()
            stats_by_pattern.to_csv(os.path.join(self.global_dir, "estadisticas_por_patron.csv"))
            
            # 4. Análisis de correlación global
            # Codificar patrones
            le = LabelEncoder()
            combined_data['pattern_code'] = le.fit_transform(combined_data['pattern'])
            
            # Codificar ciudades
            le_city = LabelEncoder()
            combined_data['city_code'] = le_city.fit_transform(combined_data['city'])
            
            # Calcular correlación entre variables
            corr_vars = ['pattern_code', 'city_code'] + available_mobility_cols
            correlation_matrix = combined_data[corr_vars].corr()
            correlation_matrix.to_csv(os.path.join(self.global_dir, "correlacion_global.csv"))
            
            # 5. ANOVA global
            anova_results_global = {}
            for mobility_type in available_mobility_cols:
                try:
                    # ANOVA por patrón
                    model_pattern = ols(f'{mobility_type} ~ C(pattern)', data=combined_data).fit()
                    anova_pattern = sm.stats.anova_lm(model_pattern, typ=2)
                    anova_results_global[f"{mobility_type}_by_pattern"] = anova_pattern
                    
                    # ANOVA por ciudad
                    model_city = ols(f'{mobility_type} ~ C(city)', data=combined_data).fit()
                    anova_city = sm.stats.anova_lm(model_city, typ=2)
                    anova_results_global[f"{mobility_type}_by_city"] = anova_city
                    
                    # ANOVA por patrón y ciudaD
                    # ANOVA por patrón y ciudad
                    model_interaction = ols(f'{mobility_type} ~ C(pattern) + C(city) + C(pattern):C(city)', data=combined_data).fit()
                    anova_interaction = sm.stats.anova_lm(model_interaction, typ=2)
                    anova_results_global[f"{mobility_type}_pattern_city_interaction"] = anova_interaction
                    
                except Exception as e:
                    logger.error(f"Error en ANOVA global para {mobility_type}: {e}")
            
            # Guardar resultados de ANOVA
            with open(os.path.join(self.global_dir, "anova_resultados_globales.txt"), 'w') as f:
                for analysis_name, anova_table in anova_results_global.items():
                    f.write(f"ANOVA para {analysis_name}:\n")
                    f.write(anova_table.to_string())
                    f.write("\n\n")
            
            # 6. Visualizaciones globales
            self._generate_global_visualizations(combined_data, available_mobility_cols)
            
            # 7. Guardar datos combinados
            combined_data.to_excel(os.path.join(self.global_dir, "datos_combinados_todas_ciudades.xlsx"), index=False)
            
            logger.info("Análisis global completado")
            return True
            
        except Exception as e:
            logger.error(f"Error en análisis global: {e}", exc_info=True)
            return False
    
    def _generate_global_visualizations(self, combined_data, mobility_cols):
        """
        Genera visualizaciones para el análisis global.
        
        Args:
            combined_data: DataFrame con datos combinados de todas las ciudades
            mobility_cols: Lista de columnas de movilidad disponibles
        """
        try:
            # 1. Boxplots de movilidad por patrón
            for i, mobility_type in enumerate(mobility_cols):
                plt.figure(figsize=(14, 8))
                sns.boxplot(x='pattern', y=mobility_type, data=combined_data)
                plt.title(f'Global: {self.mobility_columns[mobility_type]} por Patrón')
                plt.xlabel('Patrón de Calle')
                plt.ylabel(self.mobility_columns[mobility_type])
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.global_dir, f"global_boxplot_{mobility_type}_por_patron.png"))
                plt.close()
            
            # 2. Boxplots de movilidad por ciudad
            for i, mobility_type in enumerate(mobility_cols):
                plt.figure(figsize=(16, 8))
                sns.boxplot(x='city', y=mobility_type, data=combined_data)
                plt.title(f'Global: {self.mobility_columns[mobility_type]} por Ciudad')
                plt.xlabel('Ciudad')
                plt.ylabel(self.mobility_columns[mobility_type])
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.global_dir, f"global_boxplot_{mobility_type}_por_ciudad.png"))
                plt.close()
            
            # 3. Heatmap de correlación global
            corr_vars = ['pattern_code', 'city_code'] + mobility_cols
            correlation_matrix = combined_data[corr_vars].corr()
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                      linewidths=.5, mask=mask, vmin=-1, vmax=1)
            plt.title('Global: Correlación entre Patrones, Ciudades y Movilidad')
            plt.tight_layout()
            plt.savefig(os.path.join(self.global_dir, "global_heatmap.png"))
            plt.close()
            
            # 4. Gráfico de interacción entre patrón y ciudad
            for mobility_type in mobility_cols:
                plt.figure(figsize=(16, 12))
                sns.lineplot(x='pattern', y=mobility_type, hue='city', 
                           data=combined_data, markers=True, dashes=False)
                plt.title(f'Global: Interacción entre Patrón y Ciudad para {self.mobility_columns[mobility_type]}')
                plt.xlabel('Patrón de Calle')
                plt.ylabel(self.mobility_columns[mobility_type])
                plt.xticks(rotation=45)
                plt.legend(title='Ciudad', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(self.global_dir, f"global_interaccion_{mobility_type}.png"))
                plt.close()
            
            # 5. PCA global por patrón y ciudad
            if len(mobility_cols) >= 3:
                # Escalar datos para PCA
                scaler = StandardScaler()
                mobility_scaled = scaler.fit_transform(combined_data[mobility_cols])
                
                # Aplicar PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(mobility_scaled)
                
                # Crear DataFrame con resultados
                pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
                pca_df['pattern'] = combined_data['pattern'].values
                pca_df['city'] = combined_data['city'].values
                
                # PCA por patrón
                plt.figure(figsize=(14, 10))
                sns.scatterplot(x='PC1', y='PC2', hue='pattern', data=pca_df, palette='viridis', s=100)
                plt.title('Global: PCA de Movilidad por Patrón')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(title='Patrón', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(self.global_dir, "global_pca_patron.png"))
                plt.close()
                
                # PCA por ciudad
                plt.figure(figsize=(14, 10))
                sns.scatterplot(x='PC1', y='PC2', hue='city', data=pca_df, palette='tab10', s=100)
                plt.title('Global: PCA de Movilidad por Ciudad')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(title='Ciudad', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(self.global_dir, "global_pca_ciudad.png"))
                plt.close()
                
                # Guardar coeficientes de componentes principales
                pca_components = pd.DataFrame(
                    data=pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                    index=mobility_cols
                )
                pca_components.to_csv(os.path.join(self.global_dir, "global_pca_componentes.csv"))
            
        except Exception as e:
            logger.error(f"Error generando visualizaciones globales: {e}", exc_info=True)
    
    def run_analysis(self):
        """
        Ejecuta el flujo completo de análisis.
        
        Returns:
            True si el análisis fue exitoso, False en caso contrario
        """
        logger.info("Iniciando análisis de patrones de calles y movilidad urbana...")
        
        # 1. Procesar todas las ciudades
        if not self.process_cities():
            logger.error("No se pudieron procesar las ciudades")
            return False
        
        # 2. Análisis por ciudad
        for city_name, city_data in self.city_dataframes.items():
            self.analyze_city(city_data, city_name)
        
        # 3. Análisis global
        self.analyze_global_data()
        
        logger.info("Análisis completo")
        return True


if __name__ == "__main__":
    analyzer = StreetPatternMobilityAnalyzer()
    analyzer.run_analysis()