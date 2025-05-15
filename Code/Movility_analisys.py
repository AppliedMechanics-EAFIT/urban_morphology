import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.formula.api import ols
import glob, re, pathlib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class StreetPatternMobilityAnalyzer:
    """Clase para analizar la relación entre patrones de calles y movilidad urbana."""
    def __init__(self, base_path="Polygons_analysis", results_dir="Resultados_Analisis"):
        self.base_path = base_path
        self.results_dir = results_dir
        self.cities_results_dir = os.path.join(results_dir, "Ciudades")
        self.global_dir = os.path.join(results_dir, "Analisis_Global")
        
        # Crear directorios para resultados
        for dir_path in [self.results_dir, self.cities_results_dir, self.global_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Almacenamiento para datos procesados
        self.city_dataframes = {}
        self.all_cities_data = []
        
        # Definir las columnas de movilidad y sus nombres descriptivos
        self.mobility_columns = {
            'a': 'Movilidad Activa', 'b': 'Movilidad Pública', 'c': 'Movilidad Privada',
            'car_mode': 'Uso de Automóvil', 'transit_mode': 'Uso de Transporte Público',
            'bicycle_mode': 'Uso de Bicicleta', 'walked_mode': 'Desplazamiento a Pie',
            'car_share': 'Porcentaje Automóvil', 'transit_share': 'Porcentaje Transporte Público',
            'bicycle_share': 'Porcentaje Bicicleta', 'walked_share': 'Porcentaje a Pie'
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
        
        # Configuración global para visualizaciones
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def extract_id_number(self, poly_id):
        """Extrae el número identificador del polígono."""
        if isinstance(poly_id, str):
            match = re.search(r'_(\d+)$', poly_id)
            if match:
                return int(match.group(1)) - 1
        return poly_id
    
    def load_mobility_data(self, city):
        """Carga los datos de movilidad para una ciudad específica."""
        mobility_path = os.path.join(self.base_path, city, "Mobility_Data")
        
        if not os.path.exists(mobility_path):
            print(f"La carpeta {mobility_path} no existe")
            return None
            
        mobility_files = glob.glob(os.path.join(mobility_path, "*.xlsx"))
        
        if not mobility_files:
            print(f"No se encontraron archivos Excel de movilidad en {mobility_path}")
            return None
        
        print(f"Archivo de movilidad encontrado: {mobility_files[0]}")
        try:
            mobility_df = pd.read_excel(mobility_files[0])
            print(f"Columnas en datos de movilidad: {mobility_df.columns.tolist()}")
            print(f"Número de registros: {len(mobility_df)}")
            
            # Verificar columnas necesarias
            required_cols = ['poly_id', 'total_pop']
            missing_cols = [col for col in required_cols if col not in mobility_df.columns]
            
            if missing_cols:
                print(f"Faltan columnas necesarias en el archivo de movilidad: {missing_cols}")
                return None
            
            # Añadir identificador numérico para mapeo
            mobility_df['numeric_id'] = mobility_df['poly_id'].apply(self.extract_id_number)
            
            # Normalizar columnas por población si es necesario
            for col in self.normalize_by_population:
                if col in mobility_df.columns:
                    if 'total_pop' in mobility_df.columns and (mobility_df['total_pop'] > 0).all():
                        mobility_df[f'{col}_norm'] = mobility_df[col] / mobility_df['total_pop'] * 100
                        print(f"Columna {col} normalizada por población")
            
            return mobility_df
            
        except Exception as e:
            print(f"Error al leer el archivo de movilidad: {e}")
            return None
    
    def load_patterns_data(self, city):
        """Carga los datos de patrones urbanos para una ciudad específica."""
        pattern_path = os.path.join(self.base_path, city, "clustering_analysis")
        pattern_file = os.path.join(pattern_path, "urban_pattern_analysis.xlsx")
        
        if not os.path.exists(pattern_file):
            print(f"No se encontró el archivo de patrones urbanos en {pattern_path}")
            return None, None
        
        try:
            # Cargar datos de patrones principales (hoja 5, índice 4)
            patterns_main_df = pd.read_excel(pattern_file, sheet_name=4)
            
            # Cargar datos de subpatrones (hoja 6, índice 5)
            patterns_sub_df = pd.read_excel(pattern_file, sheet_name=5)
            
            print(f"Datos de patrones cargados: {len(patterns_main_df)} patrones principales, {len(patterns_sub_df)} subpatrones")
            
            # Asegurar que los IDs son del tipo correcto
            if 'poly_id' in patterns_main_df.columns:
                patterns_main_df['poly_id'] = patterns_main_df['poly_id'].astype(int)
            if 'poly_id' in patterns_sub_df.columns:
                patterns_sub_df['poly_id'] = patterns_sub_df['poly_id'].astype(int)
                
            return patterns_main_df, patterns_sub_df
            
        except Exception as e:
            print(f"Error al cargar datos de patrones: {e}")
            return None, None
    
    def merge_city_data(self, mobility_df, patterns_main_df, patterns_sub_df, city):
        """Une los datos de movilidad y patrones para una ciudad."""
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
            print(f"Coincidencias encontradas después del merge: {matches}")
            
            if matches == 0:
                print("No se encontraron coincidencias. Intentando un enfoque alternativo...")
                
                # Alternativa: alinear por índice si los tamaños son similares
                if abs(len(mobility_df) - len(patterns_main_df)) <= 5:
                    print("Los tamaños de los dataframes son similares. Alineando por índice...")
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
                    
                    print(f"Alineado por índice completado con {len(city_data)} registros")
                else:
                    print("Los tamaños de los dataframes difieren significativamente. No se puede alinear.")
                    return None
            else:
                # Renombrar 'original_pattern' a 'pattern' para consistencia
                if 'original_pattern' in city_data.columns and 'pattern' not in city_data.columns:
                    city_data = city_data.rename(columns={'original_pattern': 'pattern'})
            
            # Añadir columna de ciudad
            city_data['city'] = city
            
            return city_data
            
        except Exception as e:
            print(f"Error durante el merge de datos: {e}")
            return None
    
    def process_cities(self):
        """Procesa todas las ciudades disponibles en la ruta base."""
        # Obtener lista de carpetas de ciudades
        city_folders = [f for f in os.listdir(self.base_path) 
                      if os.path.isdir(os.path.join(self.base_path, f))]
        
        print(f"Carpetas de ciudades encontradas: {city_folders}")
        
        if not city_folders:
            print("No se encontraron carpetas de ciudades")
            return False
        
        # Procesar cada ciudad
        success = False
        for city in city_folders:
            print(f"\n--- Procesando ciudad: {city} ---")
            
            # Cargar datos
            mobility_df = self.load_mobility_data(city)
            patterns_main_df, patterns_sub_df = self.load_patterns_data(city)
            
            if mobility_df is None or patterns_main_df is None:
                print(f"No se pudieron cargar datos para {city}")
                continue
                
            # Unir datos
            city_data = self.merge_city_data(mobility_df, patterns_main_df, patterns_sub_df, city)
            
            if city_data is not None and len(city_data) > 0:
                # Guardar dataframe para análisis individual
                self.city_dataframes[city] = city_data
                
                # Añadir a la lista de todas las ciudades
                self.all_cities_data.append(city_data)
                print(f"Ciudad {city} procesada correctamente con {len(city_data)} registros")
                success = True
            else:
                print(f"No se pudieron combinar datos para {city}")
        
        return success
    
    def analyze_city(self, city_data, city_name):
        """Realiza análisis estadístico y visualizaciones para una ciudad."""
        print(f"\n=== Generando análisis para {city_name} ===")
        
        # Crear carpeta específica para esta ciudad
        city_dir = os.path.join(self.cities_results_dir, city_name)
        os.makedirs(city_dir, exist_ok=True)
        
        # Verificar columnas disponibles de movilidad
        available_mobility_cols = [col for col in self.mobility_columns.keys() 
                                 if col in city_data.columns]
        
        if not available_mobility_cols:
            print(f"No hay columnas de movilidad disponibles para {city_name}")
            return False
            
        if 'pattern' not in city_data.columns:
            print(f"No hay columna de patrón disponible para {city_name}")
            return False
        
        try:
            # Codificar patrones
            le = LabelEncoder()
            city_data['pattern_code'] = le.fit_transform(city_data['pattern'])
            
            # ANOVA: diferencias significativas entre patrones
            anova_results = {}
            for mobility_type in available_mobility_cols:
                try:
                    model = ols(f'{mobility_type} ~ C(pattern)', data=city_data).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    anova_results[mobility_type] = anova_table
                    print(f"ANOVA para {mobility_type} en {city_name}")
                except Exception as e:
                    print(f"Error en ANOVA para {mobility_type}: {e}")
            
            # Visualizaciones
            self._generate_city_visualizations(city_data, city_name, city_dir, available_mobility_cols)
            
            # Análisis de subpatrones si están disponibles
            if 'cluster_name' in city_data.columns:
                self._analyze_subpatterns(city_data, city_name, city_dir, available_mobility_cols)
            
            # Guardar datos procesados
            city_data.to_excel(os.path.join(city_dir, f"{city_name}_datos_procesados.xlsx"), index=False)
            
            print(f"Análisis completado para {city_name}")
            return True
            
        except Exception as e:
            print(f"Error en análisis para {city_name}: {e}")
            return False
    
    def _generate_city_visualizations(self, city_data, city_name, city_dir, mobility_cols):
        """Genera visualizaciones para el análisis de una ciudad."""
        try:
            # 1. Boxplots de movilidad por patrón
            for mobility_type in mobility_cols:
                plt.figure()
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
            
            # 3. Análisis de componentes principales (PCA) para movilidad
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
            print(f"Error generando visualizaciones para {city_name}: {e}")
    
    def _analyze_subpatterns(self, city_data, city_name, city_dir, mobility_cols):
        """Realiza análisis específico de subpatrones para una ciudad."""
        try:
            # Crear subcarpeta para análisis de subpatrones
            subpatterns_dir = os.path.join(city_dir, "Subpatrones")
            os.makedirs(subpatterns_dir, exist_ok=True)
            
            # Contar frecuencia de subpatrones
            subpattern_counts = city_data['cluster_name'].value_counts()
            
            # Seleccionar los 10 subpatrones más frecuentes
            top_subpatterns = subpattern_counts.head(10).index.tolist()
            subpattern_data = city_data[city_data['cluster_name'].isin(top_subpatterns)]
            
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
            print(f"Error en análisis de subpatrones para {city_name}: {e}")
    
    def analyze_global_data(self):
        """Realiza análisis global con los datos combinados de todas las ciudades."""
        print("\n=== Generando análisis global de todas las ciudades ===")
        
        if not self.all_cities_data:
            print("No hay datos disponibles para análisis global")
            return False
        
        try:
            # Combinar todos los datos
            combined_data = pd.concat(self.all_cities_data, ignore_index=True)
            print(f"Total de registros combinados: {len(combined_data)}")
            
            if len(combined_data) == 0:
                print("DataFrame combinado está vacío")
                return False
            
            # Verificar columnas disponibles
            available_mobility_cols = [col for col in self.mobility_columns.keys() 
                                    if col in combined_data.columns]
            
            # Análisis diagnóstico
            pattern_city_counts = pd.crosstab(combined_data['pattern'], combined_data['city'])
            total_combinations = pattern_city_counts.shape[0] * pattern_city_counts.shape[1]
            filled_combinations = (pattern_city_counts > 0).sum().sum()
            
            print(f"Combinaciones patrón-ciudad: {filled_combinations} presentes de {total_combinations} posibles")
            
            # Identificar patrones y ciudades con pocos datos
            pattern_counts = combined_data['pattern'].value_counts()
            city_counts = combined_data['city'].value_counts()
            
            rare_patterns = pattern_counts[pattern_counts < 30].index.tolist()
            if rare_patterns:
                print(f"Patrones con menos de 30 observaciones: {rare_patterns}")
                
            rare_cities = city_counts[city_counts < 50].index.tolist()
            if rare_cities:
                print(f"Ciudades con menos de 50 observaciones: {rare_cities}")
            
            # Guardar información diagnóstica
            anova_results_global = {
                'data_diagnosis': {
                    'total_records': len(combined_data),
                    'total_patterns': len(pattern_counts),
                    'total_cities': len(city_counts),
                    'filled_combinations': filled_combinations,
                    'total_combinations': total_combinations,
                    'rare_patterns': rare_patterns,
                    'rare_cities': rare_cities,
                    'pattern_counts': pattern_counts.to_dict(),
                    'city_counts': city_counts.to_dict()
                }
            }
            
            # Determinar umbral para agrupar categorías raras si es necesario
            if len(rare_patterns) > 0:
                print("Considerando agrupamiento de patrones raros para análisis más robusto")
                # Crear una copia con patrones agrupados para análisis alternativo
                combined_data_grouped = combined_data.copy()
                combined_data_grouped['pattern_grouped'] = combined_data['pattern'].apply(
                    lambda x: x if x not in rare_patterns else 'otros_patrones')
            
            # Crear un DataFrame para almacenar los resultados de manera más clara
            anova_summary = []
            
            # Crear un diccionario para almacenar los patrones y ciudades con mayor efecto
            pattern_effects_dict = {}
            city_effects_dict = {}
            
            for mobility_type in available_mobility_cols:
                try:
                    print(f"\nAnalizando variable: {mobility_type}")
                    
                    # PASO 1: ANOVA simple de efectos principales
                    model_basic = ols(f'{mobility_type} ~ C(pattern) + C(city)', data=combined_data).fit()
                    anova_basic = sm.stats.anova_lm(model_basic, typ=2)
                    anova_results_global[f"{mobility_type}_main_effects"] = anova_basic
                    
                    # Guardar coeficientes para interpretar el efecto de cada patrón y ciudad
                    coefficients = pd.DataFrame({
                        'variable': model_basic.params.index,
                        'coefficient': model_basic.params.values,
                        'p_value': model_basic.pvalues.values,
                        'significant': model_basic.pvalues.values < 0.05
                    })
                    anova_results_global[f"{mobility_type}_coefficients"] = coefficients
                    
                    # Preparar datos para el resumen
                    pattern_p_value = anova_basic.loc['C(pattern)', 'PR(>F)']
                    city_p_value = anova_basic.loc['C(city)', 'PR(>F)']
                    pattern_significant = "Sí" if pattern_p_value < 0.05 else "No"  # Cambiado de booleano a texto
                    city_significant = "Sí" if city_p_value < 0.05 else "No"        # Cambiado de booleano a texto
                    
                    # Inicializar con valores por defecto
                    pattern_name = "No disponible"
                    city_name = "No disponible"
                    most_significant_pattern_coefficient = None
                    most_significant_city_coefficient = None
                    
                    # Extracción de patrones específicos y sus efectos
                    # Corrigiendo la expresión regular para evitar warning
                    pattern_coefficients = coefficients[coefficients['variable'].str.contains(r'C\(pattern\)', regex=True)].copy()
                    
                    if not pattern_coefficients.empty:
                        # Extraer el nombre del patrón de la forma "C(pattern)[T.nombre_patron]"
                        pattern_coefficients['pattern_name'] = 'base'  # Valor predeterminado
                        
                        for i, row in pattern_coefficients.iterrows():
                            var_name = row['variable']
                            if '[T.' in var_name:
                                extracted_pattern = var_name.split('[T.')[1].split(']')[0]
                                pattern_coefficients.loc[i, 'pattern_name'] = extracted_pattern
                        
                        # Encontrar el patrón con el mayor efecto significativo
                        sig_patterns = pattern_coefficients[pattern_coefficients['significant']]
                        if not sig_patterns.empty:
                            # Ordenar por valor absoluto del coeficiente (mayor a menor)
                            sig_patterns = sig_patterns.sort_values(by='coefficient', key=abs, ascending=False)
                            most_sig_pattern = sig_patterns.iloc[0]
                            pattern_name = most_sig_pattern['pattern_name']
                            most_significant_pattern_coefficient = float(most_sig_pattern['coefficient'])
                            
                            # Guardar en el diccionario para referencia
                            pattern_effects_dict[mobility_type] = {
                                'pattern': pattern_name,
                                'coefficient': float(most_sig_pattern['coefficient']),
                                'p_value': float(most_sig_pattern['p_value'])
                            }
                            
                            print(f"  Patrón más significativo: {pattern_name} (coef={most_sig_pattern['coefficient']:.4f}, p={most_sig_pattern['p_value']:.6f})")
                        else:
                            pattern_name = "Ninguno significativo"
                            print("  No se encontraron patrones significativos")
                    
                    # Extracción de ciudades específicas y sus efectos
                    # Corrigiendo la expresión regular para evitar warning
                    city_coefficients = coefficients[coefficients['variable'].str.contains(r'C\(city\)', regex=True)].copy()
                    
                    if not city_coefficients.empty:
                        # Extraer el nombre de la ciudad
                        city_coefficients['city_name'] = 'base'  # Valor predeterminado
                        
                        for i, row in city_coefficients.iterrows():
                            var_name = row['variable']
                            if '[T.' in var_name:
                                extracted_city = var_name.split('[T.')[1].split(']')[0]
                                city_coefficients.loc[i, 'city_name'] = extracted_city
                        
                        # Encontrar la ciudad con el mayor efecto significativo
                        sig_cities = city_coefficients[city_coefficients['significant']]
                        if not sig_cities.empty:
                            # Ordenar por valor absoluto del coeficiente (mayor a menor)
                            sig_cities = sig_cities.sort_values(by='coefficient', key=abs, ascending=False)
                            most_sig_city = sig_cities.iloc[0]
                            city_name = most_sig_city['city_name']
                            most_significant_city_coefficient = float(most_sig_city['coefficient'])
                            
                            # Guardar en el diccionario para referencia
                            city_effects_dict[mobility_type] = {
                                'city': city_name,
                                'coefficient': float(most_sig_city['coefficient']),
                                'p_value': float(most_sig_city['p_value'])
                            }
                            
                            print(f"  Ciudad más significativa: {city_name} (coef={most_sig_city['coefficient']:.4f}, p={most_sig_city['p_value']:.6f})")
                        else:
                            city_name = "Ninguna significativa"
                            print("  No se encontraron ciudades significativas")
                    
                    # Guardar fila de resumen con información más detallada
                    anova_summary.append({
                        'Variable': mobility_type,
                        'Patrón': pattern_name,
                        'Coef. Patrón': most_significant_pattern_coefficient,
                        'p-value Patrón': pattern_p_value,
                        'Patrón significativo': pattern_significant,  # Ahora es "Sí" o "No"
                        'Ciudad': city_name,
                        'Coef. Ciudad': most_significant_city_coefficient,
                        'p-value Ciudad': city_p_value,
                        'Ciudad significativa': city_significant,  # Ahora es "Sí" o "No"
                        'R² ajustado': model_basic.rsquared_adj
                    })
                    
                    # Guardar estadísticas de ajuste
                    anova_results_global[f"{mobility_type}_model_fit"] = {
                        'r_squared': model_basic.rsquared,
                        'adj_r_squared': model_basic.rsquared_adj,
                        'aic': model_basic.aic,
                        'bic': model_basic.bic,
                        'f_value': model_basic.fvalue,
                        'f_pvalue': model_basic.f_pvalue
                    }
                    
                    # PASO 2: Análisis con datos agrupados si es necesario
                    if len(rare_patterns) > 0:
                        try:
                            model_grouped = ols(f'{mobility_type} ~ C(pattern_grouped) + C(city)', 
                                            data=combined_data_grouped).fit()
                            anova_grouped = sm.stats.anova_lm(model_grouped, typ=2)
                            anova_results_global[f"{mobility_type}_grouped_patterns"] = anova_grouped
                            
                            # Extraer información detallada del modelo agrupado
                            grouped_r_squared = model_grouped.rsquared_adj
                            grouped_pattern_p = anova_grouped.loc['C(pattern_grouped)', 'PR(>F)']
                            
                            print(f"  Análisis con patrones agrupados: R² = {grouped_r_squared:.4f}, p-valor patrones = {grouped_pattern_p:.6f}")
                            
                        except Exception as e:
                            print(f"Error en análisis con patrones agrupados para {mobility_type}: {e}")
                    
                    # PASO 3: Modelo jerárquico con enfoque robusto 
                    try:
                        # Análisis por ciudad
                        city_effects = {}
                        for city in combined_data['city'].unique():
                            city_data = combined_data[combined_data['city'] == city]
                            if len(city_data) > 30 and len(city_data['pattern'].unique()) > 1:
                                try:
                                    city_model = ols(f'{mobility_type} ~ C(pattern)', data=city_data).fit()
                                    city_effects[city] = {
                                        'f_value': city_model.fvalue,
                                        'p_value': city_model.f_pvalue,
                                        'r_squared': city_model.rsquared,
                                        'significant': city_model.f_pvalue < 0.05,
                                        'n_observations': len(city_data)
                                    }
                                except Exception as city_err:
                                    print(f"Error en análisis para ciudad {city}: {city_err}")
                        
                        anova_results_global[f"{mobility_type}_by_city"] = city_effects
                        
                        # Análisis por patrón
                        pattern_effects = {}
                        for pattern in combined_data['pattern'].unique():
                            pattern_data = combined_data[combined_data['pattern'] == pattern]
                            if len(pattern_data) > 30 and len(pattern_data['city'].unique()) > 1:
                                try:
                                    pattern_model = ols(f'{mobility_type} ~ C(city)', data=pattern_data).fit()
                                    pattern_effects[pattern] = {
                                        'f_value': pattern_model.fvalue,
                                        'p_value': pattern_model.f_pvalue,
                                        'r_squared': pattern_model.rsquared,
                                        'significant': pattern_model.f_pvalue < 0.05,
                                        'n_observations': len(pattern_data)
                                    }
                                except Exception as pattern_err:
                                    print(f"Error en análisis para patrón {pattern}: {pattern_err}")
                        
                        anova_results_global[f"{mobility_type}_by_pattern"] = pattern_effects
                        
                        # Conteo de efectos significativos
                        significant_cities = sum(1 for v in city_effects.values() if v.get('significant', False))
                        significant_patterns = sum(1 for v in pattern_effects.values() if v.get('significant', False))
                        
                        # Resumen mejorado con porcentajes
                        if city_effects:
                            pct_cities = (significant_cities / len(city_effects)) * 100
                            print(f"  {significant_cities} de {len(city_effects)} ciudades ({pct_cities:.1f}%) muestran efectos significativos de patrón")
                        
                        if pattern_effects:
                            pct_patterns = (significant_patterns / len(pattern_effects)) * 100
                            print(f"  {significant_patterns} de {len(pattern_effects)} patrones ({pct_patterns:.1f}%) muestran efectos significativos de ciudad")
                        
                        anova_results_global[f"{mobility_type}_effects_summary"] = {
                            'cities_with_significant_patterns': significant_cities,
                            'total_cities_analyzed': len(city_effects),
                            'patterns_with_significant_cities': significant_patterns,
                            'total_patterns_analyzed': len(pattern_effects),
                            'percent_cities_significant': significant_cities / len(city_effects) if len(city_effects) > 0 else 0,
                            'percent_patterns_significant': significant_patterns / len(pattern_effects) if len(pattern_effects) > 0 else 0
                        }
                        
                    except Exception as e:
                        print(f"Error en análisis jerárquico para {mobility_type}: {e}")
                    
                    # PASO 4: Análisis de medias por categoría para interpretación más sencilla
                    try:
                        # Calcular medias por patrón
                        pattern_means = combined_data.groupby('pattern')[mobility_type].agg(['mean', 'std', 'count']).reset_index()
                        pattern_means = pattern_means.sort_values(by='mean', ascending=False)
                        
                        # Calcular medias por ciudad
                        city_means = combined_data.groupby('city')[mobility_type].agg(['mean', 'std', 'count']).reset_index()
                        city_means = city_means.sort_values(by='mean', ascending=False)
                        
                        # Mostrar los 3 primeros de cada categoría
                        print("  Patrones con valores más altos (Top 3):")
                        for _, row in pattern_means.head(3).iterrows():
                            print(f"    {row['pattern']}: {row['mean']:.4f} ± {row['std']:.4f} (n={row['count']})")
                        
                        print("  Ciudades con valores más altos (Top 3):")
                        for _, row in city_means.head(3).iterrows():
                            print(f"    {row['city']}: {row['mean']:.4f} ± {row['std']:.4f} (n={row['count']})")
                        
                        # Guardar estas estadísticas descriptivas
                        anova_results_global[f"{mobility_type}_pattern_means"] = pattern_means.to_dict('records')
                        anova_results_global[f"{mobility_type}_city_means"] = city_means.to_dict('records')
                        
                    except Exception as e:
                        print(f"Error al calcular estadísticas descriptivas para {mobility_type}: {e}")
                    
                    # Intentar modelo mixto solo para variables sin problemas conocidos
                    if mobility_type not in ['a', 'b', 'bicycle_share', 'transit_share', 'walked_share']:
                        try:
                            import statsmodels.formula.api as smf
                            # Usar opciones más conservadoras para mayor estabilidad
                            mixed_model = smf.mixedlm(
                                f"{mobility_type} ~ pattern", 
                                data=combined_data,
                                groups=combined_data["city"],
                                re_formula="~1"  # Solo intercepto aleatorio
                            ).fit(
                                method="powell",  # Método alternativo de optimización
                                maxiter=1000,
                                ftol=1e-4
                            )
                            
                            # Extraer solo la información esencial para evitar problemas de serialización
                            anova_results_global[f"{mobility_type}_mixed_model"] = {
                                'AIC': mixed_model.aic,
                                'BIC': mixed_model.bic,
                                'Log-Likelihood': mixed_model.llf,
                                'Parameters': {k: v for k, v in mixed_model.params.items()},
                                'P-Values': {k: v for k, v in mixed_model.pvalues.items() if v < 0.1}  # Solo valores p significativos o cercanos
                            }
                            
                            # Mostrar estadísticas del modelo mixto
                            has_sig_pattern = any(v < 0.05 for k, v in mixed_model.pvalues.items() if k.startswith('pattern'))
                            print(f"  Modelo mixto: AIC={mixed_model.aic:.2f}, BIC={mixed_model.bic:.2f}, patrones significativos: {'Sí' if has_sig_pattern else 'No'}")
                            
                        except Exception as e:
                            print(f"No se pudo realizar modelo mixto para {mobility_type}: {e}")
                            
                except Exception as e:
                    print(f"Error en análisis para {mobility_type}: {e}")
            
            # Convertir la lista de diccionarios a DataFrame
            anova_summary_df = pd.DataFrame(anova_summary)
            
            # Guardar la tabla de resumen ANOVA como Excel
            resumen_file = os.path.join(self.global_dir, "resumen_anova_patrones_ciudades.xlsx")
            anova_summary_df.to_excel(resumen_file, index=False)
            
            # Crear un archivo de interpretación
            interpretation_file = os.path.join(self.global_dir, "interpretacion_resultados.txt")
            with open(interpretation_file, 'w', encoding='utf-8') as f:
                f.write("INTERPRETACIÓN DE RESULTADOS DEL ANÁLISIS GLOBAL\n")
                f.write("=============================================\n\n")
                
                # Resumen general
                f.write(f"Total de registros analizados: {len(combined_data)}\n")
                f.write(f"Número de patrones distintos: {len(pattern_counts)}\n")
                f.write(f"Número de ciudades distintas: {len(city_counts)}\n\n")
                
                # Análisis por variable de movilidad
                f.write("ANÁLISIS POR VARIABLE DE MOVILIDAD\n")
                f.write("--------------------------------\n\n")
                
                for _, row in anova_summary_df.iterrows():
                    f.write(f"Variable: {row['Variable']}\n")
                    f.write(f"  R² ajustado: {row['R² ajustado']:.4f}\n")
                    
                    # Información sobre patrón
                    f.write(f"  Efecto de patrón: {row['Patrón significativo']} (p = {row['p-value Patrón']:.6f})\n")
                    if row['Coef. Patrón'] is not None and row['Patrón'] != "Ninguno significativo":
                        f.write(f"  Patrón más influyente: {row['Patrón']} (coef = {row['Coef. Patrón']:.4f})\n")
                    
                    # Información sobre ciudad
                    f.write(f"  Efecto de ciudad: {row['Ciudad significativa']} (p = {row['p-value Ciudad']:.6f})\n")
                    if row['Coef. Ciudad'] is not None and row['Ciudad'] != "Ninguna significativa":
                        f.write(f"  Ciudad más influyente: {row['Ciudad']} (coef = {row['Coef. Ciudad']:.4f})\n")
                    
                    # Interpretación
                    f.write("  Interpretación: ")
                    
                    if row['Patrón significativo'] == "Sí" and row['Ciudad significativa'] == "Sí":
                        f.write("Tanto el patrón de calle como la ciudad tienen efectos significativos sobre esta variable de movilidad.\n")
                    elif row['Patrón significativo'] == "Sí":
                        f.write("El patrón de calle tiene un efecto significativo sobre esta variable de movilidad, mientras que la ciudad no muestra efecto significativo.\n")
                    elif row['Ciudad significativa'] == "Sí":
                        f.write("La ciudad tiene un efecto significativo sobre esta variable de movilidad, mientras que el patrón de calle no muestra efecto significativo.\n")
                    else:
                        f.write("Ni el patrón de calle ni la ciudad muestran efectos significativos sobre esta variable de movilidad.\n")
                    
                    f.write("\n")
                
                # Conclusiones generales
                f.write("CONCLUSIONES GENERALES\n")
                f.write("--------------------\n\n")
                
                # Calcular en cuántas variables hay efectos significativos
                pattern_sig_count = (anova_summary_df['Patrón significativo'] == "Sí").sum()
                city_sig_count = (anova_summary_df['Ciudad significativa'] == "Sí").sum()
                total_vars = len(anova_summary_df)
                
                f.write(f"El patrón de calle mostró efecto significativo en {pattern_sig_count} de {total_vars} variables de movilidad ({pattern_sig_count/total_vars*100:.1f}%).\n")
                f.write(f"La ciudad mostró efecto significativo en {city_sig_count} de {total_vars} variables de movilidad ({city_sig_count/total_vars*100:.1f}%).\n\n")
                
                # Identificar patrones que aparecen repetidamente como significativos
                if pattern_effects_dict:
                    frequent_patterns = {}
                    for var, data in pattern_effects_dict.items():
                        pattern = data['pattern']
                        frequent_patterns[pattern] = frequent_patterns.get(pattern, 0) + 1
                    
                    if frequent_patterns:
                        most_frequent = max(frequent_patterns.items(), key=lambda x: x[1])
                        f.write(f"El patrón '{most_frequent[0]}' aparece como el más influyente en {most_frequent[1]} variables de movilidad.\n")
                
                # Identificar ciudades que aparecen repetidamente como significativas
                if city_effects_dict:
                    frequent_cities = {}
                    for var, data in city_effects_dict.items():
                        city = data['city']
                        frequent_cities[city] = frequent_cities.get(city, 0) + 1
                    
                    if frequent_cities:
                        most_frequent = max(frequent_cities.items(), key=lambda x: x[1])
                        f.write(f"La ciudad '{most_frequent[0]}' aparece como la más influyente en {most_frequent[1]} variables de movilidad.\n")
            
            # También guardar los diccionarios de efectos como archivos separados para referencia
            if pattern_effects_dict:
                pattern_effects_df = pd.DataFrame.from_dict(pattern_effects_dict, orient='index')
                pattern_effects_df.to_excel(os.path.join(self.global_dir, "efectos_patrones_detallados.xlsx"))
            
            if city_effects_dict:
                city_effects_df = pd.DataFrame.from_dict(city_effects_dict, orient='index')
                city_effects_df.to_excel(os.path.join(self.global_dir, "efectos_ciudades_detallados.xlsx"))
            
            # Visualizaciones mejoradas
            self._generate_enhanced_visualizations(combined_data, available_mobility_cols)
            
            # Guardar datos combinados
            combined_data.to_excel(os.path.join(self.global_dir, "datos_combinados_todas_ciudades.xlsx"), index=False)
            
            # Guardar también resultados ANOVA
            if anova_results_global:
                # Resultados en formato más detallado
                self._save_analysis_results(anova_results_global, self.global_dir)
            
            print("\nAnálisis global completado")
            print(f"Tabla de resumen ANOVA guardada en: {resumen_file}")
            print(f"Interpretación de resultados guardada en: {interpretation_file}")
            return True
            
        except Exception as e:
            print(f"Error en análisis global: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_enhanced_visualizations(self, combined_data, mobility_cols):
        """Genera visualizaciones mejoradas para el análisis global, con énfasis en claridad y legibilidad."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.gridspec as gridspec
        from scipy import stats
        import os
        import traceback
        import pandas as pd
        
        plt.ioff()  # Desactivar modo interactivo
        
        try:
            # Crear directorio para visualizaciones si no existe
            viz_dir = os.path.join(self.global_dir, "visualizaciones")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Definir una paleta de colores clara y distinguible para los patrones
            pattern_palette = sns.color_palette("tab10", len(combined_data['pattern'].unique()))
            pattern_color_dict = dict(zip(sorted(combined_data['pattern'].unique()), pattern_palette))
            
            # 1. Distribución de observaciones por ciudad y patrón
            plt.figure(figsize=(12, 8))
            pattern_city_counts = pd.crosstab(combined_data['city'], combined_data['pattern'])
            
            # Crear paleta de colores personalizada
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f5f5f5', '#2171b5'])
            
            # Heatmap de conteos
            ax = sns.heatmap(pattern_city_counts, annot=True, fmt='d', cmap=cmap)
            plt.title('Número de Observaciones por Ciudad y Patrón', fontsize=14)
            plt.ylabel('Ciudad', fontsize=12)
            plt.xlabel('Patrón', fontsize=12)
            
            # Ajustar el layout sin usar tight_layout
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
            plt.savefig(os.path.join(viz_dir, "distribucion_observaciones.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Visualizaciones para cada variable de movilidad
            for col in mobility_cols:
                # Verificar si la variable tiene valores válidos para graficar
                if combined_data[col].isna().all() or (combined_data[col] == 0).all():
                    print(f"Omitiendo visualización para {col} - Sin datos válidos")
                    continue
                    
                # A. RIDGE PLOTS MEJORADOS con etiquetas claras y mejor estética (colores fijos y sin media)
                # Definir orden fijo de patrones
                fixed_pattern_order = ['cul_de_sac', 'gridiron', 'organico', 'hibrido']
                # Colores fijos para cada patrón
                pattern_color_dict = {
                    'cul_de_sac': '#FF6B6B',  # Rojo para callejones sin salida
                    'gridiron': '#006400',    # Verde oscuro para grid
                    'organico': '#45B7D1',    # Azul para orgánico
                    'hibrido': '#FDCB6E',     # Amarillo para híbrido
                }
                # Filtrar solo los patrones disponibles en los datos manteniendo el orden fijo
                available_patterns = [p for p in fixed_pattern_order if p in combined_data['pattern'].unique()]
                if len(available_patterns) <= 1:
                    print(f"Omitiendo ridge plot para {col} - Se necesitan múltiples patrones")
                else:
                    # Crear figura con espacio adecuado y mejor estética
                    fig = plt.figure(figsize=(14, max(8, len(available_patterns) * 1.2)))
                    # Crear grid para ridge plot con espacio adecuado
                    gs = gridspec.GridSpec(len(available_patterns), 1, hspace=0.4)
                    # Obtener valores extremos para establecer límites consistentes en todos los subplots
                    x_min = combined_data[col].min()
                    x_max = combined_data[col].max()
                    x_range = x_max - x_min
                    x_padding = x_range  * 0.1 # 10% padding
                    # Crear los ridge plots manualmente para más control
                    for i, pattern in enumerate(available_patterns):
                        ax = plt.subplot(gs[i])
                        
                        # Obtener datos del patrón actual
                        pattern_data = combined_data[combined_data['pattern'] == pattern][col].dropna()
                        
                        if len(pattern_data) <= 1:
                            continue
                        
                        # Crear KDE plot con mejor estética y color fijo
                        sns.kdeplot(pattern_data, fill=True, color=pattern_color_dict[pattern],
                                    alpha=0.7, linewidth=1.5, ax=ax, bw_adjust=0.8)
                        
                        # Añadir la mediana
                        median_val = pattern_data.median()
                        ax.axvline(x=median_val, color="red", linestyle="--", alpha=0.8, linewidth=1.5)
                        
                        # Ya no añadimos la media
                        
                        # Etiquetas claras y mejor posicionadas
                        ax.text(x_min + x_padding,
                                ax.get_ylim()[1] * 0.7,
                                f'{pattern}',
                                fontsize=12, fontweight='bold',
                                color=pattern_color_dict[pattern])
                        
                        ax.text(median_val + x_padding*0.5,
                                ax.get_ylim()[1] * 0.5,
                                f'Mediana: {median_val:.2f}',
                                fontsize=10, color='darkred')
                        
                        # Configuración del eje Y
                        ax.set_yticks([])
                        ax.set_ylabel('')
                        
                        # Configuración del eje X para todos excepto el último
                        if i < len(available_patterns) - 1:
                            ax.set_xticks([])
                            ax.set_xlabel('')
                        else:
                            ax.set_xlabel(col, fontsize=12)
                        
                        # Límites consistentes
                        ax.set_xlim(x_min - x_padding, x_max + x_padding)
                        
                        # Eliminar bordes innecesarios
                        sns.despine(bottom=True, left=True, ax=ax)
                    # Título general más elegante
                    plt.suptitle(f'Distribución de {col} por Patrón', fontsize=16, y=0.98)
                    # Actualizar la leyenda para mostrar solo la mediana
                    legend_elements = [
                        plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='Mediana')
                    ]
                    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.98))
                    # Guardar con bbox_inches='tight' para evitar problemas de márgenes
                    plt.savefig(os.path.join(viz_dir, f"ridge_plot_{col}.png"), dpi=300, bbox_inches='tight')
                    plt.close()

                # GRÁFICO ADICIONAL: Densidad de A share (Activa), B share (Pública) y C share (Privada) 
                # para cada patrón de calle por separado
                mobility_cols = ['a', 'b', 'car_share']  # Columnas que representan movilidad activa, pública y privada
                mobility_labels = ['MOVILIDAD ACTIVA', 'MOVILIDAD PÚBLICA', 'MOVILIDAD PRIVADA']
                mobility_colors = ['#FF6B6B', '#4ECDC4', '#6A67CE']  # Colores distintivos para cada tipo de movilidad

                # Aumentar el tamaño de la fuente para todos los elementos del gráfico
                plt.rcParams.update({'font.size': 11})

                # Crear un gráfico de densidad separado para cada patrón de calle
                for pattern in fixed_pattern_order:
                    # Filtrar datos para el patrón actual
                    pattern_data = combined_data[combined_data['pattern'] == pattern]
                    
                    # Verificar si hay suficientes datos para este patrón
                    if len(pattern_data) <= 1:
                        print(f"Omitiendo gráfico de densidad de movilidad para patrón {pattern} - datos insuficientes")
                        continue
                    
                    # Crear un nuevo gráfico para las densidades de movilidad de este patrón
                    plt.figure(figsize=(16, 9))
                    
                    # Crear KDE plot para cada tipo de movilidad dentro de este patrón
                    for i, (col, label, color) in enumerate(zip(mobility_cols, mobility_labels, mobility_colors)):
                        # Filtrar datos no nulos para la columna actual
                        data = pattern_data[col].dropna()
                        
                        if len(data) > 1:
                            # Crear el gráfico de densidad con etiquetas en mayúsculas
                            sns.kdeplot(data, fill=True, alpha=0.5, color=color, linewidth=2, label=label)
                            
                            # Añadir la mediana
                            median_val = data.median()
                            plt.axvline(x=median_val, color=color, linestyle="--", alpha=0.8, linewidth=1.5)
                            
                            # Altura base y separación entre etiquetas (en coordenadas de la figura, no pixeles)
                            y_base = 2.9
                            dy = 1  # Separación vertical entre etiquetas (~1 cm visual en la mayoría de figuras)

                            # Dibujar línea de la mediana
                            plt.axvline(median_val, color=color, linestyle='--', lw=1.5)

                            # Añadir etiqueta con fondo negro y texto blanco, con separación por 'i'
                            plt.text(median_val + 0.01, y_base - i * dy, f'MEDIANA {label}: {median_val:.2f}', 
                                    fontsize=11, color='black', fontweight='bold',
                                    bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))





                            # Agregar una leyenda general (solo una entrada para todas las medianas)
                            legend_elements = [
                                plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='Mediana'),
                                plt.Line2D([0], [0], color='blue', linestyle='-.', lw=1.5, label='Media')

                            ]
                            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.98))


                    
                    # Configuración del gráfico
                    plt.title(f'DISTRIBUCIÓN DE SHARES DE MOVILIDAD - PATRÓN {pattern.upper()}', fontsize=14, fontweight='bold')
                    plt.xlabel('PORCENTAJE DE SHARE', fontsize=11, fontweight='bold')
                    plt.ylabel('DENSIDAD', fontsize=11, fontweight='bold')
                    plt.legend(fontsize=11)
                    
                    # Ajustar estética
                    sns.despine()
                    plt.tight_layout()
                    
                    # Guardar el gráfico
                    plt.savefig(os.path.join(viz_dir, f"mobility_shares_density_{pattern}.png"), dpi=300, bbox_inches='tight')
                    plt.close()

                # Restaurar configuración original de tamaño de fuente si es necesario
                plt.rcParams.update({'font.size': plt.rcParams['font.size']})


                


                # B. BOXPLOT MEJORADO (una sola versión)
                plt.figure(figsize=(14, 8))
                
                # Boxplot simple y claro con notch
                ax = sns.boxplot(x='pattern', y=col, data=combined_data, notch=True,
                            palette=pattern_palette, width=0.6)
                
                # Añadir puntos de media con marcadores distinguibles sin usar swarm/strip
                for i, pattern in enumerate(ax.get_xticklabels()):
                    pattern_name = pattern.get_text()
                    pattern_data = combined_data[combined_data['pattern'] == pattern_name][col]
                    
                    if len(pattern_data) > 0:
                        # Añadir media como un diamante
                        mean_val = pattern_data.mean()
                        ax.plot(i, mean_val, marker='D', color='red', 
                            markersize=8, markeredgecolor='white', 
                            markeredgewidth=1.5, zorder=10)
                        
                        # Añadir etiqueta de la media
                        ax.text(i, mean_val, f' μ={mean_val:.2f}', 
                            color='darkred', fontsize=9,
                            va='center', ha='left')
                        
                        # Añadir etiqueta de la mediana
                        median_val = pattern_data.median()
                        ax.text(i, median_val, f' m={median_val:.2f}', 
                            color='black', fontsize=9,
                            va='center', ha='right')
                
                # Mejorar título y etiquetas
                plt.title(f'Distribución de {col} por Patrón', fontsize=14)
                plt.xlabel('Patrón', fontsize=12)
                plt.ylabel(col, fontsize=12)
                plt.xticks(rotation=45)
                
                # Añadir grid para mejor legibilidad
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                
                # Ajustar manualmente los márgenes
                plt.subplots_adjust(bottom=0.2)
                plt.savefig(os.path.join(viz_dir, f"boxplot_{col}.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # C. HEATMAP DE ESTADÍSTICAS (corregido)
                # Verificar si hay suficientes datos para un heatmap útil
                if len(combined_data['city'].unique()) > 1 and len(combined_data['pattern'].unique()) > 1:
                    # Crear pivot table directamente desde los datos originales
                    try:
                        pivot_means = pd.pivot_table(combined_data, 
                                                values=col,
                                                index='city', 
                                                columns='pattern',
                                                aggfunc='mean')
                        
                        # Verificar que el pivot table tenga datos
                        if not pivot_means.empty and not pivot_means.isna().all().all():
                            plt.figure(figsize=(14, 10))
                            
                            # Crear heatmap con valores explícitos
                            cmap_values = sns.diverging_palette(230, 20, as_cmap=True)
                            
                            # Usar un valor central explícito (la media global)
                            center_val = combined_data[col].mean()
                            
                            ax = sns.heatmap(pivot_means, annot=True, fmt='.2f', cmap=cmap_values, 
                                        center=center_val, linewidths=0.5)
                            
                            plt.title(f'Media de {col} por Ciudad y Patrón', fontsize=14)
                            plt.ylabel('Ciudad', fontsize=12)
                            plt.xlabel('Patrón', fontsize=12)
                            plt.xticks(rotation=45, ha='right')
                            
                            plt.savefig(os.path.join(viz_dir, f"heatmap_media_{col}.png"), dpi=300, bbox_inches='tight')
                            plt.close()
                            
                        else:
                            print(f"Omitiendo heatmap para {col} - Sin datos suficientes en el pivot table")
                    except Exception as e:
                        print(f"Error generando heatmap para {col}: {e}")
                        traceback.print_exc()
            
            # 3. CORRELACIÓN MEJORADA entre variables de movilidad
            # Filtrar solo las variables numéricas con suficientes datos
            valid_cols = []
            for col in mobility_cols:
                if pd.api.types.is_numeric_dtype(combined_data[col]) and combined_data[col].nunique() > 1:
                    valid_cols.append(col)
            
            if len(valid_cols) > 1:  # Necesitamos al menos 2 columnas para correlación
                mobility_data = combined_data[valid_cols]
                
                plt.figure(figsize=(14, 12))
                correlation_matrix = mobility_data.corr(numeric_only=True)
                
                # Máscara para mostrar solo la mitad del heatmap
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                
                # Heatmap de correlación mejorado
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                ax = sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, 
                            annot=True, fmt='.2f', square=True, center=0,
                            linewidths=0.5, cbar_kws={"shrink": 0.8})
                
                plt.title('Correlación entre Variables de Movilidad', fontsize=16)
                
                # Guardar con bounding box ajustada
                plt.savefig(os.path.join(viz_dir, "correlacion_variables.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. GRÁFICO DE BARRAS para TODAS las variables de movilidad (MODIFICADO)
            # 4. GRÁFICO DE BARRAS MEJORADO para TODAS las variables de movilidad
            # Definir el orden específico de los patrones (asegurando matcheo exacto con los datos)
            patron_orden = ['gridiron', 'organico', 'hibrido', 'cul_de_sac']

            # Definir los colores específicos para cada patrón
            patron_colors = {
                'cul_de_sac': '#FF6B6B',  # Rojo para callejones sin salida
                'gridiron': '#006400',    # Verde oscuro para grid
                'organico': '#45B7D1',    # Azul para orgánico
                'hibrido': '#FDCB6E',     # Amarillo para híbrido
            }

            # Crear un mapeo explícito para asegurar que los patrones coincidan independientemente de mayúsculas/minúsculas
            # y posibles variaciones en nombres
            patron_mapping = {
                'gridiron': 'gridiron',
                'Gridiron': 'gridiron',
                'GRIDIRON': 'gridiron',
                'grid': 'gridiron',
                'Grid': 'gridiron',
                'GRID': 'gridiron',
                'organico': 'organico',
                'Organico': 'organico',
                'orgánico': 'organico',
                'Orgánico': 'organico',
                'ORGANICO': 'organico',
                'hibrido': 'hibrido',
                'Hibrido': 'hibrido',
                'híbrido': 'hibrido',
                'Híbrido': 'hibrido',
                'HIBRIDO': 'hibrido',
                'cul_de_sac': 'cul_de_sac',
                'Cul_de_sac': 'cul_de_sac',
                'CUL_DE_SAC': 'cul_de_sac',
                'cul de sac': 'cul_de_sac',
                'Cul de sac': 'cul_de_sac',
                'Cul-de-sac': 'cul_de_sac',
                'cul-de-sac': 'cul_de_sac',
                'CUL DE SAC': 'cul_de_sac'
            }

            # Aplicar el mapeo a los datos para estandarizar los nombres de los patrones antes de graficar
            if 'pattern' in combined_data.columns:
                # Crear una copia temporal de patrones para no modificar los datos originales
                combined_data_temp = combined_data.copy()
                combined_data_temp['pattern_mapped'] = combined_data_temp['pattern'].map(
                    lambda x: patron_mapping.get(x, x)
                )
                
                print(f"Patrones originales: {combined_data['pattern'].unique()}")
                print(f"Patrones mapeados: {combined_data_temp['pattern_mapped'].unique()}")

            for col in mobility_cols:
                # Saltamos variables sin variación o todas NaN
                if combined_data[col].isna().all() or combined_data[col].std() == 0:
                    continue
                    
                # Crear figura y ejes con estilo mejorado
                plt.figure(figsize=(14, 8))
                
                # Aplicar estilo más moderno y limpio
                plt.style.use('seaborn-v0_8-whitegrid')
                
                # Calcular estadísticas usando los patrones mapeados para asegurar consistencia
                pattern_stats = combined_data_temp.groupby('pattern_mapped')[col].agg(['mean', 'std', 'count'])
                
                # Reordenar explícitamente según el orden definido
                # Filtrar solo los patrones que existen en los datos
                patrones_existentes = list(pattern_stats.index)
                patrones_ordenados = [p for p in patron_orden if p in patrones_existentes]
                # Añadir cualquier patrón adicional al final
                for p in patrones_existentes:
                    if p not in patrones_ordenados:
                        patrones_ordenados.append(p)
                
                print(f"Orden final de patrones para {col}: {patrones_ordenados}")
                
                # Verificación extra para asegurar que hay datos para graficar
                if not patrones_ordenados:
                    print(f"No se encontraron patrones válidos para {col}")
                    continue
                
                # Reordenar el DataFrame según el orden especificado
                pattern_stats = pattern_stats.reindex(patrones_ordenados)
                
                # Calcular error estándar
                pattern_stats['se'] = pattern_stats['std'] / np.sqrt(pattern_stats['count'])
                
                # Usar los colores específicos definidos para cada patrón
                bar_colors = [patron_colors[p] for p in patrones_ordenados]
                
                # Crear gráfico de barras con estética mejorada
                bars = plt.bar(
                    range(len(pattern_stats)), 
                    pattern_stats['mean'],
                    yerr=pattern_stats['se'],
                    color=bar_colors,
                    capsize=7,
                    width=0.7,
                    edgecolor='black',
                    linewidth=1,
                    error_kw={'elinewidth': 2, 'capthick': 2}
                )
                
                # Mejorar las etiquetas del eje X
                plt.xticks(
                    range(len(pattern_stats)), 
                    [p.replace('_', ' ').title() for p in pattern_stats.index],  # Formatear etiquetas
                    rotation=30, 
                    ha='right',
                    fontsize=11,
                    fontweight='bold'
                )
                
                # Añadir valores en las barras con mejor formato
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    mean_val = pattern_stats['mean'].iloc[i]
                    plt.text(
                        bar.get_x() + bar.get_width()/2, 
                        height + pattern_stats['se'].iloc[i] + 0.02 * max(pattern_stats['mean']),
                        f'{mean_val:.2f}', 
                        ha="center", 
                        va="bottom",
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8)
                    )
                
                # Convertir el título de la variable a mayúsculas y hacer más descriptivo
                col_mayusculas = col.upper()
                col_formatted = col_mayusculas.replace("_", " ")
                
                # Título y etiquetas mejorados
                plt.title(f'MEDIA DE {col_formatted} POR PATRÓN', 
                        fontsize=16, 
                        fontweight='bold',
                        pad=20)
                
                plt.xlabel('PATRÓN URBANO', fontsize=14, fontweight='bold', labelpad=15)
                plt.ylabel(f'MEDIA DE {col_formatted}', fontsize=14, fontweight='bold', labelpad=15)
                
                # Mejorar la cuadrícula
                plt.grid(axis='y', linestyle='--', alpha=0.3, linewidth=1)
                
                # Mejorar márgenes
                plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.15)
                
                # Añadir un recuadro alrededor de la figura
                plt.box(on=True)
                
                # Añadir una línea de referencia en y=0 si es apropiado
                if min(pattern_stats['mean']) < 0:
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Guardar con mejor calidad y formato más claro
                plt.savefig(os.path.join(viz_dir, f"barplot_{col}.png"), 
                            dpi=300, 
                            bbox_inches='tight',
                            facecolor='white')
                plt.close()
            
            # 5. GRÁFICOS RADIALES CORRECTAMENTE IMPLEMENTADOS (VERSIÓN CORREGIDA)

            # 5. GRÁFICOS RADIALES CORRECTAMENTE IMPLEMENTADOS (VERSIÓN CORREGIDA)

            # Primero, identificar todas las variables numéricas de movilidad disponibles
            available_vars = []
            for col in mobility_cols:
                if (col in combined_data.columns and 
                    pd.api.types.is_numeric_dtype(combined_data[col]) and 
                    not combined_data[col].isna().all() and 
                    combined_data[col].std() > 0):
                    available_vars.append(col)

            # Prioridad de variables (ajusta esta lista según las variables más importantes para ti)
            priority_vars = [
                'car_share', 
                'walked_share', 
                'transport_share', 
                'bicycle_share',  # Lo mantenemos en la lista, pero tendrá menor prioridad
                'mobility_a', 
                'mobility_b', 
                'mobility_c'
            ]

            # Reordenar las variables disponibles según la prioridad (si existen)
            ordered_vars = []
            for var in priority_vars:
                if var in available_vars:
                    ordered_vars.append(var)

            # Añadir cualquier otra variable disponible que no esté en la lista de prioridad
            for var in available_vars:
                if var not in ordered_vars:
                    ordered_vars.append(var)

            # Seleccionar exactamente 6 variables si es posible, o tantas como haya disponibles (mínimo 3)
            if len(ordered_vars) >= 6:
                available_vars = ordered_vars[:6]  # Tomar las 6 primeras variables priorizadas
            elif len(ordered_vars) >= 3:
                available_vars = ordered_vars  # Usar todas las variables disponibles (entre 3 y 5)
            # Si hay menos de 3, el código mostrará un mensaje de error más adelante

            # Mensaje de verificación más detallado
            print(f"Variables encontradas para el radar: {len(available_vars)}")
            print(f"Variables que se utilizarán: {available_vars}")
            print(f"Nota: Se necesitan al menos 3 variables para crear un gráfico radial. Un hexágono ideal necesita 6 variables.")

            # Si tenemos al menos 3 variables, procedemos con el gráfico
            if len(available_vars) >= 3:
                # Crear directorio específico para gráficos radiales
                radar_dir = os.path.join(viz_dir, "radar_plots")
                os.makedirs(radar_dir, exist_ok=True)
                
                # Calcular estadísticas por patrón
                pattern_means = combined_data.groupby('pattern')[available_vars].mean()
                
                # Encontrar min/max globales para cada variable para normalización consistente
                global_min = {}
                global_max = {}
                for col in available_vars:
                    global_min[col] = combined_data[col].min()
                    global_max[col] = combined_data[col].max()
                    # Evitar división por cero
                    if global_min[col] == global_max[col]:
                        global_min[col] -= 0.1
                        global_max[col] += 0.1
                
                # Función para crear gráfico radial mejorada
                def create_radar_chart(ax, pattern_data, pattern, is_comparison=False):
                    # Número de variables
                    N = len(available_vars)
                    
                    # Calcular los ángulos para cada variable (en radianes)
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # Cerrar el círculo
                    
                    # Obtener valores normalizados
                    values = []
                    real_values = []
                    for col in available_vars:
                        val = pattern_data.loc[pattern, col]
                        real_values.append(val)
                        # Normalizar a escala 0-1
                        norm_val = (val - global_min[col]) / (global_max[col] - global_min[col])
                        values.append(norm_val)
                    
                    # Completar el círculo repitiendo el primer valor
                    values += values[:1]
                    real_values += real_values[:1]
                    
                    # Color para este patrón
                    color = pattern_color_dict.get(pattern, 'blue')
                    
                    # Dibujar el gráfico principal
                    ax.plot(angles, values, 'o-', linewidth=2, color=color, markersize=6)
                    ax.fill(angles, values, color=color, alpha=0.25)
                    
                    # Dibujar líneas desde el centro a cada punto (las aristas)
                    for i, angle in enumerate(angles[:-1]):  # Excluir el último que es repetido
                        ax.plot([0, angle], [0, 1], 'k-', lw=0.8, alpha=0.2)
                    
                    # Añadir etiquetas en las posiciones exactas
                    if not is_comparison:
                        # Añadir valores reales en cada vértice
                        for i in range(N):
                            angle_in_rad = angles[i]
                            value = values[i]
                            real_value = real_values[i]
                            
                            # Posición del marcador
                            if value > 0.05:  # Si el valor no está muy cerca del centro
                                x = value * np.cos(angle_in_rad)
                                y = value * np.sin(angle_in_rad)
                                
                                # Mostrar valor real cerca del punto
                                ax.text(x*1.1, y*1.1, f"{real_value:.2f}", 
                                    color=color, fontsize=9, ha='center', va='center',
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
                    
                    # Configuración del eje
                    ax.set_ylim(0, 1)
                    
                    # Eliminar etiquetas de ejes por defecto
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    
                    # Dibujar círculos concéntricos para referencia
                    circles = [0.2, 0.4, 0.6, 0.8]
                    for circle_val in circles:
                        circle = plt.Circle((0, 0), circle_val, fill=False, color='gray', alpha=0.2)
                        ax.add_patch(circle)
                        
                        # Añadir valores de porcentaje solo en un lugar específico
                        ax.text(-0.02, circle_val, f"{int(circle_val*100)}%", 
                            ha='right', va='center', color='gray', fontsize=8, alpha=0.7)
                    
                    return ax
                
                # Crear un gráfico radial para cada patrón
                for pattern in combined_data['pattern'].unique():
                    # Verificar si este patrón existe en pattern_means
                    if pattern not in pattern_means.index:
                        continue
                    
                    # Crear figura con más espacio para acomodar las etiquetas
                    fig = plt.figure(figsize=(12, 10))
                    
                    # Crear subplot con proyección polar y amplio margen
                    ax = fig.add_subplot(111, polar=True)
                    
                    # Dibujar el gráfico radial
                    ax = create_radar_chart(ax, pattern_means, pattern)
                    
                    # Número de variables
                    N = len(available_vars)
                    
                    # Recalcular ángulos para colocar etiquetas
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    
                    # MEJORA IMPORTANTE: Colocar etiquetas en las aristas exactamente
                    for i, col in enumerate(available_vars):
                        angle = angles[i]
                        
                        # Crear etiquetas más visibles directamente en las aristas
                        # Calcular posición para la etiqueta alineada exactamente en la arista
                        label_radius = 1.2  # Distancia desde el centro hasta la etiqueta
                        x = label_radius * np.cos(angle)
                        y = label_radius * np.sin(angle)
                        
                        # Ajustar alineación según la posición angular
                        # Para que las etiquetas aparezcan alineadas con las aristas
                        if -0.1 < angle < 0.1:  # Derecha (0°)
                            ha, va = 'left', 'center'
                        elif 0.1 < angle < np.pi/2 - 0.1:  # Primer cuadrante
                            ha, va = 'left', 'bottom'
                        elif np.pi/2 - 0.1 < angle < np.pi/2 + 0.1:  # Arriba (90°)
                            ha, va = 'center', 'bottom'
                        elif np.pi/2 + 0.1 < angle < np.pi - 0.1:  # Segundo cuadrante
                            ha, va = 'right', 'bottom'
                        elif np.pi - 0.1 < angle < np.pi + 0.1:  # Izquierda (180°)
                            ha, va = 'right', 'center'
                        elif np.pi + 0.1 < angle < 3*np.pi/2 - 0.1:  # Tercer cuadrante
                            ha, va = 'right', 'top'
                        elif 3*np.pi/2 - 0.1 < angle < 3*np.pi/2 + 0.1:  # Abajo (270°)
                            ha, va = 'center', 'top'
                        else:  # Cuarto cuadrante
                            ha, va = 'left', 'top'
                        
                        # Formato mejorado para las etiquetas, con fondo más visible
                        ax.text(x, y, col, ha=ha, va=va, fontsize=14, fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.95, 
                                    boxstyle='round,pad=0.5', linewidth=1.5))
                    
                    # Añadir título
                    plt.title(f'Perfil de Movilidad: Patrón {pattern}', size=16, y=1.05)
                    
                    # Guardar el gráfico con margen adicional para asegurar que las etiquetas sean visibles
                    plt.tight_layout()
                    plt.savefig(os.path.join(radar_dir, f"radar_patron_{pattern}.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                
                # Crear gráfico de comparación de todos los patrones
                fig = plt.figure(figsize=(14, 12))
                ax = fig.add_subplot(111, polar=True)
                
                # Dibujar los gráficos de cada patrón
                legend_handles = []
                for pattern in pattern_means.index:
                    color = pattern_color_dict.get(pattern, 'blue')
                    
                    # Obtener valores normalizados
                    values = []
                    for col in available_vars:
                        val = pattern_means.loc[pattern, col]
                        norm_val = (val - global_min[col]) / (global_max[col] - global_min[col])
                        values.append(norm_val)
                    
                    # Cerrar el círculo
                    N = len(available_vars)
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]
                    values += values[:1]
                    
                    # Dibujar línea y área
                    line, = ax.plot(angles, values, 'o-', linewidth=2, color=color, label=pattern)
                    ax.fill(angles, values, color=color, alpha=0.1)
                    legend_handles.append(line)
                
                # Añadir líneas de referencia (aristas)
                N = len(available_vars)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                for angle in angles:
                    ax.plot([0, angle], [0, 1], 'k-', lw=0.8, alpha=0.2)
                
                # Añadir círculos concéntricos
                circles = [0.2, 0.4, 0.6, 0.8]
                for circle_val in circles:
                    circle = plt.Circle((0, 0), circle_val, fill=False, color='gray', alpha=0.2)
                    ax.add_patch(circle)
                    ax.text(-0.02, circle_val, f"{int(circle_val*100)}%", 
                        ha='right', va='center', color='gray', fontsize=8, alpha=0.7)
                
                # Colocar etiquetas de los ejes - MEJORA CLAVE AQUÍ
                for i, col in enumerate(available_vars):
                    angle = angles[i]
                    
                    # Posición alineada con las aristas, pero más alejada para el gráfico comparativo
                    label_radius = 1.3  # Distancia desde el centro
                    x = label_radius * np.cos(angle)
                    y = label_radius * np.sin(angle)
                    
                    # Ajustar alineación según la posición angular
                    if -0.1 < angle < 0.1:  # Derecha (0°)
                        ha, va = 'left', 'center'
                    elif 0.1 < angle < np.pi/2 - 0.1:  # Primer cuadrante
                        ha, va = 'left', 'bottom'
                    elif np.pi/2 - 0.1 < angle < np.pi/2 + 0.1:  # Arriba (90°)
                        ha, va = 'center', 'bottom'
                    elif np.pi/2 + 0.1 < angle < np.pi - 0.1:  # Segundo cuadrante
                        ha, va = 'right', 'bottom'
                    elif np.pi - 0.1 < angle < np.pi + 0.1:  # Izquierda (180°)
                        ha, va = 'right', 'center'
                    elif np.pi + 0.1 < angle < 3*np.pi/2 - 0.1:  # Tercer cuadrante
                        ha, va = 'right', 'top'
                    elif 3*np.pi/2 - 0.1 < angle < 3*np.pi/2 + 0.1:  # Abajo (270°)
                        ha, va = 'center', 'top'
                    else:  # Cuarto cuadrante
                        ha, va = 'left', 'top'
                    
                    # Etiquetas con fondo más destacado
                    ax.text(x, y, col, ha=ha, va=va, fontsize=14, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.95, 
                                boxstyle='round,pad=0.7', linewidth=1.5))
                
                # Configurar el eje
                ax.set_ylim(0, 1)
                ax.set_yticklabels([])
                ax.set_xticks([])
                
                # Añadir leyenda
                plt.legend(handles=legend_handles, loc='upper right', 
                        bbox_to_anchor=(1.3, 1.0), title="Patrones", fontsize=12)
                
                # Añadir título
                plt.title('Comparación de Perfiles de Movilidad', size=16, y=1.05)
                
                # Guardar gráfico con margen extra
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                plt.savefig(os.path.join(radar_dir, "comparacion_patrones.png"), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("No hay suficientes variables numéricas válidas para crear el gráfico radial")
                    
        except Exception as e:
            print(f"Error generando visualizaciones mejoradas: {e}")
            traceback.print_exc()

    def _save_analysis_results(self, results_dict, output_dir):
        """Guarda los resultados del análisis en formatos útiles."""
        try:
            # 1. Excel con resultados estructurados
            with pd.ExcelWriter(os.path.join(output_dir, "anova_results_global.xlsx")) as writer:
                # Hoja de resumen general
                summary_data = []
                
                for name, result in results_dict.items():
                    if 'main_effects' in name:
                        mobility_var = name.replace('_main_effects', '')
                        pattern_effect = result.loc['C(pattern)']['PR(>F)'] if 'C(pattern)' in result.index else np.nan
                        city_effect = result.loc['C(city)']['PR(>F)'] if 'C(city)' in result.index else np.nan
                        
                        summary_data.append({
                            'Variable': mobility_var,
                            'Patrón p-value': pattern_effect,
                            'Ciudad p-value': city_effect,
                            'Patrón significativo': pattern_effect < 0.05 if not np.isnan(pattern_effect) else False,
                            'Ciudad significativa': city_effect < 0.05 if not np.isnan(city_effect) else False
                        })
                
                if summary_data:
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Resumen', index=False)
                
                # Hojas individuales para cada variable
                for name, result in results_dict.items():
                    if isinstance(result, pd.DataFrame):
                        # Limitar nombre de hoja a 31 caracteres (límite de Excel)
                        sheet_name = name[:31]
                        result.to_excel(writer, sheet_name=sheet_name)
                    elif isinstance(result, dict) and not any(isinstance(v, dict) for v in result.values()):
                        # Convertir diccionarios simples a DataFrame
                        pd.DataFrame([result]).to_excel(writer, sheet_name=name[:31])
                    elif isinstance(result, dict):
                        # Para diccionarios anidados, convertir a formato tabular
                        rows = []
                        for k, v in result.items():
                            if isinstance(v, dict):
                                row = {'key': k}
                                row.update(v)
                                rows.append(row)
                        
                        if rows:
                            pd.DataFrame(rows).to_excel(writer, sheet_name=name[:31])
            
            # 2. Informe de resultados en formato Markdown
            with open(os.path.join(output_dir, "informe_análisis_global.md"), 'w', encoding='utf-8') as f:
                f.write("# Informe de Análisis Global de Movilidad\n\n")
                
                # Agrupar resultados por variable de movilidad
                mobility_vars = set()
                for name in results_dict.keys():
                    parts = name.split('_')
                    if len(parts) > 1:
                        mobility_vars.add(parts[0])
                
                for var in sorted(mobility_vars):
                    f.write(f"## Análisis para la variable: {var}\n\n")
                    
                    # Efectos principales
                    main_effects_key = f"{var}_main_effects"
                    if main_effects_key in results_dict:
                        f.write("### Efectos principales (ANOVA)\n\n")
                        df = results_dict[main_effects_key]
                        
                        # Obtener nombres de columnas reales en el DataFrame
                        available_cols = df.columns.tolist()
                        
                        # Crear encabezado de tabla con columnas disponibles
                        header = "| Fuente |"
                        separator = "|--------|"
                        
                        # Mapeo amigable para nombres de columnas
                        column_names = {
                            'df': 'gl',
                            'sum_sq': 'Suma cuadrados',
                            'MS': 'Media cuadrática',  # Alternativa a mean_sq
                            'mean_sq': 'Media cuadrática',
                            'F': 'F',
                            'PR(>F)': 'p-valor'
                        }
                        
                        for col in available_cols:
                            friendly_name = column_names.get(col, col)
                            header += f" {friendly_name} |"
                            separator += "---|"
                        
                        f.write(header + "\n")
                        f.write(separator + "\n")
                        
                        # Escribir filas
                        for idx, row in df.iterrows():
                            line = f"| {idx} |"
                            for col in available_cols:
                                val = row[col]
                                # Formatear números según el tipo de columna
                                if col == 'df':
                                    line += f" {val:.0f} |"
                                elif col in ['sum_sq', 'MS', 'mean_sq', 'F', 'PR(>F)']:
                                    line += f" {val:.4f} |"
                                else:
                                    line += f" {val} |"
                            f.write(line + "\n")
                        
                        f.write("\n")
                    
                    # Resumen de efectos
                    effects_summary_key = f"{var}_effects_summary"
                    if effects_summary_key in results_dict:
                        summary = results_dict[effects_summary_key]
                        f.write("### Resumen de efectos entre ciudades y patrones\n\n")
                        
                        f.write(f"- {summary.get('cities_with_significant_patterns', 'N/A')} de {summary.get('total_cities_analyzed', 'N/A')} ")
                        f.write("ciudades muestran diferencias significativas entre patrones\n")
                        
                        f.write(f"- {summary.get('patterns_with_significant_cities', 'N/A')} de {summary.get('total_patterns_analyzed', 'N/A')} ")
                        f.write("patrones muestran diferencias significativas entre ciudades\n\n")
                    
                    # Coeficientes
                    coefficients_key = f"{var}_coefficients"
                    if coefficients_key in results_dict:
                        coef_df = results_dict[coefficients_key]
                        if 'significant' in coef_df.columns:
                            significant_coefs = coef_df[coef_df['significant']]
                            
                            if len(significant_coefs) > 0:
                                f.write("### Efectos significativos\n\n")
                                f.write("| Variable | Coeficiente | p-value |\n")
                                f.write("|----------|-------------|--------|\n")
                                
                                for _, row in significant_coefs.iterrows():
                                    f.write(f"| {row['variable']} | {row['coefficient']:.4f} | {row['p_value']:.4f} |\n")
                                
                                f.write("\n")
                
                # Información diagnóstica
                f.write("## Diagnóstico de datos\n\n")
                for key, value in results_dict.items():
                    if "data_diagnosis" in key:
                        f.write(f"### {key}\n\n")
                        if isinstance(value, dict):
                            for k, v in value.items():
                                f.write(f"- **{k}**: {v}\n")
                        f.write("\n")
                
                f.write("## Conclusiones generales\n\n")
                f.write("- Las principales variables de movilidad muestran patrones distintivos según la estructura urbana.\n")
                f.write("- La ciudad tiene un efecto importante en las variables de movilidad, lo que indica\n")
                f.write("  que los factores contextuales locales influyen significativamente.\n")
                f.write("- Los hallazgos sugieren que las políticas de movilidad deberían adaptarse tanto al\n")
                f.write("  patrón urbano como al contexto específico de cada ciudad.\n")
                
                # Identificar hallazgos clave
                significant_effects = {}
                for key, value in results_dict.items():
                    if 'main_effects' in key:
                        var_name = key.replace('_main_effects', '')
                        if isinstance(value, pd.DataFrame):
                            pattern_significant = False
                            city_significant = False
                            
                            if 'C(pattern)' in value.index and 'PR(>F)' in value.columns:
                                if value.loc['C(pattern)', 'PR(>F)'] < 0.05:
                                    pattern_significant = True
                                    
                            if 'C(city)' in value.index and 'PR(>F)' in value.columns:
                                if value.loc['C(city)', 'PR(>F)'] < 0.05:
                                    city_significant = True
                            
                            if pattern_significant or city_significant:
                                significant_effects[var_name] = {
                                    'pattern': pattern_significant,
                                    'city': city_significant
                                }
                
                if significant_effects:
                    f.write("\n### Hallazgos significativos\n\n")
                    for var, effects in significant_effects.items():
                        f.write(f"- **{var}**: ")
                        effect_texts = []
                        if effects.get('pattern'):
                            effect_texts.append("diferencias significativas entre patrones urbanos")
                        if effects.get('city'):
                            effect_texts.append("diferencias significativas entre ciudades")
                        
                        f.write(", ".join(effect_texts) + ".\n")
            
            # 3. JSON para procesamiento posterior
            import json
            
            # Convertir dataframes y objetos numpy a formatos serializables
            def json_serializable(obj):
                if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                    return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return str(obj)
            
            # Convertir resultados a formato serializable
            json_compatible = {}
            for key, value in results_dict.items():
                if isinstance(value, pd.DataFrame):
                    json_compatible[key] = value.to_dict('records')
                elif isinstance(value, dict):
                    # Manejar diccionarios anidados
                    json_compatible[key] = json.loads(
                        json.dumps(value, default=json_serializable)
                    )
                else:
                    json_compatible[key] = json_serializable(value)
            
            with open(os.path.join(output_dir, "anova_results_global.json"), 'w', encoding='utf-8') as f:
                json.dump(json_compatible, f, indent=2)
                
        except Exception as e:
            print(f"Error guardando resultados de análisis: {e}")
            import traceback
            traceback.print_exc()

    def _generate_interaction_plots(self, combined_data, mobility_cols):
        """Genera gráficos de interacción entre patrones y ciudades."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.ioff()  # Desactivar modo interactivo para evitar mostrar gráficos en consola
        
        try:
            for col in mobility_cols:
                plt.figure(figsize=(12, 8))
                sns.boxplot(x='city', y=col, hue='pattern', data=combined_data)
                plt.title(f'Interacción Ciudad-Patrón para {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.global_dir, f"interaction_{col}.png"))
                plt.close()
                
                # Crear también una gráfica de interacción con medias y errores estándar
                plt.figure(figsize=(12, 8))
                sns.pointplot(x='city', y=col, hue='pattern', data=combined_data, 
                            dodge=True, ci=68, capsize=0.2)
                plt.title(f'Medias e Intervalos para {col} por Ciudad y Patrón')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.global_dir, f"interaction_means_{col}.png"))
                plt.close()
                
        except Exception as e:
            print(f"Error generando gráficos de interacción: {e}")
    
    def _generate_global_visualizations(self, combined_data, mobility_cols):
        """Genera visualizaciones para el análisis global."""
        try:
            # 1. Boxplots de movilidad por patrón
            for mobility_type in mobility_cols:
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
            for mobility_type in mobility_cols:
                plt.figure(figsize=(16, 8))
                sns.boxplot(x='city', y=mobility_type, data=combined_data)
                plt.title(f'Global: {self.mobility_columns[mobility_type]} por Ciudad')
                plt.xlabel('Ciudad')
                plt.ylabel(self.mobility_columns[mobility_type])
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.global_dir, f"global_boxplot_{mobility_type}_por_ciudad.png"))
                plt.close()
            
            # 3. Gráfico de interacción entre patrón y ciudad
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
            
            # 4. PCA global por patrón y ciudad
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
            print(f"Error generando visualizaciones globales: {e}")
    
    def run_analysis(self):
        """Ejecuta el flujo completo de análisis."""
        print("Iniciando análisis de patrones de calles y movilidad urbana...")
        
        # 1. Procesar todas las ciudades
        if not self.process_cities():
            print("No se pudieron procesar las ciudades")
            return False
        
        # 2. Análisis por ciudad
        for city_name, city_data in self.city_dataframes.items():
            self.analyze_city(city_data, city_name)
        
        # 3. Análisis global
        self.analyze_global_data()
        
        print("Análisis completo")
        return True


if __name__ == "__main__":
    analyzer = StreetPatternMobilityAnalyzer()
    analyzer.run_analysis()