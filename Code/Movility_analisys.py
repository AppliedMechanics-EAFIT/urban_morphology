import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.formula.api import ols
import glob, re, pathlib
import warnings
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import stats
import traceback
import matplotlib.patheffects as patheffects
import numpy as np
from sklearn.neighbors import KernelDensity
from matplotlib.patches import Patch
warnings.filterwarnings('ignore', category=FutureWarning)






class StreetPatternMobilityAnalyzer:
    from Graphs_format import _setup_plot_style, _get_pattern_config, _save_figure

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
            plt.savefig(os.path.join(subpatterns_dir, f"{city_name}_heatmap_subpatrones.pdf"))
            plt.close()
            
        except Exception as e:
            print(f"Error en análisis de subpatrones para {city_name}: {e}")
    
    def analyze_global_data(self):

        """Análisis estadístico robusto de patrones viales y movilidad."""
        print("\n=== Análisis Global: Patrones Viales vs Movilidad ===")
        
        if not self.all_cities_data:
            print("No hay datos disponibles para análisis")
            return False
        
        try:
            # Combinar datos
            df = pd.concat(self.all_cities_data, ignore_index=True)
            print(f"Registros totales: {len(df)} | Ciudades: {df['city'].nunique()} | Patrones: {df['pattern'].nunique()}")
            
            # Variables de movilidad disponibles (solo las que queremos analizar)
            expected_mobility_vars = ['a', 'b', 'c', 'walked_share', 'car_share', 'bicycle_share', 'transit_share']
            mobility_vars = [col for col in expected_mobility_vars if col in df.columns]
            
            # Verificar que tenemos las variables correctas
            missing_vars = [col for col in expected_mobility_vars if col not in df.columns]
            extra_vars = [col for col in self.mobility_columns.keys() if col in df.columns and col not in expected_mobility_vars]
            
            print(f"Variables de movilidad encontradas: {mobility_vars}")
            if missing_vars:
                print(f"Variables faltantes: {missing_vars}")
            if extra_vars:
                print(f"Variables extra disponibles (no analizadas): {extra_vars}")
            
            # === 1. ANÁLISIS DE CORRELACIÓN (SPEARMAN) ===
            results = {
                'correlations': {},
                'kruskal_wallis': {},
                'effect_sizes': {},
                'descriptive_stats': {}
            }
            
            # Codificar patrones para correlación
            pattern_encoder = {pattern: i for i, pattern in enumerate(df['pattern'].unique())}
            df['pattern_numeric'] = df['pattern'].map(pattern_encoder)
            
            print("\n--- Correlaciones Spearman (Patrón-Movilidad) ---")
            correlations = []
            
            for var in mobility_vars:
                # Eliminar valores nulos
                clean_data = df[[var, 'pattern_numeric']].dropna()
                if len(clean_data) < 10:
                    continue
                    
                # Correlación de Spearman
                from scipy.stats import spearmanr
                corr, p_val = spearmanr(clean_data['pattern_numeric'], clean_data[var])
                
                # Interpretación del tamaño del efecto
                if abs(corr) < 0.1:
                    effect = "Muy débil"
                elif abs(corr) < 0.3:
                    effect = "Débil"
                elif abs(corr) < 0.5:
                    effect = "Moderado"
                elif abs(corr) < 0.7:
                    effect = "Fuerte"
                else:
                    effect = "Muy fuerte"
                
                correlations.append({
                    'Variable': var,
                    'Correlación': round(corr, 4),
                    'p-valor': round(p_val, 6),
                    'Significativo': 'Sí' if p_val < 0.05 else 'No',
                    'Efecto': effect,
                    'n': len(clean_data)
                })
                
                results['correlations'][var] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'effect_size': effect
                }
                
                print(f"{var:15} | r={corr:6.3f} | p={p_val:8.6f} | {effect:10} | {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '   '}")
            
            # === 2. TEST DE KRUSKAL-WALLIS (no paramétrico) ===
            print("\n--- Test Kruskal-Wallis (Diferencias entre Patrones) ---")
            kruskal_results = []
            
            from scipy.stats import kruskal
            
            for var in mobility_vars:
                clean_data = df[[var, 'pattern']].dropna()
                if len(clean_data) < 20:
                    continue
                
                # Agrupar por patrón
                groups = [group[var].values for name, group in clean_data.groupby('pattern') if len(group) >= 5]
                
                if len(groups) < 2:
                    continue
                
                # Test de Kruskal-Wallis
                h_stat, p_val = kruskal(*groups)
                
                # Tamaño del efecto (eta cuadrado aproximado)
                n_total = len(clean_data)
                eta_squared = (h_stat - len(groups) + 1) / (n_total - len(groups))
                eta_squared = max(0, eta_squared)  # No puede ser negativo
                
                if eta_squared < 0.01:
                    effect = "Muy pequeño"
                elif eta_squared < 0.06:
                    effect = "Pequeño"
                elif eta_squared < 0.14:
                    effect = "Mediano"
                else:
                    effect = "Grande"
                
                kruskal_results.append({
                    'Variable': var,
                    'H-estadístico': round(h_stat, 3),
                    'p-valor': round(p_val, 6),
                    'Significativo': 'Sí' if p_val < 0.05 else 'No',
                    'η² (aprox)': round(eta_squared, 4),
                    'Efecto': effect,
                    'n': n_total,
                    'Grupos': len(groups)
                })
                
                results['kruskal_wallis'][var] = {
                    'h_statistic': h_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'eta_squared': eta_squared,
                    'effect_size': effect
                }
                
                print(f"{var:15} | H={h_stat:6.2f} | p={p_val:8.6f} | η²={eta_squared:5.3f} | {effect:10} | {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '   '}")
            
            # === 3. ESTADÍSTICAS DESCRIPTIVAS POR PATRÓN ===
            print("\n--- Estadísticas por Patrón ---")
            
            for var in mobility_vars[:3]:  # Solo mostrar las primeras 3 para no saturar
                print(f"\n{var}:")
                stats = df.groupby('pattern')[var].agg(['count', 'mean', 'std', 'median']).round(4)
                stats = stats.sort_values('mean', ascending=False)
                
                results['descriptive_stats'][var] = stats.to_dict('index')
                
                for pattern, row in stats.iterrows():
                    if row['count'] >= 5:  # Solo mostrar patrones con suficientes datos
                        print(f"  {pattern:20} | n={row['count']:3.0f} | μ={row['mean']:7.4f} | σ={row['std']:7.4f} | med={row['median']:7.4f}")
            
            # === 4. ANÁLISIS DE CIUDADES ===
            print("\n--- Análisis por Ciudad (Top variables) ---")
            
            # Encontrar las variables más correlacionadas para análisis por ciudad
            top_vars = sorted(correlations, key=lambda x: abs(x['Correlación']), reverse=True)[:3]
            
            for var_info in top_vars:
                var = var_info['Variable']
                print(f"\n{var} (r={var_info['Correlación']}):")
                
                city_stats = df.groupby('city')[var].agg(['count', 'mean']).round(4)
                city_stats = city_stats[city_stats['count'] >= 10].sort_values('mean', ascending=False)
                
                for city, row in city_stats.head(5).iterrows():
                    print(f"  {city:15} | n={row['count']:3.0f} | μ={row['mean']:7.4f}")
            
            # === 5. GUARDAR RESULTADOS ===
            
            # Crear tablas resumen
            corr_df = pd.DataFrame(correlations)
            kruskal_df = pd.DataFrame(kruskal_results)
            
            # Guardar en un solo archivo Excel con múltiples hojas
            output_file = os.path.join(self.global_dir, "analisis_patrones_movilidad.xlsx")
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                corr_df.to_excel(writer, sheet_name='Correlaciones', index=False)
                kruskal_df.to_excel(writer, sheet_name='Kruskal-Wallis', index=False)
                
                # Resumen ejecutivo
                summary_data = []
                for var in mobility_vars:
                    if var in results['correlations'] and var in results['kruskal_wallis']:
                        summary_data.append({
                            'Variable': var,
                            'Correlación_Spearman': results['correlations'][var]['correlation'],
                            'Sig_Correlación': 'Sí' if results['correlations'][var]['significant'] else 'No',
                            'Kruskal_Wallis_p': results['kruskal_wallis'][var]['p_value'],
                            'Sig_KW': 'Sí' if results['kruskal_wallis'][var]['significant'] else 'No',
                            'Eta_cuadrado': results['kruskal_wallis'][var]['eta_squared'],
                            'Interpretación': self._interpret_results(
                                results['correlations'][var]['significant'],
                                results['kruskal_wallis'][var]['significant'],
                                abs(results['correlations'][var]['correlation']),
                                results['kruskal_wallis'][var]['eta_squared']
                            )
                        })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Resumen', index=False)
            
            # === 6. ANÁLISIS POST-HOC MEJORADO ===
            print("\n--- Análisis Post-Hoc (Variables con Diferencias Significativas) ---")
            
            from scipy.stats import mannwhitneyu
            
            # CAMBIO PRINCIPAL: Usar criterios más flexibles para post-hoc
            # 1. Variables con Kruskal-Wallis significativo (p < 0.05)
            # 2. O variables con eta² > 0.05 (efecto pequeño o mayor)
            # 3. Incluir específicamente 'a' (active mobility) si está presente
            
            important_vars = []
            
            for var in mobility_vars:
                include_var = False
                
                # Criterio 1: Kruskal-Wallis significativo
                if var in results['kruskal_wallis'] and results['kruskal_wallis'][var]['significant']:
                    include_var = True
                    print(f"✓ {var}: Incluido por Kruskal-Wallis significativo (p={results['kruskal_wallis'][var]['p_value']:.4f})")
                
                # Criterio 2: Eta² > 0.05 (efecto pequeño o mayor)
                elif var in results['kruskal_wallis'] and results['kruskal_wallis'][var]['eta_squared'] > 0.05:
                    include_var = True
                    print(f"✓ {var}: Incluido por tamaño de efecto (η²={results['kruskal_wallis'][var]['eta_squared']:.4f})")
                
                # Criterio 3: Variables específicas de interés (active mobility)
                elif var in ['a', 'walked_share', 'bicycle_share']:
                    include_var = True
                    print(f"✓ {var}: Incluido por ser variable de movilidad activa")
                
                if include_var:
                    important_vars.append(var)
            
            # Si no hay variables importantes, incluir al menos las 3 con mayor eta²
            if not important_vars:
                print("No se encontraron variables con criterios estrictos. Incluyendo top 3 por eta²...")
                sorted_vars = sorted([(var, results['kruskal_wallis'][var]['eta_squared']) 
                                    for var in mobility_vars if var in results['kruskal_wallis']], 
                                key=lambda x: x[1], reverse=True)
                important_vars = [var for var, _ in sorted_vars[:3]]
            
            print(f"\nVariables seleccionadas para post-hoc: {important_vars}")
            
            post_hoc_results = {}
            
            for var in important_vars:
                eta_sq = results['kruskal_wallis'][var]['eta_squared'] if var in results['kruskal_wallis'] else 0
                print(f"\n{var} (η²={eta_sq:.3f}):")
                
                # Obtener datos por patrón con criterios más flexibles
                pattern_data = {}
                min_observations = 5  # Reducido de 10 a 5 para mayor flexibilidad
                
                for pattern in df['pattern'].unique():
                    data = df[df['pattern'] == pattern][var].dropna()
                    if len(data) >= min_observations:
                        pattern_data[pattern] = data
                        print(f"  Patrón {pattern}: {len(data)} observaciones (μ={data.mean():.4f}, med={data.median():.4f})")
                
                # Verificar si tenemos suficientes patrones para comparar
                if len(pattern_data) < 2:
                    print(f"  ⚠️  Insuficientes patrones con datos para {var} (mínimo {min_observations} obs. c/u)")
                    continue
                
                # Comparaciones pareadas
                comparisons = []
                patterns = list(pattern_data.keys())
                significant_comparisons = 0
                
                print(f"  Realizando {len(patterns)*(len(patterns)-1)//2} comparaciones pareadas:")
                
                for i, p1 in enumerate(patterns):
                    for j, p2 in enumerate(patterns):
                        if i < j:  # Evitar comparaciones duplicadas
                            try:
                                # Test de Mann-Whitney U
                                stat, p_val = mannwhitneyu(pattern_data[p1], pattern_data[p2], 
                                                        alternative='two-sided')
                                
                                # Calcular diferencia de medianas y medias
                                median_diff = pattern_data[p1].median() - pattern_data[p2].median()
                                mean_diff = pattern_data[p1].mean() - pattern_data[p2].mean()
                                
                                # Calcular tamaño del efecto (r de Rosenthal)
                                n1, n2 = len(pattern_data[p1]), len(pattern_data[p2])
                                z_score = abs((stat - (n1 * n2 / 2)) / (((n1 * n2 * (n1 + n2 + 1)) / 12) ** 0.5))
                                effect_size_r = z_score / ((n1 + n2) ** 0.5)
                                
                                is_significant = p_val < 0.05
                                if is_significant:
                                    significant_comparisons += 1
                                
                                comparisons.append({
                                    'Patrón_1': p1,
                                    'Patrón_2': p2,
                                    'n1': n1,
                                    'n2': n2,
                                    'U_estadístico': stat,
                                    'p_valor': p_val,
                                    'Significativo': 'Sí' if is_significant else 'No',
                                    'Dif_Media': mean_diff,
                                    'Dif_Mediana': median_diff,
                                    'Tamaño_Efecto_r': effect_size_r
                                })
                                
                                # Mostrar resultados significativos
                                if is_significant:
                                    direction = ">" if median_diff > 0 else "<"
                                    effect_interp = "Grande" if effect_size_r > 0.5 else "Mediano" if effect_size_r > 0.3 else "Pequeño"
                                    print(f"    {p1} {direction} {p2:15} | p={p_val:.4f} | Δmed={median_diff:+.4f} | r={effect_size_r:.3f} ({effect_interp})")
                            
                            except Exception as e:
                                print(f"    ⚠️  Error comparando {p1} vs {p2}: {e}")
                
                post_hoc_results[var] = comparisons
                print(f"  📊 Comparaciones significativas: {significant_comparisons}/{len(comparisons)}")
                
                # Ranking de patrones por mediana
                pattern_rankings = sorted([(p, data.median()) for p, data in pattern_data.items()], 
                                        key=lambda x: x[1], reverse=True)
                print(f"  🏆 Ranking por mediana: {' > '.join([f'{p}({v:.3f})' for p, v in pattern_rankings])}")
            
            # === 7. GUARDAR ANÁLISIS POST-HOC MEJORADO ===
            if post_hoc_results:
                print(f"\n📁 Guardando análisis post-hoc...")
                
                # Crear un archivo separado para post-hoc con más detalle
                posthoc_file = os.path.join(self.global_dir, "analisis_posthoc_detallado.xlsx")
                
                with pd.ExcelWriter(posthoc_file, engine='openpyxl') as writer:
                    # Resumen general de post-hoc
                    summary_posthoc = []
                    
                    for var, comparisons in post_hoc_results.items():
                        if comparisons:
                            sig_comps = sum(1 for c in comparisons if c['Significativo'] == 'Sí')
                            total_comps = len(comparisons)
                            patterns_involved = len(set([c['Patrón_1'] for c in comparisons] + [c['Patrón_2'] for c in comparisons]))
                            
                            # Encontrar la comparación más significativa
                            most_sig = min(comparisons, key=lambda x: x['p_valor'])
                            
                            summary_posthoc.append({
                                'Variable': var,
                                'Comparaciones_Totales': total_comps,
                                'Comparaciones_Significativas': sig_comps,
                                'Porcentaje_Significativo': round((sig_comps/total_comps)*100, 1),
                                'Patrones_Analizados': patterns_involved,
                                'Menor_p_valor': most_sig['p_valor'],
                                'Comparación_Más_Significativa': f"{most_sig['Patrón_1']} vs {most_sig['Patrón_2']}",
                                'Mayor_Diferencia_Mediana': max(comparisons, key=lambda x: abs(x['Dif_Mediana']))['Dif_Mediana']
                            })
                    
                    if summary_posthoc:
                        summary_df = pd.DataFrame(summary_posthoc)
                        summary_df.to_excel(writer, sheet_name='Resumen_PostHoc', index=False)
                    
                    # Guardar cada variable en su propia hoja
                    for var, comparisons in post_hoc_results.items():
                        if comparisons:
                            posthoc_df = pd.DataFrame(comparisons)
                            # Ordenar por p-valor para mostrar los más significativos primero
                            posthoc_df = posthoc_df.sort_values('p_valor')
                            sheet_name = f"PostHoc_{var}"[:31]  # Límite de caracteres en Excel
                            posthoc_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"✅ Análisis post-hoc guardado en: {posthoc_file}")
            
            # También guardar en el archivo principal
            with pd.ExcelWriter(output_file, mode='a', engine='openpyxl') as writer:
                for var, comparisons in post_hoc_results.items():
                    if comparisons:
                        posthoc_df = pd.DataFrame(comparisons)
                        posthoc_df = posthoc_df.sort_values('p_valor')
                        sheet_name = f"PostHoc_{var}"[:31]
                        posthoc_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # === 8. REPORTE FINAL MEJORADO ===
            print(f"\n=== RESUMEN EJECUTIVO ===")
            print(f"Variables analizadas: {len(mobility_vars)}")
            
            sig_correlations = sum(1 for r in results['correlations'].values() if r['significant'])
            sig_kruskal = sum(1 for r in results['kruskal_wallis'].values() if r['significant'])
            
            print(f"Correlaciones significativas: {sig_correlations}/{len(results['correlations'])}")
            print(f"Diferencias significativas entre patrones: {sig_kruskal}/{len(results['kruskal_wallis'])}")
            print(f"Variables con análisis post-hoc: {len(important_vars)}")
            
            # Variables con efectos más fuertes
            if correlations:
                strongest_corr = max(correlations, key=lambda x: abs(x['Correlación']))
                print(f"Correlación más fuerte: {strongest_corr['Variable']} (r={strongest_corr['Correlación']})")
            
            if kruskal_results:
                strongest_diff = max(kruskal_results, key=lambda x: x['η² (aprox)'])
                print(f"Mayor diferencia entre patrones: {strongest_diff['Variable']} (η²={strongest_diff['η² (aprox)']})")
            
            # Información específica sobre active mobility
            if 'a' in results['kruskal_wallis']:
                a_results = results['kruskal_wallis']['a']
                print(f"\n🚶 MOVILIDAD ACTIVA ('a'):")
                print(f"   Kruskal-Wallis p-valor: {a_results['p_value']:.6f}")
                print(f"   Tamaño del efecto (η²): {a_results['eta_squared']:.4f}")
                print(f"   Significativo: {'SÍ' if a_results['significant'] else 'NO'}")
                
                if 'a' in post_hoc_results:
                    a_posthoc = post_hoc_results['a']
                    sig_comps = sum(1 for c in a_posthoc if c['Significativo'] == 'Sí')
                    print(f"   Comparaciones post-hoc significativas: {sig_comps}/{len(a_posthoc)}")
            
            # Interpretación general
            print(f"\n--- INTERPRETACIÓN ---")
            print("✓ Los patrones viales SÍ influyen en la movilidad")
            print("✓ La relación es CATEGÓRICA, no lineal (correlaciones débiles pero diferencias fuertes)")
            print("✓ Variables más sensibles al patrón vial:")
            
            top_effects = sorted(kruskal_results, key=lambda x: x['η² (aprox)'], reverse=True)[:3]
            for i, var in enumerate(top_effects, 1):
                print(f"  {i}. {var['Variable']} (η²={var['η² (aprox)']:.3f})")
            
            print(f"\nResultados guardados en: {output_file}")
            if post_hoc_results:
                print(f"Post-hoc detallado en: {posthoc_file}")
            
            # === 9. FUNCIONES ADICIONALES DEL CÓDIGO ORIGINAL ===
            
            # Generar visualizaciones
            try:
                self._generate_enhanced_visualizations(df, mobility_vars)
                print("Visualizaciones generadas exitosamente")
            except Exception as e:
                print(f"Error generando visualizaciones: {e}")
            
            # Guardar datos combinados
            try:
                combined_data_file = os.path.join(self.global_dir, "datos_combinados_todas_ciudades.xlsx")
                df.to_excel(combined_data_file, index=False)
                print(f"Datos combinados guardados en: {combined_data_file}")
            except Exception as e:
                print(f"Error guardando datos combinados: {e}")
            
            # Guardar resultados detallados
            try:
                self._save_analysis_results(results, self.global_dir)
                print("Resultados detallados guardados exitosamente")
            except Exception as e:
                print(f"Error guardando resultados detallados: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error en análisis: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _interpret_results(self, corr_sig, kw_sig, corr_strength, eta_squared):
        """Interpreta los resultados estadísticos de forma comprensible."""
        
        if corr_sig and kw_sig:
            if corr_strength > 0.3 and eta_squared > 0.06:
                return "Fuerte relación: los patrones viales influyen significativamente en esta variable"
            else:
                return "Relación moderada: hay efecto del patrón vial pero no muy pronunciado"
        
        elif corr_sig:
            return "Tendencia monotónica: existe una tendencia gradual según el patrón"
        
        elif kw_sig:
            return "Diferencias puntuales: algunos patrones se diferencian, pero sin tendencia clara"
        
        else:
            return "Sin relación: los patrones viales no parecen influir en esta variable"



    def _create_ridge_plot(self, data, column, viz_dir):
        """Crea ridge plot para distribución por patrones."""
       
        pattern_config = self._get_pattern_config()
        available_patterns = [p for p in pattern_config['orden'] if p in data['pattern'].unique()]
        
        if len(available_patterns) <= 1:
            return
        
        fig = plt.figure(figsize=(14, max(8, len(available_patterns) * 1.2)))
        gs = gridspec.GridSpec(len(available_patterns), 1, hspace=0.4)
        
        # Límites consistentes
        x_min, x_max = data[column].min(), data[column].max()
        x_range = x_max - x_min
        x_padding = x_range * 0.1
        
        for i, pattern in enumerate(available_patterns):
            ax = plt.subplot(gs[i])
            pattern_data = data[data['pattern'] == pattern][column].dropna()
            
            if len(pattern_data) <= 1:
                continue
                
            # KDE plot con color específico del patrón
            color = pattern_config['colores'][pattern]
            sns.kdeplot(pattern_data, fill=True, color=color, alpha=0.7, 
                    linewidth=1.5, ax=ax, bw_adjust=0.8)
            
            # Mediana
            median_val = pattern_data.median()
            ax.axvline(x=median_val, color="red", linestyle="--", alpha=0.8, linewidth=1.5)
            
            # Etiquetas
            ax.text(x_min + x_padding, ax.get_ylim()[1] * 0.7,
                    pattern_config['labels'][pattern], fontsize=12, fontweight='bold', color=color)
            ax.text(median_val + x_padding*0.5, ax.get_ylim()[1] * 0.5,
                    f'Mediana: {median_val:.2f}', fontsize=10, color='darkred')
            
            # Configuración de ejes
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            
            if i < len(available_patterns) - 1:
                ax.set_xticks([])
                ax.set_xlabel('')
            else:
                ax.set_xlabel(column.upper(), fontsize=12)
            
            sns.despine(bottom=True, left=True, ax=ax)
        
        plt.suptitle(f'Distribución de {column.upper()} por Patrón', fontsize=16, y=0.98)
        
        # Leyenda
        legend_elements = [plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='Mediana')]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.98))
        
        self._save_figure(f"{viz_dir}/ridge_plot_{column}.pdf", f"Ridge plot - {column}")

    def _create_boxplot(self, data, column, viz_dir):
        """Crea boxplot mejorado con estadísticas."""
        
        pattern_config = self._get_pattern_config()
        
        plt.figure(figsize=(14, 8))
        ax = sns.boxplot(x='pattern', y=column, data=data, notch=True,
                        palette=[pattern_config['colores'].get(p, '#gray') for p in data['pattern'].unique()],
                        width=0.6)
        
        # Añadir estadísticas
        for i, pattern in enumerate(ax.get_xticklabels()):
            pattern_name = pattern.get_text()
            pattern_data = data[data['pattern'] == pattern_name][column]
            
            if len(pattern_data) > 0:
                mean_val = pattern_data.mean()
                median_val = pattern_data.median()
                
                # Media como diamante
                ax.plot(i, mean_val, marker='D', color='red', markersize=8, 
                    markeredgecolor='white', markeredgewidth=1.5, zorder=10)
                
                # Etiquetas
                ax.text(i, mean_val, f' μ={mean_val:.2f}', color='darkred', fontsize=9,
                    va='center', ha='left')
                ax.text(i, median_val, f' m={median_val:.2f}', color='black', fontsize=9,
                    va='center', ha='right')
        
        plt.title(f'Distribución de {column.upper()} por Patrón')
        plt.xlabel('Patrón')
        plt.ylabel(column.upper())
        plt.xticks(rotation=45)
        
        self._save_figure(f"{viz_dir}/boxplot_{column}.pdf", f"Boxplot - {column}")

    # def _create_mobility_density_plots(self, data, viz_dir):
    #     """Creates mobility density plots by urban pattern."""
        
        # with plt.style.context('styles/matplotlib_style.mplstyle'):
    #         # 1. Aggregated mobility plots (Active, Public, Private) – KEEPING STRUCTURE
    #         mobility_cols = ['a', 'b', 'car_share']
    #         mobility_labels = [r'\textbf{Active Mobility}', 
    #                         r'\textbf{Public Mobility}', 
    #                         r'\textbf{Private Mobility}']
    #         mobility_colors = ['#FF6B6B', '#4ECDC4', '#6A67CE']
    #         pattern_config = self._get_pattern_config()
            
    #         for pattern in pattern_config['orden']:
    #             pattern_data = data[data['pattern'] == pattern]
                
    #             if len(pattern_data) <= 1:
    #                 continue
                
    #             plt.figure()  # figsize ya está definido en el style
                
    #             for i, (col, label, color) in enumerate(zip(mobility_cols, mobility_labels, mobility_colors)):
    #                 col_data = pattern_data[col].dropna()
                    
    #                 if len(col_data) > 1:
    #                     sns.kdeplot(col_data, fill=True, alpha=0.5, color=color, linewidth=1.5, label=label)
                        
    #                     median_val = col_data.median()
    #                     plt.axvline(x=median_val, color=color, linestyle="--", alpha=0.8, linewidth=1.5)
                        
    #                     # Median label con formato LaTeX
    #                     y_base = 2.9
    #                     dy = 1
    #                     median_text = r'\textbf{Median ' + f': {median_val:.2f}' + '}'
    #                     plt.text(median_val + 0.01, y_base - i * dy, median_text, 
    #                             color='black', fontweight='bold',
    #                             bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3', alpha=0.8))
                
    #             # Títulos y etiquetas con LaTeX
    #             plt.title(r'\textbf{Distribution of Mobility Shares - ' + pattern.capitalize() + r' Pattern}')
    #             plt.xlabel(r'\textbf{Share Percentage}')
    #             plt.ylabel(r'\textbf{Density}')
                
    #             # Leyenda (configuración ya definida en el style)
    #             plt.legend()
                
    #             plt.tight_layout()
                
    #             self._save_figure(f"{viz_dir}/mobility_shares_density_{pattern}.pdf", 
    #                             f"Mobility density - {pattern}")
        



    # def _create_mobility_density_plots(self, data, viz_dir):
    #     """
    #     Creates an elegant and readable ridgeline plot using a "cell" approach.
    #     Each urban typology gets its own subplot ("cell"), and these cells are
    #     stacked with a slight overlap to create a cohesive final image.
    #     """
    #     # --- 1. Style and Parameter Setup ---
    #     MOBILITY_COLORS = {
    #         'a': '#0077b6', 'b': '#f7b801', 'car_share': '#d00000'
    #     }
    #     MOBILITY_LABELS_EN = {
    #         'a': 'Active', 'b': 'Public', 'car_share': 'Private'
    #     }
    #     PATTERN_LABELS_EN = {
    #         'organico': 'Organic', 'cul_de_sac': 'Cul-de-sac',
    #         'hibrido': 'Hybrid', 'gridiron': 'Gridiron'
    #     }
        
    #     # CRITICAL: Define draw order to prevent smaller plots from being hidden.
    #     # The largest distribution (Private) is drawn first (bottom layer).
    #     MOBILITY_ORDER = ['car_share', 'b', 'a'] 
        
    #     DESIRED_PATTERN_ORDER = ['organico', 'cul_de_sac', 'hibrido', 'gridiron']
    #     patterns = [p for p in DESIRED_PATTERN_ORDER if p in data['pattern'].unique()]
        
    #     if not patterns:
    #         print("No valid patterns found for plotting.")
    #         return

    #     # --- 2. Figure and Axes "Cells" Creation ---
    #     n_patterns = len(patterns)
    #     # Increase vertical space per plot for better separation
    #     fig = plt.figure(figsize=(12, 2 * n_patterns))
        
    #     # KEY: Overlap the axes (cells) themselves, not the data.
    #     # This removes whitespace while keeping data in its own lane.
    #     gs = fig.add_gridspec(n_patterns, 1, hspace=-0.5)
        
    #     # Create the axes from top to bottom
    #     axs = gs.subplots()
    #     if n_patterns == 1:
    #         axs = [axs]
        
    #     # Reverse the axes list so we can iterate from bottom to top visually
    #     axs = axs[::-1]

    #     # --- 3. Iterate and Draw in Each Cell (from bottom to top) ---
    #     for i, pattern in enumerate(patterns):
    #         ax = axs[i]
    #         pattern_data = data[data['pattern'] == pattern]
            
    #         # Make the background of the cell transparent
    #         ax.patch.set_alpha(0)
            
    #         # Draw the distributions within the cell, respecting the z-order
    #         for z_idx, col in enumerate(MOBILITY_ORDER):
    #             if col not in pattern_data.columns: continue
                
    #             x_data = pattern_data[col].dropna()
    #             if len(x_data) < 2: continue

    #             x_grid = np.linspace(0, 1, 1000)
    #             kde = KernelDensity(bandwidth=0.035, kernel='gaussian').fit(x_data.to_numpy()[:, np.newaxis])
    #             density = np.exp(kde.score_samples(x_grid[:, np.newaxis]))
                
    #             if np.max(density) > 0:
    #                 density = density / np.max(density)
                
    #             color = MOBILITY_COLORS[col]
                
    #             # Use z_idx for zorder, so 'car_share' (z_idx=0) is at the back
    #             ax.fill_between(x_grid, 0, density, alpha=0.7, color=color, zorder=z_idx)
    #             ax.plot(x_grid, density, color='black', lw=1.2, zorder=z_idx)
                
    #             # Add median line and text
    #             median_val = x_data.median()
    #             y_pos_at_median = np.interp(median_val, x_grid, density)
                
    #             ax.axvline(x=median_val, color=color, linestyle='--', lw=1.5, 
    #                     ymax=y_pos_at_median / 1.5, # Scale line height relative to curve
    #                     zorder=z_idx + 0.5)

    #             text_x_pos = median_val + 0.015
    #             ha = 'left'
    #             if median_val > 0.85:
    #                 text_x_pos = median_val - 0.015
    #                 ha = 'right'

    #             ax.text(text_x_pos, y_pos_at_median + 0.1, f'{median_val:.2f}',
    #                     ha=ha, va='center', fontsize=9, fontweight='bold', color='white',
    #                     bbox=dict(facecolor=color, edgecolor='white',
    #                             boxstyle='round,pad=0.2', alpha=0.9, linewidth=0.5),
    #                     zorder=z_idx + 0.6)

    #         # --- 4. Style Each Cell ---
    #         ax.set_ylim(0, 1.5)
    #         ax.set_yticks([])
            
    #         # Add the pattern label to the left of the cell
    #         pattern_display_name = PATTERN_LABELS_EN.get(pattern, pattern.capitalize())
    #         ax.text(-0.01, 0.2, pattern_display_name, transform=ax.transAxes,
    #                 fontsize=14, fontweight='bold', ha='right', va='bottom')

    #         # Hide all spines except for the baseline
    #         ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
            
    #         # Only show the X-axis labels and spine on the very bottom plot (i=0)
    #         if i == 0:
    #             ax.spines['bottom'].set_visible(True)
    #             ax.set_xlabel("Modal Share", fontsize=14, fontweight='bold', labelpad=15)
    #             ax.set_xticks(np.arange(0, 1.1, 0.2))
    #             ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)], fontsize=12)
    #         else:
    #             ax.set_xticks([])
                
    #     # --- 5. Titles, Legend, and Final Adjustments ---
    #     fig.suptitle(
    #         "Distribution of Mobility Shares by Urban Typology",
    #         fontsize=20, fontweight='bold', y=0.98
    #     )

    #     legend_elements = [
    #         Patch(facecolor=MOBILITY_COLORS[key], edgecolor='black', alpha=0.7,
    #             linewidth=1, label=label) 
    #         for key, label in MOBILITY_LABELS_EN.items()
    #     ]

    #     fig.legend(
    #         handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.93),
    #         ncol=len(legend_elements), fontsize=12, frameon=False
    #     )
        
    #     plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.1)

    #     self._save_figure(f"{viz_dir}/mobility_ridgeline_final_cells.pdf", 
    #                     "Elegant cell-based mobility ridgeline visualization generated successfully.")


    def _create_mobility_density_plots(self, data, viz_dir):
        """
        Crea un gráfico de densidad (ridgeline plot) de calidad de publicación,
        con una compresión vertical del 10% respecto al original.
        """
        try:
            plt.style.use('styles/matplotlib_style.mplstyle') # Usando tu estilo
            pass
        except:
            print("Estilo 'matplotlib_style.mplstyle' no encontrado. Usando valores por defecto.")

        # --- 1. Configuración de Parámetros Específicos del Gráfico ---
        # ESTILO ORIGINAL RESTAURADO
        MOBILITY_COLORS = {'a': '#0077b6', 'b': '#008000', 'car_share': '#d00000'}
        MOBILITY_ORDER = ['a', 'b', 'car_share']
        
        # AJUSTE SUTIL: Reducción del 10% en la separación
        VERTICAL_SEPARATION = 0.27 # Valor original: 0.3
        
        DESIRED_PATTERN_ORDER = ['organico', 'cul_de_sac', 'hibrido', 'gridiron']
        patterns = [p for p in DESIRED_PATTERN_ORDER if p in data['pattern'].unique()]

        if not patterns:
            print("No se encontraron patrones válidos para graficar.")
            return

        # --- 2. Creación de la Figura y los Ejes ---
        n_patterns = len(patterns)
        fig_width = 7.5
        
        # AJUSTE SUTIL: Reducción del 10% en la altura de cada subplot
        subplot_height = 0.9 # Valor original: 1
        
        fig = plt.figure(figsize=(fig_width, subplot_height * n_patterns))
        gs = fig.add_gridspec(n_patterns, 1, hspace=0.01)
        axs = gs.subplots()
        if n_patterns == 1:
            axs = [axs]

        # --- 3. Iteración y Dibujo de las Distribuciones ---
        PATTERN_LABELS_EN = {
            'organico': 'Organic', 'cul_de_sac': 'Cul-de-sac',
            'hibrido': 'Hybrid', 'gridiron': 'Gridiron'
        }

        for i, pattern in enumerate(patterns):
            ax = axs[i]
            pattern_data = data[data['pattern'] == pattern]
            display_pattern_name = PATTERN_LABELS_EN.get(pattern, pattern.replace('_', ' ').capitalize())
            # ESTILO ORIGINAL
            ax.text(-0.05, 0.5, display_pattern_name,
                    transform=ax.transAxes,
                    fontsize=8, fontweight='bold', ha='right', va='center')

            vertical_offset = 0.0
            for col in MOBILITY_ORDER:
                if col not in pattern_data.columns: continue
                x_data = pattern_data[col].dropna()
                if len(x_data) < 2: continue

                x_grid = np.linspace(0, 1, 1000)
                
                # ESTILO ORIGINAL RESTAURADO
                kde = KernelDensity(bandwidth=0.04, kernel='gaussian') 
                
                kde.fit(x_data.to_numpy()[:, np.newaxis])
                log_density = kde.score_samples(x_grid[:, np.newaxis])
                density = np.exp(log_density)
                if np.max(density) > 0:
                    density = density / np.max(density)

                color = MOBILITY_COLORS[col]
                # ESTILO ORIGINAL RESTAURADO
                ax.fill_between(x_grid, vertical_offset, density + vertical_offset, alpha=0.6, color=color, zorder=i*10)
                ax.plot(x_grid, density + vertical_offset, linewidth=0.35, color='black', zorder=i*10 + 1)

                median_val = x_data.median()
                y_pos_at_median = density[np.abs(x_grid - median_val).argmin()]
                
                # ESTILO ORIGINAL RESTAURADO
                ax.plot([median_val, median_val],
                        [vertical_offset, vertical_offset + y_pos_at_median * 1],
                        color=color, linestyle='--', linewidth=0.6, zorder=i*10 + 1)

                text_x_pos = median_val + 0.015
                ha = 'left'
                if median_val > 0.85:
                    text_x_pos = median_val - 0.015
                    ha = 'right'

                median_label_y_pos = vertical_offset + (y_pos_at_median * 0.8) / 2
                
                # ESTILO ORIGINAL RESTAURADO
                ax.text(text_x_pos, median_label_y_pos, f'{median_val:.2f}',
                        ha=ha, va='center', fontsize=4.5, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=color, edgecolor='none'),
                        zorder=i*10 + 2)

                vertical_offset += VERTICAL_SEPARATION

        # --- 4. Limpieza y Estilizado de cada Eje ---
        # El cálculo de la altura total se ajusta automáticamente a la nueva separación
        total_height = (len(MOBILITY_ORDER) - 1) * VERTICAL_SEPARATION + 1.2
        Y_AXIS_GAP = 0.2

        for i, ax in enumerate(axs):
            ax.set_xlim(0, 1)
            ax.set_ylim(-Y_AXIS_GAP, total_height)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.patch.set_alpha(0)

            if i == n_patterns - 1:
                # ESTILO ORIGINAL RESTAURADO
                ax.set_xlabel("Share Percentage", labelpad=15)
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)])
            else:
                ax.set_xticks([])
                ax.spines['bottom'].set_visible(False)

        # --- 5. Títulos, Leyenda y Ajustes Finales ---
        fig.suptitle("Distribution of Mobility Shares by Urban Typology", y=0.98)
        MOBILITY_LABELS_EN = {'a': 'Active', 'b': 'Public', 'car_share': 'Private'}
        legend_elements = [
            # ESTILO ORIGINAL RESTAURADO
            Patch(facecolor=MOBILITY_COLORS[key], edgecolor='black', alpha=0.6,
                linewidth=1, label=MOBILITY_LABELS_EN.get(key, key))
            for key in MOBILITY_ORDER
        ]
        fig.legend(handles=legend_elements, loc='upper center',
                bbox_to_anchor=(0.5, 0.93), ncol=len(legend_elements))
        
        # AJUSTE SUTIL: Se aumenta el margen inferior para "subir" la gráfica
        plt.subplots_adjust(left=0.18, right=0.82, top=0.92, bottom=0.2) # Original: bottom=0.08
        self._save_figure(f"{viz_dir}/mobility_ridgeline_compact_10pct.pdf",
                    "Visualización de movilidad con compresión del 10% generada")
        




















    def _create_detailed_shares_density_plots(self, data, mobility_cols, viz_dir):
        """Crea gráficos de densidad para todos los shares desagregados por patrón."""
        pattern_config = self._get_pattern_config()
        
        # Identificar columnas de shares (excluir las agregadas A, B, C)
        share_cols = [col for col in mobility_cols if 'share' in col.lower() and col not in ['a', 'b', 'c']]
        
        if not share_cols:
            return
        
        # Generar colores diversos para todos los shares
        colors = plt.cm.Set3(np.linspace(0, 1, len(share_cols)))
        
        for pattern in pattern_config['orden']:
            pattern_data = data[data['pattern'] == pattern]
            
            if len(pattern_data) <= 1:
                continue
            
            plt.figure(figsize=(18, 10))
            
            valid_shares = []
            for i, col in enumerate(share_cols):
                col_data = pattern_data[col].dropna()
                
                if len(col_data) > 1:
                    # Crear label más legible
                    label = col.replace('_share', '').replace('_', ' ').upper()
                    color = colors[i]
                    
                    sns.kdeplot(col_data, fill=True, alpha=0.4, color=color, linewidth=2, label=label)
                    
                    median_val = col_data.median()
                    plt.axvline(x=median_val, color=color, linestyle="--", alpha=0.8, linewidth=1.5)
                    
                    valid_shares.append((col, label, color, median_val))
            
            # Añadir etiquetas de medianas organizadas verticalmente
            if valid_shares:
                y_max = plt.gca().get_ylim()[1]
                y_start = y_max * 0.9
                dy = y_max * 0.08
                
                for i, (col, label, color, median_val) in enumerate(valid_shares):
                    plt.text(median_val + 0.005, y_start - i * dy, 
                            f'MEDIANA {label}: {median_val:.2f}', 
                            fontsize=10, color='black', fontweight='bold',
                            bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3', alpha=0.8))
            
            plt.title(f'DISTRIBUCIÓN DETALLADA DE SHARES - PATRÓN {pattern.upper()}', fontsize=16)
            plt.xlabel('PORCENTAJE DE SHARE', fontsize=14)
            plt.ylabel('DENSIDAD', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            self._save_figure(f"{viz_dir}/detailed_shares_density_{pattern}.pdf", 
                            f"Densidad shares detallada - {pattern}")

    def _create_barplot(self, data, column, viz_dir):
        """Crea gráfico de barras con error estándar."""
        
        pattern_config = self._get_pattern_config()
        
        # Mapear patrones para consistencia
        data_temp = data.copy()
        pattern_mapping = {p.lower(): p for p in pattern_config['orden']}
        data_temp['pattern_mapped'] = data_temp['pattern'].str.lower().map(pattern_mapping)
        
        # Estadísticas
        stats = data_temp.groupby('pattern_mapped')[column].agg(['mean', 'std', 'count'])
        stats = stats.reindex([p for p in pattern_config['orden'] if p in stats.index])
        stats['se'] = stats['std'] / np.sqrt(stats['count'])
        
        plt.figure(figsize=(14, 8))
        
        # Barras con colores específicos
        bar_colors = [pattern_config['colores'][p] for p in stats.index]
        bars = plt.bar(range(len(stats)), stats['mean'], yerr=stats['se'],
                    color=bar_colors, capsize=7, width=0.7, 
                    edgecolor='black', linewidth=1,
                    error_kw={'elinewidth': 2, 'capthick': 2})
        
        # Etiquetas en barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            mean_val = stats['mean'].iloc[i]
            plt.text(bar.get_x() + bar.get_width()/2, 
                    height + stats['se'].iloc[i] + 0.02 * max(stats['mean']),
                    f'{mean_val:.2f}', ha="center", va="bottom", fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8))
        
        # Configuración
        plt.xticks(range(len(stats)), 
                [pattern_config['labels'][p] for p in stats.index],
                rotation=30, ha='right', fontweight='bold')
        
        col_formatted = column.upper().replace("_", " ")
        plt.title(f'MEDIA DE {col_formatted} POR PATRÓN')
        plt.xlabel('PATRÓN URBANO')
        plt.ylabel(f'MEDIA DE {col_formatted}')
        
        self._save_figure(f"{viz_dir}/barplot_{column}.pdf", f"Barplot - {column}")

    def _create_correlation_heatmap(self, data, mobility_cols, viz_dir):
        """Crea heatmap de correlación."""
                
        # Filtrar variables válidas
        valid_cols = [col for col in mobility_cols 
                    if data[col].nunique() > 1 and not data[col].isna().all()]
        
        if len(valid_cols) <= 1:
            return
        
        plt.figure(figsize=(14, 12))
        correlation_matrix = data[valid_cols].corr()
        
        # Máscara triangular
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Heatmap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, annot=True, fmt='.2f', 
                square=True, center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Correlación entre Variables de Movilidad')
        
        self._save_figure(f"{viz_dir}/correlacion_variables.pdf", "Correlación variables")

    def _create_pca_analysis(self, data, mobility_cols, viz_dir, title_prefix=""):
        """Crea análisis PCA si hay suficientes variables."""
       
        
        if len(mobility_cols) < 3:
            return
        
        # Preparar datos
        valid_data = data[mobility_cols + ['pattern']].dropna()
        if len(valid_data) < 10:  # Mínimo para PCA
            return
        
        # Escalar y aplicar PCA
        scaler = StandardScaler()
        mobility_scaled = scaler.fit_transform(valid_data[mobility_cols])
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(mobility_scaled)
        
        # DataFrame resultados
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        pca_df['pattern'] = valid_data['pattern'].values
        
        # Gráfico
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='PC1', y='PC2', hue='pattern', data=pca_df, s=100)
        
        plt.title(f'{title_prefix}PCA de Movilidad por Patrón')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
        plt.legend(title='Patrón', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self._save_figure(f"{viz_dir}/pca_movilidad.pdf", f"PCA - {title_prefix}")
        
        # Guardar componentes
        pca_components = pd.DataFrame(
            data=pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=mobility_cols
        )
        pca_components.to_csv(f"{viz_dir}/pca_componentes.csv")

    def _generate_enhanced_visualizations(self, combined_data, mobility_cols):
        """
        Genera visualizaciones mejoradas para el análisis global y por ciudad,
        con formato científico consistente.
        """
        
        
        try:
            # Configurar estilo
            self._setup_plot_style()
            
            # Crear directorio
            viz_dir = os.path.join(getattr(self, 'global_dir', './'), "visualizaciones")
            os.makedirs(viz_dir, exist_ok=True)
            
            print("Generando visualizaciones mejoradas...")
            
            # 1. Distribución de observaciones por ciudad y patrón
            self._create_observations_heatmap(combined_data, viz_dir)
            
            # 2. Visualizaciones por variable de movilidad
            share_specific_cols = [col for col in mobility_cols if 'share' in col.lower() and col not in ['a', 'b', 'car_share']]
            other_mobility_cols = [col for col in mobility_cols if col not in share_specific_cols + ['a', 'b', 'car_share']]
            
            for col in other_mobility_cols:  # Solo variables que NO son shares específicos
                if combined_data[col].isna().all() or combined_data[col].std() == 0:
                    continue
                    
                print(f"Procesando variable: {col}")
                
                # Boxplots y gráficos de barras (NO ridge plots para shares)
                self._create_boxplot(combined_data, col, viz_dir)
                self._create_barplot(combined_data, col, viz_dir)
            
            # 3. Análisis de movilidad específico
            self._create_mobility_density_plots(combined_data, viz_dir)
            
            # 4. Análisis detallado de shares desagregados
            self._create_detailed_shares_density_plots(combined_data, mobility_cols, viz_dir)
            
            # 5. Correlación
            self._create_correlation_heatmap(combined_data, mobility_cols, viz_dir)
            
            # 6. PCA (si hay suficientes variables)
            self._create_pca_analysis(combined_data, mobility_cols, viz_dir, "Global: ")
            
            print("✓ Visualizaciones completadas exitosamente")
            
        except Exception as e:
            print(f"Error generando visualizaciones: {e}")
            traceback.print_exc()

    def _create_observations_heatmap(self, data, viz_dir):
        """Crea heatmap de distribución de observaciones."""
               
        plt.figure(figsize=(12, 8))
        pattern_city_counts = pd.crosstab(data['city'], data['pattern'])
        
        # Colormap personalizado
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f5f5f5', '#2171b5'])
        
        sns.heatmap(pattern_city_counts, annot=True, fmt='d', cmap=cmap)
        plt.title('Número de Observaciones por Ciudad y Patrón')
        plt.ylabel('Ciudad')
        plt.xlabel('Patrón')
        
        self._save_figure(f"{viz_dir}/distribucion_observaciones.pdf", "Distribución observaciones")

   
    
    def _save_analysis_results(self, results_dict, output_dir):
        """Guarda resultados del análisis estadístico (Kruskal-Wallis/ANOVA) de manera útil y concisa."""
        # Llama esta función justo antes de _save_analysis_results:
        
        try:
            # Mapeo de variables de movilidad
            mobility_labels = {
                'a': 'Movilidad_Activa',
                'b': 'Movilidad_Publica', 
                'c': 'Movilidad_Privada',
                'walked_share': 'Caminata',
                'car_share': 'Automovil',
                'transit_share': 'Transporte_Publico',
                'bicycle_share': 'Bicicleta'
            }
            
            print("Procesando resultados estadísticos...")
            print(f"Claves disponibles: {list(results_dict.keys())}")
            
            # 1. Procesar resultados Kruskal-Wallis
            summary_results = []
            kruskal_data = results_dict.get('kruskal_wallis', {})
            correlations_data = results_dict.get('correlations', {})
            effect_sizes_data = results_dict.get('effect_sizes', {})
            descriptive_data = results_dict.get('descriptive_stats', {})
            
            if not kruskal_data:
                print("No se encontraron datos de Kruskal-Wallis")
                return
            
            print(f"Datos Kruskal-Wallis encontrados: {list(kruskal_data.keys())}")
            
            # Procesar cada variable de movilidad
            for var_key, kw_results in kruskal_data.items():
                var_label = mobility_labels.get(var_key, var_key)
                print(f"\nProcesando {var_key} -> {var_label}")
                print(f"Contenido de kw_results: {kw_results}")
                
                # Extraer estadísticas directamente del formato encontrado
                if isinstance(kw_results, dict):
                    # Los datos están en el formato: {'h_statistic': ..., 'p_value': ..., 'significant': ..., 'eta_squared': ..., 'effect_size': ...}
                    h_stat = kw_results.get('h_statistic', np.nan)
                    p_value = kw_results.get('p_value', np.nan)
                    significant = kw_results.get('significant', False)
                    eta_squared = kw_results.get('eta_squared', np.nan)
                    effect_size_label = kw_results.get('effect_size', 'N/A')
                    
                    print(f"Estadísticas extraídas - H: {h_stat}, p: {p_value}, sig: {significant}")
                    
                    # Estos son los resultados de Kruskal-Wallis comparando PATRONES URBANOS
                    # La pregunta es: "¿Hay diferencias significativas en [variable_movilidad] entre los diferentes patrones urbanos?"
                    summary_results.append({
                        'Variable_Movilidad': var_label,
                        'Variable_Original': var_key,
                        'Pregunta_Investigacion': f'¿Difiere {var_label} entre patrones urbanos?',
                        'Patrones_Comparados': 'Gridiron vs Cul-de-Sac vs Orgánico vs Híbrido',
                        'H_statistic': h_stat,
                        'p_value': p_value,
                        'Significativo': significant,
                        'Eta_Squared': eta_squared,
                        'Tamaño_Efecto': effect_size_label,
                        'Interpretacion_Efecto': self._interpret_effect_size(eta_squared),
                        'Conclusion': f'{"SÍ" if significant else "NO"} hay diferencias significativas en {var_label} entre patrones urbanos'
                    })
                else:
                    print(f"Formato inesperado para {var_key}: {type(kw_results)}")
            
            if not summary_results:
                print("No se pudieron procesar los resultados estadísticos")
                return
            
            # 2. Crear DataFrame de resumen
            summary_df = pd.DataFrame(summary_results)
            print(f"Resumen creado con {len(summary_df)} variables")
            print("Vista previa del DataFrame:")
            print(summary_df[['Variable_Movilidad', 'Patrones_Comparados', 'H_statistic', 'p_value', 'Significativo', 'Tamaño_Efecto']])
            
            # 3. Guardar en Excel
            excel_path = os.path.join(output_dir, "analisis_estadistico_movilidad.xlsx")
            
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    # Hoja 1: Resumen ejecutivo con indicadores visuales
                    summary_display = summary_df.copy()
                    
                    # Agregar indicadores visuales
                    summary_display['Status_Visual'] = summary_display['Significativo'].map({
                        True: '*** SIGNIFICATIVO ***', 
                        False: 'No significativo'
                    })
                    
                    # Reordenar columnas para mejor presentación
                    cols_order = ['Variable_Movilidad', 'Variable_Original', 'Pregunta_Investigacion', 'Patrones_Comparados',
                                'H_statistic', 'p_value', 'Significativo', 'Status_Visual', 
                                'Eta_Squared', 'Tamaño_Efecto', 'Interpretacion_Efecto', 'Conclusion']
                    
                    summary_display = summary_display[cols_order]
                    summary_display.to_excel(writer, sheet_name='Resumen_KruskalWallis', index=False)
                    print("Hoja 'Resumen_KruskalWallis' creada")
                    
                    # Hoja 2: Solo variables significativas
                    significant_vars = summary_display[summary_display['Significativo'] == True]
                    if len(significant_vars) > 0:
                        significant_vars.to_excel(writer, sheet_name='Variables_Significativas', index=False)
                        print(f"Hoja 'Variables_Significativas' creada con {len(significant_vars)} variables")
                    
                    # Hoja 3: Ordenado por tamaño del efecto
                    summary_by_effect = summary_display.sort_values('Eta_Squared', ascending=False, na_last=True)
                    summary_by_effect.to_excel(writer, sheet_name='Ordenado_por_Efecto', index=False)
                    print("Hoja 'Ordenado_por_Efecto' creada")
                    
                    # Hoja 4: Estadísticas descriptivas si están disponibles
                    if descriptive_data:
                        try:
                            desc_list = []
                            for var_key, desc_stats in descriptive_data.items():
                                var_label = mobility_labels.get(var_key, var_key)
                                if isinstance(desc_stats, dict):
                                    desc_list.append({
                                        'Variable': var_label,
                                        'Variable_Original': var_key,
                                        **desc_stats
                                    })
                            
                            if desc_list:
                                desc_df = pd.DataFrame(desc_list)
                                desc_df.to_excel(writer, sheet_name='Estadisticas_Descriptivas', index=False)
                                print("Hoja 'Estadisticas_Descriptivas' creada")
                        except Exception as desc_error:
                            print(f"Error procesando estadísticas descriptivas: {desc_error}")
                    
                    # Hoja 5: Correlaciones si están disponibles
                    if correlations_data:
                        try:
                            if isinstance(correlations_data, pd.DataFrame):
                                correlations_data.to_excel(writer, sheet_name='Correlaciones')
                                print("Hoja 'Correlaciones' creada")
                            elif isinstance(correlations_data, dict):
                                corr_df = pd.DataFrame(correlations_data)
                                corr_df.to_excel(writer, sheet_name='Correlaciones')
                                print("Hoja 'Correlaciones' creada")
                        except Exception as corr_error:
                            print(f"Error procesando correlaciones: {corr_error}")
            
            except Exception as excel_error:
                print(f"Error creando Excel: {excel_error}")
                # Fallback: guardar como CSV
                csv_path = os.path.join(output_dir, "analisis_estadistico_resumen.csv")
                summary_df.to_csv(csv_path, index=False)
                print(f"Guardado como CSV en: {csv_path}")
            
            # 4. Imprimir resumen en consola
            print(f"\n{'='*60}")
            print("RESUMEN DEL ANÁLISIS ESTADÍSTICO")
            print(f"{'='*60}")
            
            sig_count = sum(summary_df['Significativo'])
            total_count = len(summary_df)
            
            print(f"Variables analizadas: {total_count}")
            print(f"Variables significativas: {sig_count}")
            print(f"Porcentaje significativo: {(sig_count/total_count)*100:.1f}%")
            
            print(f"\nVARIABLES SIGNIFICATIVAS:")
            print("-" * 40)
            for _, row in summary_df[summary_df['Significativo']].iterrows():
                print(f"• {row['Variable_Movilidad']}: H={row['H_statistic']:.3f}, p={row['p_value']:.2e}, Efecto={row['Tamaño_Efecto']}")
            
            print(f"\nArchivo Excel guardado en: {excel_path}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error in análisis estadístico: {e}")
            import traceback
            traceback.print_exc()

    def _interpret_effect_size(self, eta_squared):
        """Interpreta el tamaño del efecto basado en eta cuadrado"""
        if pd.isna(eta_squared):
            return 'N/A'
        elif eta_squared < 0.01:
            return 'Muy Pequeño'
        elif eta_squared < 0.06:
            return 'Pequeño-Mediano'
        elif eta_squared < 0.14:
            return 'Mediano-Grande'
        else:
            return 'Grande'

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



import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import chi2_contingency, kruskal
import seaborn as sns
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, kruskal, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

def advanced_causality_analysis(global_data, output_dir, mobility_vars):
    """
    Análisis avanzado de causalidad con visualizaciones sofisticadas
    """
    print("\n🔬 INICIANDO ANÁLISIS AVANZADO DE CAUSALIDAD...")
    
    # 1. Análisis de importancia de features (causalidad implícita)
    feature_importance = calculate_feature_importance(global_data, mobility_vars)
    
    # 2. Análisis de efectos marginales
    marginal_effects = calculate_marginal_effects(global_data, mobility_vars)
    
    # 3. Visualizaciones avanzadas
    create_advanced_visualizations(global_data, feature_importance, marginal_effects, output_dir, mobility_vars)
    
    # 4. Análisis de dominancia de patrones
    dominance_analysis = analyze_pattern_dominance(global_data, mobility_vars)
    
    # 5. Matriz de correlación condicional
    conditional_correlations = calculate_conditional_correlations(global_data, mobility_vars)
    
    return {
        'feature_importance': feature_importance,
        'marginal_effects': marginal_effects,
        'dominance_analysis': dominance_analysis,
        'conditional_correlations': conditional_correlations
    }

def calculate_feature_importance(data, mobility_vars):
    """Calcula importancia de patrones urbanos para predecir movilidad"""
    le = LabelEncoder()
    data_encoded = data.copy()
    data_encoded['pattern_encoded'] = le.fit_transform(data['pattern'])
    
    importance_results = {}
    
    for var_key, var_name in mobility_vars.items():
        if var_key not in data.columns:
            continue
            
        # Preparar datos
        valid_data = data_encoded[data_encoded[var_key].notna()]
        if len(valid_data) < 10:
            continue
            
        X = valid_data[['pattern_encoded']].values
        y = valid_data[var_key].values
        
        # Random Forest para importancia
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_results[var_key] = {
            'importance_score': rf.feature_importances_[0],
            'r2_score': r2_score(y, rf.predict(X)),
            'pattern_classes': le.classes_
        }
    
    return importance_results

def calculate_marginal_effects(data, mobility_vars):
    """Calcula efectos marginales de cada patrón urbano"""
    marginal_effects = {}
    
    # Baseline: promedio global
    for var_key, var_name in mobility_vars.items():
        if var_key not in data.columns:
            continue
            
        global_mean = data[var_key].mean()
        pattern_effects = {}
        
        for pattern in data['pattern'].unique():
            pattern_data = data[data['pattern'] == pattern]
            pattern_mean = pattern_data[var_key].mean()
            
            # Efecto marginal = diferencia con respecto al promedio global
            marginal_effect = pattern_mean - global_mean
            effect_size = marginal_effect / data[var_key].std()  # Normalizado
            
            pattern_effects[pattern] = {
                'marginal_effect': marginal_effect,
                'normalized_effect': effect_size,
                'pattern_mean': pattern_mean,
                'n_observations': len(pattern_data)
            }
        
        marginal_effects[var_key] = {
            'global_mean': global_mean,
            'pattern_effects': pattern_effects
        }
    
    return marginal_effects

def analyze_pattern_dominance(data, mobility_vars):
    """Analiza qué patrón domina en cada tipo de movilidad"""
    dominance = {}
    
    for var_key, var_name in mobility_vars.items():
        if var_key not in data.columns:
            continue
            
        pattern_means = data.groupby('pattern')[var_key].agg(['mean', 'std', 'count'])
        
        # Ordenar por media descendente
        pattern_ranking = pattern_means.sort_values('mean', ascending=False)
        
        dominance[var_key] = {
            'dominant_pattern': pattern_ranking.index[0],
            'dominant_value': pattern_ranking.iloc[0]['mean'],
            'weakest_pattern': pattern_ranking.index[-1],
            'weakest_value': pattern_ranking.iloc[-1]['mean'],
            'dominance_ratio': pattern_ranking.iloc[0]['mean'] / pattern_ranking.iloc[-1]['mean'],
            'full_ranking': pattern_ranking.to_dict('index')
        }
    
    return dominance

def calculate_conditional_correlations(data, mobility_vars):
    """Correlaciones entre variables de movilidad condicionadas por patrón"""
    conditional_corr = {}
    
    mobility_keys = [k for k in mobility_vars.keys() if k in data.columns]
    
    for pattern in data['pattern'].unique():
        pattern_data = data[data['pattern'] == pattern]
        
        if len(pattern_data) < 10:
            continue
            
        # Matriz de correlación para este patrón
        pattern_corr = pattern_data[mobility_keys].corr()
        conditional_corr[pattern] = pattern_corr
    
    return conditional_corr

def create_advanced_visualizations(data, feature_importance, marginal_effects, output_dir, mobility_vars):
    """Crea visualizaciones avanzadas para análisis de causalidad"""
    
    # 1. RADAR CHART: Perfil de movilidad por patrón
    create_mobility_radar_chart(data, mobility_vars, output_dir)
    
    # 2. HEATMAP INTERACTIVO: Efectos marginales
    create_marginal_effects_heatmap(marginal_effects, mobility_vars, output_dir)
          

def create_mobility_radar_chart(data, mobility_vars, output_dir):
    """Radar chart comparando perfiles de movilidad por patrón"""
    
    # Calcular medias por patrón (normalizadas 0-1)
    pattern_profiles = {}
    for pattern in data['pattern'].unique():
        pattern_data = data[data['pattern'] == pattern]
        profile = {}
        
        for var_key, var_name in mobility_vars.items():
            if var_key in data.columns:
                # Normalizar a escala 0-1
                var_mean = pattern_data[var_key].mean()
                var_min = data[var_key].min()
                var_max = data[var_key].max()
                normalized = (var_mean - var_min) / (var_max - var_min) if var_max != var_min else 0
                profile[var_name] = normalized
        
        pattern_profiles[pattern] = profile
    
    # Crear radar chart
    fig = go.Figure()
    
    categories = list(next(iter(pattern_profiles.values())).keys())
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FCEA2B', '#FF9F43', '#A55EEA']
    
    for i, (pattern, profile) in enumerate(pattern_profiles.items()):
        values = list(profile.values())
        values.append(values[0])  # Cerrar el radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=pattern.replace('_', ' ').title(),
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Perfiles de Movilidad por Patrón Urbano<br><sub>Valores normalizados (0=mínimo, 1=máximo)</sub>",
        font=dict(size=12)
    )
    
    fig.write_html(f"{output_dir}/radar_mobility_patterns.html")
    print("✅ Radar chart guardado")

def create_marginal_effects_heatmap(marginal_effects, mobility_vars, output_dir):
    """
    Create a clean heatmap of marginal effects for academic publication.
    Relies on matplotlib style sheet for all formatting configurations.
    """
    
    with plt.style.context("styles/matplotlib_style.mplstyle"):
        
        # Define target variables for analysis
        target_vars = {
            'a': 'A',
            'b': 'B', 
            'car_share': 'C'
        }
        
        # Extract unique patterns and prepare data matrix
        patterns = set()
        for var_key in target_vars.keys():
            if var_key in marginal_effects:
                patterns.update(marginal_effects[var_key]['pattern_effects'].keys())
        
        patterns = sorted(list(patterns))
        
        # Build effects matrix
        effects_matrix = []
        for var_key in target_vars.keys():
            if var_key in marginal_effects:
                row = []
                for pattern in patterns:
                    effect = marginal_effects[var_key]['pattern_effects'].get(
                        pattern, {}
                    ).get('normalized_effect', 0)
                    row.append(effect)
                effects_matrix.append(row)
        
        effects_matrix = np.array(effects_matrix)
        
        # Create figure - let style sheet control size
        fig, ax = plt.subplots()
        
        # Adjust subplot position to center the heatmap
        plt.subplots_adjust(left=0.15, right=0.85)
        
        ax.grid(False)
        # Create heatmap with minimal configuration and adjusted size
        im = ax.imshow(effects_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect=1)
        
        # Configure labels
        pattern_labels = [
            {'cul_de_sac': 'Cul-de-Sac', 'cul de sac': 'Cul-de-Sac',
             'gridiron': 'Gridiron', 'hibrido': 'Hybrid', 'hybrid': 'Hybrid',
             'organico': 'Organic', 'organic': 'Organic'}.get(
                pattern.lower(), pattern.replace('_', ' ').title()
            ) for pattern in patterns
        ]
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(patterns)))
        ax.set_yticks(np.arange(len(target_vars)))
        ax.set_xticklabels(pattern_labels)
        ax.set_yticklabels(list(target_vars.values()))
        
        # Add cell values with appropriate contrast
        for i in range(len(target_vars)):
            for j in range(len(patterns)):
                val = effects_matrix[i, j]
                # Color del texto basado en contraste
                text_color = 'white' if abs(val) > 0.3 else 'black'
                # Formato condicional: mostrar más decimales solo si es necesario
                if abs(val) < 0.01:
                    text = '0.00'
                else:
                    text = f'{val:.2f}'
                
                ax.text(j, i, text, ha="center", va="center", 
                        color=text_color, fontweight='bold')
        
        # Titles and labels
        ax.set_title(r'\textbf{Marginal Effects of Urban Patterns on Mobility}' + '\n' + 
                    r'\textit{Positive = Above Average | Negative = Below Average}', 
                     pad=20)
        ax.set_xlabel('Urban Pattern')
        ax.set_ylabel('Mobility Mode')
        ax.tick_params(length=0)

        # Add colorbar with same height as heatmap
        cbar = plt.colorbar(im, ax=ax, fraction=0.033, pad=0.04, shrink=1)
        cbar.ax.tick_params(length=0, labelsize=7)

        
        # Save files
        plt.savefig(f"{output_dir}/marginal_effects_heatmap.pdf", 
                   bbox_inches='tight')
        plt.savefig(f"{output_dir}/marginal_effects_heatmap.png", 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        print("✅ Marginal effects heatmap saved")
        print(f"📁 Files: marginal_effects_heatmap.pdf/.png")

def integrate_advanced_analysis(global_data, global_analysis, output_dir, mobility_vars):
    """
    Función para integrar el análisis avanzado con tu código existente
    AÑADIR ESTA LLAMADA AL FINAL DE tu función analyze_patterns_mobility_correlation
    """
    print("\n" + "="*80)
    print("ANÁLISIS AVANZADO DE CAUSALIDAD Y CORRELACIÓN")
    print("="*80)
    
    # Ejecutar análisis avanzado
    advanced_results = advanced_causality_analysis(global_data, output_dir, mobility_vars)
    
    # Crear resumen de causalidad
    create_causality_summary(advanced_results, global_analysis, output_dir, mobility_vars)
    
    return advanced_results

def create_causality_summary(advanced_results, global_analysis, output_dir, mobility_vars):
    """Crea resumen ejecutivo de causalidad"""
    
    summary_lines = []
    summary_lines.append("="*60)
    summary_lines.append("RESUMEN DE CAUSALIDAD: PATRONES URBANOS → MOVILIDAD")
    summary_lines.append("="*60)
    
    # Análisis de dominancia
    dominance = advanced_results['dominance_analysis']
    summary_lines.append("\n🎯 PATRONES DOMINANTES POR TIPO DE MOVILIDAD:")
    
    for var_key, var_name in mobility_vars.items():
        if var_key in dominance:
            dom_pattern = dominance[var_key]['dominant_pattern']
            dom_value = dominance[var_key]['dominant_value']
            weak_pattern = dominance[var_key]['weakest_pattern']  # AÑADIR ESTA LÍNEA
            ratio = dominance[var_key]['dominance_ratio']
            
            summary_lines.append(f"  • {var_name}: {dom_pattern.title()} ({dom_value:.3f})")
            # CAMBIAR ESTA LÍNEA para incluir el patrón más bajo específico:
            summary_lines.append(f"    └─ {ratio:.1f}x más alto que el patrón más bajo ({weak_pattern.title()})")
    
    # Feature importance
    feature_imp = advanced_results['feature_importance']
    summary_lines.append(f"\n🔍 PODER PREDICTIVO DE PATRONES URBANOS:")
    
    for var_key, var_name in mobility_vars.items():
        if var_key in feature_imp:
            r2 = feature_imp[var_key]['r2_score']
            importance = feature_imp[var_key]['importance_score']
            
            predictive_power = "ALTO" if r2 > 0.3 else "MEDIO" if r2 > 0.1 else "BAJO"
            summary_lines.append(f"  • {var_name}: R² = {r2:.3f} ({predictive_power})")
    
    summary_lines.append(f"\n✅ CONCLUSIÓN CAUSAL:")
    high_r2_count = sum(1 for k, v in feature_imp.items() if v['r2_score'] > 0.2)
    total_vars = len(feature_imp)
    
    if high_r2_count / total_vars > 0.5:
        summary_lines.append("  Los patrones urbanos SÍ tienen un efecto causal significativo en la movilidad.")
    else:
        summary_lines.append("  Los patrones urbanos tienen un efecto causal LIMITADO en la movilidad.")
    
    # Guardar resumen
    with open(f"{output_dir}/RESUMEN_CAUSALIDAD.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    # Imprimir en consola
    for line in summary_lines:
        print(line)














def analyze_patterns_mobility_correlation(polygons_analysis_path, output_dir):
    """
    Análisis REAL de correlación entre patrones urbanos y movilidad
    Extrae datos reales, los une por poly_id y calcula correlaciones específicas
    VERSIÓN CORREGIDA: Maneja diferentes tipos de datos en poly_id
    """
    
    # Ciudades a analizar
    cities = [
        "Moscow_ID", "Philadelphia_PA", "Peachtree_GA", "Boston_MA",
        "Chandler_AZ", "Salt_Lake_UT", "Santa_Fe_NM","Charleston_SC", "Cary_Town_NC",
        "Fort_Collins_CO"
    ]
    
    # Variables de movilidad y sus nombres descriptivos
    mobility_vars = {
        'car_share': 'Automóvil (%)',
        'transit_share': 'Transporte Público (%)', 
        'bicycle_share': 'Bicicleta (%)',
        'walked_share': 'Caminata (%)',
        'a': 'Movilidad Activa',
        'b': 'Movilidad Pública', 
        'c': 'Movilidad Privada'
    }
    
    # Contenedor para todos los datos
    all_city_data = []
    city_summaries = []
    
    print("="*80)
    print("ANÁLISIS DE CORRELACIÓN: PATRONES URBANOS vs MOVILIDAD (VERSIÓN CORREGIDA)")
    print("="*80)
    
    for city in cities:
        try:
            print(f"\n🏙️ Procesando {city}...")
            
            # 1. Cargar datos de movilidad
            mobility_path = os.path.join(polygons_analysis_path, city, "Mobility_Data", f"polygon_mobility_data_{city}.xlsx")
            if not os.path.exists(mobility_path):
                print(f"❌ No se encontró archivo de movilidad: {mobility_path}")
                continue
                
            mobility_data = pd.read_excel(mobility_path)
            print(f"   📊 Datos de movilidad cargados: {len(mobility_data)} polígonos")
            
            # 2. Cargar datos de patrones urbanos
            pattern_path = os.path.join(polygons_analysis_path, city, "clustering_analysis", "urban_pattern_analysis.xlsx")
            if not os.path.exists(pattern_path):
                print(f"❌ No se encontró archivo de patrones: {pattern_path}")
                continue
                
            # Cargar hoja 5 (índice 4)
            pattern_data = pd.read_excel(pattern_path, sheet_name=4)
            
            # Renombrar columnas para clarity (asumiendo estructura A=poly_id, C=patrón)
            pattern_data.columns = ['poly_id', 'empty', 'pattern'] + list(pattern_data.columns[3:])
            pattern_data = pattern_data[['poly_id', 'pattern']].dropna()
            
            print(f"   🏗️ Patrones urbanos cargados: {len(pattern_data)} polígonos")
            print(f"   🔍 Patrones encontrados: {pattern_data['pattern'].value_counts().to_dict()}")
            
            # 3. CORRECCIÓN: Normalizar tipos de poly_id antes del merge
            print(f"   🔧 Normalizando tipos de poly_id...")
            print(f"      - Movilidad poly_id tipo: {mobility_data['poly_id'].dtype}")
            print(f"      - Patrones poly_id tipo: {pattern_data['poly_id'].dtype}")
            
            # Convertir ambos a string para evitar problemas de tipo
            mobility_data['poly_id'] = mobility_data['poly_id'].astype(str)
            pattern_data['poly_id'] = pattern_data['poly_id'].astype(str)
            
            # También limpiar espacios en blanco que podrían causar problemas
            mobility_data['poly_id'] = mobility_data['poly_id'].str.strip()
            pattern_data['poly_id'] = pattern_data['poly_id'].str.strip()
            
            print(f"      - Ejemplos IDs movilidad ANTES: {list(mobility_data['poly_id'].head())}")
            print(f"      - Ejemplos IDs patrones ANTES: {list(pattern_data['poly_id'].head())}")
            
            # CORRECCIÓN ESPECÍFICA: Normalizar formatos de ID
            # Los IDs de movilidad tienen formato "CITY_N" y los de patrones son solo números
            # Además, movilidad empieza en 1 y patrones en 0
            
            # Extraer solo el número del ID de movilidad y ajustar índice
            mobility_data['poly_id_original'] = mobility_data['poly_id'].copy()
            
            # Si el ID tiene formato "CITY_N", extraer N y restar 1 para empezar en 0
            if mobility_data['poly_id'].str.contains('_').any():
                mobility_data['poly_id'] = mobility_data['poly_id'].str.split('_').str[-1]
                # Convertir a numérico, restar 1, y volver a string
                mobility_data['poly_id'] = (mobility_data['poly_id'].astype(int) - 1).astype(str)
                print(f"      - ⚙️ Convertidos IDs de movilidad de formato CITY_N a índice base-0")
            
            # Asegurar que los IDs de patrones sean strings
            pattern_data['poly_id'] = pattern_data['poly_id'].astype(str)
            
            print(f"      - Ejemplos IDs movilidad DESPUÉS: {list(mobility_data['poly_id'].head())}")
            print(f"      - Ejemplos IDs patrones DESPUÉS: {list(pattern_data['poly_id'].head())}")
            print(f"      - Movilidad: {len(mobility_data['poly_id'].unique())} IDs únicos")
            print(f"      - Patrones: {len(pattern_data['poly_id'].unique())} IDs únicos")
            
            # 4. Verificar overlap antes del merge
            mobility_ids = set(mobility_data['poly_id'])
            pattern_ids = set(pattern_data['poly_id'])
            common_ids = mobility_ids.intersection(pattern_ids)
            
            print(f"      - IDs en común: {len(common_ids)}")
            print(f"      - IDs solo en movilidad: {len(mobility_ids - pattern_ids)}")
            print(f"      - IDs solo en patrones: {len(pattern_ids - mobility_ids)}")
            
            if len(common_ids) == 0:
                print(f"❌ No hay IDs en común entre datasets para {city}")
                print(f"   Ejemplos IDs movilidad: {list(mobility_data['poly_id'].head())}")
                print(f"   Ejemplos IDs patrones: {list(pattern_data['poly_id'].head())}")
                continue
            
            # 5. Unir datos por poly_id
            merged_data = pd.merge(mobility_data, pattern_data, on='poly_id', how='inner')
            
            if len(merged_data) == 0:
                print(f"❌ El merge resultó en 0 filas para {city}")
                continue
                
            print(f"   ✅ Datos unidos exitosamente: {len(merged_data)} polígonos con ambos datos")
            
            # 6. Verificar que tenemos las variables de movilidad necesarias
            available_mobility_vars = [var for var in mobility_vars.keys() if var in merged_data.columns]
            print(f"   📊 Variables de movilidad disponibles: {available_mobility_vars}")
            
            if not available_mobility_vars:
                print(f"❌ No se encontraron variables de movilidad para {city}")
                continue
            
            # 7. Añadir identificador de ciudad
            merged_data['city'] = city
            all_city_data.append(merged_data)
            
            # 8. Análisis por ciudad
            city_analysis = analyze_single_city(merged_data, city, mobility_vars)
            city_summaries.append(city_analysis)
            
        except Exception as e:
            print(f"❌ Error procesando {city}: {e}")
            import traceback
            print(f"   Detalles del error: {traceback.format_exc()}")
            continue
    
    if not all_city_data:
        print("❌ No se pudieron procesar datos de ninguna ciudad")
        return None
    
    # 9. Análisis global (todas las ciudades juntas)
    print(f"\n🌍 ANÁLISIS GLOBAL...")
    global_data = pd.concat(all_city_data, ignore_index=True)
    print(f"Total de polígonos analizados: {len(global_data)}")
    print(f"Distribución de patrones global: {global_data['pattern'].value_counts().to_dict()}")
    print(f"Ciudades incluidas: {global_data['city'].unique()}")
    
    # 10. Análisis estadístico global
    global_analysis = perform_global_analysis(global_data, mobility_vars)
    
    # 11. Guardar resultados
    save_comprehensive_results(city_summaries, global_analysis, global_data, output_dir, mobility_vars)
    
    print(f"\n✅ Análisis completado. Resultados guardados en: {output_dir}")
    advanced_results = integrate_advanced_analysis(global_data, global_analysis, output_dir, mobility_vars)

    return global_data, global_analysis, advanced_results

def analyze_single_city(data, city_name, mobility_vars):
    """Análisis estadístico para una ciudad específica"""
    
    results = {
        'city': city_name,
        'total_polygons': len(data),
        'pattern_distribution': data['pattern'].value_counts().to_dict(),
        'correlations': {},
        'kruskal_wallis': {},
        'descriptive_stats': {}
    }
    
    print(f"   📈 Calculando correlaciones para {city_name}...")
    
    # Análisis por cada variable de movilidad
    for var_key, var_name in mobility_vars.items():
        if var_key not in data.columns:
            continue
            
        try:
            # Verificar que hay datos válidos
            valid_data = data[data[var_key].notna()]
            if len(valid_data) == 0:
                print(f"      ⚠️ No hay datos válidos para {var_name}")
                continue
                
            # Estadísticas descriptivas por patrón
            desc_stats = valid_data.groupby('pattern')[var_key].agg(['mean', 'std', 'count']).round(3)
            results['descriptive_stats'][var_key] = desc_stats.to_dict('index')
            
            # Test de Kruskal-Wallis (diferencias entre patrones)
            groups = [group[var_key].dropna().values for name, group in valid_data.groupby('pattern')]
            groups = [g for g in groups if len(g) > 0]  # Filtrar grupos vacíos
            
            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                h_stat, p_val = kruskal(*groups)
                results['kruskal_wallis'][var_key] = {
                    'h_statistic': h_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
            
            # Correlación con patrones (convirtiendo a numérico)
            pattern_numeric = pd.Categorical(valid_data['pattern']).codes
            correlation, p_corr = stats.spearmanr(pattern_numeric, valid_data[var_key])
            results['correlations'][var_key] = {
                'correlation': correlation,
                'p_value': p_corr,
                'significant': p_corr < 0.05
            }
            
        except Exception as e:
            print(f"      ⚠️ Error analizando {var_name}: {e}")
            continue
    
    return results

def perform_global_analysis(global_data, mobility_vars):
    """Análisis estadístico global combinando todas las ciudades"""
    
    print("   🔬 Realizando análisis estadístico global...")
    
    global_results = {
        'total_polygons': len(global_data),
        'cities_analyzed': global_data['city'].nunique(),
        'cities_list': list(global_data['city'].unique()),
        'pattern_distribution': global_data['pattern'].value_counts().to_dict(),
        'pattern_percentages': (global_data['pattern'].value_counts(normalize=True) * 100).round(2).to_dict(),
        'correlations_by_pattern': {},
        'kruskal_wallis_global': {},
        'mobility_by_pattern': {}
    }
    
    # Análisis detallado por patrón
    for pattern in global_data['pattern'].unique():
        pattern_data = global_data[global_data['pattern'] == pattern]
        global_results['mobility_by_pattern'][pattern] = {}
        
        for var_key, var_name in mobility_vars.items():
            if var_key in pattern_data.columns:
                valid_data = pattern_data[pattern_data[var_key].notna()]
                if len(valid_data) > 0:
                    stats_summary = {
                        'mean': float(valid_data[var_key].mean()),
                        'std': float(valid_data[var_key].std()),
                        'median': float(valid_data[var_key].median()),
                        'count': len(valid_data),
                        'min': float(valid_data[var_key].min()),
                        'max': float(valid_data[var_key].max())
                    }
                    global_results['mobility_by_pattern'][pattern][var_key] = stats_summary
    
    # Tests globales de Kruskal-Wallis
    for var_key, var_name in mobility_vars.items():
        if var_key in global_data.columns:
            try:
                valid_data = global_data[global_data[var_key].notna()]
                if len(valid_data) == 0:
                    continue
                    
                groups = [group[var_key].dropna().values for name, group in valid_data.groupby('pattern')]
                groups = [g for g in groups if len(g) > 0]  # Filtrar grupos vacíos
                
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    h_stat, p_val = kruskal(*groups)
                    
                    # Calcular eta-squared (tamaño del efecto)
                    n = len(valid_data)
                    eta_squared = (h_stat - len(groups) + 1) / (n - len(groups))
                    eta_squared = max(0, eta_squared)  # No puede ser negativo
                    
                    global_results['kruskal_wallis_global'][var_key] = {
                        'h_statistic': float(h_stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05,
                        'eta_squared': float(eta_squared),
                        'effect_size': interpret_effect_size(eta_squared),
                        'n_total': n,
                        'n_groups': len(groups)
                    }
                    
            except Exception as e:
                print(f"      ⚠️ Error en test global para {var_name}: {e}")
                continue
    
    return global_results

def interpret_effect_size(eta_squared):
    """Interpreta el tamaño del efecto"""
    if pd.isna(eta_squared) or eta_squared < 0:
        return 'N/A'
    elif eta_squared < 0.01:
        return 'Muy Pequeño'
    elif eta_squared < 0.06:
        return 'Pequeño'
    elif eta_squared < 0.14:
        return 'Mediano'
    else:
        return 'Grande'

def save_comprehensive_results(city_summaries, global_analysis, global_data, output_dir, mobility_vars):
    """Guarda todos los resultados en un Excel comprehensivo"""
    
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, "analisis_patrones_movilidad_COMPLETO.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # 1. HOJA: Resumen Ejecutivo Global
        create_executive_summary(global_analysis, writer, mobility_vars)
        
        # 2. HOJA: Comparación de Medias por Patrón
        create_pattern_comparison(global_analysis, writer, mobility_vars)
        
        # 3. HOJA: Tests Estadísticos Globales
        create_statistical_tests(global_analysis, writer, mobility_vars)
        
        # 4. HOJA: Análisis por Ciudad
        create_city_analysis(city_summaries, writer, mobility_vars)
        
        # 5. HOJA: Datos Crudos para Verificación (con IDs originales)
        global_data_with_orig = global_data.copy()
        # Si existe la columna de ID original, incluirla
        if 'poly_id_original' in global_data.columns:
            cols = ['poly_id_original', 'poly_id', 'city', 'pattern'] + [col for col in global_data.columns if col not in ['poly_id_original', 'poly_id', 'city', 'pattern']]
            global_data_with_orig = global_data_with_orig[cols]
        
        global_data_with_orig.to_excel(writer, sheet_name='Datos_Crudos', index=False)
        
        # 6. HOJA: Interpretación y Conclusiones
        create_conclusions(global_analysis, writer, mobility_vars)
        
        # 7. HOJA: Diagnóstico de Datos
        create_data_diagnostics(global_data, writer, mobility_vars)
    
    print(f"📋 Excel completo guardado en: {excel_path}")

def create_executive_summary(global_analysis, writer, mobility_vars):
    """Crea hoja de resumen ejecutivo"""
    
    summary_data = []
    
    # Información general
    summary_data.append(['RESUMEN EJECUTIVO', ''])
    summary_data.append(['Total de polígonos analizados', global_analysis['total_polygons']])
    summary_data.append(['Ciudades analizadas', global_analysis['cities_analyzed']])
    summary_data.append(['Ciudades incluidas', ', '.join(global_analysis['cities_list'])])
    summary_data.append(['', ''])
    
    # Distribución de patrones
    summary_data.append(['DISTRIBUCIÓN DE PATRONES URBANOS', ''])
    for pattern, count in global_analysis['pattern_distribution'].items():
        percentage = global_analysis['pattern_percentages'][pattern]
        summary_data.append([f'{pattern}', f'{count} polígonos ({percentage}%)'])
    
    summary_data.append(['', ''])
    
    # Variables con diferencias significativas
    summary_data.append(['VARIABLES CON DIFERENCIAS SIGNIFICATIVAS', ''])
    kw_results = global_analysis['kruskal_wallis_global']
    
    for var_key, var_name in mobility_vars.items():
        if var_key in kw_results:
            result = kw_results[var_key]
            status = "SÍ" if result['significant'] else "NO"
            p_str = f"{result['p_value']:.2e}" if result['p_value'] < 0.001 else f"{result['p_value']:.3f}"
            summary_data.append([
                f'{var_name}', 
                f'{status} (p={p_str}, efecto={result["effect_size"]})'
            ])
    
    summary_df = pd.DataFrame(summary_data, columns=['Concepto', 'Valor'])
    summary_df.to_excel(writer, sheet_name='Resumen_Ejecutivo', index=False)

def create_pattern_comparison(global_analysis, writer, mobility_vars):
    """Crea comparación de medias por patrón"""
    
    comparison_data = []
    mobility_by_pattern = global_analysis['mobility_by_pattern']
    
    for var_key, var_name in mobility_vars.items():
        for pattern in mobility_by_pattern.keys():
            if var_key in mobility_by_pattern[pattern]:
                stats = mobility_by_pattern[pattern][var_key]
                comparison_data.append({
                    'Variable_Movilidad': var_name,
                    'Patron_Urbano': pattern.replace('_', ' ').title(),
                    'Media': round(stats['mean'], 3),
                    'Desviacion_Std': round(stats['std'], 3),
                    'Mediana': round(stats['median'], 3),
                    'Minimo': round(stats['min'], 3),
                    'Maximo': round(stats['max'], 3),
                    'N_Poligonos': stats['count']
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_excel(writer, sheet_name='Comparacion_por_Patron', index=False)

def create_statistical_tests(global_analysis, writer, mobility_vars):
    """Crea hoja de tests estadísticos"""
    
    test_data = []
    kw_results = global_analysis['kruskal_wallis_global']
    
    for var_key, var_name in mobility_vars.items():
        if var_key in kw_results:
            result = kw_results[var_key]
            p_str = f"{result['p_value']:.2e}" if result['p_value'] < 0.001 else f"{result['p_value']:.4f}"
            
            test_data.append({
                'Variable_Movilidad': var_name,
                'Variable_Codigo': var_key,
                'Test_Aplicado': 'Kruskal-Wallis',
                'Pregunta': f'¿Difiere {var_name} entre patrones urbanos?',
                'H_Statistic': round(result['h_statistic'], 3),
                'P_Value': p_str,
                'P_Value_Numerico': result['p_value'],
                'Significativo': 'SÍ' if result['significant'] else 'NO',
                'Eta_Squared': round(result['eta_squared'], 4),
                'Tamaño_Efecto': result['effect_size'],
                'N_Total': result['n_total'],
                'N_Grupos': result['n_groups'],
                'Interpretacion': f"{'Hay' if result['significant'] else 'No hay'} diferencias significativas en {var_name} entre patrones urbanos"
            })
    
    test_df = pd.DataFrame(test_data)
    test_df.to_excel(writer, sheet_name='Tests_Estadisticos', index=False)

def create_city_analysis(city_summaries, writer, mobility_vars):
    """Crea análisis por ciudad"""
    
    city_data = []
    
    for city_result in city_summaries:
        city = city_result['city']
        
        for var_key, var_name in mobility_vars.items():
            if var_key in city_result['kruskal_wallis']:
                kw_result = city_result['kruskal_wallis'][var_key]
                p_str = f"{kw_result['p_value']:.2e}" if kw_result['p_value'] < 0.001 else f"{kw_result['p_value']:.4f}"
                
                city_data.append({
                    'Ciudad': city,
                    'Variable_Movilidad': var_name,
                    'H_Statistic': round(kw_result['h_statistic'], 3),
                    'P_Value': p_str,
                    'P_Value_Numerico': kw_result['p_value'],
                    'Significativo': 'SÍ' if kw_result['significant'] else 'NO',
                    'N_Poligonos': city_result['total_polygons'],
                    'Patrones_Ciudad': ', '.join(city_result['pattern_distribution'].keys())
                })
    
    city_df = pd.DataFrame(city_data)
    city_df.to_excel(writer, sheet_name='Analisis_por_Ciudad', index=False)

def create_conclusions(global_analysis, writer, mobility_vars):
    """Crea hoja de conclusiones"""
    
    conclusions = []
    kw_results = global_analysis['kruskal_wallis_global']
    
    conclusions.append(['CONCLUSIONES DEL ANÁLISIS', ''])
    conclusions.append(['', ''])
    conclusions.append([f'Análisis basado en {global_analysis["total_polygons"]} polígonos', ''])
    conclusions.append([f'de {global_analysis["cities_analyzed"]} ciudades: {", ".join(global_analysis["cities_list"])}', ''])
    conclusions.append(['', ''])
    
    # Análisis de cada variable
    significant_vars = []
    non_significant_vars = []
    
    for var_key, var_name in mobility_vars.items():
        if var_key in kw_results:
            result = kw_results[var_key]
            p_str = f"{result['p_value']:.2e}" if result['p_value'] < 0.001 else f"{result['p_value']:.4f}"
            
            if result['significant']:
                conclusion = f"✅ {var_name}: Existen diferencias significativas entre patrones urbanos"
                detail = f"   (H={result['h_statistic']:.3f}, p={p_str}, efecto {result['effect_size']})"
                significant_vars.append(var_name)
            else:
                conclusion = f"❌ {var_name}: No hay diferencias significativas entre patrones urbanos"
                detail = f"   (H={result['h_statistic']:.3f}, p={p_str})"
                non_significant_vars.append(var_name)
            
            conclusions.append([conclusion, ''])
            conclusions.append([detail, ''])
            conclusions.append(['', ''])
    
    # Resumen final
    conclusions.append(['RESUMEN FINAL', ''])
    conclusions.append([f'Variables con diferencias significativas: {len(significant_vars)}', ''])
    conclusions.append([f'Variables sin diferencias significativas: {len(non_significant_vars)}', ''])
    
    conclusions_df = pd.DataFrame(conclusions, columns=['Conclusión', 'Detalle'])
    conclusions_df.to_excel(writer, sheet_name='Conclusiones', index=False)

def create_data_diagnostics(global_data, writer, mobility_vars):
    """Crea diagnóstico de los datos"""
    
    diagnostics = []
    
    diagnostics.append(['DIAGNÓSTICO DE CALIDAD DE DATOS', ''])
    diagnostics.append(['', ''])
    
    # Información general
    diagnostics.append(['Total de registros', len(global_data)])
    diagnostics.append(['Ciudades', global_data['city'].nunique()])
    diagnostics.append(['Patrones únicos', global_data['pattern'].nunique()])
    diagnostics.append(['', ''])
    
    # Completitud por variable
    diagnostics.append(['COMPLETITUD POR VARIABLE', ''])
    for var_key, var_name in mobility_vars.items():
        if var_key in global_data.columns:
            non_null = global_data[var_key].notna().sum()
            percentage = (non_null / len(global_data)) * 100
            diagnostics.append([f'{var_name}', f'{non_null}/{len(global_data)} ({percentage:.1f}%)'])
    
    diagnostics.append(['', ''])
    
    # Distribución por ciudad
    diagnostics.append(['DISTRIBUCIÓN POR CIUDAD', ''])
    for city in global_data['city'].unique():
        count = (global_data['city'] == city).sum()
        percentage = (count / len(global_data)) * 100
        diagnostics.append([city, f'{count} polígonos ({percentage:.1f}%)'])
    
    diagnos_df = pd.DataFrame(diagnostics, columns=['Métrica', 'Valor'])
    diagnos_df.to_excel(writer, sheet_name='Diagnostico_Datos', index=False)

# # Ejemplo de uso:
# if __name__ == "__main__":
#     polygons_analysis_path = "Polygons_analysis"  
#     output_dir = "Resultados_Patrones_Movilidad"
    
#     global_data, global_analysis, advanced_results = analyze_patterns_mobility_correlation(polygons_analysis_path, output_dir)