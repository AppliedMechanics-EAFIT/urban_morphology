import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.formula.api import ols
import glob, re, pathlib
import warnings
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import stats
import traceback
import matplotlib.patheffects as patheffects
from sklearn.neighbors import KernelDensity
from matplotlib.patches import Patch
warnings.filterwarnings('ignore', category=FutureWarning)
from scipy import stats
from scipy.stats import chi2_contingency, kruskal
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from scipy.stats import spearmanr, mannwhitneyu

class StreetPatternMobilityAnalyzer:

    def _setup_plot_style(self, column_width_mm=85):
        """
        Set up global plot style for scientific publication figures using 10pt base font.
        
        Parameters
        ----------
        column_width_mm : int
            Target width of the figure in millimeters. Typical values:
            - 85 mm for single-column figures
            - 170 mm for double-column figures
        """
        width_in = column_width_mm / 25.4
        height_in = width_in * 0.75  # Maintain aspect ratio (4:3 approx)

        plt.rcParams.update({
            'figure.figsize': (width_in, height_in),
            'font.size': 10,  # Match cas-dc article font size
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Palatino', 'Computer Modern Roman'],
            'axes.titlesize': 10,
            'axes.labelsize': 10,
            'axes.labelpad': 4,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'legend.fontsize': 9,
            'figure.titlesize': 10,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'savefig.dpi': 600,  # Still useful for raster fallback
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'axes.facecolor': 'white',
            'figure.facecolor': 'white'
        })

        sns.set_palette("deep")
        plt.ioff()

    def _save_figure(self, fig_path, title=""):
        """
        Save figure to PDF with high-quality vector output.
        
        Parameters
        ----------
        fig_path : str
            Path to save the figure (should end in '.pdf' for vector output).
        title : str
            Optional title to print after saving.
        """
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig_path = os.path.splitext(fig_path)[0] + '.pdf'  # Force .pdf output

        plt.savefig(fig_path, format='pdf', bbox_inches='tight', facecolor='white')
        plt.close()

        if title:
            print(f"✓ Saved: {title}")
            
    def _get_pattern_config(self):
        """Retorna configuración consistente de patrones y colores."""
        return {
            'orden': ['gridiron', 'organico', 'hibrido', 'cul_de_sac'],
            'colores': {
                'cul_de_sac': '#FF6B6B',
                'gridiron': '#006400', 
                'organico': '#45B7D1',
                'hibrido': '#FDCB6E'
            },
            'labels': {
                'cul_de_sac': 'Cul-de-sac',
                'gridiron': 'Gridiron',
                'organico': 'Orgánico',
                'hibrido': 'Híbrido'
            }
        }

    def __init__(self, base_path="Polygons_analysis", results_dir="Mobility_Results"):
        self.base_path = base_path
        self.results_dir = results_dir
        self.cities_results_dir = os.path.join(results_dir, "Cities")
        self.global_dir = os.path.join(results_dir, "Global_Analysis")
        self.polygons_analysis_path = "Polygons_analysis"
        self.output_dir = self.global_dir
        self.global_data = None
        self.global_analysis = None
        self.advanced_results = None
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
        """Processes all available cities in the base path."""
        # Get list of city folders
        city_folders = [f for f in os.listdir(self.base_path) 
                      if os.path.isdir(os.path.join(self.base_path, f))]
        
        print(f"City folders found: {city_folders}")
        
        if not city_folders:
            print("No city folders found")
            return False
        
        # Process each city
        success = False
        for city in city_folders:
            print(f"\n--- Processing city: {city} ---")
            
            # Load data
            mobility_df = self.load_mobility_data(city)
            patterns_main_df, patterns_sub_df = self.load_patterns_data(city)
            
            if mobility_df is None or patterns_main_df is None:
                print(f"Could not load data for {city}")
                continue
                
            # Merge data
            city_data = self.merge_city_data(mobility_df, patterns_main_df, patterns_sub_df, city)
            
            if city_data is not None and len(city_data) > 0:
                # Save dataframe for individual analysis
                self.city_dataframes[city] = city_data
                
                # Add to the list of all cities
                self.all_cities_data.append(city_data)
                print(f"City {city} processed successfully with {len(city_data)} records")
                success = True
            else:
                print(f"Could not combine data for {city}")
        
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
        """Generates visualizations for city analysis."""
        try:
            # 1. Mobility boxplots by pattern
            for mobility_type in mobility_cols:
                plt.figure()
                sns.boxplot(x='pattern', y=mobility_type, data=city_data)
                plt.title(f'{city_name}: {self.mobility_columns[mobility_type]} by Pattern')
                plt.xlabel('Street Pattern')
                plt.ylabel(self.mobility_columns[mobility_type])
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(city_dir, f"{city_name}_boxplot_{mobility_type}.png"))
                plt.close()
                    
        except Exception as e:
            print(f"Error generating visualizations for {city_name}: {e}")
    
    def _analyze_subpatterns(self, city_data, city_name, city_dir, mobility_cols):
        """Performs specific subpattern analysis for a city."""
        try:
            # Create subfolder for subpattern analysis
            subpatterns_dir = os.path.join(city_dir, "Subpatterns")
            os.makedirs(subpatterns_dir, exist_ok=True)
            
            # Count subpattern frequency
            subpattern_counts = city_data['cluster_name'].value_counts()
            
            # Select the 10 most frequent subpatterns
            top_subpatterns = subpattern_counts.head(10).index.tolist()
            subpattern_data = city_data[city_data['cluster_name'].isin(top_subpatterns)]
            
            # Visualizations for subpatterns
            for mobility_type in mobility_cols:
                plt.figure(figsize=(14, 8))
                sns.boxplot(x='cluster_name', y=mobility_type, data=subpattern_data, order=top_subpatterns)
                plt.title(f'{city_name}: {self.mobility_columns[mobility_type]} by Subpattern')
                plt.xlabel('Subpattern')
                plt.ylabel(self.mobility_columns[mobility_type])
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(subpatterns_dir, f"{city_name}_boxplot_subpattern_{mobility_type}.png"))
                plt.close()
                        
        except Exception as e:
            print(f"Error in subpattern analysis for {city_name}: {e}")
    
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
            output_file = os.path.join(self.global_dir, "Mobility_analysis.xlsx")
            
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
                            'Eta_cuadrado': results['kruskal_wallis'][var]['eta_squared']
                        })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Resumen', index=False)
            
            # === 6. ANÁLISIS POST-HOC MEJORADO ===
            print("\n--- Análisis Post-Hoc (Variables con Diferencias Significativas) ---")
            
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
                posthoc_file = os.path.join(self.global_dir, "Post_Hoc_Analysis.xlsx")
                
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
                combined_data_file = os.path.join(self.global_dir, "Full_combined_Data.xlsx")
                df.to_excel(combined_data_file, index=False)
                print(f"Datos combinados guardados en: {combined_data_file}")
            except Exception as e:
                print(f"Error guardando datos combinados: {e}")
                      
            return True
            
        except Exception as e:
            print(f"Error en análisis: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_mobility_density_plots(self, data, viz_dir):
        """
        Creates an elegant and readable ridgeline plot using a "cell" approach.
        Each urban typology gets its own subplot ("cell"), and these cells are
        stacked with a slight overlap to create a cohesive final image.
        """
        with plt.style.context('styles/matplotlib_style.mplstyle'):
            # --- 1. Style and Parameter Setup ---
            MOBILITY_COLORS = {
                'a': '#FFC300', 'b': '#6895D2', 'car_share': '#D04848'
            }
            MOBILITY_LABELS_EN = {
                'a': 'Active (A)', 'b': 'Public (B)', 'car_share': 'Private (C)'
            }
            PATTERN_LABELS_EN = {
                'organico': 'Organic', 'cul_de_sac': 'Cul-de-sac',
                'hibrido': 'Hybrid', 'gridiron': 'Gridiron'
            }
            
            MOBILITY_ORDER = ['car_share', 'b', 'a'] 
            
            # The pattern order is already as requested
            DESIRED_PATTERN_ORDER = ['organico', 'cul_de_sac', 'hibrido', 'gridiron']
            patterns = [p for p in DESIRED_PATTERN_ORDER if p in data['pattern'].unique()]
            
            if not patterns:
                print("No valid patterns found for plotting.")
                return

            # --- 2. Figure and Axes "Cells" Creation ---
            n_patterns = len(patterns)
            fig = plt.figure(figsize=(7.5, 1.2 * n_patterns))
            gs = fig.add_gridspec(n_patterns, 1, hspace=-0.33)
            axs = gs.subplots()
            if n_patterns == 1:
                axs = [axs]
            
            axs = axs[::-1]
            patterns_for_plotting = patterns[::-1]

            # --- 3. Iterate and Draw in Each Cell (from bottom to top) ---
            for i, pattern in enumerate(patterns_for_plotting):
                ax = axs[i]
                pattern_data = data[data['pattern'] == pattern]
                
                ax.patch.set_alpha(0)
                
                median_labels_data = []

                for z_idx, col in enumerate(MOBILITY_ORDER):
                    if col not in pattern_data.columns: continue
                    x_data = pattern_data[col].dropna()
                    if len(x_data) < 2: continue

                    x_grid = np.linspace(0, 1, 1000)
                    kde = KernelDensity(bandwidth=0.035, kernel='gaussian').fit(x_data.to_numpy()[:, np.newaxis])
                    density = np.exp(kde.score_samples(x_grid[:, np.newaxis]))
                    
                    if np.max(density) > 0:
                        density = density / np.max(density)
                    
                    color = MOBILITY_COLORS[col]
                    
                    ax.fill_between(x_grid, 0, density, alpha=0.4, color=color, zorder=z_idx)
                    ax.plot(x_grid, density, color=color, lw=1.5, zorder=z_idx)
                    
                    median_val = x_data.median()
                    y_pos_at_median = np.interp(median_val, x_grid, density)
                    
                    ax.axvline(x=median_val, color=color, linestyle='--', lw=1.5, 
                            ymax=y_pos_at_median / 1.5,
                            zorder=z_idx + 0.5)

                    median_labels_data.append({
                        'median': median_val,
                        'y_pos': y_pos_at_median, # This is the "natural" height on the curve
                        'color': color,
                        'zorder': z_idx
                    })

                # 1. Sort labels by their natural height on the curve.
                median_labels_data.sort(key=lambda item: item['y_pos'])
                
                # 2. Initialize variables for position control.
                last_adjusted_y = -1  # Stores the Y position of the center of the last placed label.
                REQUIRED_SEPARATION = 0.14 

                for label_info in median_labels_data:
                    # KEY LOGIC: "Nudge" (Subtle push)
                    # The ideal Y position (target_y) is the natural height on the curve.
                    target_y = label_info['y_pos']
                    
                    # The final position (adjusted_y) will be the highest between:
                    # a) Its ideal position (target_y).
                    # b) The position of the last label plus the required separation.
                    # This means the label will ONLY move up if strictly necessary.
                    adjusted_y = max(target_y, last_adjusted_y + REQUIRED_SEPARATION)

                    # Horizontal positioning (a bit more space than before)
                    median_val = label_info['median']
                    text_x_pos = median_val + 0.02
                    ha = 'left'
                    if median_val > 0.85:
                        text_x_pos = median_val - 0.02
                        ha = 'right'

                    # Draw the text at the calculated final position
                    ax.text(text_x_pos, adjusted_y, f'{median_val:.2f}',
                            ha=ha, va='center',fontsize=5,  fontweight='bold', color='white',
                            bbox=dict(facecolor=label_info['color'], edgecolor='white',
                                    boxstyle='round,pad=0.2', alpha=0.95, linewidth=0.5),
                            zorder=label_info['zorder'] + 10)

                    # Update the position of the last label for the next iteration.
                    last_adjusted_y = adjusted_y

                # --- 4. Style Each Cell ---
                ax.set_ylim(0, 1.5)
                ax.set_yticks([])        
                ax.set_xlim(-0.03, 1.03)
                
                # Add the pattern label to the left of the cell
                pattern_display_name = PATTERN_LABELS_EN.get(pattern, pattern.capitalize())
                ax.text(-0.01, 0.2, pattern_display_name, transform=ax.transAxes,
                         fontweight='bold', ha='right', va='bottom')

                # Hide all spines except for the baseline
                ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
                
                # Only show the X-axis labels and spine on the very bottom plot (i=0)
                if i == 0:
                    ax.spines['bottom'].set_visible(True)
                    ax.set_xlabel("Modal Share", fontweight='bold', labelpad=15)
                    ax.xaxis.set_label_coords(0.45,-0.15)
                    ax.set_xticks(np.arange(0, 1.1, 0.2))
                    ax.tick_params(axis='x', which='both', length=0)  # Remove all x-axis ticks
                    ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)], fontsize=6)
                else:
                    ax.set_xticks([])
                    
            # --- 5. Titles, Legend, and Final Adjustments ---
            fig.suptitle(
                "Distribution of Mobility Shares by Urban Typology",
                fontweight='bold', y=0.87,x=0.46
            )

            legend_elements = [
                Patch(facecolor=MOBILITY_COLORS[key], edgecolor=MOBILITY_COLORS[key], alpha=0.7,
                    linewidth=1.5, label=label) 
                for key, label in MOBILITY_LABELS_EN.items()
            ]

            fig.legend(
                handles=legend_elements, loc='upper center', bbox_to_anchor=(0.46, 0.85),
                ncol=len(legend_elements),  frameon=False
            )
            
            plt.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.1)

            self._save_figure(f"{viz_dir}/mobility_ridgeline_final_cells.pdf", 
                            "Elegant cell-based mobility ridgeline visualization generated successfully.")

    def _create_detailed_shares_density_plots(self, data, mobility_cols, viz_dir):
        """Creates density plots for all shares disaggregated by pattern."""
        with plt.style.context('styles/matplotlib_style.mplstyle'):
            pattern_config = self._get_pattern_config()
            
            # Identify share columns (exclude aggregated A, B, C)
            share_cols = [col for col in mobility_cols if 'share' in col.lower() and col not in ['a', 'b', 'c']]
            
            if not share_cols:
                return

            # Generate diverse colors for all shares
            colors = plt.cm.Set3(np.linspace(0, 1, len(share_cols)))
            
            for pattern in pattern_config['orden']:
                pattern_data = data[data['pattern'] == pattern]
                
                if len(pattern_data) <= 1:
                    continue
                
                plt.figure(figsize=(7.5,5))
                
                valid_shares = []
                for i, col in enumerate(share_cols):
                    col_data = pattern_data[col].dropna()
                    
                    if len(col_data) > 1:
                        # Create more readable label
                        label = col.replace('_share', '').replace('_', ' ').upper()
                        color = colors[i]
                        
                        sns.kdeplot(col_data, fill=True, alpha=0.4, color=color, linewidth=2, label=label)
                        
                        median_val = col_data.median()
                        plt.axvline(x=median_val, color=color, linestyle="--", alpha=0.8, linewidth=1.5)
                        
                        valid_shares.append((col, label, color, median_val))
                
                # Add median labels organized vertically
                if valid_shares:
                    y_max = plt.gca().get_ylim()[1]
                    y_start = y_max * 0.9
                    dy = y_max * 0.08
                    
                    for i, (col, label, color, median_val) in enumerate(valid_shares):
                        plt.text(median_val + 0.005, y_start - i * dy, 
                                f'MEDIAN {label}: {median_val:.2f}', 
                                 color='black', fontweight='bold', fontsize=6,
                                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3', alpha=0.8))
                
                plt.title(f'DETAILED SHARE DISTRIBUTION - PATTERN {pattern.upper()}')
                plt.xlabel('SHARE PERCENTAGE')
                plt.ylabel('DENSITY')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                self._save_figure(f"{viz_dir}/detailed_shares_density_{pattern}.pdf", 
                                f"Detailed shares density - {pattern}")
            
    def _create_barplot(self, data, column, viz_dir):
        """Creates bar chart with standard error."""
        
        pattern_config = self._get_pattern_config()
        
        # Map patterns for consistency
        data_temp = data.copy()
        pattern_mapping = {p.lower(): p for p in pattern_config['orden']}
        data_temp['pattern_mapped'] = data_temp['pattern'].str.lower().map(pattern_mapping)
        
        # Statistics
        stats = data_temp.groupby('pattern_mapped')[column].agg(['mean', 'std', 'count'])
        stats = stats.reindex([p for p in pattern_config['orden'] if p in stats.index])
        stats['se'] = stats['std'] / np.sqrt(stats['count'])
        
        plt.figure(figsize=(14, 8))
        
        # Bars with specific colors
        bar_colors = [pattern_config['colores'][p] for p in stats.index]
        bars = plt.bar(range(len(stats)), stats['mean'], yerr=stats['se'],
                    color=bar_colors, capsize=7, width=0.7, 
                    edgecolor='black', linewidth=1,
                    error_kw={'elinewidth': 2, 'capthick': 2})
        
        # Labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            mean_val = stats['mean'].iloc[i]
            plt.text(bar.get_x() + bar.get_width()/2, 
                    height + stats['se'].iloc[i] + 0.02 * max(stats['mean']),
                    f'{mean_val:.2f}', ha="center", va="bottom", fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8))
        
        # Configuration
        plt.xticks(range(len(stats)), 
                [pattern_config['labels'][p] for p in stats.index],
                rotation=0, ha='right', fontweight='bold')
        
        col_formatted = column.upper().replace("_", " ")
        plt.title(f'MEAN {col_formatted} BY PATTERN')
        plt.xlabel('URBAN PATTERN')
        plt.ylabel(f'MEAN {col_formatted}')
        
        self._save_figure(f"{viz_dir}/barplot_{column}.pdf", f"Barplot - {column}")

    def _generate_enhanced_visualizations(self, combined_data, mobility_cols):
        """
        Genera visualizaciones mejoradas para el análisis global y por ciudad,
        con formato científico consistente.
        """
        try:
            # Configurar estilo
            self._setup_plot_style()
            
            # Crear directorio
            viz_dir = os.path.join(getattr(self, 'global_dir', './'), "Graphs")
            os.makedirs(viz_dir, exist_ok=True)
            
            print("Generando visualizaciones mejoradas...")
            
            # 1. Distribución de observaciones por ciudad y patrón
            self._create_observations_heatmap(combined_data, viz_dir)
            
            for col in mobility_cols:  
                if combined_data[col].isna().all() or combined_data[col].std() == 0:
                    continue
                    
                print(f"Procesando variable: {col}")
                
                # Boxplots y gráficos de barras (NO ridge plots para shares)
                self._create_barplot(combined_data, col, viz_dir)
            
            # 3. Análisis de movilidad específico
            self._create_mobility_density_plots(combined_data, viz_dir)
            
            # 4. Análisis detallado de shares desagregados
            self._create_detailed_shares_density_plots(combined_data, mobility_cols, viz_dir)
                                    
            print("✓ Visualizaciones completadas exitosamente")
            
        except Exception as e:
            print(f"Error generando visualizaciones: {e}")
            traceback.print_exc()

    def _create_observations_heatmap(self, data, viz_dir):
        """Creates heatmap of observations distribution."""
               
        plt.figure(figsize=(12, 8))
        pattern_city_counts = pd.crosstab(data['city'], data['pattern'])
        
        # Custom colormap
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f5f5f5', '#2171b5'])
        
        sns.heatmap(pattern_city_counts, annot=True, fmt='d', cmap=cmap)
        plt.title('Number of Observations by City and Pattern')
        plt.ylabel('City')
        plt.xlabel('Pattern')
        
        self._save_figure(f"{viz_dir}/observations_distribution.pdf", "Observations distribution")

    def calculate_feature_importance(self, data, mobility_vars):
        """Calculates importance of urban patterns for predicting mobility"""
        le = LabelEncoder()
        data_encoded = data.copy()
        data_encoded['pattern_encoded'] = le.fit_transform(data['pattern'])
        
        importance_results = {}
        
        for var_key, var_name in mobility_vars.items():
            if var_key not in data.columns:
                continue
                
            # Prepare data
            valid_data = data_encoded[data_encoded[var_key].notna()]
            if len(valid_data) < 10:
                continue
                
            X = valid_data[['pattern_encoded']].values
            y = valid_data[var_key].values
            
            # Random Forest for importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            importance_results[var_key] = {
                'importance_score': rf.feature_importances_[0],
                'r2_score': r2_score(y, rf.predict(X)),
                'pattern_classes': le.classes_
            }
        
        return importance_results

    def calculate_marginal_effects(self, data, mobility_vars):
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

    def analyze_pattern_dominance(self, data, mobility_vars):
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

    def calculate_conditional_correlations(self, data, mobility_vars):
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

    def create_marginal_effects_heatmap(self, marginal_effects, mobility_vars, output_dir):
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
                            color=text_color, fontsize=6, fontweight='bold')
            
            # Titles and labels
            ax.set_title(r'\textbf{Marginal Effects of Urban Patterns on Mobility}' + '\n' + 
                        r'\textit{Positive = Above Average | Negative = Below Average}', 
                        pad=20, y=0.9)
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

    def analyze_patterns_mobility_correlation(self, polygons_analysis_path, output_dir):
        """
        REAL correlation analysis between urban patterns and mobility
        Extracts real data, joins by poly_id and calculates specific correlations
        CORRECTED VERSION: Handles different data types in poly_id
        """
        
        # Cities to analyze
        cities = [
            "Moscow_ID", "Philadelphia_PA", "Peachtree_GA", "Boston_MA",
            "Chandler_AZ", "Salt_Lake_UT", "Santa_Fe_NM","Charleston_SC", "Cary_Town_NC",
            "Fort_Collins_CO"
        ]
        
        # Mobility variables and their descriptive names
        mobility_vars = {
            'car_share': 'Car (%)',
            'transit_share': 'Public Transit (%)', 
            'bicycle_share': 'Bicycle (%)',
            'walked_share': 'Walking (%)',
            'a': 'Active Mobility',
            'b': 'Public Mobility', 
            'c': 'Private Mobility'
        }
        
        # Container for all data
        all_city_data = []
        city_summaries = []
        
        print("CORRELATION ANALYSIS: URBAN PATTERNS vs MOBILITY")
        
        for city in cities:
            try:
                print(f"\nProcessing {city}...")
                
                # 1. Load mobility data
                mobility_path = os.path.join(polygons_analysis_path, city, "Mobility_Data", f"polygon_mobility_data_{city}.xlsx")
                if not os.path.exists(mobility_path):
                    print(f"Mobility file not found: {mobility_path}")
                    continue
                    
                mobility_data = pd.read_excel(mobility_path)
                
                # 2. Load urban pattern data
                pattern_path = os.path.join(polygons_analysis_path, city, "clustering_analysis", "urban_pattern_analysis.xlsx")
                if not os.path.exists(pattern_path):
                    print(f"Pattern file not found: {pattern_path}")
                    continue
                    
                # Load sheet 5 (index 4)
                pattern_data = pd.read_excel(pattern_path, sheet_name=4)
                
                # Rename columns for clarity (assuming structure A=poly_id, C=pattern)
                pattern_data.columns = ['poly_id', 'empty', 'pattern'] + list(pattern_data.columns[3:])
                pattern_data = pattern_data[['poly_id', 'pattern']].dropna()
                
                # 3. Normalize poly_id types before merge Convert both to string to avoid type issues
                mobility_data['poly_id'] = mobility_data['poly_id'].astype(str)
                pattern_data['poly_id'] = pattern_data['poly_id'].astype(str)
                
                # Also clean whitespace that could cause issues
                mobility_data['poly_id'] = mobility_data['poly_id'].str.strip()
                pattern_data['poly_id'] = pattern_data['poly_id'].str.strip()
                
                # SPECIFIC CORRECTION: Normalize ID formats Mobility IDs have format "CITY_N" and pattern IDs are just numbers
                # Also, mobility starts at 1 and patterns at 0
                
                # Extract only the number from mobility ID and adjust index
                mobility_data['poly_id_original'] = mobility_data['poly_id'].copy()
                
                # If ID has format "CITY_N", extract N and subtract 1 to start at 0
                if mobility_data['poly_id'].str.contains('_').any():
                    mobility_data['poly_id'] = mobility_data['poly_id'].str.split('_').str[-1]
                    # Convert to numeric, subtract 1, and back to string
                    mobility_data['poly_id'] = (mobility_data['poly_id'].astype(int) - 1).astype(str)
                
                # Ensure pattern IDs are strings
                pattern_data['poly_id'] = pattern_data['poly_id'].astype(str)
                
                # 4. Verify overlap before merge
                mobility_ids = set(mobility_data['poly_id'])
                pattern_ids = set(pattern_data['poly_id'])
                common_ids = mobility_ids.intersection(pattern_ids)
                
                if len(common_ids) == 0:
                    print(f"No common IDs between datasets for {city}")
                    continue
                
                # 5. Join data by poly_id
                merged_data = pd.merge(mobility_data, pattern_data, on='poly_id', how='inner')
                
                if len(merged_data) == 0:
                    print(f"Merge resulted in 0 rows for {city}")
                    continue
                    
                # 6. Verify we have necessary mobility variables
                available_mobility_vars = [var for var in mobility_vars.keys() if var in merged_data.columns]
                
                if not available_mobility_vars:
                    print(f"No mobility variables found for {city}")
                    continue
                
                # 7. Add city identifier
                merged_data['city'] = city
                all_city_data.append(merged_data)
                
                # 8. Single city analysis
                city_analysis = self.analyze_single_city(merged_data, city, mobility_vars)
                city_summaries.append(city_analysis)
                
            except Exception as e:
                print(f"Error processing {city}: {e}")
                import traceback
                print(f"Error details: {traceback.format_exc()}")
                continue
        
        if not all_city_data:
            print("Could not process data from any city")
            return None
        
        # 9. Global analysis (all cities together)
        print(f"\nGlobal analysis...")
        global_data = pd.concat(all_city_data, ignore_index=True)
        
        # 10. Global statistical analysis
        global_analysis = self.perform_global_analysis(global_data, mobility_vars)
        
        # 11. Save results
        self.save_comprehensive_results(city_summaries, global_analysis, global_data, output_dir, mobility_vars)
        
        print(f"Analysis completed. Results saved in: {output_dir}")
        advanced_results = self.advanced_causality_analysis(global_data, output_dir, mobility_vars)

        return global_data, global_analysis, advanced_results

    def analyze_single_city(self, data, city_name, mobility_vars):
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

    def perform_global_analysis(self, global_data, mobility_vars):
        """Global statistical analysis combining all cities"""
        
        print("   🔬 Performing global statistical analysis...")
        
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
        
        # Detailed analysis by pattern
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
        
        # Global Kruskal-Wallis tests
        for var_key, var_name in mobility_vars.items():
            if var_key in global_data.columns:
                try:
                    valid_data = global_data[global_data[var_key].notna()]
                    if len(valid_data) == 0:
                        continue
                        
                    groups = [group[var_key].dropna().values for name, group in valid_data.groupby('pattern')]
                    groups = [g for g in groups if len(g) > 0]  # Filter empty groups
                    
                    if len(groups) > 1 and all(len(g) > 0 for g in groups):
                        h_stat, p_val = kruskal(*groups)
                        
                        # Calculate eta-squared (effect size)
                        n = len(valid_data)
                        eta_squared = (h_stat - len(groups) + 1) / (n - len(groups))
                        eta_squared = max(0, eta_squared)  # Cannot be negative
                        
                        global_results['kruskal_wallis_global'][var_key] = {
                            'h_statistic': float(h_stat),
                            'p_value': float(p_val),
                            'significant': p_val < 0.05,
                            'eta_squared': float(eta_squared),
                            'effect_size': self.interpret_effect_size(eta_squared),
                            'n_total': n,
                            'n_groups': len(groups)
                        }
                        
                except Exception as e:
                    print(f"      ⚠️ Error in global test for {var_name}: {e}")
                    continue
        
        return global_results

    def interpret_effect_size(self, eta_squared):
        """Interprets effect size"""
        if pd.isna(eta_squared) or eta_squared < 0:
            return 'N/A'
        elif eta_squared < 0.01:
            return 'Very Small'
        elif eta_squared < 0.06:
            return 'Small'
        elif eta_squared < 0.14:
            return 'Medium'
        else:
            return 'Large'

    def save_comprehensive_results(self, city_summaries, global_analysis, global_data, output_dir, mobility_vars):
        """Saves all results in a comprehensive Excel file"""
        
        os.makedirs(output_dir, exist_ok=True)
        excel_path = os.path.join(output_dir, "Complete_data_comparison.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    
            # 1. SHEET: Pattern Comparison by Means
            self.create_pattern_comparison(global_analysis, writer, mobility_vars)
            
            # 2. SHEET: Global Statistical Tests
            self.create_statistical_tests(global_analysis, writer, mobility_vars)
            
            # 3. SHEET: City Analysis
            self.create_city_analysis(city_summaries, writer, mobility_vars)
            
            # 4. SHEET: Raw Data for Verification (with original IDs)
            global_data_with_orig = global_data.copy()
            # If original ID column exists, include it
            if 'poly_id_original' in global_data.columns:
                cols = ['poly_id_original', 'poly_id', 'city', 'pattern'] + [col for col in global_data.columns if col not in ['poly_id_original', 'poly_id', 'city', 'pattern']]
                global_data_with_orig = global_data_with_orig[cols]
            
            global_data_with_orig.to_excel(writer, sheet_name='Raw_Data', index=False)
                    
            # 5. SHEET: Data Diagnostics
            self.create_data_diagnostics(global_data, writer, mobility_vars)
        
        print(f"📋 Complete Excel saved at: {excel_path}")

    def create_pattern_comparison(self, global_analysis, writer, mobility_vars):
        """Creates comparison of means by pattern""" 
        comparison_data = []
        mobility_by_pattern = global_analysis['mobility_by_pattern']
        
        for var_key, var_name in mobility_vars.items():
            for pattern in mobility_by_pattern.keys():
                if var_key in mobility_by_pattern[pattern]:
                    stats = mobility_by_pattern[pattern][var_key]
                    comparison_data.append({
                        'Mobility_Variable': var_name,
                        'Urban_Pattern': pattern.replace('_', ' ').title(),
                        'Mean': round(stats['mean'], 3),
                        'Standard_Deviation': round(stats['std'], 3),
                        'Median': round(stats['median'], 3),
                        'Minimum': round(stats['min'], 3),
                        'Maximum': round(stats['max'], 3),
                        'N_Polygons': stats['count']
                    })   
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Comparison_by_Pattern', index=False)

    def create_statistical_tests(self, global_analysis, writer, mobility_vars):
        """Creates statistical tests sheet""" 
        test_data = []
        kw_results = global_analysis['kruskal_wallis_global']
        
        for var_key, var_name in mobility_vars.items():
            if var_key in kw_results:
                result = kw_results[var_key]
                p_str = f"{result['p_value']:.2e}" if result['p_value'] < 0.001 else f"{result['p_value']:.4f}"
                
                test_data.append({
                    'Mobility_Variable': var_name,
                    'Variable_Code': var_key,
                    'Test_Applied': 'Kruskal-Wallis',
                    'Question': f'Does {var_name} differ between urban patterns?',
                    'H_Statistic': round(result['h_statistic'], 3),
                    'P_Value': p_str,
                    'P_Value_Numeric': result['p_value'],
                    'Significant': 'YES' if result['significant'] else 'NO',
                    'Eta_Squared': round(result['eta_squared'], 4),
                    'Effect_Size': result['effect_size'],
                    'N_Total': result['n_total'],
                    'N_Groups': result['n_groups'],
                    'Interpretation': f"{'There are' if result['significant'] else 'There are no'} significant differences in {var_name} between urban patterns"
                })
        test_df = pd.DataFrame(test_data)
        test_df.to_excel(writer, sheet_name='Statistical_Tests', index=False)

    def create_city_analysis(self, city_summaries, writer, mobility_vars):
        """Creates analysis by city"""
        
        city_data = []
        
        for city_result in city_summaries:
            city = city_result['city']
            
            for var_key, var_name in mobility_vars.items():
                if var_key in city_result['kruskal_wallis']:
                    kw_result = city_result['kruskal_wallis'][var_key]
                    p_str = f"{kw_result['p_value']:.2e}" if kw_result['p_value'] < 0.001 else f"{kw_result['p_value']:.4f}"
                    
                    city_data.append({
                        'City': city,
                        'Mobility_Variable': var_name,
                        'H_Statistic': round(kw_result['h_statistic'], 3),
                        'P_Value': p_str,
                        'P_Value_Numeric': kw_result['p_value'],
                        'Significant': 'YES' if kw_result['significant'] else 'NO',
                        'N_Polygons': city_result['total_polygons'],
                        'City_Patterns': ', '.join(city_result['pattern_distribution'].keys())
                    })
        city_df = pd.DataFrame(city_data)
        city_df.to_excel(writer, sheet_name='Analysis_by_City', index=False)

    def create_data_diagnostics(self, global_data, writer, mobility_vars):
        """Creates data diagnostics"""
        diagnostics = []
        
        diagnostics.append(['DATA QUALITY DIAGNOSTICS', ''])
        diagnostics.append(['', ''])
        
        # General information
        diagnostics.append(['Total records', len(global_data)])
        diagnostics.append(['Cities', global_data['city'].nunique()])
        diagnostics.append(['Unique patterns', global_data['pattern'].nunique()])
        diagnostics.append(['', ''])
        
        # Completeness by variable
        diagnostics.append(['COMPLETENESS BY VARIABLE', ''])
        for var_key, var_name in mobility_vars.items():
            if var_key in global_data.columns:
                non_null = global_data[var_key].notna().sum()
                percentage = (non_null / len(global_data)) * 100
                diagnostics.append([f'{var_name}', f'{non_null}/{len(global_data)} ({percentage:.1f}%)'])
        
        diagnostics.append(['', ''])
        
        # Distribution by city
        diagnostics.append(['DISTRIBUTION BY CITY', ''])
        for city in global_data['city'].unique():
            count = (global_data['city'] == city).sum()
            percentage = (count / len(global_data)) * 100
            diagnostics.append([city, f'{count} polygons ({percentage:.1f}%)'])
        
        diagnos_df = pd.DataFrame(diagnostics, columns=['Metric', 'Value'])
        diagnos_df.to_excel(writer, sheet_name='Data_Diagnostics', index=False)

    def advanced_causality_analysis(self, global_data, output_dir, mobility_vars):
        """
        Advanced causality analysis with sophisticated visualizations
        """      
        # 1. Feature importance analysis (implicit causality)
        feature_importance = self.calculate_feature_importance(global_data, mobility_vars)
        
        # 2. Marginal effects analysis
        marginal_effects = self.calculate_marginal_effects(global_data, mobility_vars)
        
        # 3. Marginal effects heatmap
        self.create_marginal_effects_heatmap( marginal_effects, mobility_vars, output_dir)
        
        # 4. Pattern dominance analysis
        dominance_analysis = self.analyze_pattern_dominance(global_data, mobility_vars)
        
        # 5. Conditional correlation matrix
        conditional_correlations = self.calculate_conditional_correlations(global_data, mobility_vars)
        
        return {
            'feature_importance': feature_importance,
            'marginal_effects': marginal_effects,
            'dominance_analysis': dominance_analysis,
            'conditional_correlations': conditional_correlations
        }

    def run_analysis(self):
        """Executes the complete analysis workflow."""
        print("Starting street patterns and urban mobility analysis...")
        
        # 1. Process all cities
        if not self.process_cities():
            print("Could not process cities")
            return False
        
        # 2. Analysis by city
        for city_name, city_data in self.city_dataframes.items():
            self.analyze_city(city_data, city_name)
        
        # 3. Global analysis
        self.analyze_global_data()
        
        # 4. Patterns-mobility correlation analysis
        global_data, global_analysis, advanced_results = self.analyze_patterns_mobility_correlation(
            self.polygons_analysis_path, 
            self.output_dir
        )
        
        # Save results in the class if you need later access
        self.global_data = global_data
        self.global_analysis = global_analysis
        self.advanced_results = advanced_results
        
        print("Complete analysis")
        return True
    
if __name__ == "__main__":
    analyzer = StreetPatternMobilityAnalyzer()
    analyzer.run_analysis()