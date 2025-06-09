import pandas as pd
import os
from collections import defaultdict

def analyze_urban_patterns_complete():
    """Analiza tanto patrones teóricos (hoja 5) como clustering (hoja 6)"""
    base_path = "Polygons_Analysis"
    results = {}
    
    city_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    print("Ciudades encontradas:", city_folders)
    print("="*80)
    
    for city_folder in city_folders:
        print(f"\n🏙️  ANALIZANDO: {city_folder}")
        print("-" * 60)
        
        excel_path = os.path.join(base_path, city_folder, "clustering_analysis", "urban_pattern_analysis.xlsx")
        
        if not os.path.exists(excel_path):
            print(f"❌ No se encontró el archivo: {excel_path}")
            continue
            
        try:
            # ===== HOJA 5: PATRONES TEÓRICOS/ORIGINALES =====
            print(f"\n📚 ANALIZANDO PATRONES TEÓRICOS (Hoja 5)")
            print("-" * 40)
            
            try:
                df_teoricos = pd.read_excel(excel_path, sheet_name=4)  # Hoja 5 = índice 4
                
                # Verificar estructura
                print(f"📏 Dimensiones: {df_teoricos.shape}")
                print(f"📋 Columnas: {list(df_teoricos.columns)}")
                
                # Asumir que las columnas son: poly_id, vacio, original_pattern
                if len(df_teoricos.columns) >= 3:
                    # Renombrar para claridad
                    df_teoricos.columns = ['poly_id', 'vacio', 'patron_teorico'] + list(df_teoricos.columns[3:])
                    
                    # Limpiar datos
                    df_teoricos_clean = df_teoricos.dropna(subset=['patron_teorico'])
                    total_teoricos = len(df_teoricos_clean)
                    
                    print(f"📊 Total propiedades válidas: {total_teoricos}")
                    
                    # Análisis de patrones teóricos
                    teoricos_counts = df_teoricos_clean['patron_teorico'].value_counts()
                    teoricos_percentages = (teoricos_counts / total_teoricos * 100).round(2)
                    
                    print(f"\n🎯 PATRONES TEÓRICOS:")
                    for patron, percentage in teoricos_percentages.items():
                        count = teoricos_counts[patron]
                        is_hybrid = 'híbrido' in str(patron).lower()
                        marker = "🔥" if is_hybrid else "  "
                        print(f"{marker} {patron}: {percentage}% ({count} propiedades)")
                
                else:
                    print("❌ Estructura inesperada en hoja 5")
                    df_teoricos_clean = pd.DataFrame()
                    teoricos_percentages = pd.Series()
                    total_teoricos = 0
                    
            except Exception as e:
                print(f"❌ Error leyendo hoja 5: {str(e)}")
                df_teoricos_clean = pd.DataFrame()
                teoricos_percentages = pd.Series()
                total_teoricos = 0
            
            # ===== HOJA 6: PATRONES DE CLUSTERING =====
            print(f"\n🤖 ANALIZANDO PATRONES DE CLUSTERING (Hoja 6)")
            print("-" * 40)
            
            try:
                df_clustering = pd.read_excel(excel_path, sheet_name=5)  # Hoja 6 = índice 5
                
                print(f"📏 Dimensiones: {df_clustering.shape}")
                
                # Renombrar columnas como antes
                df_clustering.columns = ['poly_id', 'vacia', 'cluster', 'patron', 'subpatron'] + list(df_clustering.columns[5:])
                
                # Limpiar datos
                df_clustering_clean = df_clustering.dropna(subset=['patron', 'subpatron'])
                total_clustering = len(df_clustering_clean)
                
                print(f"📊 Total propiedades válidas: {total_clustering}")
                
                # Análisis de patrones de clustering
                clustering_counts = df_clustering_clean['patron'].value_counts()
                clustering_percentages = (clustering_counts / total_clustering * 100).round(2)
                
                print(f"\n🎯 PATRONES DE CLUSTERING:")
                for patron, percentage in clustering_percentages.items():
                    count = clustering_counts[patron]
                    print(f"   {patron}: {percentage}% ({count} propiedades)")
                
                # Análisis de subpatrones de clustering
                subclustering_counts = df_clustering_clean['subpatron'].value_counts()
                subclustering_percentages = (subclustering_counts / total_clustering * 100).round(2)
                
                print(f"\n🎯 SUBPATRONES DE CLUSTERING:")
                for subpatron, percentage in subclustering_percentages.items():
                    count = subclustering_counts[subpatron]
                    print(f"   {subpatron}: {percentage}% ({count} propiedades)")
                
            except Exception as e:
                print(f"❌ Error leyendo hoja 6: {str(e)}")
                clustering_percentages = pd.Series()
                subclustering_percentages = pd.Series()
                total_clustering = 0
            
            # Guardar resultados de ambas hojas
            results[city_folder] = {
                'teoricos': {
                    'total_properties': total_teoricos,
                    'patrones': teoricos_percentages.to_dict() if not teoricos_percentages.empty else {}
                },
                'clustering': {
                    'total_properties': total_clustering,
                    'patrones': clustering_percentages.to_dict() if not clustering_percentages.empty else {},
                    'subpatrones': subclustering_percentages.to_dict() if not subclustering_percentages.empty else {}
                }
            }
            
        except Exception as e:
            print(f"❌ Error general procesando {city_folder}: {str(e)}")
    
    return results

def print_global_summary(results):
    """Imprime resumen global separando teóricos vs clustering"""
    print("\n" + "="*100)
    print("📋 RESUMEN GLOBAL POR CIUDAD")
    print("="*100)
    
    for city, data in results.items():
        print(f"\n🏙️  {city.replace('_', ' ').upper()}")
        print("-" * 50)
        
        # Patrones teóricos
        if data['teoricos']['total_properties'] > 0:
            print(f"📚 PATRONES TEÓRICOS ({data['teoricos']['total_properties']} propiedades):")
            sorted_teoricos = sorted(data['teoricos']['patrones'].items(), key=lambda x: x[1], reverse=True)
            for patron, pct in sorted_teoricos:
                is_hybrid = 'híbrido' in str(patron).lower()
                marker = "🔥" if is_hybrid else "  "
                print(f"{marker} {patron}: {pct}%")
        else:
            print("📚 PATRONES TEÓRICOS: No se pudieron procesar")
        
        # Patrones de clustering
        if data['clustering']['total_properties'] > 0:
            print(f"\n🤖 PATRONES DE CLUSTERING ({data['clustering']['total_properties']} propiedades):")
            sorted_clustering = sorted(data['clustering']['patrones'].items(), key=lambda x: x[1], reverse=True)
            for patron, pct in sorted_clustering:
                print(f"   {patron}: {pct}%")
            
            print(f"\n🎯 TOP 5 SUBPATRONES DE CLUSTERING:")
            sorted_subclustering = sorted(data['clustering']['subpatrones'].items(), key=lambda x: x[1], reverse=True)[:5]
            for subpatron, pct in sorted_subclustering:
                print(f"   {subpatron}: {pct}%")
        else:
            print("🤖 PATRONES DE CLUSTERING: No se pudieron procesar")

def calculate_combined_percentages(results):
    """Calcula porcentajes globales combinando todas las ciudades"""
    print("\n" + "="*100)
    print("🌍 ANÁLISIS GLOBAL COMBINADO")
    print("="*100)
    
    # ===== PATRONES TEÓRICOS GLOBALES =====
    print(f"\n📚 PATRONES TEÓRICOS GLOBALES:")
    global_teoricos = defaultdict(int)
    total_teoricos_global = 0
    
    for city, data in results.items():
        if data['teoricos']['total_properties'] > 0:
            total_teoricos_global += data['teoricos']['total_properties']
            for patron, pct in data['teoricos']['patrones'].items():
                count = int(pct * data['teoricos']['total_properties'] / 100)
                global_teoricos[patron] += count
    
    if total_teoricos_global > 0:
        print(f"Total global: {total_teoricos_global} propiedades")
        for patron, count in sorted(global_teoricos.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_teoricos_global * 100)
            is_hybrid = 'híbrido' in str(patron).lower()
            marker = "🔥" if is_hybrid else "  "
            print(f"{marker} {patron}: {pct:.2f}% ({count} propiedades)")
    
    # ===== PATRONES DE CLUSTERING GLOBALES =====
    print(f"\n🤖 PATRONES DE CLUSTERING GLOBALES:")
    global_clustering = defaultdict(int)
    total_clustering_global = 0
    
    for city, data in results.items():
        if data['clustering']['total_properties'] > 0:
            total_clustering_global += data['clustering']['total_properties']
            for patron, pct in data['clustering']['patrones'].items():
                count = int(pct * data['clustering']['total_properties'] / 100)
                global_clustering[patron] += count
    
    if total_clustering_global > 0:
        print(f"Total global: {total_clustering_global} propiedades")
        for patron, count in sorted(global_clustering.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_clustering_global * 100)
            print(f"   {patron}: {pct:.2f}% ({count} propiedades)")
    
    # ===== SUBPATRONES DE CLUSTERING GLOBALES =====
    print(f"\n🎯 TOP 10 SUBPATRONES DE CLUSTERING GLOBALES:")
    global_subclustering = defaultdict(int)
    
    for city, data in results.items():
        if data['clustering']['total_properties'] > 0:
            for subpatron, pct in data['clustering']['subpatrones'].items():
                count = int(pct * data['clustering']['total_properties'] / 100)
                global_subclustering[subpatron] += count
    
    if total_clustering_global > 0:
        sorted_global_subclustering = sorted(global_subclustering.items(), key=lambda x: x[1], reverse=True)[:10]
        for subpatron, count in sorted_global_subclustering:
            pct = (count / total_clustering_global * 100)
            print(f"   {subpatron}: {pct:.2f}% ({count} propiedades)")

# Ejecutar el análisis completo
if __name__ == "__main__":
    print("🚀 Iniciando análisis completo de patrones urbanos...")
    print("📚 Hoja 5: Patrones teóricos/originales")
    print("🤖 Hoja 6: Patrones de clustering")
    print("="*80)
    
    results = analyze_urban_patterns_complete()
    
    if results:
        print_global_summary(results)
        calculate_combined_percentages(results)
        print(f"\n✅ Análisis completado para {len(results)} ciudades")
    else:
        print("❌ No se pudieron procesar datos de ninguna ciudad")