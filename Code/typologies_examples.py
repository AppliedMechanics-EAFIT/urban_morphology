import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
import numpy as np

def generate_urban_typologies_plot(style_path='styles/matplotlib_style.mplstyle', 
                                   output_filename='urban_typologies_single_column',
                                   output_format='pdf',  # 'pdf', 'png', or 'both'
                                   locations=None):
    """
    Genera un gráfico de tipologías urbanas usando OSMnx y guarda en el formato especificado.
    
    Parameters:
    -----------
    style_path : str
        Ruta al archivo de estilo de matplotlib (.mplstyle)
    output_filename : str
        Nombre base del archivo de salida (sin extensión)
    output_format : str
        Formato de salida: 'pdf', 'png', o 'both'
    locations : dict, optional
        Diccionario con las ubicaciones. Si es None, usa las ubicaciones por defecto.
    
    Returns:
    --------
    str or list
        Ruta(s) del archivo(s) generado(s)
    """
    
    # Configurar OSMnx
    ox.settings.use_cache = True
    
    # Ubicaciones por defecto si no se proporcionan
    if locations is None:
        locations = {
            'Grid': ('Midtown Manhattan, New York City, NY, USA', 1000),
            'Cul-De-Sac': ('Mission Viejo, CA, USA', 1200),
            'Organic': ('Alfama, Lisbon, Portugal', 600),
            'Hybrid': ('Canberra, Australia', 2000)
        }
    
    # Usar el estilo personalizado con context manager
    with plt.style.context(style_path):
        
        # SOLUCIÓN 1: Forzar configuración de fuentes después de cargar el estilo
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'font.size': 7.0
        })
        
        # Crear la figura con el tamaño correcto para la figura completa
        fig, axes = plt.subplots(2, 2)
        axes_flat = axes.flatten()
        
        for i, (pattern_type, (location, distance)) in enumerate(locations.items()):
            print(f"Processing: {pattern_type}")
            ax = axes_flat[i]
            
            try:
                # Obtener y proyectar el grafo
                G = ox.graph_from_address(location, dist=distance, network_type='drive')
                G_proj = ox.project_graph(G)
                
                # Plotear el grafo
                ox.plot_graph(G_proj, ax=ax, node_size=0, edge_color='black',
                              edge_linewidth=0.25, bgcolor='white', show=False, close=False)
                
                # SOLUCIÓN 2: Establecer el título DESPUÉS de ox.plot_graph
                # para asegurar que use la configuración correcta
                ax.set_title(pattern_type, fontsize=7, fontfamily='serif')
                
                # SOLUCIÓN 3: Alternativa con LaTeX explícito
                # ax.set_title(rf'\textbf{{{pattern_type}}}', fontsize=7)
                
            except Exception as e:
                print(f"Error with {pattern_type}: {str(e)}")
                # Mostrar mensaje de error en caso de fallo
                ax.text(0.5, 0.5, f'Error\n{pattern_type}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=6,
                        fontfamily='serif',  # Especificar familia de fuente
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
                ax.set_title(pattern_type, fontsize=7, fontfamily='serif')
            
            # Configurar el subplot
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout(pad=0.255, w_pad=0.1, h_pad=0.595)

        # Lista para almacenar archivos generados
        generated_files = []
        
        # Guardar según el formato especificado
        if output_format.lower() in ['pdf', 'both']:
            pdf_filename = f"{output_filename}.pdf"
            with PdfPages(pdf_filename) as pdf:
                pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
            generated_files.append(pdf_filename)
            print(f"Gráfico guardado como PDF: {pdf_filename}")
        
        if output_format.lower() in ['png', 'both']:
            png_filename = f"{output_filename}.png"
            fig.savefig(png_filename, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
            generated_files.append(png_filename)
            print(f"Gráfico guardado como PNG: {png_filename}")
        
               
        # Cerrar la figura para liberar memoria
        plt.close(fig)
    
    # Retornar la lista de archivos generados o un solo archivo si es uno
    return generated_files if len(generated_files) > 1 else generated_files[0]


# # Ejemplo de uso:
# if __name__ == "__main__":
    
#     # Ubicaciones personalizadas
#     custom_locations = {
#         'Grid': ('Midtown Manhattan, New York City, NY, USA', 1000),
#         'Cul-De-Sac': ('Mission Viejo, CA, USA', 1200),
#         'Organic': ('Alfama, Lisbon, Portugal', 600),
#         'Hybrid': ('Canberra, Australia', 2000)
#     }
    
#     # Generar ambos formatos
#     generate_urban_typologies_plot(
#         style_path='styles/matplotlib_style.mplstyle',
#         output_filename='urban_typologies_single_column',
#         output_format='pdf',
#         locations=custom_locations
#     )



def generate_legend_with_style(style_path='styles/matplotlib_style.mplstyle',
                              output_filename='street_legend_updated',
                              output_format='png',  # 'pdf', 'png', or 'both'
                              figsize=(7.5, 1.8)):
    """
    Genera una leyenda usando el estilo personalizado con figsize manual.
    
    Parameters:
    -----------
    style_path : str
        Ruta al archivo de estilo de matplotlib (.mplstyle)
    output_filename : str
        Nombre base del archivo de salida (sin extensión)
    output_format : str
        Formato de salida: 'pdf', 'png', o 'both'
    figsize : tuple
        Tamaño de la figura (ancho, alto)
    
    Returns:
    --------
    matplotlib.figure.Figure
        La figura generada
    """
    
    # Usar el estilo personalizado con context manager
    with plt.style.context(style_path):
        
        # Sobrescribir SOLO el figsize, manteniendo todo lo demás del estilo
        # (fonts, tamaños, colores, etc. vienen del .mplstyle)
        
        # Base colors for each main category
        base_colors = {
            'organic': "#2399CF",     # Blue for Organic
            'cul_de_sac': "#F13F3F",  # Red for Cul-de-sac
            'hybrid': "#E4CD4D",      # Yellow for Hybrid
            'gridiron': "#0C850C",    # Green for Gridiron
        }

        fig, ax = plt.subplots(figsize=figsize)  # Usar figsize personalizado
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 2)
        ax.axis('off')

        y_center = 1
        rect_width = 0.65
        rect_height = 0.8
        spacing = 2.1
        x = 0.3

        def draw_hatched_patch(x, color, label):
            # Fondo blanco
            bg_rect = plt.Rectangle((x, y_center - rect_height / 2), rect_width, rect_height,
                                    facecolor='white', edgecolor='black', linewidth=0.5)
            ax.add_patch(bg_rect)

            # Hachurado del color deseado
            hatch_rect = plt.Rectangle((x, y_center - rect_height / 2), rect_width, rect_height,
                                    facecolor='none', edgecolor=color, hatch='/////', linewidth=0.5)
            ax.add_patch(hatch_rect)

            # El texto usará automáticamente el font.size del estilo (7.0)
            # Si quieres un tamaño específico para la leyenda, puedes ajustarlo aquí
            ax.text(x + rect_width + 0.05, y_center, label, ha='left', va='center', 
                   fontsize=8)  # Ligeramente más grande que el font.size base para legibilidad

        # Organic
        draw_hatched_patch(x, base_colors['organic'], 'Organic')
        x += spacing

        # Organic Street⁺
        base_color = np.array(mcolors.to_rgb(base_colors['organic']))
        adjusted_color = base_color * 0.55
        adjusted_color = np.clip(adjusted_color, 0, 1)
        draw_hatched_patch(x, adjusted_color, 'Street$^+$')
        x += spacing

        # Cul-de-sac
        draw_hatched_patch(x, base_colors['cul_de_sac'], 'Cul-de-sac')
        x += spacing

        # Cul-de-sac Street⁺
        base_color = np.array(mcolors.to_rgb(base_colors['cul_de_sac']))
        adjusted_color = base_color * 0.5
        adjusted_color = np.clip(adjusted_color, 0, 1)
        draw_hatched_patch(x, adjusted_color, 'Street$^+$')
        x += spacing

        # Hybrid
        draw_hatched_patch(x, base_colors['hybrid'], 'Hybrid')
        x += spacing

        # Gridiron
        draw_hatched_patch(x, base_colors['gridiron'], 'Gridiron')
        x += spacing

        # Gridiron Street⁺
        base_color = np.array(mcolors.to_rgb(base_colors['gridiron']))
        adjusted_color = base_color * 0.55
        adjusted_color = np.clip(adjusted_color, 0, 1)
        draw_hatched_patch(x, adjusted_color, 'Street$^+$')
        x += spacing

        # Gridiron Street⁻
        base_color = np.array(mcolors.to_rgb(base_colors['gridiron']))
        adjusted_color = base_color + (1 - base_color) * 0.45
        adjusted_color = np.clip(adjusted_color, 0, 1)
        draw_hatched_patch(x, adjusted_color, 'Street$^-$')

        plt.tight_layout()
        
        # Lista para almacenar archivos generados
        generated_files = []
        
        # Guardar según el formato especificado
        if output_format.lower() in ['pdf', 'both']:
            pdf_filename = f"{output_filename}.pdf"
            fig.savefig(pdf_filename, dpi=300, bbox_inches='tight', facecolor='white')
            generated_files.append(pdf_filename)
            print(f"Leyenda guardada como PDF: {pdf_filename}")
        
        if output_format.lower() in ['png', 'both']:
            png_filename = f"{output_filename}.png"
            fig.savefig(png_filename, dpi=300, bbox_inches='tight', facecolor='white')
            generated_files.append(png_filename)
            print(f"Leyenda guardada como PNG: {png_filename}")
        
        # Mostrar el gráfico
        plt.show()
        
        return fig

# Función simplificada para mantener compatibilidad con tu código anterior
def save_legend_styled(filename='street_legend_updated.png', dpi=300, 
                      style_path='styles/matplotlib_style.mplstyle'):
    """
    Función simplificada que mantiene la interfaz similar a tu código original.
    """
    # Determinar formato basado en la extensión del archivo
    if filename.endswith('.pdf'):
        output_format = 'pdf'
        output_filename = filename[:-4]  # Remover extensión
    elif filename.endswith('.png'):
        output_format = 'png'
        output_filename = filename[:-4]  # Remover extensión
    else:
        output_format = 'png'
        output_filename = filename
    
    fig = generate_legend_with_style(
        style_path=style_path,
        output_filename=output_filename,
        output_format=output_format,
        figsize=(7.5, 1.8)
    )
    
    return fig

# Ejemplo de uso manteniendo tu interfaz original:
if __name__ == "__main__":
    
    
    # O usando la función más completa:
    generate_legend_with_style(
        style_path='styles/matplotlib_style.mplstyle',
        output_filename='street_legend_updated',
        output_format='pdf',  # Genera PDF y PNG
        figsize=(7.5, 0.8)
    )