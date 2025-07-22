import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
import numpy as np
import os
import matplotlib.patches as mpatches
import geopandas as gpd
from Polygon_clustering import procesar_poligonos_y_generar_grafos, load_polygon_stats_from_txt, classify_polygon, plot_polygons_classification_png


def generate_urban_typologies_plot(style_path='styles/matplotlib_style.mplstyle',
                                   output_filename='urban_typologies_single_column',
                                   output_format='pdf',  # 'pdf', 'png', or 'both'
                                   locations=None):
    """
    Generates an urban typologies figure using OSMnx and saves it in the specified format.
    
    Parameters:
    -----------
    style_path : str
        Path to the matplotlib style file (.mplstyle).
    output_filename : str
        Base name for the output file (without extension).
    output_format : str
        Output format: 'pdf', 'png', or 'both'.
    locations : dict, optional
        Dictionary with locations. If None, default locations are used.
    
    Returns:
    --------
    str or list
        Path(s) to the generated file(s).
    """

    # Ensure output folder exists
    output_dir = 'Supporting_figures'
    os.makedirs(output_dir, exist_ok=True)

    # Enable OSMnx cache
    ox.settings.use_cache = True

    # Default locations if none are provided
    if locations is None:
        locations = {
        'Cuadrícula': ('Midtown Manhattan, New York City, NY, USA', 1000),
        'Callejón sin salida': ('Mission Viejo, CA, USA', 1200),
        'Orgánico': ('Alfama, Lisbon, Portugal', 600),
        'Híbrido': ('Canberra, Australia', 2000)
    }

    # Apply custom style with context manager
    with plt.style.context(style_path):
        fig, axes = plt.subplots(2, 2)
        axes_flat = axes.flatten()

        for i, (pattern_type, (location, distance)) in enumerate(locations.items()):
            print(f"Processing: {pattern_type}")
            ax = axes_flat[i]
            try:
                G = ox.graph_from_address(location, dist=distance, network_type='drive')
                G_proj = ox.project_graph(G)

                ox.plot_graph(G_proj, ax=ax, node_size=0, edge_color='black',
                              edge_linewidth=0.25, bgcolor='white', show=False, close=False)

                ax.set_title(pattern_type, fontsize=7, fontfamily='serif')

            except Exception as e:
                print(f"Error with {pattern_type}: {str(e)}")
                ax.text(0.5, 0.5, f'Error\n{pattern_type}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=6,
                        fontfamily='serif',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
                ax.set_title(pattern_type, fontsize=7, fontfamily='serif')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout(pad=0.255, w_pad=0.1, h_pad=0.595)

        generated_files = []

        # Save in selected format(s)
        if output_format.lower() in ['pdf', 'both']:
            pdf_path = os.path.join(output_dir, f"{output_filename}.pdf")
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
            generated_files.append(pdf_path)
            print(f"Figure saved as PDF: {pdf_path}")

        if output_format.lower() in ['png', 'both']:
            png_path = os.path.join(output_dir, f"{output_filename}.png")
            fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
            generated_files.append(png_path)
            print(f"Figure saved as PNG: {png_path}")

        plt.close(fig)

    return generated_files if len(generated_files) > 1 else generated_files[0]

def generate_street_legend(style_path='styles/matplotlib_style.mplstyle',
                           output_filename='street_legend_updated',
                           output_format='png',  # 'pdf', 'png', or 'both'
                           figsize=(7.5, 1.8)):
    """
    Generates a legend for street typologies using a custom style and saves it in the desired format.

    Parameters:
    -----------
    style_path : str
        Path to the matplotlib style file (.mplstyle).
    output_filename : str
        Base name of the output file (without extension).
    output_format : str
        Output format: 'pdf', 'png', or 'both'.
    figsize : tuple
        Figure size in inches (width, height).

    Returns:
    --------
    str or list
        Path(s) to the saved file(s).
    """

    # Ensure the output directory exists
    output_dir = 'Supporting_figures'
    os.makedirs(output_dir, exist_ok=True)

    # Use custom matplotlib style
    with plt.style.context(style_path):

        # Base colors for each main category
        base_colors = {
            'organic': "#2399CF",     # Blue for Organic
            'cul_de_sac': "#F13F3F",  # Red for Cul-de-sac
            'hybrid': "#E4CD4D",      # Yellow for Hybrid
            'gridiron': "#0C850C",    # Green for Gridiron
        }

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 2)
        ax.axis('off')

        y_center = 1
        rect_width = 0.65
        rect_height = 0.8
        spacing = 2.1
        x = 0.3

        def draw_hatched_patch(x, color, label):
            # White background
            bg_rect = plt.Rectangle((x, y_center - rect_height / 2), rect_width, rect_height,
                                    facecolor='white', edgecolor='black', linewidth=0.5)
            ax.add_patch(bg_rect)

            # Colored hatching
            hatch_rect = plt.Rectangle((x, y_center - rect_height / 2), rect_width, rect_height,
                                       facecolor='none', edgecolor=color, hatch='/////', linewidth=0.5)
            ax.add_patch(hatch_rect)

            ax.text(x + rect_width + 0.05, y_center, label, ha='left', va='center',
                    fontsize=8)

        # Organic
        draw_hatched_patch(x, base_colors['organic'], 'Organic')
        x += spacing

        # Organic Street⁺
        adjusted_color = np.clip(np.array(mcolors.to_rgb(base_colors['organic'])) * 0.55, 0, 1)
        draw_hatched_patch(x, adjusted_color, 'Street$^+$')
        x += spacing

        # Cul-de-sac
        draw_hatched_patch(x, base_colors['cul_de_sac'], 'Cul-de-sac')
        x += spacing

        # Cul-de-sac Street⁺
        adjusted_color = np.clip(np.array(mcolors.to_rgb(base_colors['cul_de_sac'])) * 0.5, 0, 1)
        draw_hatched_patch(x, adjusted_color, 'Street$^+$')
        x += spacing

        # Hybrid
        draw_hatched_patch(x, base_colors['hybrid'], 'Hybrid')
        x += spacing

        # Gridiron
        draw_hatched_patch(x, base_colors['gridiron'], 'Gridiron')
        x += spacing

        # Gridiron Street⁺
        adjusted_color = np.clip(np.array(mcolors.to_rgb(base_colors['gridiron'])) * 0.55, 0, 1)
        draw_hatched_patch(x, adjusted_color, 'Street$^+$')
        x += spacing

        # Gridiron Street⁻
        adjusted_color = np.clip(np.array(mcolors.to_rgb(base_colors['gridiron'])) + (1 - np.array(mcolors.to_rgb(base_colors['gridiron']))) * 0.45, 0, 1)
        draw_hatched_patch(x, adjusted_color, 'Street$^-$')

        plt.tight_layout()

        # Save figure
        generated_files = []

        if output_format.lower() in ['pdf', 'both']:
            pdf_path = os.path.join(output_dir, f"{output_filename}.pdf")
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
            generated_files.append(pdf_path)
            print(f"Legend saved as PDF: {pdf_path}")

        if output_format.lower() in ['png', 'both']:
            png_path = os.path.join(output_dir, f"{output_filename}.png")
            fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
            generated_files.append(png_path)
            print(f"Legend saved as PNG: {png_path}")

        plt.close(fig)

        return generated_files if len(generated_files) > 1 else generated_files[0]

def plot_polygons_classification_png(
    geojson_path,
    stats_dict,
    classify_func,
    output_png="polygons_classification.png",
    graph_dict=None
):
    """
    Reads a GeoDataFrame (geojson_path), assigns a 'class' to each polygon
    based on statistics in 'stats_dict', the 'classify_func', and graphs
    in 'graph_dict', and draws to a PNG (Matplotlib) with different colors per class.

    Parameters:
    -----------
    geojson_path : str
        Path to the GeoJSON file containing the polygons
    stats_dict : dict
        Dictionary with statistics for each polygon. Keys are (idx, sub_idx)
    classify_func : function
        Classification function that receives statistics and a graph as parameters
    output_png : str, optional
        Path to save the resulting image
    graph_dict : dict, optional
        Dictionary with road network graphs for each polygon.
        Keys may have a different format than those in stats_dict.

    Returns:
    --------
    gdf : GeoDataFrame
        The GeoDataFrame with an additional 'pattern' column containing the classification
    """

    # Load the GeoDataFrame
    gdf = gpd.read_file(geojson_path)
    
    # Create 'pattern' column with the class
    patterns = []
    
    for idx, row in gdf.iterrows():
        # Identify the polygon
        poly_id = row['poly_id'] if 'poly_id' in gdf.columns else idx
        key = (idx, 0)  # Assuming each row = sub-polygon 0
        
        if key in stats_dict:
            poly_stats = stats_dict[key]
            
            # Improved graph_dict handling
            G = None
            if graph_dict is not None:
                # 1. Extract main polygon ID
                if isinstance(poly_id, tuple) and len(poly_id) >= 1:
                    main_id = poly_id[0]
                else:
                    main_id = poly_id
                
                # 2. Generate possible key formats
                possible_keys = [main_id, str(main_id), idx, str(idx), key]
                
                # 3. Search the graph using possible keys
                for possible_key in possible_keys:
                    if possible_key in graph_dict:
                        G = graph_dict[possible_key]
                        break
                
                # 4. Check that G is a valid graph object
                if G is not None and not hasattr(G, 'number_of_nodes'):
                    print(f"Warning: Object for polygon {poly_id} is not a valid graph.")
                    G = None
            
            # Classify the polygon using the classification function
            # Passing both the statistics and the graph
            category = classify_func(poly_stats, G)
        else:
            print(f"Warning: No statistics found for polygon {poly_id}")
            category = "unknown"
            
        patterns.append(category)

    # Add the pattern column to the GeoDataFrame
    gdf["pattern"] = patterns

    # Count how many polygons per category
    pattern_counts = gdf["pattern"].value_counts()
    print("Pattern counts:")
    for pattern, count in pattern_counts.items():
        print(f"  - {pattern}: {count}")

    # Map each class to a color
    color_map = {
        'cul_de_sac': '#FF6B6B',   # Red for cul-de-sacs
        'gridiron': '#006400',     # Dark green for grid
        'organico': '#45B7D1',     # Blue for organic
        'hibrido': '#FDCB6E',      # Yellow for hybrid
        'unknown': '#CCCCCC'       # Gray for unknown
    }

    # Function to get color by category
    def get_color(cat):
        return color_map.get(cat, "black")

    # Get colors for each polygon
    plot_colors = [get_color(cat) for cat in gdf["pattern"]]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(
        ax=ax,
        color=plot_colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7  # Transparency for better visibility
    )

    # Manual legend with count for each category
    legend_patches = []
    for cat, col in color_map.items():
        count = pattern_counts.get(cat, 0)
        if count > 0:  # Only show present categories in legend
            patch = mpatches.Patch(
                color=col, 
                label=f"{cat} ({count})"
            )
            legend_patches.append(patch)
    
    ax.legend(
        handles=legend_patches, 
        title="Urban Fabric Types",
        loc="upper right",
        frameon=True,
        framealpha=0.9
    )

    ax.set_title("Morphological Classification of Urban Fabrics", fontsize=16)
    ax.set_axis_off()

    # Save image
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved to: {output_png}")
    
    return gdf


# Example usage
if __name__ == "__main__":
    generate_street_legend(
        style_path='styles/matplotlib_style.mplstyle',
        output_filename='street_legend_updated',
        output_format='pdf',  # or 'png', or 'both'
        figsize=(7.5, 0.8)
    )
    custom_locations = {
        'Grid': ('Midtown Manhattan, New York City, NY, USA', 1000),
        'Cul-De-Sac': ('Mission Viejo, CA, USA', 1200),
        'Organic': ('Alfama, Lisbon, Portugal', 600),
        'Hybrid': ('Canberra, Australia', 2000)
    }
    generate_urban_typologies_plot(
        style_path='styles/matplotlib_style.mplstyle',
        output_filename='urban_typologies_single_column',
        output_format='pdf',
        locations=custom_locations
    )





# 1. Load GeoJSON polygons
geojson_path = "GeoJSON_Export/medellin_ant/tracts/medellin_ant_tracts.geojson"
gdf = gpd.read_file(geojson_path)

# 2. Process polygons and generate graphs
graph_dict = procesar_poligonos_y_generar_grafos(gdf)

# 3. Load polygon statistics from .txt file
stats_txt = "Polygons_analysis/Medellin_ANT/stats/Polygon_Analisys_Medellin_ANT_sorted.txt"
stats_dict = load_polygon_stats_from_txt(stats_txt)

# 4. Classify polygons and save visualization
plot_polygons_classification_png(
    geojson_path=geojson_path,
    stats_dict=stats_dict,
    classify_fn=classify_polygon,
    output_png="morphological_classification.png",
    graph_dict=graph_dict
)
