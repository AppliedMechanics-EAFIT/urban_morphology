import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
import numpy as np
import os

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
