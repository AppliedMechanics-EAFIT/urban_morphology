
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

