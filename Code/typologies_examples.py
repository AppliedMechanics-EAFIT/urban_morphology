
# import osmnx as ox
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# # Configurar OSMnx
# ox.settings.use_cache = True

# # Usar LaTeX en matplotlib para texto (necesita tener instalado LaTeX en sistema)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Times"],
#     "axes.titlesize": 10,
#     "axes.titleweight": "bold"
# })

# locations = {
#     'Grid': ('Midtown Manhattan, New York City, NY, USA', 1000),
#     'Cul-De-Sac': ('Mission Viejo, CA, USA', 1200),
#     'Organic': ('Alfama, Lisbon, Portugal', 600),
#     'Hybrid': ('Canberra, Australia', 2000)
# }

# # Figura pequeña para 1 columna ~ 3.3 x 3.3 pulgadas (~8.4 cm)
# fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.3))
# axes_flat = axes.flatten()

# for i, (pattern_type, (location, distance)) in enumerate(locations.items()):
#     print(f"Processing: {pattern_type}")
#     ax = axes_flat[i]
#     try:
#         G = ox.graph_from_address(location, dist=distance, network_type='drive')
#         G_proj = ox.project_graph(G)
#         ox.plot_graph(G_proj, ax=ax, node_size=0, edge_color='black',
#                       edge_linewidth=0.25, bgcolor='white', show=False, close=False)
#         ax.set_title(pattern_type)
#     except Exception as e:
#         print(f"Error with {pattern_type}: {str(e)}")
#         ax.text(0.5, 0.5, f'Error\n{pattern_type}', ha='center', va='center',
#                 transform=ax.transAxes, fontsize=8,
#                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
#         ax.set_title(pattern_type)

#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_aspect('equal')
#     for spine in ax.spines.values():
#         spine.set_visible(False)

# plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.6)  # subí h_pad para más espacio vertical

# output_filename = 'urban_typologies_single_column.pdf'
# with PdfPages(output_filename) as pdf:
#     pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)

# plt.show()
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
    "text.color": "black",        

})

def generate_legend():
    # Base colors for each main category
    base_colors = {
        'organic': "#2399CF",     # Blue for Organic
        'cul_de_sac': "#F13F3F",  # Red for Cul-de-sac
        'hybrid': "#E4CD4D",      # Yellow for Hybrid
        'gridiron': "#0C850C",    # Green for Gridiron
    }

    fig, ax = plt.subplots(figsize=(16, 2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 2)
    ax.axis('off')

    y_center = 1
    rect_width = 0.4
    rect_height = 0.4
    spacing = 2
    x = 0.5


    def draw_hatched_patch(x, color,  label):
        # Fondo blanco
        bg_rect = plt.Rectangle((x, y_center - rect_height / 2), rect_width, rect_height,
                                facecolor='white', edgecolor='black', linewidth=0.5)
        ax.add_patch(bg_rect)

        # Hachurado del color deseado
        hatch_rect = plt.Rectangle((x, y_center - rect_height / 2), rect_width, rect_height,
                                facecolor='none', edgecolor=color, hatch='/////', linewidth=0.5)
        ax.add_patch(hatch_rect)

        ax.text(x + rect_width + 0.05, y_center, label, ha='left', va='center', fontsize=16)

    def draw_patch(x, color, label):
        rect = plt.Rectangle((x, y_center - rect_height / 2), rect_width, rect_height,
                             facecolor=color, edgecolor='black', linewidth=0.5)
        rect.set_hatch('/')
        ax.add_patch(rect)
        ax.text(x + rect_width + 0.05, y_center, label, ha='left', va='center', fontsize=10)

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
    return fig

# Save legend image
def save_legend(filename='street_legend_updated.png', dpi=300):
    fig = generate_legend()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Leyenda guardada como: {filename}")

# Run
save_legend()
