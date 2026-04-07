import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
import sys
from datetime import datetime
import argparse

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

THEME = None


def load_fonts():
    """
    Load Roboto fonts from the fonts directory.
    Returns dict with font paths for different weights.
    """
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }

    for _, path in fonts.items():
        if not os.path.exists(path):
            print(f"⚠ Font not found: {path}")
            return None
    return fonts


FONTS = load_fonts()


def generate_output_filename(city, theme_name, street_detail):
    """
    Generate unique output filename with city, theme, detail mode, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = str(city).lower().replace(' ', '_')
    filename = f"{city_slug}_{theme_name}_{street_detail}_{timestamp}.png"
    return os.path.join(POSTERS_DIR, filename)


def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []

    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith('.json'):
            theme_name = file[:-5]
            themes.append(theme_name)
    return themes


def load_theme(theme_name="feature_based"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")

    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "road_default": "#3A3A3A"
        }

    with open(theme_file, 'r', encoding='utf-8') as f:
        theme = json.load(f)

    print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
    if 'description' in theme:
        print(f"  {theme['description']}")
    return theme


def create_gradient_fade(ax, color, location='bottom', zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))

    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]

    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    y_range = ylim[1] - ylim[0]
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end

    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect='auto',
        cmap=custom_cmap,
        zorder=zorder,
        origin='lower'
    )


def normalize_highway(highway):
    """
    Normalize highway tags that can be strings or lists.
    """
    if isinstance(highway, list):
        return highway[0] if highway else 'unclassified'
    return highway if highway else 'unclassified'


def get_edge_colors_by_type(G):
    """
    Assign colors to edges based on road type hierarchy.
    """
    edge_colors = []
    for _, _, data in G.edges(data=True):
        highway = normalize_highway(data.get('highway', 'unclassified'))

        if highway in ['motorway', 'motorway_link']:
            color = THEME['road_motorway']
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            color = THEME['road_primary']
        elif highway in ['secondary', 'secondary_link']:
            color = THEME['road_secondary']
        elif highway in ['tertiary', 'tertiary_link']:
            color = THEME['road_tertiary']
        elif highway in ['residential', 'living_street', 'unclassified', 'service']:
            color = THEME['road_residential']
        else:
            color = THEME['road_default']

        edge_colors.append(color)
    return edge_colors


def get_edge_widths_by_type(G, street_detail='standard'):
    """
    Assign line widths to edges based on road type.
    Detailed and ultra modes make smaller roads more visible.
    """
    edge_widths = []

    for _, _, data in G.edges(data=True):
        highway = normalize_highway(data.get('highway', 'unclassified'))

        if street_detail == 'ultra':
            if highway in ['motorway', 'motorway_link']:
                width = 1.6
            elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
                width = 1.3
            elif highway in ['secondary', 'secondary_link']:
                width = 1.1
            elif highway in ['tertiary', 'tertiary_link']:
                width = 0.95
            elif highway in ['residential', 'living_street', 'unclassified']:
                width = 0.85
            elif highway == 'service':
                width = 0.75
            else:
                width = 0.6

        elif street_detail == 'detailed':
            if highway in ['motorway', 'motorway_link']:
                width = 1.5
            elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
                width = 1.2
            elif highway in ['secondary', 'secondary_link']:
                width = 1.0
            elif highway in ['tertiary', 'tertiary_link']:
                width = 0.85
            elif highway in ['residential', 'living_street', 'unclassified']:
                width = 0.7
            elif highway == 'service':
                width = 0.6
            else:
                width = 0.5

        else:
            if highway in ['motorway', 'motorway_link']:
                width = 1.2
            elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
                width = 1.0
            elif highway in ['secondary', 'secondary_link']:
                width = 0.8
            elif highway in ['tertiary', 'tertiary_link']:
                width = 0.6
            else:
                width = 0.4

        edge_widths.append(width)

    return edge_widths


def get_coordinates(city, country):
    """
    Fetch coordinates for a given city and country using geopy.
    """
    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster_advance")
    time.sleep(1)

    location = geolocator.geocode(f"{city}, {country}")

    if location:
        print(f"✓ Found: {location.address}")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        return (location.latitude, location.longitude)

    raise ValueError(f"Could not find coordinates for {city}, {country}")


def get_street_network(point, dist, street_detail='standard'):
    """
    Fetch street network with selectable detail levels.
    """
    if street_detail == 'ultra':
        print("Using ultra street detail mode...")
        road_filter = (
            '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|'
            'secondary|secondary_link|tertiary|tertiary_link|residential|unclassified|'
            'living_street|service"]'
        )
        G = ox.graph_from_point(
            point,
            dist=dist,
            dist_type='bbox',
            custom_filter=road_filter,
            simplify=False,
            retain_all=True,
            truncate_by_edge=True
        )

    elif street_detail == 'detailed':
        print("Using detailed street detail mode...")
        road_filter = (
            '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|'
            'secondary|secondary_link|tertiary|tertiary_link|residential|unclassified|'
            'living_street|service"]'
        )
        G = ox.graph_from_point(
            point,
            dist=dist,
            dist_type='bbox',
            custom_filter=road_filter,
            simplify=False,
            retain_all=True,
            truncate_by_edge=True
        )

    else:
        print("Using standard street detail mode...")
        G = ox.graph_from_point(
            point,
            dist=dist,
            dist_type='bbox',
            network_type='all'
        )

    return G


def create_poster(city, country, point, dist, output_file, orientation='portrait', street_detail='standard'):
    print(f"\nGenerating map for {city}, {country}...")
    print(f"Street detail mode: {street_detail}")

    with tqdm(total=3, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        pbar.set_description("Downloading street network")
        G = get_street_network(point, dist, street_detail=street_detail)
        pbar.update(1)
        time.sleep(0.5)

        pbar.set_description("Downloading water features")
        try:
            water = ox.features_from_point(
                point,
                tags={'natural': 'water', 'waterway': 'riverbank'},
                dist=dist
            )
        except Exception:
            water = None
        pbar.update(1)
        time.sleep(0.3)

        pbar.set_description("Downloading parks/green spaces")
        try:
            parks = ox.features_from_point(
                point,
                tags={'leisure': 'park', 'landuse': 'grass'},
                dist=dist
            )
        except Exception:
            parks = None
        pbar.update(1)

    print("✓ All data downloaded successfully!")
    print("Rendering map...")

    if orientation == 'landscape':
        fig, ax = plt.subplots(figsize=(20, 12), facecolor=THEME['bg'])
    else:
        fig, ax = plt.subplots(figsize=(12, 16), facecolor=THEME['bg'])

    ax.set_facecolor(THEME['bg'])
    ax.set_position([0, 0, 1, 1])

    if water is not None and not water.empty:
        water.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1)

    if parks is not None and not parks.empty:
        parks.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=2)

    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(G)
    edge_widths = get_edge_widths_by_type(G, street_detail=street_detail)

    ox.plot_graph(
        G,
        ax=ax,
        bgcolor=THEME['bg'],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False,
        close=False
    )

    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)

    if FONTS:
        font_main = FontProperties(fname=FONTS['bold'], size=60)
        font_sub = FontProperties(fname=FONTS['light'], size=22)
        font_coords = FontProperties(fname=FONTS['regular'], size=14)
        font_attr = FontProperties(fname=FONTS['light'], size=8)
    else:
        font_main = FontProperties(family='monospace', weight='bold', size=60)
        font_sub = FontProperties(family='monospace', weight='normal', size=22)
        font_coords = FontProperties(family='monospace', size=14)
        font_attr = FontProperties(family='monospace', size=8)

    spaced_city = " ".join(list(str(city).upper()))

    ax.text(
        0.5, 0.14, spaced_city,
        transform=ax.transAxes,
        color=THEME['text'],
        ha='center',
        fontproperties=font_main,
        zorder=11
    )

    ax.text(
        0.5, 0.10, str(country).upper(),
        transform=ax.transAxes,
        color=THEME['text'],
        ha='center',
        fontproperties=font_sub,
        zorder=11
    )

    lat, lon = point
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    coords = f"{abs(lat):.4f}° {lat_dir} / {abs(lon):.4f}° {lon_dir}"

    ax.text(
        0.5, 0.07, coords,
        transform=ax.transAxes,
        color=THEME['text'],
        alpha=0.7,
        ha='center',
        fontproperties=font_coords,
        zorder=11
    )

    ax.plot(
        [0.4, 0.6], [0.125, 0.125],
        transform=ax.transAxes,
        color=THEME['text'],
        linewidth=1,
        zorder=11
    )

    ax.text(
        0.98, 0.02,
        "© OpenStreetMap contributors",
        transform=ax.transAxes,
        color=THEME['text'],
        alpha=0.5,
        ha='right',
        va='bottom',
        fontproperties=font_attr,
        zorder=11
    )

    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, facecolor=THEME['bg'])
    plt.close()

    print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    print("""
City Map Poster Generator - Advanced
====================================
Usage:
  python create_map_poster_advance.py --city <name> --country <name> [options]

Examples:
  python create_map_poster_advance.py --city "Abu Dhabi" --country "UAE" --theme noir --distance 8000 --street-detail detailed
  python create_map_poster_advance.py --lat 24.2517 --lng 54.3429 --theme warm_beige --distance 8000 --orientation landscape --street-detail detailed
  python create_map_poster_advance.py --lat 24.2517 --lng 54.3429 --theme noir --distance 5000 --orientation landscape --street-detail ultra
  python create_map_poster_advance.py --list-themes

Options:
  --city, -c                City name
  --country, -C             Country name
  --theme, -t               Theme name (default: feature_based)
  --distance, -d            Map radius in meters (default: 29000)
  --lat                     Custom latitude coordinate
  --lng                     Custom longitude coordinate
  --orientation, -o         portrait or landscape
  --street-detail           standard, detailed, or ultra (default: standard)
  --list-themes             List all available themes
""")


def list_themes():
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return

    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, 'r', encoding='utf-8') as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except Exception:
            display_name = theme_name
            description = ''

        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate advanced map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster_advance.py --city "Abu Dhabi" --country "UAE" --theme noir --street-detail detailed
  python create_map_poster_advance.py --lat 24.2517 --lng 54.3429 --theme warm_beige --distance 8000 --orientation landscape --street-detail detailed
  python create_map_poster_advance.py --lat 24.2517 --lng 54.3429 --theme noir --distance 5000 --street-detail ultra
  python create_map_poster_advance.py --list-themes
"""
    )

    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--theme', '-t', type=str, default='feature_based', help='Theme name (default: feature_based)')
    parser.add_argument('--distance', '-d', type=int, default=29000, help='Map radius in meters (default: 29000)')
    parser.add_argument('--lat', type=float, help='Custom latitude coordinate')
    parser.add_argument('--lng', type=float, help='Custom longitude coordinate')
    parser.add_argument('--orientation', '-o', type=str, choices=['portrait', 'landscape'], default='portrait',
                        help='Poster orientation: portrait or landscape (default: portrait)')
    parser.add_argument('--street-detail', type=str, choices=['standard', 'detailed', 'ultra'], default='standard',
                        help='Street detail mode: standard, detailed, or ultra (default: standard)')
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)

    if args.list_themes:
        list_themes()
        sys.exit(0)

    if args.lat is None and args.lng is None:
        if not args.city or not args.country:
            print("Error: Either provide --lat and --lng, OR provide --city and --country\n")
            print_examples()
            sys.exit(1)

    if (args.lat is None) != (args.lng is None):
        print("Error: You must provide both --lat and --lng together.\n")
        sys.exit(1)

    available_themes = get_available_themes()
    if args.theme not in available_themes:
        print(f"Error: Theme '{args.theme}' not found.")
        print(f"Available themes: {', '.join(available_themes)}")
        sys.exit(1)

    print("=" * 50)
    print("City Map Poster Generator - Advanced")
    print("=" * 50)

    try:
        THEME = load_theme(args.theme)

        if args.lat is not None and args.lng is not None:
            coords = (args.lat, args.lng)
            print(f"✓ Using custom coordinates: {args.lat}, {args.lng}")
        else:
            coords = get_coordinates(args.city, args.country)

        city_name = args.city if args.city else f"{args.lat}_{args.lng}"
        country_name = args.country if args.country else "custom"
        output_file = generate_output_filename(city_name, args.theme, args.street_detail)

        print("\n" + "=" * 50)
        create_poster(
            city_name,
            country_name,
            coords,
            args.distance,
            output_file,
            args.orientation,
            args.street_detail
        )
        print("✓ Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
