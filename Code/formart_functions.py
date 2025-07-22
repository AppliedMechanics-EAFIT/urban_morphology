import os
import geopandas as gpd

def convert_shapefile_to_geojson(shapefile_paths, output_directory):

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    geojson_data = {}

    for shapefile_path in shapefile_paths:
        gdf = gpd.read_file(shapefile_path)

        # Convert to EPSG:4326 if needed
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        # Convert the geodataframe to GeoJSON format
        geojson_string = gdf.to_json()

        # Define the output JSON file name
        shapefile_name = os.path.basename(shapefile_path).replace(".shp", ".geojson")
        output_path = os.path.join(output_directory, shapefile_name)

        # Save the GeoJSON file
        with open(output_path, "w") as f:
            f.write(geojson_string)

        # Store the file path in the result dictionary
        geojson_data[shapefile_path] = output_path

    return geojson_data

def filter_periphery_polygons(in_geojson, out_geojson, area_threshold=5.0):
    """
    Reads a GeoJSON (in_geojson), removes polygons with area >= area_threshold (km²),
    and saves a new GeoJSON to out_geojson with the filtered polygons.
    Returns a GeoDataFrame with the result.

    Parameters:
    -----------
    in_geojson : path to the original GeoJSON file.
    out_geojson: path where the filtered GeoJSON will be saved.
    area_threshold: float, area threshold in km²; 
                    polygons with area >= threshold are considered "rural" and excluded.

    Returns:
    --------
    GeoDataFrame with the “urban” polygons (area < area_threshold).
    """

    # 1. Load the GeoDataFrame
    gdf = gpd.read_file(in_geojson)
    print(f"Read: {in_geojson} with {len(gdf)} total polygons.")

    # 2. Reproject to a metric system to calculate area in km² (e.g., EPSG:3395 or 3857)
    #    EPSG:3395 (World Mercator) or 3857 (Pseudo Mercator). Adjust according to your region for greater precision.
    gdf_merc = gdf.to_crs(epsg=3395)

    # 3. Calculate area in km²
    gdf["area_km2"] = gdf_merc.geometry.area / 1e6

    # 4. Filter
    mask_urban = gdf["area_km2"] < area_threshold
    gdf_filtered = gdf[mask_urban].copy()
    print(f"{len(gdf) - len(gdf_filtered)} polygons excluded for being >= {area_threshold} km².")

    # 5. Save as new GeoJSON
    #    (if you don’t want the "area_km2" column in the result, drop it beforehand)
    gdf_filtered.drop(columns=["area_km2"], inplace=True)
    gdf_filtered.to_file(out_geojson, driver="GeoJSON")
    print(f"Filtered file saved to: {out_geojson} with {len(gdf_filtered)} polygons.\n")

    return gdf_filtered

def match_polygons_by_area(gdfA, gdfB, area_ratio_threshold=0.9, out_csv=None):
    """
    Matches each polygon from gdfA with the polygon from gdfB that has the most
    intersection area (area(A ∩ B) / area(A)).

    - It is assumed that gdfA and gdfB share the same projection (CRS).
    - To avoid conflicts with B's geometry, we clone B's geometry
      into a column 'geomB' before the sjoin.
    - At the end, we generate a DataFrame with (indexA, indexB, area_ratio).
    - If area_ratio < area_ratio_threshold it is not considered a match.

    Parameters:
    -----------
    gdfA, gdfB : GeoDataFrames
        With Polygon geometries
    area_ratio_threshold : float
        Minimum threshold to consider a match.
    out_csv : str | None
        Path to save CSV, or None if not desired.

    Returns:
    --------
    DataFrame with columns:
      - indexA : index of polygon A
      - indexB : index of polygon B
      - area_ratio : fraction of A's area overlapping with B
    """

    # Local copies (to avoid modifying the original gdfs)
    gdfA = gdfA.copy()
    gdfB = gdfB.copy()

    # Ensure indices for tracking
    gdfA["indexA"] = gdfA.index
    gdfB["indexB"] = gdfB.index

    # 1. Clone B's geometry into a regular column 'geomB'
    #    to avoid depending on how sjoin renames B's geometry
    gdfB["geomB"] = gdfB.geometry

    # 2. Perform sjoin with intersects
    #    - main geometry in B remains as 'geometry'
    #    - sjoin will use that geometry for intersection
    joined = gpd.sjoin(
        gdfA,
        gdfB.drop(columns="geomB"),  # native geometry for sjoin
        how="inner",
        predicate="intersects",
        lsuffix="_A",
        rsuffix="_B"
    )

    # 'joined' has 'geometry' = A's geometry
    # and B's columns except geomB (because we dropped it).
    # However, we kept 'geomB' in gdfB to reassign it after the sjoin

    # 3. Join B's 'geomB' column to joined manually:
    #    Each row in joined has an 'indexB' that indicates which polygon from B it was.
    #    We can do a merge with gdfB[["indexB","geomB"]] (in memory).
    joined = joined.merge(
        gdfB[["indexB", "geomB"]],
        on="indexB",
        how="left"
    )

    # 4. Calculate area_ratio = area(A ∩ geomB) / area(A)
    def compute_area_ratio(row):
        geomA = row["geometry"]   # polygon from A
        geomB_ = row["geomB"]     # original polygon from B
        if geomA is None or geomB_ is None:
            return 0.0
        inter = geomA.intersection(geomB_)
        if inter.is_empty:
            return 0.0
        areaA = geomA.area
        if areaA == 0:
            return 0.0
        return inter.area / areaA

    joined["area_ratio"] = joined.apply(compute_area_ratio, axis=1)

    # 5. For each indexA, keep the B polygon with the highest area_ratio
    best_matches = joined.loc[
        joined.groupby("indexA")["area_ratio"].idxmax()
    ].copy()

    # 6. Filter by threshold
    best_matches = best_matches[best_matches["area_ratio"] >= area_ratio_threshold]

    # 7. Extract a DataFrame with indexA, indexB, area_ratio
    df_out = best_matches[["indexA", "indexB", "area_ratio"]].reset_index(drop=True)

    # 8. Save CSV if desired
    if out_csv:
        df_out.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} with {len(df_out)} matches. Threshold={area_ratio_threshold}")

    return df_out

def add_polygon_ids_to_geojson(
    input_geojson_path,
    output_geojson_path=None,
    id_field_name="Py_poly_id"
):
    gdf = gpd.read_file(input_geojson_path)
    gdf[id_field_name] = None
    gdf["is_multipolygon"] = False
    gdf["num_sub_polygons"] = 1

    id_mapping = {}

    for idx, row in gdf.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            gdf.at[idx, id_field_name] = f"empty_{idx}"
            continue

        if geom.geom_type == "Polygon":
            poly_id = f"{idx}-0"
            gdf.at[idx, id_field_name] = poly_id
            id_mapping[poly_id] = {
                "original_idx": idx,
                "sub_idx": 0,
                "area": geom.area,
                "perimeter": geom.length,
                "tuple_id": f"({idx}, 0)"
            }

        elif geom.geom_type == "MultiPolygon":
            subpolys = list(geom.geoms)
            gdf.at[idx, "is_multipolygon"] = True
            gdf.at[idx, "num_sub_polygons"] = len(subpolys)

            id_list = []
            for sub_idx, _ in enumerate(subpolys):
                poly_id = f"{idx}-{sub_idx}"
                id_list.append(poly_id)
                id_mapping[poly_id] = {
                    "original_idx": idx,
                    "sub_idx": sub_idx,
                    "tuple_id": f"({idx}, {sub_idx})"
                }

            gdf.at[idx, id_field_name] = "|".join(id_list)

    if output_geojson_path is None:
        base, ext = os.path.splitext(input_geojson_path)
        output_geojson_path = f"{base}_with_poly_ids{ext}"

    gdf.to_file(output_geojson_path, driver="GeoJSON")
    return output_geojson_path



# if __name__ == "__main__":
#     geojson_file = "GeoJSON_Export/medellin_ant/tracts/medellin_ant_tracts.geojson"
#     add_polygon_ids_to_geojson(input_geojson_path=geojsson_file, id_field_name="Py_poly_id")
#     print("GeoJson file with poly_id included processed")


# gdf_filtrado = filter_periphery_polygons(
#     in_geojson="Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA.geojson",
#     out_geojson="Poligonos_Medellin/Json_files/EOD_2017_SIT_only_AMVA_URBANO.geojson",
#     area_threshold=5.0  # Ajusta a tu criterio
# )



# # ========================================
# if __name__ == "__main__":

#     shpA = "Poligonos_Medellin/EOD_2017_SIT_only_AMVA.shp"
#     shpB = "Poligonos_Medellin/eod_gen_trips_mode.shp"

#     print("Leyendo shapefiles A y B...")
#     gdfA = gpd.read_file(shpA)
#     gdfB = gpd.read_file(shpB)

#     # Asegurar mismo CRS
#     if gdfA.crs != gdfB.crs:
#         gdfB = gdfB.to_crs(gdfA.crs)
#         print(f"Reproyectado B a {gdfA.crs}")

#     # Llamar función con threshold=0.9
#     print("Iniciando match_polygons_by_area con area_ratio_threshold=0.9")
#     df_matches = match_polygons_by_area(
#         gdfA,
#         gdfB,
#         area_ratio_threshold=0.9,
#         out_csv="Poligonos_Medellin/Resultados/Matchs_A_B/matches_by_area.csv"
#     )

#     print("Algunos matches:")
#     print(df_matches.head())
#     print(f"Total matches: {len(df_matches)}")