import pandas as pd

def clean_and_filter_data(file_path, output_file_ABC, report_file_ABC):
    # Cargar el archivo Excel
    df = pd.read_excel(file_path, dtype=str)  # Leer como string para evitar errores de tipo

    # ---- Eliminar duplicados manteniendo la versión más actualizada ----
    key_columns = df.columns[4:10]  # Columnas E a J para comparación
    removed_rows = pd.DataFrame()  # DataFrame para almacenar filas eliminadas

    # Identificar duplicados basados en las columnas clave
    duplicates = df.duplicated(subset=key_columns, keep=False)  # Marca ambos duplicados como `True`

    # Filtrar los casos donde haya duplicados y uno de ellos tenga "YES" en la columna D
    for _, group in df[duplicates].groupby(list(key_columns)):
        if "YES" in group.iloc[:, 3].values:  # Columna D está en el índice 3
            to_remove = group[group.iloc[:, 3] == "NO"]  # Conservar "YES", eliminar "NO"
            removed_rows = pd.concat([removed_rows, to_remove])
            df = df.drop(to_remove.index)

    # ---- Eliminar filas con 3 o más valores vacíos en las columnas P a U ----
    key_columns_missing = df.columns[15:21]  # Columnas P a U (índices 15 a 20 en 0-based indexing)
    missing_values_rows = df[df[key_columns_missing].isna().sum(axis=1) >= 3]  # Filtrar filas con 3+ vacíos

    # Eliminar esas filas del DataFrame original y agregarlas al reporte
    removed_rows = pd.concat([removed_rows, missing_values_rows])
    df = df.drop(missing_values_rows.index)

    # Guardar los datos limpios
    df.to_excel(output_file_ABC, index=False)
    
    # Guardar las filas eliminadas en un informe
    removed_rows.to_excel(report_file_ABC, index=False)

    print(f"Total de filas eliminadas: {len(removed_rows)}")