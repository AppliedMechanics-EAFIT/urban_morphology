import pandas as pd

def remove_duplicate_observations(file_path, output_file_ABC, report_file_ABC):
    # Cargar el archivo Excel
    df = pd.read_excel(file_path, dtype=str)  # Leer todo como string para evitar errores

    # Definir las columnas clave para la comparación (E a J)
    key_columns = df.columns[4:10]  # Tomamos columnas E a J (índices 4 a 9 en 0-based indexing)

    # Crear un DataFrame para almacenar las filas eliminadas
    removed_rows = pd.DataFrame()

    # Identificar duplicados basados en las columnas clave
    duplicates = df.duplicated(subset=key_columns, keep=False)  # `keep=False` marca ambos duplicados como `True`

    # Filtrar los casos donde haya duplicados y uno de ellos tenga "YES" en la columna D
    for _, group in df[duplicates].groupby(list(key_columns)):
        # Si hay un "YES" en la columna D, conservarlo y eliminar los "NO"
        if "YES" in group.iloc[:, 3].values:  # Columna D está en el índice 3
            to_remove = group[group.iloc[:, 3] == "NO"]
            removed_rows = pd.concat([removed_rows, to_remove])
            df = df.drop(to_remove.index)  # Eliminar las filas con "NO"

    # Guardar los datos limpios
    df.to_excel(output_file_ABC, index=False)
    
    # Guardar las filas eliminadas en un informe
    removed_rows.to_excel(report_file_ABC, index=False)
