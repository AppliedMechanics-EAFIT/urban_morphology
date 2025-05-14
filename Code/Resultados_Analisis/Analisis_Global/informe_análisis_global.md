# Informe de Análisis Global de Movilidad

## Análisis para la variable: a

### Efectos principales (ANOVA)

| Fuente | Suma cuadrados | gl | F | p-valor |
|--------|---|---|---|---|
| C(pattern) | 1.4813 | 3 | 24.0843 | 0.0000 |
| C(city) | 3.7306 | 5 | 36.3925 | 0.0000 |
| Residual | 24.8281 | 1211 | nan | nan |

### Resumen de efectos entre ciudades y patrones

- 5 de 5 ciudades muestran diferencias significativas entre patrones
- 4 de 4 patrones muestran diferencias significativas entre ciudades

### Efectos significativos

| Variable | Coeficiente | p-value |
|----------|-------------|--------|
| Intercept | 0.1691 | 0.0000 |
| C(pattern)[T.gridiron] | 0.0597 | 0.0004 |
| C(city)[T.Chandler_AZ] | -0.1351 | 0.0000 |
| C(city)[T.Peachtree_GA] | -0.1554 | 0.0001 |
| C(city)[T.Philadelphia_PA] | -0.1179 | 0.0000 |
| C(city)[T.Salt_Lake_UT] | -0.1341 | 0.0000 |
| C(city)[T.Santa_Fe_NM] | -0.1137 | 0.0000 |

## Análisis para la variable: b

### Efectos principales (ANOVA)

| Fuente | Suma cuadrados | gl | F | p-valor |
|--------|---|---|---|---|
| C(pattern) | 2.0480 | 3 | 49.2105 | 0.0000 |
| C(city) | 5.6440 | 5 | 81.3717 | 0.0000 |
| Residual | 16.7992 | 1211 | nan | nan |

### Resumen de efectos entre ciudades y patrones

- 2 de 5 ciudades muestran diferencias significativas entre patrones
- 4 de 4 patrones muestran diferencias significativas entre ciudades

### Efectos significativos

| Variable | Coeficiente | p-value |
|----------|-------------|--------|
| Intercept | 0.2106 | 0.0000 |
| C(pattern)[T.gridiron] | 0.0898 | 0.0000 |
| C(city)[T.Chandler_AZ] | -0.2049 | 0.0000 |
| C(city)[T.Peachtree_GA] | -0.2029 | 0.0000 |
| C(city)[T.Philadelphia_PA] | -0.0997 | 0.0000 |
| C(city)[T.Salt_Lake_UT] | -0.2063 | 0.0000 |
| C(city)[T.Santa_Fe_NM] | -0.2013 | 0.0000 |

## Análisis para la variable: bicycle

## Análisis para la variable: c

### Efectos principales (ANOVA)

| Fuente | Suma cuadrados | gl | F | p-valor |
|--------|---|---|---|---|
| C(pattern) | 4.9542 | 3 | 37.5034 | 0.0000 |
| C(city) | 20.1968 | 5 | 91.7342 | 0.0000 |
| Residual | 53.3243 | 1211 | nan | nan |

### Resumen de efectos entre ciudades y patrones

- 5 de 5 ciudades muestran diferencias significativas entre patrones
- 4 de 4 patrones muestran diferencias significativas entre ciudades

### Efectos significativos

| Variable | Coeficiente | p-value |
|----------|-------------|--------|
| Intercept | 0.5258 | 0.0000 |
| C(pattern)[T.gridiron] | -0.0699 | 0.0047 |
| C(pattern)[T.hibrido] | 0.0828 | 0.0057 |
| C(pattern)[T.organico] | 0.0728 | 0.0040 |
| C(city)[T.Chandler_AZ] | 0.3781 | 0.0000 |
| C(city)[T.Peachtree_GA] | 0.4389 | 0.0000 |
| C(city)[T.Philadelphia_PA] | 0.2113 | 0.0000 |
| C(city)[T.Salt_Lake_UT] | 0.3717 | 0.0000 |
| C(city)[T.Santa_Fe_NM] | 0.3625 | 0.0000 |

## Análisis para la variable: car

## Análisis para la variable: data

## Análisis para la variable: transit

## Análisis para la variable: walked

## Diagnóstico de datos

### data_diagnosis

- **total_records**: 1220
- **total_patterns**: 4
- **total_cities**: 6
- **filled_combinations**: 23
- **total_combinations**: 24
- **rare_patterns**: []
- **rare_cities**: ['Santa_Fe_NM', 'Peachtree_GA']
- **pattern_counts**: {'gridiron': 666, 'organico': 340, 'hibrido': 119, 'cul_de_sac': 95}
- **city_counts**: {'Philadelphia_PA': 627, 'Boston_MA': 356, 'Chandler_AZ': 107, 'Salt_Lake_UT': 72, 'Santa_Fe_NM': 42, 'Peachtree_GA': 16}

## Conclusiones generales

- Las principales variables de movilidad muestran patrones distintivos según la estructura urbana.
- La ciudad tiene un efecto importante en las variables de movilidad, lo que indica
  que los factores contextuales locales influyen significativamente.
- Los hallazgos sugieren que las políticas de movilidad deberían adaptarse tanto al
  patrón urbano como al contexto específico de cada ciudad.

### Hallazgos significativos

- **a**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
- **b**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
- **c**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
- **car_mode**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
- **transit_mode**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
- **bicycle_mode**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
- **walked_mode**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
- **car_share**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
- **transit_share**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
- **bicycle_share**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
- **walked_share**: diferencias significativas entre patrones urbanos, diferencias significativas entre ciudades.
