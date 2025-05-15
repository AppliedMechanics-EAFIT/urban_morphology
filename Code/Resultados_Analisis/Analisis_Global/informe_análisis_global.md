# Informe de Análisis Global de Movilidad

## Análisis para la variable: a

### Efectos principales (ANOVA)

| Fuente | Suma cuadrados | gl | F | p-valor |
|--------|---|---|---|---|
| C(pattern) | 1.0143 | 3 | 16.1861 | 0.0000 |
| C(city) | 4.0793 | 5 | 39.0592 | 0.0000 |
| Residual | 25.2952 | 1211 | nan | nan |

### Resumen de efectos entre ciudades y patrones

- 4 de 5 ciudades muestran diferencias significativas entre patrones
- 4 de 4 patrones muestran diferencias significativas entre ciudades

### Efectos significativos

| Variable | Coeficiente | p-value |
|----------|-------------|--------|
| Intercept | 0.1819 | 0.0000 |
| C(pattern)[T.gridiron] | 0.0473 | 0.0000 |
| C(pattern)[T.hibrido] | -0.0353 | 0.0412 |
| C(city)[T.Chandler_AZ] | -0.1520 | 0.0000 |
| C(city)[T.Peachtree_GA] | -0.1705 | 0.0000 |
| C(city)[T.Philadelphia_PA] | -0.1181 | 0.0000 |
| C(city)[T.Salt_Lake_UT] | -0.1360 | 0.0000 |
| C(city)[T.Santa_Fe_NM] | -0.1285 | 0.0000 |

## Análisis para la variable: b

### Efectos principales (ANOVA)

| Fuente | Suma cuadrados | gl | F | p-valor |
|--------|---|---|---|---|
| C(pattern) | 2.1245 | 3 | 51.2834 | 0.0000 |
| C(city) | 6.6957 | 5 | 96.9763 | 0.0000 |
| Residual | 16.7227 | 1211 | nan | nan |

### Resumen de efectos entre ciudades y patrones

- 2 de 5 ciudades muestran diferencias significativas entre patrones
- 4 de 4 patrones muestran diferencias significativas entre ciudades

### Efectos significativos

| Variable | Coeficiente | p-value |
|----------|-------------|--------|
| Intercept | 0.2359 | 0.0000 |
| C(pattern)[T.gridiron] | 0.0774 | 0.0000 |
| C(pattern)[T.organico] | -0.0239 | 0.0192 |
| C(city)[T.Chandler_AZ] | -0.2210 | 0.0000 |
| C(city)[T.Peachtree_GA] | -0.2254 | 0.0000 |
| C(city)[T.Philadelphia_PA] | -0.1046 | 0.0000 |
| C(city)[T.Salt_Lake_UT] | -0.2139 | 0.0000 |
| C(city)[T.Santa_Fe_NM] | -0.2156 | 0.0000 |

## Análisis para la variable: bicycle

## Análisis para la variable: c

### Efectos principales (ANOVA)

| Fuente | Suma cuadrados | gl | F | p-valor |
|--------|---|---|---|---|
| C(pattern) | 4.0995 | 3 | 30.5441 | 0.0000 |
| C(city) | 22.9350 | 5 | 102.5279 | 0.0000 |
| Residual | 54.1789 | 1211 | nan | nan |

### Resumen de efectos entre ciudades y patrones

- 4 de 5 ciudades muestran diferencias significativas entre patrones
- 4 de 4 patrones muestran diferencias significativas entre ciudades

### Efectos significativos

| Variable | Coeficiente | p-value |
|----------|-------------|--------|
| Intercept | 0.5242 | 0.0000 |
| C(pattern)[T.gridiron] | -0.0697 | 0.0000 |
| C(pattern)[T.hibrido] | 0.1077 | 0.0000 |
| C(pattern)[T.organico] | 0.0661 | 0.0003 |
| C(city)[T.Chandler_AZ] | 0.4116 | 0.0000 |
| C(city)[T.Peachtree_GA] | 0.4511 | 0.0000 |
| C(city)[T.Philadelphia_PA] | 0.2105 | 0.0000 |
| C(city)[T.Salt_Lake_UT] | 0.3715 | 0.0000 |
| C(city)[T.Santa_Fe_NM] | 0.3885 | 0.0000 |

## Análisis para la variable: car

## Análisis para la variable: data

## Análisis para la variable: transit

## Análisis para la variable: walked

## Diagnóstico de datos

### data_diagnosis

- **total_records**: 1220
- **total_patterns**: 4
- **total_cities**: 6
- **filled_combinations**: 22
- **total_combinations**: 24
- **rare_patterns**: []
- **rare_cities**: ['Santa_Fe_NM', 'Peachtree_GA']
- **pattern_counts**: {'gridiron': 542, 'cul_de_sac': 380, 'organico': 209, 'hibrido': 89}
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
