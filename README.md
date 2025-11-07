# 🏙️ Urban Morphology and Mobility Patterns  
### *A Topological and Spatial Network Analysis*


### *Pre-Print*
[![arXiv](https://img.shields.io/badge/arXiv-2507.19648-b31b1b.svg)](https://arxiv.org/abs/2507.19648)


---

##  Overview

This repository hosts the full computational framework and data workflow developed for the research project  
**“Urban Morphology and Mobility Patterns: A Topological and Spatial Network Analysis”**,  
conducted at **Universidad EAFIT**, within the **Applied Mechanics and Engineering Physics Group**.

The project integrates **urban network topology**, **spatial geometry**, and **mobility analytics**  
to explore how morphological attributes of street networks relate to modal distributions of travel and accessibility across cities.

## ⚙️ Computational Workflow

The analytical pipeline is structured as follows:

1. **Data Acquisition and Cleaning**
   - Extraction of street networks via [OSMnx](https://github.com/gboeing/osmnx)
   - Integration of census tracts (*tracts*) and modal share data

2. **Morphological Metric Computation**
   - **Spatial metrics:** street density, angular variation, circularity  
   - **Topological metrics:** node degree, dead-end ratio, entropy of orientation

3. **Typology Classification**
   - Multi-Attribute Decision Making (MADM) framework  
   - Automatic identification of morphological types (Grid, Organic, Hybrid, Dendritic)

4. **Mobility Correlation Analysis**
   - Non-parametric tests (Kruskal–Wallis, Mann–Whitney)  
   - Estimation of modal effects and effect sizes (η)

5. **Visualization and Export**
   - t-SNE cluster visualization and heatmaps  
   - GeoJSON export for GIS interoperability

