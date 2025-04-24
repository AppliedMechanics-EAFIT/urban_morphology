## Functions to read the xlsx format of mobility data
import os
import matplotlib.pyplot as plt
from openpyxl import load_workbook, Workbook
import pandas as pd

## Class to define the different mobility information (The ABC of mobility)
class CiudadesABC:
    def __init__(self, index, ObsID, year, LastObservation, City, metro_names, Country, continent, 
                 region, subregion, state_name, state_abbr, population, longitude, latitude, 
                 Walking, Cycling, Motorbikes, Active, Bus, Car, IncomeGroup, DataSource, DataLink, GDPPP_2022):
        self.index = index  # Format like M10001
        self.ObsID = ObsID  # Format like ID0005
        self.year = year  # Year in numeric format
        self.LastObservation = LastObservation  # Variable text
        self.City = City  # City name
        self.metro_names = metro_names  # Metropolitan area name
        self.Country = Country  # Country
        self.continent = continent  # Continent
        self.region = region  # Region
        self.subregion = subregion  # Subregion
        self.state_name = str(state_name) if state_name not in [None, "", " "] else "Not applicable" # State name
        self.state_abbr = str(state_abbr) if state_abbr not in [None, "", " "] else "Not applicable"  # State abbreviation
        self.population = int(population) if population else None  # Population
        self.longitude = float(longitude)  # Geographic longitude
        self.latitude = float(latitude)  # Geographic latitude
        self.Walking = float(Walking) if Walking != "NA" else None  # Percentage of walking
        self.Cycling = float(Cycling) if Cycling != "NA" else None  # Percentage of cycling
        self.Motorbikes = float(Motorbikes) if Motorbikes != "NA" else None  # Percentage of motorbike use
        self.Active = float(Active)  # Percentage of active mobility
        self.Bus = float(Bus)  # Percentage of bus use
        self.Car = float(Car)  # Percentage of car use
        self.IncomeGroup = IncomeGroup  # Income group
        self.DataSource = DataSource  # Data source
        self.DataLink = DataLink  # Data link
        self.GDPPP_2022 = float(GDPPP_2022)  # Adjusted GDP per capita in 2022

    def __repr__(self):
        return (
            f"CiudadesABC(index={self.index}, ObsID={self.ObsID}, year={self.year}, "
            f"LastObservation={self.LastObservation}, City={self.City}, "
            f"metro_names={self.metro_names}, Country={self.Country}, continent={self.continent}, "
            f"region={self.region}, subregion={self.subregion}, state_name={self.state_name}, "
            f"state_abbr={self.state_abbr}, population={self.population}, longitude={self.longitude}, "
            f"latitude={self.latitude}, Walking={self.Walking}, Cycling={self.Cycling}, "
            f"Motorbikes={self.Motorbikes}, Active={self.Active}, Bus={self.Bus}, Car={self.Car}, "
            f"IncomeGroup={self.IncomeGroup}, DataSource={self.DataSource}, DataLink={self.DataLink}, "
            f"GDPPP_2022={self.GDPPP_2022})"
        )

## Function to read the xlsx file with the information
def read_nodes_from_excel(filename, sheet_name):
    wb = load_workbook(filename=filename, data_only=True)
    sheet = wb[sheet_name]

    nodes = []

    # Iterate over rows starting from the second (excluding headers)
    for row in sheet.iter_rows(min_row=2, values_only=True):
        node = CiudadesABC(
            index=row[0],
            ObsID=row[1],
            year=row[2],
            LastObservation=row[3],
            City=row[4],
            metro_names=row[5],
            Country=row[6],
            continent=row[7],
            region=row[8],
            subregion=row[9],
            state_name=row[10],
            state_abbr=row[11],
            population=row[12],
            longitude=row[13],
            latitude=row[14],
            Walking=row[15],
            Cycling=row[16],
            Motorbikes=row[17],
            Active=row[18],
            Bus=row[19],
            Car=row[20],
            IncomeGroup=row[21],
            DataSource=row[22],
            DataLink=row[23],
            GDPPP_2022=row[24],
        )
        nodes.append(node)

    wb.close()
    return nodes

def clean_and_filter_data(file_path, output_file_ABC, report_file_ABC):
    # Load the Excel file
    df = pd.read_excel(file_path, dtype=str)  # Read as string to avoid type errors

    # ---- Remove duplicates keeping the most updated version ----
    key_columns = df.columns[4:10]  # Columns E to J for comparison
    removed_rows = pd.DataFrame()  # DataFrame to store removed rows

    # Identify duplicates based on key columns
    duplicates = df.duplicated(subset=key_columns, keep=False)  # Mark both duplicates as `True`

    # Filter cases where duplicates exist and one of them has "YES" in column D
    for _, group in df[duplicates].groupby(list(key_columns)):
        if "YES" in group.iloc[:, 3].values:  # Column D is at index 3
            to_remove = group[group.iloc[:, 3] == "NO"]  # Keep "YES", remove "NO"
            removed_rows = pd.concat([removed_rows, to_remove])
            df = df.drop(to_remove.index)

    # ---- Remove rows with 3 or more missing values in columns P to U ----
    key_columns_missing = df.columns[15:21]  # Columns P to U (indices 15 to 20 in 0-based indexing)
    missing_values_rows = df[df[key_columns_missing].isna().sum(axis=1) >= 3]  # Filter rows with 3+ missing values

    # Remove those rows from the original DataFrame and add them to the report
    removed_rows = pd.concat([removed_rows, missing_values_rows])
    df = df.drop(missing_values_rows.index)

    # Save the cleaned data
    df.to_excel(output_file_ABC, index=False)
    
    # Save the removed rows in a report
    removed_rows.to_excel(report_file_ABC, index=False)

    print(f"Total removed rows: {len(removed_rows)}")
