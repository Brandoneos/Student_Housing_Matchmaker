import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

geocoded_colleges_df = pd.read_csv('chicago_colleges_geoceded.csv')

# Cleaned Chicago Rentals ------------------
chicago_rentals_df = pd.read_csv("Zillow Rental (Chicago Area 03_03_2025).csv")
chicago_rentals_df = chicago_rentals_df[['Price', 'Address', 'Listing URL', 'Short Address', 'Zip', 'Beds', 'Baths', 'Raw Property Details', 'Latitude', 'Longitude']].reset_index(drop=True)

# Keep Beds and Baths as floats, cleaned NaN baths
chicago_rentals_df['Baths'] = chicago_rentals_df['Baths'].fillna(1)
chicago_rentals_df['Beds'] = chicago_rentals_df['Beds'].astype(float)

# Dropped undisclosed Address from database so left with 2731 rows.
chicago_rentals_df = chicago_rentals_df[chicago_rentals_df['Short Address'] != '(undisclosed Address)'].reset_index(drop=True)

# Cleaned Price column to show as type int
chicago_rentals_df['Price'] = (
    chicago_rentals_df['Price']
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)

chicago_rentals_df['Price'] = pd.to_numeric(chicago_rentals_df['Price'], errors='coerce')
chicago_rentals_df['Price'] = chicago_rentals_df['Price'].astype(float)


divvy_stations_df = pd.read_csv("Divvy_Bicycle_Stations_20250405.csv")
divvy_stations_df = divvy_stations_df[['Station Name', 'Latitude', 'Longitude', 'Location']]


def haversine(lat1, long1, lat2, long2):
    R = 3963.19  # Radius of the Earth in miles
    lat1 = np.radians(lat1)
    long1 = np.radians(long1)
    lat2 = np.radians(lat2)
    long2 = np.radians(long2)

    dlong = long2 - long1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def distance_to_nearest_divvy(row):
    distances = haversine(
        row['Latitude'], row['Longitude'], divvy_stations_df['Latitude'].values, divvy_stations_df['Longitude'].values
    )

    if len(distances) > 0:
        return np.min(distances)
    
    return np.nan

chicago_rentals_df['nearest_divvy'] = chicago_rentals_df.apply(distance_to_nearest_divvy, axis=1)
print(chicago_rentals_df[['nearest_divvy']])
sns.histplot(chicago_rentals_df['nearest_divvy'], bins=30, kde=True)
plt.xlabel("Distance to Nearest Divvy Station (miles)")
plt.title("Distribution of Distance to Nearest Divvy Station for Apartment Listings")
plt.axvline(0.25, color='red', linestyle='dashed', linewidth=1, label='0.25 Mile Threshold')
plt.legend()
plt.show()
