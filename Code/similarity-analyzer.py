import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

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

# End of Cleaned Chicago Rentals --------------------

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


# Start of our Recommendation System

# Get the School you want to find recommendations for
school = geocoded_colleges_df[geocoded_colleges_df['Name'] == 'University of Illinois Chicago'].iloc[0]
school_lat = school['Latitude']
school_long = school['Longitude']

# We're going to create a dataframe (our feature vector) for preferences to perform a similarity scoring analysis on so we can get best recommendation options
trimmed_chicago_rentals_df = chicago_rentals_df[['Price', 'Address', 'Listing URL', 'Latitude', 'Longitude', 'Beds', 'Baths']].copy()

# Check the distance from our rentals to the university
def distance_to_university(row):
    distances = haversine(
        row['Latitude'], row['Longitude'], school_lat, school_long
    )

    row['distance_to_university'] = distances
    return row
trimmed_chicago_rentals_df = trimmed_chicago_rentals_df.apply(distance_to_university, axis=1)


# Now add nearby divvy stations, our threshold for "nearby" is going to be 0.5 miles away from the listing
nearby_threshold = 0.5

def count_nearby_divvy_stations(row):
    distances = haversine(
        row['Latitude'], row['Longitude'], divvy_stations_df['Latitude'].values, divvy_stations_df['Longitude'].values
    )

    # Return the amount of nearby divvy stations to the apartment listing
    return np.sum(distances <= nearby_threshold)

trimmed_chicago_rentals_df['nearby_divvy'] = trimmed_chicago_rentals_df.apply(count_nearby_divvy_stations, axis=1)
trimmed_chicago_rentals_df = trimmed_chicago_rentals_df[['Price', 'Address', 'Listing URL', 'Beds', 'Baths', 'distance_to_university', 'nearby_divvy']]

# Now with our edited dataframe, we are going to normalize our features
location = trimmed_chicago_rentals_df[['Address', 'Listing URL']]
trimmed_chicago_rentals_df = trimmed_chicago_rentals_df.dropna()
trimmed_chicago_rentals_df = trimmed_chicago_rentals_df.drop(columns=['Address', 'Listing URL'])

scaler = MinMaxScaler()

normalized_listings = scaler.fit_transform(trimmed_chicago_rentals_df)
# print(normalized_listings)

rentals_normalized_df = pd.DataFrame(normalized_listings, columns=trimmed_chicago_rentals_df.columns)
# print(rentals_normalized_df)

# Student Preferences
student_preferences = {
    'Price': [1400],
    'Beds': [1],
    'Baths': [1],
    'distance_to_university': [.5], # in miles
    'nearby_divvy': [2] # at least two stations nearby
}

student_df = pd.DataFrame(student_preferences)

normalized_student = scaler.transform(student_df)

student_normalized_df = pd.DataFrame(normalized_student, columns=trimmed_chicago_rentals_df.columns)
print(student_normalized_df)

similarity_scores = cosine_similarity(rentals_normalized_df, student_normalized_df)

similarity_df = pd.DataFrame(similarity_scores, columns=['similarity'], index=trimmed_chicago_rentals_df.index)
results = pd.concat([similarity_df, location], axis=1)
results = results.sort_values(by='similarity', ascending=False)
print(results[['Listing URL']].values)
