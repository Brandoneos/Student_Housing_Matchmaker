import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def haversine(lat1, long1, lat2, long2):
    R = 3963.19
    # lat1, long1, lat2, long2 = map(np.radians, [lat1, long1, lat2, long2])
    lat1 = np.radians(lat1)
    long1 = np.radians(long1)
    lat2 = np.radians(lat2)
    long2 = np.radians(long2)

    dlong = long2 - long1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def calculateAndInsertDistance(df, university):
    df['distance'] = df.apply(lambda x: haversine(x['Latitude'], x['Longitude'], university['Latitude'], university['Longitude']), axis=1)

geocoded_colleges_df = pd.read_csv('chicago_colleges_geoceded.csv')
school_name = "University of Illinois Chicago"
university = geocoded_colleges_df[geocoded_colleges_df['Name'] == school_name].copy()
geocoded_colleges_df = pd.read_csv('chicago_colleges_geoceded.csv')

# # 41.873779 -87.651001 - UIC
geocoded_colleges_df1 = geocoded_colleges_df[['Name','Latitude','Longitude']].copy()
# geocoded_colleges_df1['distance'] = np.sqrt((abs(geocoded_colleges_df1['Latitude'] - 41.873779) ** 2 )+ 
#                                           (abs(geocoded_colleges_df1['Longitude'] - -87.651001) ** 2))

calculateAndInsertDistance(geocoded_colleges_df1, university)
# geocoded_colleges_df1
# subset = geocoded_colleges_df.iloc[10:62]

# print(subset)
# (University of Illinois Chicago) exists in universities

ax = sns.scatterplot(x='Name', y='distance', data=geocoded_colleges_df1)
ax.set_xticks([])
ax.set_xlabel("Universities")