import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare the dataset
chicago_rentals_df = pd.read_csv("Zillow Rental (Chicago Area 03_03_2025).csv")
chicago_rentals_df = chicago_rentals_df[['Price', 'Address', 'Listing URL', 'Short Address', 'Zip', 'Beds', 'Baths', 'Raw Property Details', 'Latitude', 'Longitude']].reset_index(drop=True)

# Clean the data
chicago_rentals_df['Baths'] = chicago_rentals_df['Baths'].fillna(1)
chicago_rentals_df['Beds'] = chicago_rentals_df['Beds'].astype(float)

df = chicago_rentals_df
df['Beds'] = df['Beds'].astype(float)
df['Baths'] = df['Baths'].astype(float)
df = df[df['Price'].notnull() & (df['Price'] > 0)]

# Simulate adding a "Colleges Within 2 Miles" feature
# In a real scenario, you would calculate this based on college location data
# For this simulation, we'll generate data that correlates somewhat with price and location
np.random.seed(42)  # For reproducibility

# Generate simulated college proximity data
# Higher values in certain areas of the city, with some randomness
def simulate_college_proximity(lat, long):
    # Central Chicago approximate coordinates
    chicago_center_lat, chicago_center_long = 41.8781, -87.6298
    
    # Calculate distance from city center (rough proxy for urban density)
    dist_from_center = np.sqrt((lat - chicago_center_lat)**2 + (long - chicago_center_long)**2)
    
    # More colleges near city center, fewer in outlying areas
    # Adding randomness to make it realistic
    base_count = max(0, 5 - (dist_from_center * 20))
    college_count = int(max(0, base_count + np.random.normal(0, 1)))
    
    return college_count

# Apply the function to create the new feature
df['Colleges_Within_2mi'] = df.apply(
    lambda row: simulate_college_proximity(row['Latitude'], row['Longitude']), 
    axis=1
)

# One-hot encode ZIP codes
df['Zip'] = df['Zip'].astype(str)
df = pd.get_dummies(df, columns=['Zip'], drop_first=True)

# Drop rows with missing values
df = df.dropna()

# Define features, including the new college proximity feature
features_original = ['Beds', 'Baths', 'Latitude', 'Longitude']
features_with_colleges = ['Beds', 'Baths', 'Latitude', 'Longitude', 'Colleges_Within_2mi']
target = 'Price'

#lr_orig, rf_orig, X_test_orig, y_test_orig = run_analysis(features_original, "Original Features")
# Run analysis with and without the college proximity feature
def run_analysis(features_list, feature_set_name):
    print(f"\n=== Analysis using {feature_set_name} ===")
    
    X = df[features_list]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Baseline Model
    mean_price = y_train.mean()
    baseline_preds = [mean_price] * len(y_test)
    
    # Model 2: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    
    # Model 3: Random Forest
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    
    # Evaluate models
    def evaluate_model(name, true, pred):
        rmse = np.sqrt(mean_squared_error(true, pred))  # Correct RMSE calculation
        mae = mean_absolute_error(true, pred)
        print(f"{name} → RMSE: ${rmse:.2f}, MAE: ${mae:.2f}")
    
    print(f"=== Model Performance on Rent Prediction ({feature_set_name}) ===")
    evaluate_model("Baseline (Mean)", y_test, baseline_preds)
    evaluate_model("Linear Regression", y_test, lr_preds)
    evaluate_model("Random Forest", y_test, rf_preds)
    
    # Feature importance analysis for Random Forest
    importances = rf.feature_importances_
    top_features = pd.Series(importances, index=features_list).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f"Feature Importance (Random Forest) - {feature_set_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()
    
    return lr, rf, X_test, y_test

# Run analysis without college proximity feature
lr_orig, rf_orig, X_test_orig, y_test_orig = run_analysis(features_original, "Original Features")

# Run analysis with college proximity feature
lr_college, rf_college, X_test_college, y_test_college = run_analysis(features_with_colleges, "Features with College Proximity")

# Additional analysis to isolate the impact of college proximity
# Create scatter plot of college proximity vs. price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Colleges_Within_2mi', y='Price', data=df, alpha=0.5)
plt.title("Relationship between College Proximity and Rental Price")
plt.xlabel("Number of Colleges Within 2 Miles")
plt.ylabel("Rental Price ($)")
plt.tight_layout()
plt.show()

# Calculate correlation
correlation = df['Colleges_Within_2mi'].corr(df['Price'])
print(f"\nCorrelation between College Proximity and Price: {correlation:.4f}")

# Compare average prices by college proximity
college_price_groups = df.groupby('Colleges_Within_2mi')['Price'].agg(['mean', 'count']).reset_index()
college_price_groups.columns = ['Colleges Within 2mi', 'Average Price', 'Count']
print("\nAverage Rental Prices by College Proximity:")
print(college_price_groups.sort_values('Colleges Within 2mi'))

# Plot average price by number of colleges
plt.figure(figsize=(12, 6))
bars = sns.barplot(x='Colleges Within 2mi', y='Average Price', data=college_price_groups)
plt.title("Average Rental Price by Number of Colleges Within 2 Miles")
plt.xlabel("Number of Colleges Within 2 Miles")
plt.ylabel("Average Rental Price ($)")

# Add count labels above each bar
for i, (_, row) in enumerate(college_price_groups.iterrows()):
    if not np.isnan(row['Average Price']):
        bars.text(i, row['Average Price'] + 50, f"n={int(row['Count'])}", 
                  ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Calculate feature impact through permutation importance
from sklearn.inspection import permutation_importance

# For the model with college proximity
result = permutation_importance(
    rf_college, X_test_college, y_test_college, 
    n_repeats=10, random_state=42, n_jobs=-1
)

# Get the importances specific to the college feature
college_importance = result.importances_mean[features_with_colleges.index('Colleges_Within_2mi')]
print(f"\nPermutation Importance of College Proximity Feature: {college_importance:.4f}")

# Print conclusion about whether college proximity is a good predictor
print("\n=== CONCLUSION ===")
print(f"Adding college proximity data {'improved' if college_importance > 0.05 else 'did not significantly improve'} rental price predictions.")
print(f"College proximity ranked {list(top_features.index).index('Colleges_Within_2mi')+1 if 'Colleges_Within_2mi' in top_features.index else 'N/A'} out of {len(features_with_colleges)} features in importance.")