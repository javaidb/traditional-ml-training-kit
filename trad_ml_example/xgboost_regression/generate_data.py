import pandas as pd
import numpy as np
import os

def generate_housing_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic housing data."""
    np.random.seed(random_state)
    
    # Generate features
    square_feet = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    year_built = np.random.randint(1950, 2024, n_samples)
    lot_size = np.random.normal(8000, 2000, n_samples)
    
    # Generate categorical features
    neighborhoods = ['Downtown', 'Suburb', 'Rural', 'Urban', 'Coastal']
    property_types = ['Single Family', 'Townhouse', 'Condo', 'Multi-Family']
    
    neighborhood = np.random.choice(neighborhoods, n_samples)
    property_type = np.random.choice(property_types, n_samples)
    
    # Generate target (price) with some noise
    base_price = (
        200 * square_feet +
        50000 * bedrooms +
        75000 * bathrooms +
        100 * (year_built - 1950) +
        5 * lot_size
    )
    
    # Add neighborhood and property type effects
    neighborhood_effects = {
        'Downtown': 100000,
        'Suburb': 50000,
        'Rural': -50000,
        'Urban': 75000,
        'Coastal': 150000
    }
    
    property_type_effects = {
        'Single Family': 50000,
        'Townhouse': 25000,
        'Condo': 0,
        'Multi-Family': 75000
    }
    
    for n in neighborhoods:
        mask = neighborhood == n
        base_price[mask] += neighborhood_effects[n]
    
    for p in property_types:
        mask = property_type == p
        base_price[mask] += property_type_effects[p]
    
    # Add noise
    price = base_price + np.random.normal(0, 50000, n_samples)
    price = np.maximum(price, 100000)  # Ensure minimum price
    
    # Create DataFrame
    df = pd.DataFrame({
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'lot_size': lot_size,
        'neighborhood': neighborhood,
        'property_type': property_type,
        'price': price
    })
    
    return df

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('temp_data', exist_ok=True)
    
    # Generate and save data
    df = generate_housing_data()
    df.to_csv('temp_data/housing_data.csv', index=False)
    print(f"Generated {len(df)} samples of housing data")
    print("\nFeature statistics:")
    print(df.describe()) 