import requests
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# fetch data UK police API for multiple locations and months
def fetch_crime_data_multi(months, locations):
    url = "https://data.police.uk/api/crimes-street/all-crime"
    
    all_data = []
    
    for lat, lng in locations:
        for m in months:
            print(f"Fetching {m} for ({lat}, {lng})...")
            params = {"lat": lat, "lng": lng, "date": m}
            
            response = requests.get(url, params)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"{m}: {len(data)} records")
                    all_data.extend(data)
                except:
                    pass
            
                time.sleep(1)
            
    return all_data

# convert raw json data into dataframe
def create_dataframe(data):
    df = pd.DataFrame(data)
    print("Shape: ", df.shape)
    return df

# clean data + extract latitude & longitude from nested
def clean_data(df):
    df = df[['category', 'month', 'location']].copy()
    
    df['latitude'] = df['location'].apply(
        lambda x: x['latitude'] if isinstance(x, dict) else None
    )
    
    df['longitude'] = df['location'].apply(
        lambda x: x['longitude'] if isinstance(x, dict) else None
    )
    
    df = df.drop(columns=['location'])
    df = df.dropna(subset=['latitude', 'longitude'])
    
    return df

# create grid system (spatial grouping of coordinates)
def create_grid_id(lat, lon, min_lat, min_lon, grid_size=0.009):
    row = int((float(lat) - min_lat) / grid_size)
    col = int((float(lon) - min_lon) / grid_size)
    return f"{row}_{col}"

def apply_grid(df):
    min_lat = df['latitude'].astype(float).min()
    min_lon = df['longitude'].astype(float).min()
    
    df['grid_id'] = df.apply(
        lambda x: create_grid_id(x['latitude'], x['longitude'], min_lat, min_lon),
        axis = 1
    )
    
    return df

# add temporal feature 
def add_week(df):
    df['month'] = pd.to_datetime(df['month'])
    df['week'] = df['month'].dt.isocalendar().week
    return df

# assign weights to crime types (severity-based)

def add_weights(df):
    weights = {
        'violent-crime': 3,
        'burglary': 2,
        'robbery': 2,
        'vehicle-crime': 1.5,
        'anti-social-behaviour': 1,
        'other-theft': 1,
        'shoplifting': 1,
        'criminal-damage-arson': 2,
        'public-order': 1,
        'drugs': 1.2
    }
    
    df['weight'] = df['category'].map(weights).fillna(1)
    return df

# aggregate data -> grid + week level crime score
def aggregate_data(df):
    agg_df = df.groupby(['grid_id', 'week']).agg(
        crime_score=('weight', 'sum')
    ).reset_index()
    return agg_df

# classify risk based on crime score
def assign_risk(score):
    if score < 40:
        return "Low"
    elif score < 100:
        return "Medium"
    else:
        return "High"

def add_risk_label(df):
    df['risk'] = df['crime_score'].apply(assign_risk)
    return df

# encode grid id for model input
def encode_grid(df):
    le = LabelEncoder()
    df['grid_id_encoded'] = le.fit_transform(df['grid_id'])
    return df

# split grid into row and column (spatial features)
def split_grid(df):
    df[['row', 'col']] = df['grid_id'].str.split('_', expand=True).astype(int)
    return df

# add temporal history features (no leakage - shifted)
def add_prev_features(df):
    df = df.sort_values(by=['grid_id', 'week'])
    df['prev_crime'] = df.groupby('grid_id')['crime_score'].shift(1)
    df['rolling_avg'] = df.groupby('grid_id')['crime_score'].rolling(3).mean().reset_index(level=0, drop=True)
    df = df.dropna()
    return df

# prepare feature and target
def prepare_data(df):
    X = df[['grid_id_encoded', 'week', 'row', 'col']]
    y_class = df['risk']
    y_reg = df['crime_score']
    
    return X, y_class, y_reg

# train classifcation model (time-based split)
def train_classification(X, y, df):
    df = df.sort_values("week")
    
    split = int (len(df) * 0.8)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [8, 12, 16],
        'min_samples_leaf': [2, 4, 6]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_}")
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    print("Random Forest Classification model trained")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    

def main():
    '''
    months = [
        "2026-02","2026-01","2025-12",
        "2025-11","2025-10","2025-09",
        "2025-08","2025-07","2025-06",
        "2025-05","2025-04","2025-03",
        "2025-02","2025-01","2024-12",
        "2024-11","2024-10","2024-09"
    ]
    
    locations = [
        (51.5074,-0.1278),
        (51.5155,-0.0922),
        (51.5033,-0.1195),
        (51.5200,-0.1000),
        (51.4900,-0.0800)
    ]
    
    data = fetch_crime_data_multi(months, locations)
    
    df = create_dataframe(data)
    df = clean_data(df)
    df = apply_grid(df)
    df = add_week(df)
    df = add_weights(df)
    
    agg_df = aggregate_data(df)
    agg_df = add_risk_label(agg_df)
    agg_df = encode_grid(agg_df)
    agg_df = split_grid(agg_df)
    agg_df = add_prev_features(agg_df)
    
    coords = df[['grid_id', 'latitude', 'longitude']].drop_duplicates()
    agg_df = agg_df.merge(coords, on='grid_id', how='left')
    
    agg_df.to_csv('final_crime_dataset.csv', index=False)
    '''
    
    agg_df = pd.read_csv("./final_crime_dataset.csv");
    
    X, y_class, y_reg = prepare_data(agg_df)
    
    train_classification(X, y_class, agg_df)
    
if __name__ == "__main__":
    main()
    
    