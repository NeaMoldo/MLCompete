import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('./dataset_train.csv')
test = pd.read_csv('./dataset_eval.csv')

# ============= REQUIREMENT 1 =============
task1 = train.copy()
# Use Elapsed Time (total time) instead of Moving Time
task1['speed'] = task1['Distance'] / (task1['Elapsed Time'] / 3600)

# FIXED: Correct month order
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
rows = []

for m in months:
    month_data = task1[task1['Activity Date'].str.contains(m)].copy()
    avg_speed = month_data['speed'].mean() if not month_data.empty else 0
    avg_speed = np.floor(avg_speed * 1e5) / 1e5
    rows.append({'subtaskID': 1, 'Answer1': m, 'Answer2': avg_speed})

req1 = pd.DataFrame(rows)

# ============= REQUIREMENT 2 =============
def add_features(df):
    df = df.copy()
    
    # Parse date
    df['datetime'] = pd.to_datetime(df['Activity Date'], format='%b %d, %Y, %I:%M:%S %p')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Speed features
    df['avg_speed'] = df['Distance'] / (df['Moving Time'] / 3600)
    df['speed_ratio'] = df['Moving Time'] / df['Elapsed Time']
    
    # Distance between start and end (haversine formula)
    lat1, lon1 = np.radians(df['Starting Latitude']), np.radians(df['Starting Longitude'])
    lat2, lon2 = np.radians(df['Finish Latitude']), np.radians(df['Finish Longitude'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['start_end_distance'] = 6371 * c  # Earth radius in km
    
    # Is it a round trip? (start ≈ end)
    df['is_round_trip'] = (df['start_end_distance'] < 0.1).astype(int)
    
    return df

# Add features to both datasets
train_features = add_features(train)
test_features = add_features(test)

# Prepare training data
feature_cols = ['Distance', 'Elapsed Time', 'Moving Time', 
                'Starting Latitude', 'Starting Longitude',
                'Finish Latitude', 'Finish Longitude',
                'hour', 'day_of_week', 'is_weekend',
                'avg_speed', 'speed_ratio', 'start_end_distance', 'is_round_trip']

X_train = train_features[feature_cols]
y_train = train_features['Label']

# Train model
forest = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
forest.fit(X_train, y_train)

# Predict
X_test = test_features[feature_cols]
predictions = forest.predict(X_test)

# Create requirement 2 output
req2 = pd.DataFrame({
    'subtaskID': 2,
    'Answer1': test['Activity ID'],
    'Answer2': predictions
})

# ============= SUBMISSION =============
submission = pd.concat([req1, req2], ignore_index=True)
submission.to_csv('submission.csv', index=False)

print("Submission file created!")
print(f"\nRequirement 1 sample:\n{req1.head()}")
print(f"\nRequirement 2 sample:\n{req2.head()}")