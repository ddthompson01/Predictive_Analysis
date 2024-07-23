import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Define file paths
file_paths = {
    "ADA_CNP_FL": "/Users/daniellethompson/Desktop/Test/ADA_CNP_FL.txt",
    "ADA_CNP_LA": "/Users/daniellethompson/Desktop/Test/ADA_CNP_LA.txt",
    "ADA_CNP_MIDLAND": "/Users/daniellethompson/Desktop/Test/ADA_CNP_MIDLAND.txt",
    "ADA_CNP_OH": "/Users/daniellethompson/Desktop/Test/ADA_CNP_OH.txt",
    "ADA_CNP_TX": "/Users/daniellethompson/Desktop/Test/ADA_CNP_TX.txt",
    "SALES_DATA": "/Users/daniellethompson/Desktop/Test/Dec-Jan SalesData.csv",
    "REGIONS": "/Users/daniellethompson/Desktop/Test/Regions.csv",
    "SCHOOLS": "/Users/daniellethompson/Desktop/Test/Schools.csv",
    "STATES": "/Users/daniellethompson/Desktop/Test/States.csv",
    "OUTPUT_FILE": "/Users/daniellethompson/Desktop/Test/Combined_Data.xlsx"
}

def load_data(file_path, delimiter='\t'):
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        print(f"Loaded data from {file_path} successfully.")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

ada_cnp_fl = load_data(file_paths['ADA_CNP_FL'])
ada_cnp_la = load_data(file_paths['ADA_CNP_LA'])
ada_cnp_midland = load_data(file_paths['ADA_CNP_MIDLAND'])
ada_cnp_oh = load_data(file_paths['ADA_CNP_OH'])
ada_cnp_tx = load_data(file_paths['ADA_CNP_TX'])
sales_data = load_data(file_paths['SALES_DATA'], delimiter=',')
regions = load_data(file_paths['REGIONS'], delimiter=',')
schools = load_data(file_paths['SCHOOLS'], delimiter=',')
states = load_data(file_paths['STATES'], delimiter=',')

# Function to clean and format datasets
def clean_data(df, date_column='ATTDATE'):
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce').dt.strftime('%m/%d/%Y')
    df['ATTENDANCE'] = df['MEMBERSHIP'] - df['ABSENCES']
    df.dropna(inplace=True)
    df.fillna(0, inplace=True)
    return df

ada_cnp_fl = clean_data(ada_cnp_fl)
ada_cnp_la = clean_data(ada_cnp_la)
ada_cnp_midland = clean_data(ada_cnp_midland)
ada_cnp_oh = clean_data(ada_cnp_oh)
ada_cnp_tx = clean_data(ada_cnp_tx)

combined_data = pd.concat([ada_cnp_fl, ada_cnp_la, ada_cnp_midland, ada_cnp_oh, ada_cnp_tx])

combined_data.rename(columns={'SCHOOLNAME': 'Campus', 'ATTDATE': 'Date'}, inplace=True)
sales_data.rename(columns={'SaleDate': 'Date', 'School': 'Campus'}, inplace=True)

# Ensure 'Date' and 'Campus' columns are of the same format
combined_data['Date'] = pd.to_datetime(combined_data['Date'], format='%m/%d/%Y')
sales_data['Date'] = pd.to_datetime(sales_data['Date'], format='%m/%d/%Y')

if 'Campus' in combined_data.columns and 'Campus' in sales_data.columns:
    combined_data['Campus'] = combined_data['Campus'].str.strip().str.lower()
    sales_data['Campus'] = sales_data['Campus'].str.strip().str.lower()

combined_data = combined_data.merge(schools, on='Campus', how='left', suffixes=('', '_schools'))
combined_data = combined_data.merge(regions, on='RegionId', how='left', suffixes=('', '_regions'))
combined_data = combined_data.merge(states, on='StateId', how='left', suffixes=('', '_states'))
combined_data = combined_data.merge(sales_data, on=['Campus', 'Date'], how='left', suffixes=('', '_sales'))

#no missing values and correct data types
combined_data.dropna(inplace=True)
combined_data.fillna(0, inplace=True)
combined_data = combined_data.convert_dtypes()

# Calculate participation rate
combined_data['ParticipationRate'] = combined_data['TotalMealCount'] / combined_data['ATTENDANCE']

# Save the final merged data to a new Excel file
combined_data.to_excel(file_paths['OUTPUT_FILE'], index=False)

# Print preview of final merged data
print("Final Merged Data preview:\n", combined_data.head())

#Check if features correlate with participation rates
features = ['FreeCount', 'ReducedCount', 'PaidCount', 'TotalMealCount', 'ATTENDANCE']

# Let's see what correlates with participation rates?????
for feature in features:
    sns.regplot(data=combined_data, x='ParticipationRate', y=feature)
    plt.title(f"Participation Rate and {feature}")
    plt.show()

# Model to predict participation rates based on meal types
features = ['FreeCount', 'ReducedCount', 'PaidCount']
target = 'ParticipationRate'

# Prepare data for modeling
model_data = combined_data.dropna(subset=features + [target])

# Train-test split
train_data = model_data.sample(frac=0.8, random_state=1)
test_data = model_data.drop(train_data.index)

# Initialize and train the model
model = LinearRegression()
model.fit(train_data[features], train_data[target])

# Predict on test data
preds = model.predict(test_data[features])
preds = pd.Series(preds, index=test_data.index)

# Join predictions back to the test data
test_data['preds'] = preds
rmse = mean_squared_error(test_data[target], test_data['preds'])**0.5
r2 = pearsonr(test_data[target], test_data['preds'])[0]**2
print(f"RMSE: {rmse}\nR-squared: {r2}")

sns.regplot(data=test_data, x=target, y='preds')
plt.title('Actual vs Predicted Participation Rates')
plt.show()

print(test_data[[target, 'preds']].head(10))

# Function to check data integrity
def check_data_integrity(df):
    missing_values = df.isnull().sum().sum()
    data_types = df.dtypes
    print(f"Missing values: {missing_values}")
    print(f"Data types:\n{data_types}")

# Regularly check data integrity
check_data_integrity(combined_data)
