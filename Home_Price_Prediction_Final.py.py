#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# To display plots inline in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
# Set default plot size
matplotlib.rcParams["figure.figsize"] = (20,10)

# Load the dataset
df1 = pd.read_csv('Bengaluru_House_Data.csv')
df1.head()  # Display the first few rows of the dataset

# Check the shape of the dataset (number of rows and columns)
df1.shape

# Group by 'area_type' and count the number of occurrences
df1.groupby("area_type")["area_type"].agg("count")

# Drop unnecessary columns from the dataset
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
df2.head()

# Check for missing values in the dataset
df2.isnull().sum()

# Drop rows with missing values
df3 = df2.dropna()
df3.isnull().sum()  # Verify that there are no more missing values

# Check the shape of the dataset after dropping missing values
df3.shape

# Display unique values in the 'size' column
df3['size'].unique()

# Convert 'size' column to an integer number of BHKs (Bedrooms, Halls, Kitchens)
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

df3.head()  # Display the first few rows after transformation

# Check unique values in the 'bhk' column
df3['bhk'].unique()

# Check for outliers where the number of BHKs is unusually high
df3[df3.bhk > 20]

# Display unique values in the 'total_sqft' column
df3.total_sqft.unique()

# Function to check if a value can be converted to a float
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# Display rows where 'total_sqft' cannot be converted to a float
df3[~df3['total_sqft'].apply(is_float)].head(10)

# Function to convert 'total_sqft' values to a number, handling ranges and invalid strings
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

# Test the conversion function with a sample value
convert_sqft_to_num('2350')

# Test the conversion function with a value that includes text
convert_sqft_to_num('34.46Sq. Meter')

# Make a copy of the dataset for further processing
df4 = df3.copy()

# Apply the conversion function to the 'total_sqft' column
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()

# Inspect a specific row to check the transformation
df4.loc[2]

# Create a copy of the dataset for further processing
df5 = df4.copy()

# Add a new column for price per square foot
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
df5.head()

# Check the number of unique locations
len(df5.location.unique())

# Remove any leading or trailing spaces from the 'location' column
df5.location = df5.location.apply(lambda x: x.strip())
location_status = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_status

# Check how many locations have 10 or fewer properties
len(location_status[location_status <= 10])

# Create a list of locations with 10 or fewer properties
location_status_less_than_10 = location_status[location_status <= 10]
location_status_less_than_10

# Check the number of unique locations again
len(df5.location.unique())

# Replace locations with fewer than 10 properties with 'other'
df5.location = df5.location.apply(lambda x: 'other' if x in location_status_less_than_10 else x)
len(df5.location.unique())

# Display the first 10 rows of the dataset to inspect the changes
df5.head(10)

# Identify outliers where the area per BHK is unusually low
df5[df5.total_sqft / df5.bhk < 300].head()

# Check the shape of the dataset before removing outliers
df5.shape

# Remove outliers where the area per BHK is unusually low
df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
df6.shape

# Display summary statistics for the 'price_per_sqft' column
df6.price_per_sqft.describe()

# Function to remove outliers based on price per square foot within each location
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

# Apply the outlier removal function
df7 = remove_pps_outliers(df6)
df7.shape

# Function to plot scatter plots for 2 BHK and 3 BHK properties in a given location
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("TOTAL SQUARE FEET AREA")
    plt.ylabel("PRICE PER SQUARE FEET")
    plt.title(location)
    plt.legend()

# Plot scatter chart for properties in Rajaji Nagar
plot_scatter_chart(df7, "Rajaji Nagar")

# Plot scatter chart for properties in Hebbal
plot_scatter_chart(df7, "Hebbal")

# Function to remove outliers based on the price per square foot for properties with different BHKs
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

# Apply the BHK outlier removal function
df8 = remove_bhk_outliers(df7)
df8.shape

# Plot scatter chart for properties in Hebbal after removing outliers
plot_scatter_chart(df8, "Hebbal")

# Plot histogram of price per square foot
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft, rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")

# Display unique values in the 'bath' column
df8.bath.unique()

# Plot histogram of the number of bathrooms
plt.hist(df8.bath, rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")

# Identify properties with an unusually high number of bathrooms
df8[df8.bath > 10]

# Identify properties where the number of bathrooms exceeds the number of BHKs + 2
df8[df8.bath > df8.bhk + 2]

# Remove properties with an unusually high number of bathrooms
df9 = df8[df8.bath < df8.bhk + 2]
df9.shape

# Display the first two rows of the cleaned dataset
df9.head(2)

# Drop the 'size' and 'price_per_sqft' columns as they are no longer needed
df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
df10.head(3)

# Create dummy variables for the 'location' column
dummies = pd.get_dummies(df10.location)
dummies.head(3)

# Add dummy variables to the dataset and drop the original 'location' column
df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
df11.head()

# Drop the 'location' column as it has been replaced by dummy variables
df12 = df11.drop('location', axis='columns')
df12.head(2)

# Check the shape of the final dataset
df12.shape

# Separate the features (X) and the target variable (y)
X = df12.drop(['price'], axis='columns')
X.head(3)

y = df12.price
y.head(3)

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Import linear regression model and cross-validation method
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Initialize and fit the linear regression model
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# Evaluate the model using cross-validation
cross_val_score(lr_clf, X_train, y_train, cv=5)

# Define a function to evaluate model performance using multiple algorithms
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

def find_best_model_using_gridsearchcv(X, y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.tree import DecisionTreeRegressor
    
    models = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for model_name, mp in models.items():
        clf = GridSearchCV(mp['model'], mp['params'], cv=cv, return_train_score=False)
        clf.fit(X, y)
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Call the function to find the best model
find_best_model_using_gridsearchcv(X, y)

# Initialize the model with the best hyperparameters found
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1, selection='cyclic')
lasso.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso.predict(X_test)

# Evaluate model performance on the test set
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

