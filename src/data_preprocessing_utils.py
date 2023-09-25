# -*- coding: utf-8 -*-
#utility function
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def is_float(n):
    try:
        float_n = float(n)
    except ValueError:
        return False
    else:
        return True

def preprocess_size(size):
    try:
        original_size = size  # Keep a copy of the original size for error reporting
        size = size.lower().strip()  # Convert to lowercase for uniform processing and remove spaces

        quantity = 1

        # Handle '750mL + 2/' format
        if '+' in size:
            size, _ = size.split('+')  # Only consider the main component
            size = size.strip()  # remove any potential spaces
        # Handle '750mL 4 Pk', '187mL 4 Pk', '250mL 4 Pk', '750mL 12 P', and '750mL  3' formats
        elif 'pk' in size or ('p' in size.lower()) or (' ' in size and re.search(r'\d', size.split()[-1])):
            size_parts = re.split(r'\s+', size)  # Split on one or more spaces
            if len(size_parts) >= 2:  # If there are at least two parts
                size = size_parts[0]
                quantity = int(re.sub(r'\D', '', size_parts[1]))  # Keep only the digits
        # Convert gallons and ounces to mL
        elif 'gal' in size:
            size = re.sub(r'gal', '', size).strip()  # remove any potential spaces
            if is_float(size):
                size = float(size) * 3785.41  # Convert gallons to mL
                return size  # If we're here, we have a final result
            else:
                print(f"Size value is not numeric: '{original_size}'")
                return None
        elif 'oz' in size:
            size = re.sub(r'oz', '', size).strip()  # remove any potential spaces
            if is_float(size):
                size = float(size) * 29.5735  # Convert ounces to mL
                return size  # If we're here, we have a final result
            elif '/' in size:
                numerator, denominator = map(float, size.split('/'))  # Split the fraction and convert to float
                size = (numerator / denominator) * 29.5735  # Convert ounces to mL
                return size  # If we're here, we have a final result
            else:
                print(f"Size value is not numeric: '{original_size}'")
                return None

        # Handle '3/100mL' format
        elif '/' in size and 'pk' not in size:
            if 'oz' in size or 'ml' in size or 'l' in size:
                numerator, denominator = size.split('/')[0], re.split(r'[a-z]+', size.split('/')[1])[0]
                size = str(float(numerator) / float(denominator)) + "".join(re.findall(r'[a-z]+', size.split('/')[1]))
            else:
                quantity, size = size.split('/')
                quantity = int(quantity.strip())  # remove any potential spaces
                size = size.strip()  # remove any potential spaces

        # Convert sizes in liters to mL
        if 'ml' in size and is_float(re.sub(r'ml', '', size).strip()):
            size = re.sub(r'ml', '', size).strip()  # remove any potential spaces
            size = float(size) * quantity
        elif 'l' in size and is_float(re.sub(r'l', '', size).strip()):
            size = re.sub(r'l', '', size).strip()  # remove any potential spaces
            size = float(size) * 1000 * quantity
        elif size == 'liter':
            size = 1000.0
        elif is_float(size):  # If the size is just a floating point number, it's likely in mL
            size = float(size)
        else:
            print(f"Size value is not numeric: '{original_size}'")
            return None

        return size

    except Exception as e:
        print(f"Error processing size: '{original_size}'. Error: {e}")
        return None  # If any error occurs, return None



def preprocess_sales_df(sales_df):
    # Create a copy to avoid changes to the original dataframe
    sales_df = sales_df.copy()

    # Standardize 'Size' to milliliters
    sales_df['Size'] = sales_df['Size'].apply(preprocess_size)

    # Standardize text columns
    for col in ['Description', 'VendorName']:
        sales_df[col] = sales_df[col].str.strip()


    # Convert date columns to datetime format
    sales_df['SalesDate'] = pd.to_datetime(sales_df['SalesDate'])

    return sales_df



def data_for_training(processing_data):

    data = processing_data.copy()

    # take 3000 products said to loa limitation materiels, 
    # if you have a large amount of ram you can delete the next two lines to train the model on the 6890 products.
    top_100_categories=data['Description'].value_counts().head(3000).index.tolist()
    data=data[data['Description'].isin(top_100_categories)]


    # Select the necessary columns for the model training
    selected_features = ['Year', 'Month', 'Description', 'Store', 'Classification', 'ExciseTax', 'Size', 'PurchasePrice', 'SalesPrice','VendorName']
    X = data[selected_features]
    y = data['SalesQuantity']

    # One-Hot Encoding for the categorical columns
    X_encoded = pd.get_dummies(X, columns=['Description', 'Store', 'Classification','VendorName'])

    # Divide the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.3, random_state=27)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=27)

    # Normalize the numerical data
    scaler = StandardScaler()
    X_train[['ExciseTax', 'Size', 'PurchasePrice', 'SalesPrice']] = scaler.fit_transform(X_train[['ExciseTax', 'Size', 'PurchasePrice', 'SalesPrice']])
    X_val[['ExciseTax', 'Size', 'PurchasePrice', 'SalesPrice']] = scaler.transform(X_val[['ExciseTax', 'Size', 'PurchasePrice', 'SalesPrice']])
    X_test[['ExciseTax', 'Size', 'PurchasePrice', 'SalesPrice']] = scaler.transform(X_test[['ExciseTax', 'Size', 'PurchasePrice', 'SalesPrice']])


    return X_train,X_test,X_val,y_train,y_test,y_val


