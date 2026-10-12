import pandas as pd
import numpy as np

def preprocess_data(data, train_data):

    # Drop Columns with Missing Values
    missing_values_count = data.isna().mean()
    data_cols_w_missing_vals = data.loc[:, missing_values_count > 0 ].columns

    num_cat_fts_lst = ['MSSubClass', 'MoSold']
    for col in data_cols_w_missing_vals:
        if train_data[col].dtype == 'object' or col in num_cat_fts_lst:
            impute_val = train_data[col].mode().iloc[0] # pandas ignores missing value when calculating mode
        else:
            impute_val = train_data[col].median() # pandas ignores missing value when calculating mode

        data[col] = data[col].fillna(impute_val)

    ### Encode Ordinal Categorical Features ###
    # Step 1: Specify the Order of Values for Each Column
    # Step 2: Use pandas.map() to Map the Orders to Values
    quality_dict = {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'Po': 0
    }

    data['KitchenQual'] = data['KitchenQual'].map(quality_dict)
    data['ExterQual'] = data['ExterQual'].map(quality_dict)
    data['ExterCond'] = data['ExterCond'].map(quality_dict)
    data['HeatingQC'] = data['HeatingQC'].map(quality_dict)

    street_dict = {
        'Grvl': 0,
        'Pave': 1
    }

    data['Street'] = data['Street'].map(street_dict)


    utilities_dict = {
        'AllPub': 0,
        'NoSeWa': 1
    }
    data['Utilities'] = data['Utilities'].map(utilities_dict)

    central_air_dict = {
        'N':0 ,
        'Y':1
        }

    data['CentralAir'] = data['CentralAir'].map(central_air_dict)

    y_n_p_dict = {
        'N':0 ,
        'P':1 ,
        'Y':2 ,
        }

    data['PavedDrive'] = data['PavedDrive'].map(y_n_p_dict)

    lotshape_dict = {
        'Reg': 3,
        'IR1': 2,
        'IR2': 1,
        'IR3': 0
        }

    data['LotShape'] = data['LotShape'].map(lotshape_dict)

    functional_dict = {
        'Sev': 0,
        'Maj2': 1,
        'Maj1': 2,
        'Mod': 3,
        'Min2': 4,
        'Min1': 5,
        'Typ': 6
    }

    data['Functional'] = data['Functional'].map(functional_dict)

    nominal_cat_ft = [ # Require One-Hot Encoding
        'LandSlope',
        'LandContour',
        'MSZoning',
        'LotConfig',
        'BldgType',
        'RoofStyle',
        'Heating',
        'SaleCondition',
        'Foundation',
        'RoofMatl',
        'Condition2',
        'HouseStyle',
        'SaleType',
        'Condition1',
        'Exterior1st',
        'Exterior2nd',
        'Neighborhood'
    ]
    ### Encode Nominal Categorical Features & Numerical Categorical Features
    for cat_ft in nominal_cat_ft + ['MSSubClass', 'MoSold']: # numerical_categoricals_to_encode
        dummies = pd.get_dummies(data[cat_ft], prefix=cat_ft, dtype=int)
        data = pd.concat([data, dummies], axis=1)
        data.drop(cat_ft, axis=1, inplace=True)

    return data


# def train_test_split(data, split_ratio=0.8):

#     # Shuffle Indices
#     indices = np.random.permutation(len(data))

#     # Determine Train/Test Split
#     split = int(split_ratio * len(data))

#     # Split data into train/test
#     train_indices = indices[:split]
#     test_indices = indices[split:]

#     train_data = data.iloc[train_indices] # complete training dataset including Ids and Targets
#     test_data = data.iloc[test_indices] # complete test dataset including Ids and Targets

#     # Split data into X_train, y_train, X_test, y_test
#     # Drop Id and Target labels from X_train and X_test
#     train_ids = train_data['Id']
#     test_ids = test_data['Id']

#     X_train = train_data.drop(['SalePrice', 'Id'], axis=1) # input features used by the model
#     y_train = train_data['SalePrice']

#     X_test = test_data.drop(['SalePrice', 'Id'], axis=1) # input features used by the model
#     y_test = test_data['SalePrice']


#     return train_ids, test_ids, X_train, y_train, X_test, y_test

def scale_features(X_train, X_test):
    features_to_scale = [
        'LotArea',
        'YearBuilt',
        'YearRemodAdd',
        'BsmtFinSF1',
        'BsmtFinSF2',
        'BsmtUnfSF',
        'TotalBsmtSF',
        '1stFlrSF',
        '2ndFlrSF',
        'LowQualFinSF',
        'GrLivArea',
        'BsmtFullBath',
        'BsmtHalfBath',
        'FullBath',
        'HalfBath',
        'BedroomAbvGr',
        'KitchenAbvGr',
        'TotRmsAbvGrd',
        'Fireplaces',
        'GarageCars',
        'GarageArea',
        'WoodDeckSF',
        'OpenPorchSF',
        'EnclosedPorch',
        '3SsnPorch',
        'ScreenPorch',
        'PoolArea',
        'MiscVal',
        'YrSold'
    ]

    ordinal_features_to_scale = [
        'LotShape',
        'OverallQual',
        'OverallCond',
        'ExterQual',
        'ExterCond',
        'HeatingQC',
        'KitchenQual',
        'Functional',
        'PavedDrive'
    ]

    features_to_scale = features_to_scale + ordinal_features_to_scale

    X_mean = X_train[features_to_scale].mean()
    X_std = X_train[features_to_scale].std()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[features_to_scale] = (X_train[features_to_scale] - X_mean) / X_std
    X_test_scaled[features_to_scale] = (X_test[features_to_scale] - X_mean) / X_std

    return X_train_scaled, X_test_scaled, X_mean, X_std