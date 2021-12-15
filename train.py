import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def inch2meter(inch):
    return 0.0254 * inch

def ht2meter(height):
    ft = height // 100
    inch = height % 100

    return inch2meter(12 * ft + inch)

def pounds2kg(pound):
    return pound * 0.4535924

def isRule1(row):
    return 1 if 18 <= row["Ins_Age"] <= 39 and (row["BMI"] < 17.39 or row["BMI"] > 38.5) else 0

def isRule2(row):
    return 1 if 40 <= row["Ins_Age"] <= 59 and (row["BMI"] < 18.49 or row["BMI"] > 38.5) else 0

def isRule3(row):
    return 1 if row["Ins_Age"] >= 60 and (row["BMI"] < 18.49 or row["BMI"] > 45.5) else 0

def preprocessing(df):
    df["Ht_meter"] = df["Ht"].apply(ht2meter)
    df["Wt_kg"] = df["Wt"].apply(pounds2kg)
    df["BMI"] = df["Wt_kg"] / (df["Ht_meter"] ** 2)
    df["isMale"] = pd.get_dummies(df["Ins_Gender"], drop_first=True)

    return df

def add_BMI_rules(df):
    # Add three features `rule1`, `rule2` and `rule3` to incorporate the first 3 business rules for calculating BMI  
    df["rule1"] = df.apply(isRule1, axis = 1)
    df["rule2"] = df.apply(isRule2, axis = 1)
    df["rule3"] = df.apply(isRule3, axis = 1)

    return df

def standardize(df):
    scaler = MinMaxScaler()
    # Normalize `Ins_Age`, `Ht_meter`, `Wt`
    df[["Ins_Age", "Ht_meter", "Wt"]] = scaler.fit_transform(df[["Ins_Age", "Ht_meter", "Wt"]])

    return df

def train_rf(X_train, y_train):
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, y_train)

    return rf_reg

def validate(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    return mse

if __name__ == "__main__":
    filename = "Dummy-Data.csv"
    df = pd.read_csv(filename)

    df = preprocessing(df)
    df = add_BMI_rules(df)
    df = standardize(df)

    cols = ["Ins_Age", "isMale", "Ht_meter", "Wt", "rule1", "rule2", "rule3"]
    X = df[cols]
    y = df["BMI"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    model = train_rf(X_train, y_train)
    train_mse = validate(model, X_train, y_train)
    print("Train MSE: {:.4f}".format(train_mse))

    test_mse = validate(model, X_test, y_test)
    print("Test MSE: {:.4f}".format(test_mse))

    # save model as a pkl file
    with open('model.pkl','wb') as f:
        pickle.dump(model, f)
    