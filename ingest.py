from model import predict_price
import pandas as pd
import os

data_file = 'used_car_predictions.csv'

def analyze_predicted_price():
    df = pd.DataFrame()
    specs = pd.read_csv('usedcar_data.csv')
    specs.drop(specs[specs['fuelType'] == 'Other'].index, inplace = True)
    specs.drop(specs[specs['fuelType'] == 'Electric'].index, inplace = True)
    specs.drop(specs[specs['transmission'] == 'Other'].index, inplace = True)
    df = specs.loc[:,['brand', 'year', 'transmission', 'mileage', 'fuelType', 'engineSize']]
    df = predict_price(df)
    df.to_csv(data_file, index=False)