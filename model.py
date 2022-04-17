import pickle

def predict_price(df):
    pipe = pickle.load(open('pipeline.pkl', 'rb'))
    predicted_price = pipe.predict(df)
    df['price'] = predicted_price
    return df
