import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """
    Loads prepared_churn_data file from week 2 into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df):
    """
    This uses pycaret best model to predict the data in the df dataframe.
    """
    new_data = df.iloc[-2:-1].copy() 
    model = load_model('LDA')
    predictions = predict_model(model, data=new_data)
    predictions.rename({'Label': 'churn_prediction'}, axis=1, inplace=True)
    predictions['churn_prediction'].replace({1: 'Churn', 0: 'No churn'},
                                            inplace=True)
    return predictions['churn_prediction']


if __name__ == "__main__":
    df = load_data('new_churn_data.csv')
    new_data = df.iloc[-2:-1].copy()
    model = load_model('LDA')
    predictions = predict_model(model, data=new_data)
    print('predictions:')
    print(predictions)