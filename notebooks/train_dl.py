'''
Train DL models on dengue and chik data an apply the transfer learning
'''
 
import pandas as pd
from train_models import train_dl_model, train_transf_chik
import os


df = pd.read_csv('s_cities.csv', index_col = 'Unnamed: 0')

end_date = '2023-11-01'
RETRAIN = True

for city, ini_date, end_date_train in zip(
    df.geocode, df.start_train_chik, df.end_train_chik):
    print(f"Training {city} model")

    # train the dengue model
    if RETRAIN or not os.path.exists(f'../saved_models/lstm/trained_{city}_dengue_msle.keras'):
        train_dl_model(city,   doenca = 'dengue', end_date_train = end_date_train , end_date = end_date, plot = False)

    # apply transfer
    if  RETRAIN or not os.path.exists(f'../saved_models/lstm/trained_{city}_chik_tranf_msle.keras'):
        train_transf_chik(city, ini_date = ini_date, end_date_train = end_date_train , end_date = end_date, plot =False)

    # train the chik model
    if RETRAIN or not os.path.exists(f'../saved_models/lstm/trained_{city}_chik_msle.keras'):
        train_dl_model(city,   doenca = 'chik', end_date_train = end_date_train , end_date = end_date, plot = False)
