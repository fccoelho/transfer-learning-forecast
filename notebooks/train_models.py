import sys
import tensorflow.keras as keras
import pandas as pd

sys.path.append('../')
from plots_lstm import plot_loss
from lstm import build_model, train_model, apply_model
from lstm import build_model, transf_chik, transf_model
from pgbm_model import pgbm_train, cross_dengue_chik_prediction

PREDICT_N = 4  # number of new days predicted
LOOK_BACK = 12  # number of last days used to make the prediction
BATCH_SIZE = 1
EPOCHS = 400
HIDDEN = 8
L1 = 1e-5
L2 = 1e-5


def train_dl_model(city, doenca='dengue', end_date_train='2022-11-01', end_date='2023-11-01', ini_date=None, plot=True, lr =0.0001 ):
    FILENAME_DATA = f'../data/{doenca}_{city}_cluster.csv.gz'
    cols = pd.read_csv(FILENAME_DATA, index_col='Unnamed: 0').shape[1]

    FEAT = cols  # number of features

    model = build_model(l1=L1, l2=L2, hidden=HIDDEN, features=FEAT, predict_n=PREDICT_N, look_back=LOOK_BACK,
                        batch_size=BATCH_SIZE, loss='msle', lr=lr)

    model, hist = train_model(model, city, doenca=doenca, epochs=EPOCHS, end_train_date=end_date_train,
                                              ini_date=ini_date,
                                              ratio=None, end_date=end_date,
                                              predict_n=PREDICT_N, look_back=LOOK_BACK, label='msle',
                                              batch_size=BATCH_SIZE,
                                              filename=FILENAME_DATA, verbose=1)
    if plot:
        plot_loss(hist, title=F'Model loss - MSLE - {city}')


def train_transf_chik(city, ini_date, end_date_train, end_date, plot=True):
    FILENAME_DATA = f'../data/chik_{city}_cluster.csv.gz'

    BATCH_SIZE = 1

    cols = pd.read_csv(FILENAME_DATA, index_col='Unnamed: 0').shape[1]

    FEAT = cols # number of features

    filename = f'../saved_models/lstm/trained_{city}_dengue_msle.keras'
    model = transf_model(filename, L1, L2, HIDDEN, FEAT, PREDICT_N, LOOK_BACK, batch_size=BATCH_SIZE, lr=0.0001)

    # apply transf model 
    hist = transf_chik(model, city, ini_date=ini_date, end_train_date=end_date_train,
                                   end_date=end_date, epochs=EPOCHS,
                                   predict_n=PREDICT_N, look_back=LOOK_BACK, validation_split=0.0,
                                   monitor='loss',
                                   min_delta=0.002,
                                   label=f'transf_msle', ratio=None, filename_data=FILENAME_DATA, verbose=0)

    if plot:
        plot_loss(hist, title=f'Model loss - MSLE - {city}')


def train_pgbm_model(city, doenca, ini_date, end_date_train, end_date):
    FILENAME_DATA = f'../data/{doenca}_{city}_cluster.csv.gz'
    # ini_date=ini_date, end_train_date=end_date_train,
    # end_date=end_date, filename=FILENAME_DATA, verbose=1)
    #
    #                   look_back=LOOK_BACK, ini_date=ini_date,
    #                   end_date=end_date, filename=FILENAME_DATA)

    pgbm_train(city, PREDICT_N, LOOK_BACK, doenca=doenca, ini_date=ini_date, end_train_date=end_date_train,
               end_date=end_date, filename=FILENAME_DATA, verbose=0)


def apply_dengue_pgbm_on_chik(city, ini_date, end_date, plot=True):
    FILENAME_DATA = f'../data/chik_{city}_cluster.csv.gz'
    preds, preds25, preds975, X_data, targets = cross_dengue_chik_prediction(city, predict_n=PREDICT_N,
                                                                             look_back=LOOK_BACK, ini_date=ini_date,
                                                                             end_date=end_date, filename=FILENAME_DATA,
                                                                                plot=plot)

