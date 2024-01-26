import pandas as pd
import geopandas as gpd
from forecast import apply_forecast_macro

# from train_models_to_for import LOOK_BACK, PREDICT_N
PREDICT_N = 10  # number of new days predicted
LOOK_BACK = 12

df_muni = gpd.read_file('../muni_br.gpkg')

dfs = pd.read_csv('../macro_saude.csv')
ini_date = None
end_date = '2024-01-07'
# for macro in dfs.loc[dfs.state=='MG'].code_macro.unique():
for macro in dfs.code_macro.unique():
    print(f'Forecasting: {macro}')

    filename = f'../data/dengue_{macro}.csv'
    model_name = f'trained_{macro}_dengue_macro'

    df_for = apply_forecast_macro(macro, ini_date, end_date, look_back=LOOK_BACK, predict_n=PREDICT_N,
                                  filename=filename, model_name=model_name, df_muni=df_muni)
