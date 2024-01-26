'''Script to merge files in this folder into a single country file'''

import pandas as pd
from glob import glob
import datetime

dfs = glob('forecast*.csv')


for i, pth in enumerate(dfs):
    if i == 0:
        df = pd.read_csv(pth)
    else:
        df = pd.concat([df, pd.read_csv(pth)])

df.to_csv(f'Forecast_Brasil_{datetime.date.today().isoformat()}.csv.gz', index = False)