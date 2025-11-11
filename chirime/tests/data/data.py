import os

import pandas as pd

file = os.path.join(os.path.dirname(__file__), "prefecture_pnt.csv")
PREF_DF = pd.read_csv(file)
