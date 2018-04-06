import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.colors as colors, matplotlib.colorbar as colorbar
import matplotlib.cm as cm, matplotlib.font_manager as fm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
# from mpl_toolkits.basemap import Basemap
# %matplotlib inline
FILENAME = 'AGE_TASK_DMM/age_data.xls'

sheet_all = 'both; 1950-2005, estimates'
sheet_male = 'm; 1950-2005, estimates'
sheet_female = 'f; 1950-2005, estimates'


df_all = pd.read_excel(FILENAME,sheetname=sheet_all, skiprows=range(6))
df_male = pd.read_excel(FILENAME,sheetname=sheet_male, skiprows=range(6))
df_female = pd.read_excel(FILENAME,sheetname=sheet_female, skiprows=range(6))

rus_all = (df_all.loc[df_all.iloc[:,2] == 'Russian Federation'])
rus_all = rus_all.iloc[:,5:]
rus_all.set_index(rus_all.columns[0],inplace=True,drop=True)

rus_f = (df_female.loc[df_female.iloc[:,2] == 'Russian Federation'])
rus_f = rus_f.iloc[:,5:]
rus_f.set_index(rus_f.columns[0],inplace=True,drop=True)

rus_m = (df_male.loc[df_male.iloc[:,2] == 'Russian Federation'])
rus_m = rus_m.iloc[:,5:]
rus_m.set_index(rus_m.columns[0],inplace=True,drop=True)



def mortality_rate(dataframe):
    prev = dataframe.iloc[10,:21].values
    curr = dataframe.iloc[11,1:].values
    index = dataframe.iloc[11, 1:].index
    print(index)
    mortality = dict()
    for p,c,i in zip(curr,prev,index):
        mortality_rate = c/p
        mortality[i]=(mortality_rate)
    return mortality

print('2005 to 2000 Moratality rates', mortality_rate(rus_m))




