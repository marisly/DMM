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


df_all = pd.read_excel(FILENAME,sheet_name=sheet_all, skiprows=range(6))
df_male = pd.read_excel(FILENAME,sheet_name=sheet_male, skiprows=range(6))
df_female = pd.read_excel(FILENAME,sheet_name=sheet_female, skiprows=range(6))

rus_all = (df_all.loc[df_all.iloc[:,2] == 'Russian Federation'])
rus_all = pd.DataFrame(rus_all.iloc[:,5:])
rus_all.set_index(rus_all.columns[0],inplace=True,drop=True)

rus_f = (df_female.loc[df_female.iloc[:,2] == 'Russian Federation'])
rus_f = pd.DataFrame(rus_f.iloc[:,5:])
rus_f.set_index(rus_f.columns[0],inplace=True,drop=True)

rus_m = (df_male.loc[df_male.iloc[:,2] == 'Russian Federation'])
rus_m = pd.DataFrame(rus_m.iloc[:,5:])
rus_m.set_index(rus_m.columns[0],inplace=True,drop=True)


def mortality_rate_2005(dataframe):
    prev = dataframe.iloc[10,:21].values
    curr = dataframe.iloc[11,1:].values
    index = dataframe.iloc[11, 1:].index

    mortality = dict()

    for p,c,i in zip(curr,prev,index):
        mortality_rate = c/p
        mortality[i]=(mortality_rate)
    return mortality


men = mortality_rate_2005(rus_m)
women = men = mortality_rate_2005(rus_f)
#
# # plt.bar(range(len(men)), list(men.values()), align='center')
# # plt.xticks(range(len(men)), list(men.keys()))
# # plt.show()
#
# print('2005 to 2000 Moratality rates men',men)
# print('2005 to 2000 Moratality rates women', women)
# #

def fertility():
    fem = rus_f.loc[:,'20 - 24':'35 - 39']
    fem = fem.sum(axis = 1)
    fert = pd.DataFrame(columns=['fertility'],index=['year'])

    children = list(rus_all.loc[:,'0 - 4'].values)
    # print(children)

    index = 0
    for year,row in(fem.iteritems()):
        # print(row,children[index])
        fert.loc[year] = children[index]/row
        index += 1
    return fert

fertility_indexes = fertility()
# print('Fertility coeff for women 20-39   ', fertility_indexes)


def boys_to_girls():
    boys = rus_m.loc[:,'0 - 4']
    girls = rus_f.loc[:,'0 - 4']
    boys_to_girls = pd.DataFrame
    boys_to_girls = boys
    for year,boy in boys.iteritems():
        girl = girls.loc[year]
        perc = 100 / (boy + girl) * boy
        boys_to_girls.loc[year] = perc
    mean = boys_to_girls.mean(axis =0)

    return boys_to_girls,mean

boy_probability,boy_probability_mean = boys_to_girls()
# print('Boys to girls birth probability    ', boy_probability, boy_probability_mean)

# Спрогнозировать изменение численности населения страны и демографический профиль на 100 лет!
# Перевести коэффициенты к шагу 1 год



def population():
    index = 2005
    future_f = pd.DataFrame(rus_f)
    future_m = pd.DataFrame(rus_m)
    # print('population prediction')
    while index < 2100:

        # print(future_f.loc[index])
        for range in future_f.loc[index]:
            print('Range',range)

        index = index + 5
        break


    # future_f =

population()






