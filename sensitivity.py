import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.colors as colors, matplotlib.colorbar as colorbar
import matplotlib.cm as cm, matplotlib.font_manager as fm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy import interpolate
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
rus_all = pd.DataFrame(rus_all.iloc[:,5:])
rus_all.set_index(rus_all.columns[0],inplace=True,drop=True)

rus_f = (df_female.loc[df_female.iloc[:,2] == 'Russian Federation'])
rus_f = pd.DataFrame(rus_f.iloc[:,5:])
rus_f.set_index(rus_f.columns[0],inplace=True,drop=True)

rus_m = (df_male.loc[df_male.iloc[:,2] == 'Russian Federation'])
rus_m = pd.DataFrame(rus_m.iloc[:,5:])
rus_m.set_index(rus_m.columns[0],inplace=True,drop=True)



def calc_mortality_matrix(dataframe):
    mortality = (dataframe).copy()
    for col in range(1,11):
        prev = dataframe.iloc[col-1, :21].values
        curr = dataframe.iloc[col, 1:].values
        index = dataframe.iloc[col, 1:].index
        row_index = dataframe.index[col]
        #
        for p, c, i in zip(curr, prev, index):
            if p == '-' or c == '-':
                mortality.at[row_index,i]  = np.nan
            else:
                mortality_rate = float(p) / float(c)
                mortality.at[row_index,i] = (mortality_rate)

    mortality = mortality.iloc[1:11,1:]
    return mortality



mortality_women = calc_mortality_matrix(rus_f)
mortality_men = calc_mortality_matrix(rus_m)

# mortality_women.boxplot()
# plt.xticks(rotation='vertical',fontsize=8)
# plt.margins(0.2)
# plt.show()
#
# mortality_men.boxplot()
# plt.xticks(rotation='vertical',fontsize=8)
# plt.margins(0.2)
# plt.show()


def fertility():
    fem = rus_f.loc[:,'15 - 19':'40 - 44']
    fem = fem.sum(axis = 1)
    fert = pd.DataFrame(columns=['fertility'],index=['year'])

    children = list(rus_all.loc[:,'0 - 4'].values)
    # print(children)

    index = 0
    for year,row in(fem.iteritems()):
        # print(row,children[index])
        fert.loc[year] = ((children[index]))/(row)
        index += 1
    return fert

fertility_indexes = fertility()
# print('Fertility coeff for women 20-39   ', fertility_indexes)
#


def boys_to_girls():
    boys = rus_m.loc[:,'0 - 4']
    girls = rus_f.loc[:,'0 - 4']
    # boys_to_girls = pd.DataFrame
    boys_to_girls =  pd.DataFrame(columns=['boys_to_girls'], index=['year'])
    # fert = pd.DataFrame(columns=['fertility'], index=['year'])
    for year,boy in boys.iteritems():
        girl = girls.loc[year]
        perc = 100 / (boy + girl) * boy
        boys_to_girls.loc[year] = perc/100
    mean = boys_to_girls.mean(axis =0)

    return boys_to_girls,mean

boy_probability,boy_probability_mean = boys_to_girls()
# print('Boys to girls birth probability    ', boy_probability, boy_probability_mean)

# result = pd.concat([mortality_women,fertility_indexes,boy_probability])
# result.boxplot(figsize=[10,8])
# plt.xticks(rotation='vertical',fontsize=11)
# plt.margins(0.2)
# plt.ylim(ymin=0.1,ymax=1.1)
# plt.show()



mortality_men = mortality_men.iloc[0]
mortality_women = mortality_women.iloc[0]
# FERTILITY = 0.23

def population(FERTILITY, BOY_TO_GIRL,w_mort_20_24):
    index = 2005
    future_f = pd.DataFrame(rus_f)
    future_m = pd.DataFrame(rus_m)

    # print('population prediction', future_f.loc[2000].iteritems)
    columns = (future_f.columns)

    while index < 2100:
        column = 1

        for range in future_f.loc[index]:
            if column < 21:
                if columns[column] == '20-24':
                    prediction_women = range * w_mort_20_24
                    future_f.loc[index + 5, columns[column]] = prediction_women

                    prediction_men = range * mortality_men[columns[column]]
                    future_m.loc[index + 5, columns[column]] = prediction_men
                    column += 1
                else:
                    prediction_women = range * mortality_women[columns[column]]
                    future_f.loc[index + 5,columns[column]]=prediction_women

                    prediction_men = range * mortality_men[columns[column]]
                    future_m.loc[index + 5, columns[column]] = prediction_men
                    column += 1
            else:

                fertile_women = pd.DataFrame(future_f.loc[index,'15 - 19':'40 - 44'])
                # print(fertile_women)
                # print(fertile_women.sum(axis=0).values)

                boys_number = float(BOY_TO_GIRL  * FERTILITY * fertile_women.sum(axis=0).values)

                girls_number = float((1 - BOY_TO_GIRL)  * FERTILITY * fertile_women.sum(axis=0).values)

                # print(future_m.loc[index + 5, '5 - 9'])
                # print('BOY', index, boys_number)
                # print('GIRL', index, girls_number)
                # print(type(future_f))
                future_m.loc[index + 5, '0 - 4'] = boys_number
                future_f.loc[index + 5, '0 - 4'] = girls_number
                column += 1

        index += 5

    YEAR = 2100
    total_m = pd.DataFrame(future_m.loc[YEAR])
    total_men = total_m.sum(axis = 0).values

    total_f = pd.DataFrame(future_f.loc[YEAR])
    total_women = total_f.sum(axis = 0).values

    total = total_men+total_women
    return total

total_base_scenario = (population(FERTILITY=0.23,BOY_TO_GIRL=0.5,w_mort_20_24=0.997277))
print('TOTAL PEOPLE', total_base_scenario)


plt.style.use('seaborn-whitegrid')


#
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np



# Define the model inputs
problem = {
    'num_vars': 3,
    'names': ['fertility', 'w_mort_20_24', 'boys_girls'],
    'bounds': [[0.194608, 0.472530],
               [0.990274,  1.018228],
               [0.508009, 0.516034]]
}


def evaluate(values):
    Y = np.zeros([values.shape[0]])

    for i, X in enumerate(values):
        print("TEST",i,X)
        Y[i] = population(X[0],X[1],X[2])
    return Y



# Generate samples
param_values = saltelli.sample(problem, 10)
# print(len(param_values),param_values)

# Run model (example)
Y = evaluate(param_values)

# Perform analysis
Si = sobol.analyze(problem, Y,print_to_console=True)

# Print the first-order sensitivity indices
print(Si['S1'])

#
# Выполнить анализ чувствительности для демографической модели относительно набора параметров: коэффициент фертильности, соотношение мальчиков/девочек, коэффициент “выживаемости” для различных возрастных групп (можно взять не все). Выход модели: число жителей на заданный год. Протестировать на итоговых значениях прогноза на 10, 20, 50, 100 лет.
# Определить диапазоны значений параметров модели из данных за предыдущие периоды (1950-2000)
# На основе всех диапазонов значений параметров выполнить анализ неопределенности в форме графика с доверительными интервалами результатов. Значения между границами можно считать распределенными равномерно.
#

