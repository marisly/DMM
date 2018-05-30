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



def mortality_rate_2005(dataframe):
    prev = dataframe.iloc[10,:21].values
    curr = dataframe.iloc[11,1:].values
    index = dataframe.iloc[11, 1:].index

    mortality = dict()

    for p,c,i in zip(curr,prev,index):
        mortality_rate = p/c
        mortality[i]=(mortality_rate)
    return mortality


mortality_men = mortality_rate_2005(rus_m)
mortality_women  = mortality_rate_2005(rus_f)
#
# plt.bar(range(len(rus_m)), list(rus_m.values()), align='center')
# plt.xticks(range(len(rus_m)), list(rus_m.keys()))
# plt.show()

# print('2005 to 2000 Moratality rates men',mortality_men)
# print('2005 to 2000 Moratality rates women', mortality_women)
# print(mortality_women['5 - 9'])
# #

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


def boys_to_girls():
    boys = rus_m.loc[:,'0 - 4']
    girls = rus_f.loc[:,'0 - 4']
    # boys_to_girls = pd.DataFrame
    boys_to_girls = pd.DataFrame(boys)
    for year,boy in boys.iteritems():
        girl = girls.loc[year]
        perc = 100 / (boy + girl) * boy
        boys_to_girls.loc[year] = perc
    mean = boys_to_girls.mean(axis =0)

    return boys_to_girls,mean

boy_probability,boy_probability_mean = boys_to_girls()
# print('Boys to girls birth probability    ', boy_probability, boy_probability_mean)


FERTILITY = 0.23

def population():
    index = 2005
    future_f = pd.DataFrame(rus_f)
    future_m = pd.DataFrame(rus_m)
    # print('population prediction', future_f.loc[2000].iteritems)
    columns = (future_f.columns)

    while index < 2100:
        column = 1

        for range in future_f.loc[index]:
            if column < 21:
                prediction_women = range * mortality_women[columns[column]]
                future_f.loc[index + 5,columns[column]]=prediction_women

                prediction_men = range * mortality_men[columns[column]]
                future_m.loc[index + 5, columns[column]] = prediction_men
                column += 1
            else:
                fertile_women = pd.DataFrame(future_f.loc[index,'15 - 19':'40 - 44'])
                # print(fertile_women)
                # print(fertile_women.sum(axis=0).values)

                boys_number = float(boy_probability_mean /100 * FERTILITY  * fertile_women.sum(axis=0).values)

                girls_number = float((100 - boy_probability_mean) / 100 * FERTILITY * fertile_women.sum(axis=0).values)

                # print(future_m.loc[index + 5, '5 - 9'])
                # print('BOY', index, boys_number)
                # print('GIRL', index, girls_number)
                # print(type(future_f))
                future_m.loc[index + 5, '0 - 4'] = boys_number
                future_f.loc[index + 5, '0 - 4'] = girls_number
                column += 1

        index += 5

    return future_f,future_m


future_f,future_m = (population())


YEAR = 2050
print("BY 5 year intervals prediction")
total_m = pd.DataFrame(future_m.loc[YEAR])
total_men = total_m.sum(axis = 0).values
print('Total men    ', YEAR, total_men)


total_f = pd.DataFrame(future_f.loc[YEAR])
total_women = total_f.sum(axis = 0).values

print('Total women    ', YEAR, total_women)
print('Total     ', YEAR, total_women + total_men)

# print(future_f)

plt.style.use('seaborn-whitegrid')

# ax = total_f.plot(color = 'orange', label="women")
# ax = total_m.plot(ax=ax,color = 'lightblue', label='men')
# plt.xticks(range(len(total_f)), list(total_f.index),rotation='vertical')
# ax.tick_params(axis='both', which='major', labelsize=8)
# leg = ax.legend()
# plt.show()

#Show graphs UN vs MINE

if YEAR < 2051:
    estimate = pd.read_excel(FILENAME, sheetname='both; 2010-50, medium-fertility', skiprows=range(6), index_col=0)
    est_un = (estimate.loc[estimate.iloc[:, 1] == 'Russian Federation'])
    est_un = pd.DataFrame(est_un.iloc[:, 4:])
    est_un.set_index(est_un.columns[0], inplace=True, drop=True)
    est_un = pd.DataFrame(est_un.loc[YEAR])

    total_un = est_un.sum(axis = 0).values

    print('Total   by UN prediction  ', YEAR, total_un)
    # total = total_f.add(total_m)
    # total_all = total.iloc[0]
    # ax = est_un.plot(color = 'grey', label="UN Prediction")
    # ax = total.plot(ax=ax, color ='green', label='Model by 5 years period')
    # plt.xticks(range(len(est_un)), list(est_un.index),rotation='vertical')
    # ax.tick_params(axis='both', which='major', labelsize=8)
    # leg = ax.legend()
    # plt.show()



# Перевести коэффициенты к шагу 1 год


print("All to 1 YEAR" )

m_2005 = pd.DataFrame(rus_m.loc[2005])
f_2005 = pd.DataFrame(rus_f.loc[2005])

reindex_m = pd.DataFrame()
reindex_f = pd.DataFrame()




def interpolate_to_1_year(dataframe,year):
    x = dict()
    year = year
    for index, value in dataframe.itertuples():
        # print('PROCESSING     ', index, value)
        if index == '100+':
            start_intv = 100
            end_intv = 104
            x[end_intv] = value/5
        else:
            (start_intv, end_intv) = [int(s) for s in index.split(' - ')]
        if start_intv == 0:
            x[start_intv] = value/5
        x[(start_intv + end_intv) / 2] = value/5

    y = list(x.values())
    x = list(x.keys())
    interpolation = interpolate.interp1d(x, y, kind='linear')
    reindex_dataframe=pd.DataFrame()

    for index,value in dataframe.itertuples():
        # print('PROCESSING     ',index,value)
        if index == '100+':
            start_intv = 100
            end_intv = 104
        else:
            (start_intv, end_intv) = [int(s) for s in index.split(' - ')]

        while start_intv <= end_intv:
            value = interpolation(start_intv)
            reindex_dataframe.loc[year,start_intv]=value
            start_intv += 1
    return reindex_dataframe

#SIMPLY divide by 5
def divide_to_1_year(dataframe,year):
    reindex_dataframe = pd.DataFrame()
    year = year
    for index, value in dataframe.itertuples():
        # print('PROCESSING     ', index, value)
        if index == '100+':
            start_intv = 100
            end_intv = 104
        else:
            (start_intv, end_intv) = [int(s) for s in index.split(' - ')]
        for i in range(start_intv,end_intv+1):
            reindex_dataframe.loc[year,i]=value/5
            i += 1

    return reindex_dataframe

reindex_f = divide_to_1_year(f_2005,2005)
reindex_m = divide_to_1_year(m_2005,2005)

m_2000 = pd.DataFrame(rus_m.loc[2000])
f_2000 = pd.DataFrame(rus_f.loc[2000])

reindex_f_2000 = divide_to_1_year(f_2000,2000)
reindex_m_2000 = divide_to_1_year(m_2000,2000)

reindex_f = reindex_f_2000.append(reindex_f)
reindex_m = reindex_m_2000.append(reindex_m)
# print("REINDEXED FEMALE",reindex_f)
# print("REINDEXED MEN",reindex_m)

# reindex_f.loc[2005].plot()
# reindex_m.loc[2005].plot()
# plt.show()
reindex_all = reindex_f + reindex_m

# plt.xaxis.set_ticks(np.arange(1, 100, 1))
# reindex_m.loc[2005].plot()
# reindex_m.loc[2000].plot()
# # reindex_f.loc[2005].plot()
# plt.xticks(np.arange(0, 100, step=5))
# plt.show()

def recalc_index(x):
    x = 1 - (1-x)/5
    return x


# MORTALITY BY REINDEX
def mortality_rate_2005(dataframe):
    prev = dataframe.loc[2000,].values
    curr = dataframe.loc[2005,5:].values
    index = dataframe.loc[2005, 5:].index
    mortality = dict()

    for i in range(1,5):
        if str(dataframe) == str(reindex_m):
            mortality[i]= recalc_index(0.9968467586899926)
            # dataframe.loc['mort', i] =recalc_index(0.9968467586899926)
        else:
            mortality[i] =recalc_index(0.9984944205912538)
            # dataframe.loc['mort', i] = recalc_index(0.9984944205912538)
        i += 1


    for p,c,i in zip(curr,prev,index):
        mortality_rate = 1 - (1 - p/c)/6
        mortality[i] = (mortality_rate)
        dataframe.loc['mort', i] = mortality_rate

    return mortality

mortality_men = mortality_rate_2005(reindex_m)
mortality_female = mortality_rate_2005(reindex_f)


# print('Mortality men',mortality_men)
# print('Mortality women',mortality_female)


# PLOT MOrtality index by 1 year

# lists = sorted(mortality_female.items())
# x, y = zip(*lists) # unpack a list of pairs into two tuples
# plt.plot(x, y,color='orange', label='Women')
# lists = sorted(mortality_men.items())
# x, y = zip(*lists) # unpack a list of pairs into two tuples
# plt.plot(x, y,color='lightblue', label='Men')
#
# # plt.xticks(range(len(total_un_by_year)), list(total_un_by_year.index), rotation='vertical')
# plt.tick_params(axis='both', which='major', labelsize=8)
# leg = plt.legend()
# plt.show()
#


FERTILITY = 0.33
START_YEAR = 2005
END_YEAR = 2101
#
def population_by_year():

    index = 2005

    columns = (reindex_f.columns)
    print()

    while index < END_YEAR:

        # range_f=reindex_f.loc[index]

        columns = list(reindex_f.loc[index].index)
        # print('TEST',index,columns)

        for column in columns:

            if column == 0:
                fertile_women = pd.DataFrame(reindex_f.loc[index, 15:40])
                boys_number = float(boy_probability_mean / 100 * FERTILITY/5 * fertile_women.sum(axis=0).values)
                girls_number = float((100-boy_probability_mean)/ 100 * FERTILITY/5 * fertile_women.sum(axis=0).values)

                reindex_m.loc[index + 1, 0] = boys_number
                reindex_f.loc[index + 1, 0] = girls_number
                reindex_all.loc[index + 1, 0] = girls_number + boys_number
                # print("BOYS", boys_number, "GIRLS", girls_number)

                if index == 2005:
                    range_f = reindex_f.loc[2000, column]
                    range_m = reindex_m.loc[2000, column]

                else:
                    range_f = reindex_f.loc[index, column]
                    range_m = reindex_m.loc[index, column]

                prediction_women = range_f * mortality_female[column+1]
                reindex_f.loc[index + 1, columns[1]] = prediction_women

                prediction_men = range_m * mortality_men[column+1]
                reindex_m.loc[index + 1, columns[1]] = prediction_men

                reindex_all.loc[index + 1, columns[column+1]] = prediction_men + prediction_women
                # print(index, column, range_f,range_m,prediction_women, prediction_men)

            elif column == 104:
                pass

            else:

                range_f = reindex_f.loc[index, column]
                range_m = reindex_m.loc[index, column]

                prediction_women = range_f * mortality_female[column+1]
                reindex_f.loc[index + 1, columns[column+1]] = prediction_women

                prediction_men = range_m * mortality_men[column+1]
                reindex_m.loc[index + 1, columns[column+1]] = prediction_men

                reindex_all.loc[index + 1, columns[column+1]] = prediction_men+prediction_women
                # print(index,column, prediction_women,prediction_men)

        index += 1

    return reindex_f, reindex_m
#
#
#
#
reindex_f,reindex_m = population_by_year()
reindex_all = reindex_f.add(reindex_m, fill_value=0)
# print("RESULTS", reindex_m,reindex_f)
#
writer = pd.ExcelWriter('output.xlsx')
reindex_m.to_excel(writer,'Men')
reindex_f.to_excel(writer,'Women')
reindex_all.to_excel(writer,'All')

writer.save()
#
#
print("INDEXED BY 1 YEAR")
YEAR = 2050
total_m = pd.DataFrame(reindex_m.loc[YEAR])
total_men = total_m.sum(axis = 0).values
print('Total men    ', YEAR, total_men)

total_f = pd.DataFrame(reindex_f.loc[YEAR])
total_women = total_f.sum(axis = 0).values

print('Total women    ', YEAR, total_women)
print('Total     ', YEAR, total_women + total_men)

plt.figure(figsize=(18,5))
plt.xlabel('Number of requests every 10 minutes')

ax = total_f.plot(color = 'orange', label="women")
ax = total_m.plot(ax=ax,color = 'lightblue', label='men')

plt.xticks(range(len(total_f)), list(total_f.index),rotation='vertical')
ax.tick_params(axis='both', which='major', labelsize=7)
leg = ax.legend()

plt.show()
#
if YEAR > 2050:
    YEAR = 2050
    print('Total   by UN prediction  ', YEAR, total_un)
    total_un_by_year = divide_to_1_year(est_un,YEAR).loc[YEAR]
    print('BY YEAR UN_____', type(total_un_by_year))
    #caclulate popolation by own method
    # total_un_by_year = total.iloc[0]

    total_model = reindex_all.loc[YEAR]
    print('BY YEAR UN_____', type(total_model))
    plt.figure(figsize=(18, 5))

    ax = total_un_by_year.plot(color = 'grey', label="UN")
    ax = total_model.plot(ax=ax, color ='green', label='model')
    plt.xticks(range(len(total_un_by_year)), list(total_un_by_year.index),rotation='vertical')
    ax.tick_params(axis='both', which='major', labelsize=7)
    leg = ax.legend()
    plt.show()
