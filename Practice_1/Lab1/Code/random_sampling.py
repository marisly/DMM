import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random

xk = np.arange(6)+1
pk = (0.1, 0.2, 0.3, 0.25, 0.1, 0.05)
custm = stats.rv_discrete(name='custm', values=(xk, pk))

def DrawSampleHistogram(R,title,bins=None):
    fig = plt.figure()
    x = np.arange(len(R)) 
    plt.grid()       
    if(bins is None):    
        plt.hist(R, range=None, normed=False)
    else:
        plt.hist(R, bins=bins, range=None, normed=False)
    plt.title(title)
    plt.show()

def DrawSample(R,title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(R))     
    ax.plot(x, R, 'ro', ms=12)
    plt.grid()
    plt.title(title)
    plt.show()

# ------- UNIFORM DISTRIBUTION -------
size=100000
minVal,maxVal = 1,8
# R = [ random.randint(minVal,maxVal) for i in range(size)]
# DrawSample(R,'SAMPLE FROM UNIFORM DISTRIBUTION')
# DrawSampleHistogram(R,title='UNIFORM SAMPLE HISTOGRAM',bins = maxVal-minVal+1)
#
# # ------- USER-DEFINES DISCRETE DISTRIBUTION -------
# fig = plt.figure()
# ax = fig.add_subplot(111)
# width = 0.8
# ax.set_ylim(0,max(custm.pmf(xk))*1.3)
# ax.bar(xk-width/2, custm.pmf(xk),width)
# plt.grid()
# plt.show()

# ------- SAMPLING -------
size=100
R = custm.rvs(size = size)
#R = custm.rvs()

DrawSample(R,'SAMPLE FROM DISCRETE DISTRIBUTION')
DrawSampleHistogram(R,title='SAMPLE FROM DISCRETE DISTRIBUTION',bins=len(pk))