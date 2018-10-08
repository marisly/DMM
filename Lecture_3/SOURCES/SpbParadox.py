# -*- coding: cp1251 -*-
# Monty Hall problem: https://en.wikipedia.org/wiki/Monty_Hall_problem
import random
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# N - the number of games
N=1000
allWin = []
for i in range(0,N):
    counter = 1
    while True:
     toss  = random.randrange(0,2)               # random result
     if toss == 0:
         win = 2**counter
         allWin.append(win)
         break
     else:
         counter = counter + 1

print(np.mean(allWin))
# ------------------------------------
plt.hist(allWin, np.arange(min(allWin), 2**8+1))
plt.grid()
plt.xlabel("Sum of Win")
plt.ylabel("Probability")
plt.show()
