# -*- coding: cp1251 -*-
# Two envelopes problem https://en.wikipedia.org/wiki/Two_envelopes_problem
import random
import numpy as np
from enum import Enum

class Decision(Enum):
    CHANGE = 1
    NOTCHANGE = 2

# Strategy: always change
N,sumWin,sumLost=100000,0,0
M1 = 1
M2 = 0.5                                # half of 1
totalMoney = 0                          # total money to win
prevVal = None
for i in range(0,N):
     money = random.randrange(1,100)           # random value to win in one game
     index = random.randrange(0,2)             # where the value is greater
     conv = [money*(M1 if index==0 else M2),
             money*(M1 if index==1 else M2)]
     totalMoney = totalMoney + sum(conv)

     decision = Decision.NOTCHANGE
     if prevVal is not None and conv[0]<prevVal:
         decision = Decision.CHANGE

     if(decision == Decision.NOTCHANGE):
         sumWin = sumWin + conv[0]
         sumLost = sumLost + conv[1]
         #prevVal = conv[0]

     if(decision == Decision.CHANGE):           
         sumWin = sumWin + conv[1]
         sumLost = sumLost + conv[0]
         #prevVal = conv[1]

     #prevVal = sumWin
     prevVal = conv[0]

print("{0:.3}".format(sumWin/totalMoney))