# -*- coding: cp1251 -*-
# �������� ����� �����: https://ru.wikipedia.org/wiki/��������_�����_�����
import random
import numpy as np
from enum import Enum

class Decision(Enum):
    CHANGE = 1
    NOTCHANGE = 2

# ������� 2: ���������
N,sumWin,sumLost=10000,0,0
M1 = 1
M2 = 0.5 # ����������� ����������
totalMoney = 0                          # ������ ��� ������������� 1.5 ��. �����
prevVal = []
for i in range(0,N):
     money = random.randrange(1,1000)           # ��������� �������� ����� (1..100)
     index = random.randrange(0,2)             # � ����� ��������� ������ (0,1)
     conv = [money*(M1 if index==0 else M2),
             money*(M1 if index==1 else M2)]
     totalMoney = totalMoney + sum(conv)
     K = 1.2# 1.1

     decision = Decision.NOTCHANGE
     if len(prevVal)>0 and conv[0]<np.mean(prevVal):
         decision = Decision.CHANGE

     if(decision == Decision.NOTCHANGE):        # � ��������� �������� ������ �����
         sumWin = sumWin + conv[0]
         sumLost = sumLost + conv[1]
         prevVal.append(conv[0]*K)

     if(decision == Decision.CHANGE):           # � ��������� �������� ������ �����
         sumWin = sumWin + conv[1]
         sumLost = sumLost + conv[0]
         prevVal.append(conv[1]*K)

     #prevVal.append(conv[0])

print("Win % {0:.3}".format(sumWin/totalMoney))