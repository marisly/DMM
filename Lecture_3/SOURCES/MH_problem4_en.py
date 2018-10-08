# -*- coding: cp1251 -*-
# Monty Hall problem: https://en.wikipedia.org/wiki/Monty_Hall_problem
import random
import numpy as np

# N - the number of repetitions
# suc - the number of successful outcomes
# notsuc - the number of unsuccessful outcomes
N,suc,notsuc=10000,0,0
for i in range(0,N):
     car = random.randrange(0,3)               # random position of a car
     choice = random.randrange(0,3)            # random choice of the player
     ind,newChoice,open = [0,1,2],-1,-1
     if(car != choice):                        # player has guessed
         ind.remove(choice)                    # car can not be opened
         open = random.sample(ind,1)[0]        # open random door of the two remaining
         ind.remove(open)                      # remove from final choice an open door
         choice = ind[0]                       # final playr's choice
     else:                                     # player hasn't guessed
         if(random.random()>0.5):
             ind.remove(choice)                # remove current choice
             choice = random.sample(ind, 1)[0] # open random door of the two remaining

     if(choice==car):                          # check the result
         suc = suc + 1
     else:
         notsuc = notsuc + 1

# print stat (probabilities)
print(suc/N)
print(notsuc/N)