# -*- coding: cp1251 -*-
# Monty Hall problem: https://en.wikipedia.org/wiki/Monty_Hall_problem
import random
import numpy as np

# N - the number of repetitions
# suc - the number of successful outcomes
# notsuc - the number of unsuccessful outcomes
N,suc,notsuc=10000,0,0
for i in range(0,N):
     car = random.randrange(0,3)                       # random position of a car
     choice = random.randrange(0,3)                    # random choice of the player
     ind,newChoice,open = [0,1,2],-1,-1
     goat_ind = [x for x in ind if x!=car]
     monty_choice = random.sample(goat_ind, 1)[0]      # choose random door with goat
     if(monty_choice != choice):
         change_ind = [x for x in ind if x!=choice ]   # open the door with goat (may be the same as choice)
         choice = random.sample(change_ind, 1)[0]      # final player's choice

     if(choice==car):                                  # check the result
         suc = suc + 1
     else:
         notsuc = notsuc + 1

# print stat (probabilities)
print(suc/N)
print(notsuc/N)