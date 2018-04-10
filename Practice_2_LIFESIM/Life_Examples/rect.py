import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

N=20
matrix = np.zeros((N,N))
cmap = colors.ListedColormap(['white', 'black'])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(matrix, cmap=cmap)
ticks = [i+0.5 for i in range(0,N-1)]
plt.xticks(ticks)
plt.yticks(ticks)    
ax.grid()

def onmouseclick(event):        
    x,y = int(round(event.xdata)),int(round(event.ydata))               
    matrix[y,x]=1
    ax.matshow(matrix, cmap=cmap)
    plt.xticks(ticks)
    plt.yticks(ticks) 
    fig.canvas.draw()
    
def onkeypress(event):   
    # --- MODEL (TO DO)---

    if(matrix[0,0]==0):
        matrix[0,0]=1              
    else: 
        matrix[0,0]=0              
    ax.matshow(matrix, cmap=cmap)
    plt.xticks(ticks)
    plt.yticks(ticks)  
    fig.canvas.draw()
            
fig.canvas.mpl_connect('button_press_event', onmouseclick)
fig.canvas.mpl_connect('key_press_event', onkeypress)

plt.show()
