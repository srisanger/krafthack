from IPython import display
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import numpy as np

# plotting function:
def trainingplot(tag, losshistory:list, predictionhistory:list, targethistory:list, periods:list, trainingtime:list, targets:list,
                 plotlen:int=5000, figsize:tuple=(15,15)):
    L = np.array(losshistory)
    P = np.array(predictionhistory)
    T = np.array(targethistory)
    I = np.array(periods)
    trainingtime = np.array(trainingtime)
    
    display.clear_output(wait=True)
    fig, axs = plt.subplots(1+P.shape[1], figsize=figsize)
    axs[0].plot(I, L, '-') 
    avgloss = np.nanmean(L)
    avgtime = np.nanmean(trainingtime) * 60 * 60 * 24
    axs[0].title.set_text(f'loss - episode {tag} - average: {avgloss} | training time (seconds/day) - {avgtime}')
    axs[0].grid()
    
    for i in range(P.shape[1]):
        axs[1+i].plot(I[-plotlen:], T[:,i][-plotlen:], '-', label='outcome') 
        axs[1+i].plot(I[-plotlen:], P[:,i][-plotlen:], '-', alpha=0.66, label='prediction') 
        axs[1+i].title.set_text(targets[i])
    plt.show()
    
# plotting function:
def lossplot(tag, losshistory:list, plotlen:int=5000, figsize:tuple=(15,15)):
    L = np.array(losshistory)
    display.clear_output(wait=True)
    fig, axs = plt.subplots(1, figsize=figsize)
    axs.plot(np.array(L)[-plotlen:], L[-plotlen:], '-') 
    avgloss = np.nanmean(L)
    axstitle.set_text(f'loss - episode {tag} - average: {avgloss}')
    axs.grid()
    plt.show()