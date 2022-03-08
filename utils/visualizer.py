from IPython import display
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import numpy as np

# plotting function:
def lossplot(tag, losshistory:list, plotlen:int=5000, figsize:tuple=(5,5)):
    L = np.array(losshistory)
    display.clear_output(wait=True)
    fig, axs = plt.subplots(1, figsize=figsize)
    plt.plot(np.arange(len(L[-plotlen:])), L[-plotlen:], '-') 
    avgloss = np.nanmean(L)
    plt.title(f"loss - episode {tag} - average: {format(avgloss, '.3g')}")
    plt.grid()
    plt.show()