#%%
import pandas as pd

import matplotlib as mpl
# Use the pgf backend (must be set before pyplot imported)
#mpl.use('pgf')

import matplotlib.pyplot as plt



data = pd.read_csv('../results/mean_nmse_time.txt', delimiter='|', skipinitialspace=True)
data.columns = data.columns.str.strip()
data.set_index('MICROPHONES', inplace=True)
plt.figure(figsize=(10, 6))
for col in data.columns[0:]:
    plt.plot(data.index, data[col], marker='o', label=col)

plt.legend()
plt.xlabel('MICROPHONES')
plt.ylabel('NMSE(dB)')
plt.title('NMSE for each channel and model')
plt.grid(True)
plt.xticks(data.index)  
plt.tight_layout()


#plt.savefig('../results/image/nmse_for_each_channel.pgf')
plt.savefig('../results/image/nmse_for_each_channel.png')

plt.show()
# %%
import numpy as np
# Leggi i dati dal file.txt
data = {}
with open('../results/nmse_freq.txt', 'r') as file:
    nmse = []
    for i,line in enumerate(file):
        if line.strip():  # Ignora le righe vuote
            parts = line.split('|')
            numero = parts[0].strip()
            if i == 0:
                freq = [float(x) for x in parts[1].split()]
            else:
                nmse.append([float(x) for x in parts[1].split()])

# Creazione dei quattro subplot
fig, axs = plt.subplots(2, 2, figsize=(10, 8))


legenda = ['Siren', 'Siren+pde', 'Siren+pde+rowdy']
# Plot per ogni numero
for i, ax in enumerate(axs.flat):
    for j in range(3):
        ax.plot(freq,10*np.log(nmse[i*3 + j]),label=legenda[j])
        ax.set_title(f"Subplot {i+1}")
        ax.legend()

# Imposta i titoli globali per gli assi x e y
fig.text(0.5, 0.04, 'Frequenza', ha='center')
fig.text(0.04, 0.5, 'Valore', va='center', rotation='vertical')

# Mostra il grafico
plt.show()

plt.tight_layout()
plt.show()

# %%
