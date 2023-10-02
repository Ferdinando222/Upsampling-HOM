import matplotlib.pyplot as plt
import numpy as np
import wandb

def plot_model(azimuth,colatitude,normalized_output_data,previsions,points_sampled,pinn=False):
    # Crea un grafico 2D della pressione in funzione di azimuth e colatitude
    fig, (ax,ax1) = plt.subplots(1, 2, figsize=(12, 5))
    sc = ax.scatter(azimuth, colatitude, c=np.abs(previsions), cmap='viridis')
    sc1 = ax1.scatter(azimuth, colatitude, c=np.abs(normalized_output_data), cmap='viridis')

    # Imposta manualmente i segnaposti e le etichette sugli assi x e y
    azimuth_ticks = [0, np.pi / 2, np.pi, 2 * np.pi]
    colatitude_ticks = [0, np.pi / 2, np.pi]

    ax.set_xticks(azimuth_ticks)
    ax.set_yticks(colatitude_ticks)
    ax1.set_xticks(azimuth_ticks)
    ax1.set_yticks(colatitude_ticks)
    # Imposta manualmente le etichette degli assi
    ax.set_xticklabels(['0', 'π/2', 'π', '2π'])
    ax.set_yticklabels(['0', 'π/2', 'π'])
    ax1.set_xticklabels(['0', 'π/2', 'π', '2π'])
    ax1.set_yticklabels(['0', 'π/2', 'π'])


    ax.set_xlabel('Azimuth (radians)')
    ax.set_ylabel('Colatitude (radians)')
    ax.set_title('Previsions')
    ax.grid(True)

    ax1.set_xlabel('Azimuth (radians)')
    ax1.set_ylabel('Colatitude (radians)')
    ax1.set_title('Ground Truth')
    ax1.grid(True)

    plt.colorbar(sc1,label='Pressure')
    plt.colorbar(sc, label='Pressure')

    if pinn:
        plt.savefig(f"../src/image/Pinn_{points_sampled}.png")
    else:
        plt.savefig(f"../src/image/NoPinn_{points_sampled}.png")

    plt.show()
