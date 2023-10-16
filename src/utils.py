import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sound_field_analysis import utils
import global_variables as gb
from sklearn.preprocessing import MinMaxScaler

def plot_model(data,previsions,points_sampled,pinn=False):

    # prev_real = np.real(previsions)
    # prev_imag = np.imag(previsions)
    # original_real = gb.scaler_r_s.inverse_transform(prev_real)
    # original_imag = gb.scaler_i_s.inverse_transform(prev_imag)
    # previsions  = original_real + 1j * original_imag
    input_sampled = data.X_sampled
    x = input_sampled[:,0].cpu().detach().numpy()
    y =input_sampled[:,1].cpu().detach().numpy()
    z = input_sampled[:,2].cpu().detach().numpy()

    azimuth_sampled,colatitude_sampled,_ = utils.cart2sph((x,y,z))
    microphone_positions = np.column_stack((azimuth_sampled, colatitude_sampled))

    # Crea un grafico 2D della pressione in funzione di azimuth e colatitude
    pressure_difference = np.abs(data.NORMALIZED_OUTPUT)-np.abs(previsions)
    print(pressure_difference.shape)

    fig, (ax,ax1,ax2) = plt.subplots(1, 3, figsize=(12, 5))
    sc = ax.scatter(data.azimuth, data.colatitude, c=np.abs(previsions), cmap='viridis')
    sc1 = ax1.scatter(data.azimuth, data.colatitude, c=np.abs(data.NORMALIZED_OUTPUT), cmap='viridis')
    
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(1, 0, 0),(1, 1, 1), (1, 0, 1)], N=256)
    sc2 = ax2.scatter(data.azimuth, data.colatitude, c=pressure_difference, cmap=cmap,vmax=1,vmin=-1)

    ax.scatter(microphone_positions[:, 0], microphone_positions[:, 1], color='red', marker='o', label='Microphones')

    # Imposta manualmente i segnaposti e le etichette sugli assi x e y
    azimuth_ticks = [0, np.pi / 2, np.pi, 2 * np.pi]
    colatitude_ticks = [0, np.pi / 2, np.pi]

    ax.set_xticks(azimuth_ticks)
    ax.set_yticks(colatitude_ticks)
    ax1.set_xticks(azimuth_ticks)
    ax1.set_yticks(colatitude_ticks)
    ax2.set_xticks(azimuth_ticks)
    ax2.set_yticks(colatitude_ticks)
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

    ax2.set_xticklabels(['0', 'π/2', 'π', '2π'])
    ax2.set_yticklabels(['0', 'π/2', 'π'])
    ax2.set_xlabel('Azimuth (radians)')
    ax2.set_ylabel('Colatitude (radians)')
    ax2.set_title('Pressure Difference')
    ax2.grid(True)

    plt.colorbar(sc1,label='Pressure')
    plt.colorbar(sc, label='Pressure')
    
    cbar2 = plt.colorbar(sc2, label='Pressure Difference', ax=ax2)
    cbar2.set_ticks([-1,-0.5,0,0.5, 1])
    cbar2.set_ticklabels(['-1','-o.5','0', '0.5', '1'])

    frequency_label = f"Frequenza: {gb.frequency} Hz"
    ax.text(1, 0, frequency_label, transform=ax.transAxes, ha='right', va='bottom', color='black', fontsize=12)
    ax1.text(1, 0, frequency_label, transform=ax1.transAxes, ha='right', va='bottom', color='black', fontsize=12)


    if pinn:
        plt.savefig(f"../src/image/Pinn_{points_sampled}_{gb.frequency}.png")
    else:
        plt.savefig(f"../src/image/NoPinn_{points_sampled}_{gb.frequency}.png")

    plt.show()


def inverse_normalize(part, min,max):
    return gb.scaler.inverse_transform(part.reshape(-1, 1)).flatten()



