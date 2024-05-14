#%%
import numpy as np
from sound_field_analysis import io, gen, process, plot, sph, utils   

def compute_binaural_signal(Pnm,rf_nfft,sh_max_order,configuration,is_apply_rfi):
    rf_amp_max_db = 20
    HRIR = io.read_SOFA_file("../dataset/hrir/HRIR_L2702.sofa")
    FS = int(HRIR.l.fs)
    NFFT = HRIR.l.signal.shape[-1]
    # %%


    ## COMPUTE SH FOR HRTF

    Hnm = np.stack(
        [
            process.spatFT(
                process.FFT(HRIR.l.signal, fs=FS, NFFT=NFFT, calculate_freqs=False),
                position_grid=HRIR.grid,
                order_max=sh_max_order,
                kind="complex",
            ),
            process.spatFT(
                process.FFT(HRIR.r.signal, fs=FS, NFFT=NFFT, calculate_freqs=False),
                position_grid=HRIR.grid,
                order_max=sh_max_order,
                kind="complex",
            ),
        ]
    )

    # compute radial filters
    dn = gen.radial_filter_fullspec(
        max_order=sh_max_order,
        NFFT=rf_nfft,
        fs=FS,
        array_configuration=configuration,
        amp_maxdB=rf_amp_max_db,
    )
    if is_apply_rfi:
        # improve radial filters (remove DC offset and make casual) [1]
        dn, _, dn_delay_samples = process.rfi(dn, kernelSize=rf_nfft-1)
    else:
        # make radial filters causal
        dn_delay_samples = rf_nfft / 2
        dn *= gen.delay_fd(target_length_fd=dn.shape[-1], delay_samples=dn_delay_samples)

    # SH grades stacked by order
    m = sph.mnArrays(sh_max_order)[0]

    # reverse indices for stacked SH grades
    m_rev_id = sph.reverseMnIds(sh_max_order)

    # select azimuth head orientations to compute (according to SSR BRIR requirements)
    azims_SSR_rad = np.deg2rad(np.arange(0, 360) - 37)
    Pnm_dn_Hnm = np.float_power(-1.0, m)[:, np.newaxis] * Pnm[m_rev_id] * dn * Hnm

    # loop over all head orientations that are to be computed
    # this could be done with one inner product but the loop helps clarity
    S = np.zeros([len(azims_SSR_rad), Hnm.shape[0], Hnm.shape[-1]], dtype=Hnm.dtype)
    
    for azim_id, alpha in enumerate(azims_SSR_rad):
        alpha_exp = np.exp(-1j * m * alpha)[:, np.newaxis]
        # these are the spectra of the ear signals
        S[azim_id] = np.sum(Pnm_dn_Hnm * alpha_exp, axis=1)

    # IFFT to yield ear impulse responses
    BRIR = process.iFFT(S)
    
    # normalize BRIRs
    BRIR *= 0.9 / np.max(np.abs(BRIR))

    return BRIR