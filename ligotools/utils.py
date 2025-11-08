"""
ligotools.utils
Utility functions for Stat 159 HW3 – LIGO tutorial
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt


# ---------- Whitening ----------
def whiten(strain, interp_psd, dt):
    """Whiten a strain time series using an interpolated PSD."""
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


# ---------- Frequency shift ----------
def reqshift(data, fshift, sample_rate):
    """Shift the frequency of data by fshift Hz."""
    t = np.arange(data.size) / sample_rate
    return data * np.cos(2 * np.pi * fshift * t)


# ---------- Write WAV ----------
def write_wavfile(path, fs, data):
    """Write a mono WAV file, scaling floats to int16 for playback."""
    arr = np.asarray(data)
    if arr.dtype.kind == "f":
        mx = np.max(np.abs(arr)) or 1.0
        arr = (arr / mx * 32767).astype(np.int16)
    wavfile.write(path, int(fs), arr)


# ---------- PSD plot helper ----------
def plot_psd_section(ax, strain, fs, NFFT=4*4096, NOVL=4*2048):
    """Plot a Power Spectral Density (PSD) on a given Axes object."""
    from scipy import signal
    freqs, pxx = signal.welch(strain, fs=fs, nperseg=NFFT, noverlap=NOVL)
    ax.semilogy(freqs, pxx)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.grid(True, which="both", ls=":")
    return freqs, pxx


# ---------- PSD and Template Plot Helper ----------
def plot_psd_and_template(det, freqs, data_psd, datafreq, template_fft, d_eff, fs, eventname, plottype, outdir="figures"):
    """
    Plot the amplitude spectral density (ASD) and overlay the template
    for a given detector (H1 or L1). Saves the figure in outdir.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    pcolor = "g" if det == "L1" else "r"

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    # --- prepare template data for plotting ---
    template_f = np.abs(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff

    # --- make plot ---
    plt.figure(figsize=(10, 6))
    plt.loglog(datafreq, template_f, "k", label="template(f)*√f")
    plt.loglog(freqs, np.sqrt(data_psd), pcolor, label=f"{det} ASD")
    plt.xlim(20, fs / 2)
    plt.ylim(1e-24, 1e-20)
    plt.grid(True, which="both", ls=":")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("strain noise ASD (strain/√Hz), template h(f)*√f")
    plt.legend(loc="upper left")
    plt.title(f"{det} ASD and template around event")

    outpath = outdir / f"{eventname}_{det}_matchfreq.{plottype}"
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved PSD+template plot: {outpath}")

