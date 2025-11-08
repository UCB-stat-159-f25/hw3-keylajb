import numpy as np
from pathlib import Path
from scipy.io import wavfile
from ligotools.utils import whiten, reqshift, write_wavfile

def test_whiten_basic():
    """Check that whiten() preserves array length and returns finite values."""
    x = np.random.randn(2048)
    interp_psd = lambda f: np.ones_like(f)   # flat PSD
    dt = 1 / 4096
    w = whiten(x, interp_psd, dt)
    assert len(w) == len(x)
    assert np.all(np.isfinite(w))

def test_reqshift_changes_phase():
    """Check that reqshift() shifts frequencies (roughly changes waveform)."""
    fs = 4096
    t = np.arange(0, 1, 1/fs)
    tone = np.sin(2*np.pi*100*t)
    shifted = reqshift(tone, 200, fs)
    assert shifted.shape == tone.shape
    assert not np.allclose(shifted, tone)

def test_write_wavfile_creates_file(tmp_path):
    """Check that write_wavfile() writes a valid WAV file."""
    data = np.random.randn(4096)
    fs = 4096
    outpath = tmp_path / "test.wav"
    write_wavfile(outpath, fs, data)
    # confirm file exists and header is readable
    sr, arr = wavfile.read(outpath)
    assert sr == fs
    assert arr.ndim == 1
    assert arr.dtype == np.int16
