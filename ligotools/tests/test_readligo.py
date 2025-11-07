# ligotools/tests/test_readligo.py
import numpy as np
from pathlib import Path
from ligotools import readligo as rl

DATA = Path("data")

def test_loaddata_returns_arrays():
    """Check that loaddata returns numpy arrays of equal length."""
    fn = DATA / "H-H1_LOSC_4_V2-1126259446-32.hdf5"
    strain, time, chan_dict = rl.loaddata(fn)
    assert isinstance(strain, np.ndarray)
    assert isinstance(time, np.ndarray)
    assert strain.size == time.size
    assert "DATA" in chan_dict 

def test_time_spacing_reasonable():
    """Check that time spacing is constant and reasonable."""
    fn = DATA / "L-L1_LOSC_4_V2-1126259446-32.hdf5"
    strain, time, _ = rl.loaddata(fn)
    dt = np.diff(time)
    assert np.allclose(dt, dt[0], atol=1e-6)
    fs = 1.0 / dt[0]
    assert 1000 < fs < 10000
