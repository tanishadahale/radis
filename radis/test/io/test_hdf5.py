# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 20:36:38 2021

@author: erwan
"""

import pytest

from radis.api.hdf5 import DataFileManager, hdf2df
from radis.io.hitran import fetch_hitran
from radis.misc.config import getDatabankEntries
from radis.misc.utils import NotInstalled, not_installed_vaex_args

try:
    import vaex
except ImportError:
    vaex = NotInstalled(*not_installed_vaex_args)


@pytest.mark.fast
@pytest.mark.skipif(isinstance(vaex, NotInstalled), reason="Vaex not available")
def test_hdf5_io_engines(*args, **kwargs):
    """Test different engines implemented in :py:class:`radis.api.hdf5.DataFileManager`"""

    import os
    from os.path import exists

    for file in ["test_pytables.h5", "test_h5py.h5"]:
        if exists(file):
            os.remove(file)

    # Test data
    import h5py
    import numpy as np
    import pandas as pd

    df0 = pd.DataFrame({"a": np.arange(10) ** 2, "b": np.arange(10) ** 3})
    metadata0 = {"some_metadata": True}
    # ... a Pandas HDFStore file
    df0.to_hdf("test_pytables.h5", "df")
    # ... a h5py file with same content
    with h5py.File("test_h5py.h5", "w") as f:
        # group = f.create_group("df")
        for c in df0.columns:
            f.create_dataset(c, data=df0[c])

    # Test Pytables engine : add_metadata ; load ; read_metadata
    manager = DataFileManager(engine="pytables")
    manager.add_metadata("test_pytables.h5", metadata0)

    df = manager.read("test_pytables.h5")
    assert (df == df0).all().all()
    assert manager.read_metadata("test_pytables.h5") == metadata0

    # Test h5py engine : add_metadata ; load ; read_metadata
    manager = DataFileManager(engine="h5py")
    manager.add_metadata("test_h5py.h5", metadata0, key=None)
    df = manager.read("test_h5py.h5", key=None)
    assert (df == df0).all().all()
    assert manager.read_metadata("test_h5py.h5") == metadata0

    # Test vaex engine : add_metadata ; load ; read_
    # .. also able to read h5py files
    manager = DataFileManager(engine="vaex")
    manager.add_metadata("test_h5py.h5", metadata0, key=None)
    df = manager.read("test_h5py.h5", key=None)
    # this time; df is a vaex DataFrame
    # ... it keeps the file open so we should close the file handle
    assert (df.to_pandas_df() == df0).all().all()
    df.close()
    assert manager.read_metadata("test_h5py.h5", key=None) == metadata0

    # Test get_columns function of DataFileManager
    # For vaex
    manager = DataFileManager(engine="vaex")
    assert list(manager.get_columns("test_h5py.h5")) == ["a", "b"]

    # For pytables
    manager = DataFileManager(engine="pytables")
    assert list(manager.get_columns("test_pytables.h5")) == ["a", "b"]


@pytest.mark.needs_connection
def test_local_hdf5_lines_loading(*args, **kwargs):
    """
    We use the OH HITRAN line database to test :py:func:`~radis.io.hitemp.fetch_hitran`
    and :py:func:`~radis.api.hdf5.hdf2df`

    - Partial loading (only specific wavenumbers)
    - Only certain isotopes
    - Only certain columns

    """

    fetch_hitran("OH")  # to initialize the database

    path = getDatabankEntries("HITRAN-OH")["path"]

    # Initialize the database
    fetch_hitran("OH")
    path = getDatabankEntries("HITRAN-OH")["path"][0]
    df = hdf2df(path)
    wmin, wmax = df.wav.min(), df.wav.max()
    assert wmin < 2300  # needed for next test to be valid
    assert wmax > 2500  # needed for next test to be valid
    assert len(df.columns) > 5  # many columns loaded by default
    assert len(df.iso.unique()) > 1

    # Test loading only certain columns
    df = hdf2df(path, columns=["wav", "int"])
    assert len(df.columns) == 2 and "wav" in df.columns and "int" in df.columns

    # Test loading only certain isotopes
    df = hdf2df(path, isotope="2")
    assert df.iso.unique() == 2

    # Test partial loading of wavenumbers
    df = hdf2df(path, load_wavenum_min=2300, load_wavenum_max=2500)
    assert df.wav.min() >= 2300
    assert df.wav.max() <= 2500

    # Test with only one
    assert hdf2df(path, load_wavenum_min=2300).wav.min() >= 2300

    # Test with the other
    assert hdf2df(path, load_wavenum_max=2500).wav.max() <= 2500


@pytest.mark.fast
def test_feather_io_engine(*args, **kwargs):
    """Test Feather engine implemented in :py:class:`radis.api.hdf5.DataFileManager`

    Tests write, read, add_metadata, read_metadata, and multithreaded read
    for the Feather format.
    """

    import os
    from os.path import exists

    test_file = "test_feather_cache.feather"
    if exists(test_file):
        os.remove(test_file)

    import numpy as np
    import pandas as pd

    df0 = pd.DataFrame({"a": np.arange(10) ** 2, "b": np.arange(10) ** 3})
    metadata0 = {"some_metadata": "True", "version": "0.9.35", "wavenum_min": 2300.5}

    # Test feather engine: write
    manager = DataFileManager(engine="feather")
    manager.write(test_file, df0)

    # Test feather engine: read (with multithreading)
    df = manager.read(test_file)
    assert (df == df0).all().all()

    # Test feather engine: add_metadata + read_metadata
    manager.add_metadata(test_file, metadata0)
    metadata_read = manager.read_metadata(test_file)

    # Check metadata round-trip (values are stored as strings then parsed back)
    assert metadata_read["some_metadata"] == "True"
    assert metadata_read["version"] == "0.9.35"
    assert metadata_read["wavenum_min"] == 2300.5

    # Test that data is still intact after metadata was added
    df = manager.read(test_file)
    assert (df == df0).all().all()

    # Test guess_engine
    engine = DataFileManager.guess_engine(test_file, verbose=False)
    assert engine == "feather"

    # Test cache_file returns .feather suffix
    import pathlib

    assert manager.cache_file("test.par") == pathlib.Path("test.feather")

    # Cleanup
    if exists(test_file):
        os.remove(test_file)


@pytest.mark.fast
def test_feather_cache_save_load(*args, **kwargs):
    """Test save_to_hdf and load_h5_cache_file with Feather engine."""

    import os
    from os.path import exists

    import numpy as np
    import pandas as pd

    from radis.api.cache_files import load_h5_cache_file, save_to_hdf

    test_file = "test_cache.feather"
    if exists(test_file):
        os.remove(test_file)

    df0 = pd.DataFrame(
        {"wav": np.linspace(2300, 2500, 100), "int": np.random.rand(100)}
    )
    metadata0 = {"wavenum_min": 2300.0, "wavenum_max": 2500.0}

    # Save with feather engine
    save_to_hdf(df0, test_file, metadata=metadata0, engine="feather", verbose=False)
    assert exists(test_file)

    # Load back (use current radis version to avoid future-version error)
    import radis

    df = load_h5_cache_file(
        test_file,
        use_cached=True,
        valid_if_metadata_is=metadata0,
        current_version=radis.__version__,
        last_compatible_version=radis.config["OLDEST_COMPATIBLE_VERSION"],
        engine="feather",
        verbose=False,
    )
    assert df is not None
    assert len(df) == 100
    assert (df["wav"].values == df0["wav"].values).all()

    # Cleanup
    if exists(test_file):
        os.remove(test_file)


@pytest.mark.fast
@pytest.mark.skipif(isinstance(vaex, NotInstalled), reason="Vaex not available")
def test_hdf5_to_feather_migration(*args, **kwargs):
    """Test update_hdf5_to_feather conversion function."""

    import os
    from os.path import exists

    import numpy as np
    import pandas as pd

    from radis.api.hdf5 import update_hdf5_to_feather

    test_h5 = "test_migration.h5"
    test_feather = "test_migration.feather"
    for f in [test_h5, test_feather]:
        if exists(f):
            os.remove(f)

    df0 = pd.DataFrame({"a": np.arange(10) ** 2, "b": np.arange(10) ** 3})
    metadata0 = {"some_metadata": "True", "version": "0.9.35"}

    # Create an HDF5 file with metadata
    manager = DataFileManager(engine="pytables")
    manager.write(test_h5, df0)
    manager.add_metadata(test_h5, metadata0)

    # Convert to feather
    fname_feather = update_hdf5_to_feather(test_h5, verbose=False)
    assert fname_feather == test_feather
    assert exists(test_feather)

    # Verify data and metadata preserved
    feather_manager = DataFileManager(engine="feather")
    df = feather_manager.read(test_feather)
    assert (df == df0).all().all()

    metadata_read = feather_manager.read_metadata(test_feather)
    assert metadata_read["some_metadata"] == "True"
    assert metadata_read["version"] == "0.9.35"

    # Cleanup
    for f in [test_h5, test_feather]:
        if exists(f):
            os.remove(f)


if __name__ == "__main__":
    test_hdf5_io_engines()
    test_local_hdf5_lines_loading()
    test_feather_io_engine()
    test_feather_cache_save_load()
    test_hdf5_to_feather_migration()
