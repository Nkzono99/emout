import pytest
import h5py
import numpy as np


def create_h5file(filename, name, timesteps, shape):
    h5 = h5py.File(filename, 'w')
    group = h5.create_group(name)
    for i in range(timesteps):
        group.create_dataset('{:04}'.format(
            i), data=np.zeros(shape), dtype='f')
    h5.close()


@pytest.fixture
def emdir(tmpdir):
    print(tmpdir)
    phisp_path = tmpdir.join('phisp00_0000.h5')
    ex_path = tmpdir.join('ex00_0000.h5')

    create_h5file(phisp_path, 'phisp', 5, (100, 30, 30))
    create_h5file(ex_path, 'ex', 5, (100, 30, 30))
    return tmpdir
