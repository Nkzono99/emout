import emout
import f90nml
import h5py
import numpy as np
import pytest

nml = '''!!key dx=[0.5],to_c=[10000.0]
&tmgrid
    dt = 0.0020000000000000005
    nx = 64
    ny = 64
    nz = 512
/
&mpi
    nodes(1:3) = 4, 4, 32
/
'''
time_steps = 5
data_shape = (100, 30, 30)


def create_h5file(filename, name, timesteps, shape):
    h5 = h5py.File(filename, 'w')
    group = h5.create_group(name)
    for i in range(timesteps):
        group.create_dataset('{:04}'.format(
            i), data=np.zeros(shape), dtype='f')
    h5.close()


def create_inpfile(filename):
    inp = f90nml.reads(nml)
    with open(filename, 'w', encoding='utf-8') as f:
        f90nml.write(inp, f, force=True)


@pytest.fixture
def emdir(tmpdir):
    phisp_path = tmpdir.join('phisp00_0000.h5')
    ex_path = tmpdir.join('ex00_0000.h5')
    inpfile_path = tmpdir.join('plasma.inp')

    create_h5file(phisp_path, 'phisp', time_steps, data_shape)
    create_h5file(ex_path, 'ex', time_steps, data_shape)
    create_inpfile(inpfile_path)
    return tmpdir


@pytest.fixture
def data(emdir):
    return emout.Emout(emdir)
