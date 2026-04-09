import emout

from emout.core.data import GridDataSeries
from tests.conftest import create_h5file, create_inpfile


def test_open_vector3d_data(tmpdir):
    """{name}xyz 形式で 3成分 VectorData を取得できることを確認する。"""
    create_h5file(tmpdir.join("jx00_0000.h5"), "jx", timesteps=3, shape=(8, 6, 4))
    create_h5file(tmpdir.join("jy00_0000.h5"), "jy", timesteps=3, shape=(8, 6, 4))
    create_h5file(tmpdir.join("jz00_0000.h5"), "jz", timesteps=3, shape=(8, 6, 4))
    create_inpfile(tmpdir.join("plasma.inp"))

    data = emout.Emout(tmpdir)
    vec = data.jxyz

    assert len(vec.objs) == 3
    assert vec.x_data.name == "jx"
    assert vec.y_data.name == "jy"
    assert vec.z_data.name == "jz"

    vec3d = vec[-1]
    assert vec3d.shape == (8, 6, 4)
    assert vec3d.x_data.shape == (8, 6, 4)
    assert vec3d.y_data.shape == (8, 6, 4)
    assert vec3d.z_data.shape == (8, 6, 4)


def test_xyz_suffix_can_still_load_scalar_data(tmpdir):
    """末尾 xyz のスカラー名でも GridDataSeries をロードできることを確認する。"""
    create_h5file(
        tmpdir.join("fooxyz00_0000.h5"),
        "fooxyz",
        timesteps=2,
        shape=(4, 3, 2),
    )
    create_inpfile(tmpdir.join("plasma.inp"))

    data = emout.Emout(tmpdir)
    scalar = data.fooxyz

    assert isinstance(scalar, GridDataSeries)
