# emout
EMSESの出力ファイルを取り扱うパッケージ

* Documentation: https://nkzono99.github.io/emout/

## Requirement
* numpy
* h5py
* matplotlib
* f90nml

## Installation
```
> pip install emout

or

> pip install git+https://github.com/Nkzono99/emout.git
```

## Usage
以下のようなフォルダ構成の場合のサンプルコード.
```
.
└── output_dir
    ├── plasma.inp
    ├── phisp00_0000.h5
    ├── nd1p00_0000.h5
    ├── nd2p00_0000.h5
    ├── j1x00_0000.h5
    ├── j1y00_0000.h5
    ...
    └── bz00_0000.h5
```

### データをロードする
```
>>> import emout
>>> data = emout.Emout('output_dir')
>>>
>>> data.phisp  # data of "phisp00_0000.h5"
>>> len(data.phisp)
11
>>> data.phisp[0].shape
(513, 65, 65)
>>> data.j1x  # data of "j1x00_0000.h5"
>>> data.bz  # data of "bz00_0000.h5"
```

### データをプロットする
```
>>> x, y, z = 32, 32, 100
>>> data.phisp[1][z, :, :].plot()  # plot xy-plane at z=100
>>> data.phisp[1][:, y, x].plot()  # plot center line along z-axis
```

### パラメータファイル(plasma.inp)を取得する
```
>>> data.inp  # namelist of 'plasma.inp'
>>> data.inp['tmgrid']['nx']  # inp[group_name][parameter_name]
64
>>> data.inp['nx']  # can omit group name
64
```

### 単位変換を行う
```
>>> data.unit.v.trans(1)  # velocity: Physical unit to EMSES unit
3.3356409519815205e-05
>>> data.unit.v.reverse(1)  # velocity: EMSES unit to Physical unit
29979.2458
```

### データマスクを適用する
```
>>> # mask below average values
>>> data.phisp[1].masked(lambda phi: phi < phi.mean())
>>>
>>> # above code does the same as this code
>>> phi = data.phisp[1].copy()
>>> phi[phi < phi.mean()] = np.nan
```

### グループ化する
```
>>> from emout.utils import Group
>>> data1 = emout.Emout('output_dir1')
>>> data2 = emout.Emout('output_dir2')
>>>
>>> # grouping
>>> group = Group([data1, data2])
>>> group.phisp[10][100, :, :].plot(show=True)
>>>
>>> # above code does the same as this code
>>> data1.phisp[10][100, :, :].plot(show=True)
>>> data2.phisp[10][100, :, :].plot(show=True)
```