# emout
EMSESの出力ファイルを取り扱うパッケージ

## Requirement
* numpy
* h5py
* matplotlib
* f90nml

## Installation
```
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
```
>>> import emout
>>> data = emout.Emout('output_dir')
>>>
>>> # getting data
>>> data.phisp  # data of "phisp00_0000.h5"
>>> len(data.phisp)
11
>>> data.phisp[0].shape
(513, 65, 65)
>>> data.j1x  # data of "j1x00_0000.h5"
>>> data.bz  # data of "bz00_0000.h5"
>>>
>>> # getting inp file
>>> data.inp  # namelist of 'plasma.inp'
>>> data.inp['tmgrid']['nx']  # inp[group_name][parameter_name]
64
>>> data.inp['nx']  # can omit group name
64
```
