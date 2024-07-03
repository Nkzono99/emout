# emout
EMSESの出力ファイルを取り扱うパッケージ

* Documentation: https://nkzono99.github.io/emout/

## Installation
```
pip install emout
```

## Example code

-  [Visualization of simulation results for lunar surface charging](https://nbviewer.org/github/Nkzono99/examples/blob/main/examples/emout/example.ipynb)

## Usage
以下のようなフォルダ構成の場合の使い方.
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

>>> data.phisp[1][z, :, :].plot(use_si=True)  # can plot with SI-unit (such as x[m], y[m], phisp[V])

>>> data.phisp[1][z, :, :].plot(show=True)  # to view the plot on the fly (same as matplotlib.pyplot.show())
>>> data.phisp[1][z, :, :].plot(savefilename='phisp.png')  # to save to the file
```

### パラメータファイル(plasma.inp)を取得する
```
>>> data.inp  # namelist of 'plasma.inp'
>>> data.inp['tmgrid']['nx']  # inp[group_name][parameter_name]
64
>>> data.inp['nx']  # can omit group name
64
>>> data.inp.tmgrid.nx  # can access like attribute
>>> data.inp.nx  # can also omit group name
```

### 単位変換を行う
> [!NOTE]
> パラメータファイル (plasma.inp) の一行目に以下を記述している場合のみ、EMSES単位からSI単位系への変換がサポートされます。
> 
> ```
> !!key dx=[0.5],to_c=[10000.0]
> ```
> 
> ```dx```: グリッド幅 [m]
> ```to_c```: EMSES内部での光速の規格化された値

```
>>> data.unit.v.trans(1)  # velocity: Physical unit to EMSES unit
3.3356409519815205e-05
>>> data.unit.v.reverse(1)  # velocity: EMSES unit to Physical unit
29979.2458
```

### SI単位系への変換
> [!NOTE]
> パラメータファイル (plasma.inp) の一行目に以下を記述している場合のみ、EMSES単位からSI単位系への変換がサポートされます。
> 
> ```
> !!key dx=[0.5],to_c=[10000.0]
> ```
> 
> ```dx```: グリッド幅 [m]
> ```to_c```: EMSES内部での光速の規格化された値

```
>>> # SI単位系に変換した値を取得する
>>> phisp_volt = data.phisp[-1, :, :, :].val_si
>>> j1z_A_per_m2 = data.j1z[-1, :, :, :].val_si
>>> nd1p_per_cc = data.nd1p[-1, :, :, :].val_si
```

### 継続したシミュレーション結果を扱う
```
>>> import emout
>>> data = emout.Emout('output_dir', append_directories=['output_dir2', 'output_dir3'])
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
