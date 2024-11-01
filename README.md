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
``` python
>>> import emout
>>> data = emout.Emout('output_dir')

>>> data.phisp  # data of "phisp00_0000.h5"
>>> len(data.phisp)
11
>>> data.phisp[0].shape
(513, 65, 65)
>>> data.j1x  # data from "j1x00_0000.h5"
>>> data.bz  # data from "bz00_0000.h5"
>>> data.j1xy  # vector data object from "j1x00_0000.h5" and "j1y00_0000.h5"

>>> data.rex  # data from "rex00_0000.h5", created by relocating 'ex00_0000.h5'

>>> data.icur  # data from "icur" as pandas.DataFrame
>>> data.pbody  # data from "pbody" as pandas.DataFrame
```

### パラメータファイル(plasma.inp)を取得する
```python
>>> data.inp  # namelist of 'plasma.inp'
>>> data.inp['tmgrid']['nx']  # inp[group_name][parameter_name]
64
>>> data.inp['nx']  # can omit group name
64
>>> data.inp.tmgrid.nx  # can access like attribute
>>> data.inp.nx  # can also omit group name
```

### データをプロットする
```python
>>> x, y, z = 32, 32, 100
>>> data.phisp[-1, z, :, :].plot()  # plot xy-plane at z=100
>>> data.phisp[-1, :, y, x].plot()  # plot center line along z-axis

>>> data.phisp[-1, z, :, :].plot(use_si=True)  # can plot with SI-unit (such as x[m], y[m], phisp[V])
>>> data.phisp[-1, z, :, :].plot()  # use_si=True by default

>>> data.phisp[-1, z, :, :].plot(show=True)  # to view the plot on the fly (same as matplotlib.pyplot.show())
>>> data.phisp[-1, z, :, :].plot(savefilename='phisp.png')  # to save to the file

>>> data.j1xy[-1, z, :, :].plot() # can plot vector data as a streamline
```

### データのアニメーションを作成する
```python
>>> x, y, z = 32, 32, 100
>>> data.phisp[:, z, :, :].gifplot() # can generate time-series animation

>>> data.phisp[:, z, :, :].gifplot(axis=0) # Selectable data axes to animate
# (if axis=0, the first axis, i.e. the time axis, is selected, by default axis=0)

>>> data.phisp[:, z, :, :].gifplot(action='save', filename='phisp.gif') # for save on a file

>>> data.phisp[:, z, :, :].gifplot(action='to_html') # for display on jupyter

# If you want to annimation several data at once,
# prepare multiple frame update objects for each data at first.
>>> updater0 = data.phisp[:, z, :, :].gifplot(action='frames', mode='cmap')
>>> updater1 = data.phisp[:, z, :, :].build_frame_updater(mode='cont') # == gifplot(action='frames', mode='cont')
>>> updater2 = data.nd1p[:, z, :, :].build_frame_updater(mode='cmap', vmin=1e-3, vmax=20, norm='log')
>>> updater3 = data.nd2p[:, z, :, :].build_frame_updater(mode='cmap', vmin=1e-3, vmax=20, norm='log')
>>> updater4 = data.j2xy[:, z, :, :].build_frame_updater(mode='stream')
>>> layout = [[[updater0, updater1], [updater2], [updater3, updater4]]]
>>> animator = updater0.to_animator(layout=layout) # create animator object from frame object (phisp: cmap+cont, nd1p: cmap, nd2p: cmap+current-stream)
>>> animator.plot(action='to_html') # write plot function like gifplot 
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

``` python
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

``` python
>>> # SI単位系に変換した値を取得する
>>> phisp_volt = data.phisp[-1, :, :, :].val_si
>>> j1z_A_per_m2 = data.j1z[-1, :, :, :].val_si
>>> nd1p_per_cc = data.nd1p[-1, :, :, :].val_si
```

### 継続したシミュレーション結果を扱う
``` python
>>> import emout
>>> data = emout.Emout('output_dir', append_directories=['output_dir2', 'output_dir3'])
```

### データマスクを適用する
``` python
>>> # mask below average values
>>> data.phisp[1].masked(lambda phi: phi < phi.mean())
>>>
>>> # above code does the same as this code
>>> phi = data.phisp[1].copy()
>>> phi[phi < phi.mean()] = np.nan
```

### 3次元電荷分布から3次元電位分布を計算する. (Poisson's equation solver)
``` python
>>> from emout import poisson
>>> import scipy.constants as cn
>>> data = emout.Emout()

>>> dx = data.inp.dx # Grid width [m]
>>> btypes = ["pdn"[i] for i in data.inp.mtd_vbnd] # boundary conditions
>>> rho = data.rho[-1].val_si # Charge distribution [C/m^3]

>>> phisp = poisson(data.rho[-1].val_si, dx=dx, btypes, epsilon_0=cn.epsilon_0)

>>> np.allclose(phisp, data.phisp[-1])
True # (maybe True because there may be slight numerical errors...)
```