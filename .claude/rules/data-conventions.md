<important if="working with grid data arrays, axis ordering, file I/O, unit conversion, or f90nml namelist parsing">

# データ規約

## 軸順序

- グリッドデータ: `(t, z, y, x)`。3D ボリューム: `(z, y, x)`。
- `Data3d.axisunits`: `[-1]`=x, `[-2]`=y, `[-3]`=z の `UnitTranslator`。

## ファイル命名

- グリッド: `{name}00_0000.h5`
- 粒子: `p{species}{comp}(e?){seg}_{part}.h5`

## 単位変換

`plasma.inp` 1 行目の `!!key dx=[...],to_c=[...]` に依存。`unit is None` のケースを壊さない。

- grid→SI: `data.unit.length.reverse(x)`
- SI→grid: `data.unit.length.trans(x)`

## f90nml の疎配列

`_get_scalar` / `_get_vector` ヘルパを使う。直接触るなら:

- 1D: `start_index[name] == [start_for_dim1]`
- 2D: `start_index[name] == [None, start_for_dim2]`（dim1 完全記述、dim2 が疎）

</important>
