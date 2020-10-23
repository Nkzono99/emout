.. emout documentation master file, created by
   sphinx-quickstart on Wed Oct 21 03:47:03 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to emout's documentation!
=================================

EMSESの出力ファイルを取り扱うパッケージ.

Requirement
-----------
   
   * numpy
   * h5py
   * matplotlib
   * f90nml

Installation
------------

   ::
     
     $ pip install emout

Usage
-----

以下のようなフォルダ構成の場合のサンプルコード

::

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

データのロード
^^^^^^^^^^^^^^

.. code-block:: python

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

データのプロット
^^^^^^^^^^^^^^^^

.. code-block:: python

   >>> x, y, z = 32, 32, 100
   >>> data.phisp[1][z, :, :].plot()  # plot xy-plane at z=100
   >>> data.phisp[1][:, y, x].plot()  # plot center line along z-axis


パラメータファイル(plasma.inp)の取得
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   >>> data.inp  # namelist of 'plasma.inp'
   >>> data.inp['tmgrid']['nx']  # inp[group_name][parameter_name]
   64
   >>> data.inp['nx']  # can omit group name
   64

単位変換(EMSES単位系 <=> SI単位系)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   >>> data.unit.v.trans(1)  # velocity: Physical unit to EMSES unit
   3.3356409519815205e-05
   >>> data.unit.v.reverse(1)  # velocity: EMSES unit to Physical unit
   29979.2458

データマスクの適用
^^^^^^^^^^^^^^^^^

.. code-block:: python

   >>> # mask below average values
   >>> data.phisp[1].masked(lambda phi: phi < phi.mean())
   >>>
   >>> # above code does the same as this code
   >>> phi = data.phisp[1].copy()
   >>> phi[phi < phi.mean()] = np.nan

グループ化
^^^^^^^^^^^

.. code-block:: python

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

.. toctree::
   :caption: Modules:


   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
