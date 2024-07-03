import re

import matplotlib.pyplot as plt
import numpy as np

import emout.plot.basic_plot as emplt
import matplotlib.animation as animation
import emout.utils as utils

class VectorData(utils.Group):
    def __init__(self, objs, name=None, attrs=None):
        x_data, y_data = objs

        if attrs is None:
            attrs = dict()

        if name:
            attrs["name"] = name
        elif "name" in attrs:
            pass
        elif hasattr(x_data, "name"):
            attrs["name"] = name
        else:
            attrs["name"] = ""

        super().__init__([x_data, y_data], attrs=attrs)
        self.x_data = x_data
        self.y_data = y_data

    def __setattr__(self, key, value):
        if key in ("x_data", "y_data"):
            super().__dict__[key] = value
            return
        super().__setattr__(key, value)

    @property
    def name(self):
        return self.attrs["name"]
    
    def gifplot(
        self,
        fig=None,
        axis=0,
        show=False,
        savefilename=None,
        interval=200,
        repeat=True,
        title=None,
        notitle=False,
        offsets=None,
        use_si=True,
        to_html: bool = False,
        **kwargs,
    ):
        """gifアニメーションを作成する

        Parameters
        ----------
        fig : Figure
            アニメーションを描画するFigure(Noneの場合新しく作成する), by default None
        axis : int, optional
            アニメーションする軸, by default 0
        show : bool, optional
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default True
        savefilename : str, optional
            保存するファイル名(Noneの場合保存しない), by default None
        interval : int, optional
            フレーム間のインターバル(ミリ秒), by default 400
        repeat : bool
            アニメーションをループするならTrue, by default True
        title : str, optional
            タイトル(Noneの場合データ名(phisp等)), by default None
        notitle : bool, optional
            タイトルを付けない場合True, by default False
        offsets : (float or str, float or str, float or str)
            プロットのx,y,z軸のオフセット('left': 最初を0, 'center': 中心を0, 'right': 最後尾を0, float: 値だけずらす), by default None
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default False
        to_html : bool
            アニメーションをHTMLとして返す. (使用例: Jupyter Notebook等でアニメーションを描画する際等)
        """
        if self.objs[0].valunit is None:
            use_si = False

        def _offseted(line, offset):
            if offset == "left":
                line -= line[0]
            elif offset == "center":
                line -= line[len(line) // 2]
            elif offset == "right":
                line -= line[-1]
            else:
                line += offset
            return line

        def _update(i):
            plt.clf()

            # 指定した軸でスライス
            slices = [slice(None)] * len(self.objs[0].shape)
            slices[axis] = i
            val = self[tuple(slices)]

            # タイトルの設定
            if notitle:
                _title = title if len(title) > 0 else None
            else:
                ax = self.objs[0].slice_axes[axis]
                slc = self.objs[0].slices[ax]
                maxlen = self.objs[0].shape[axis]

                line = np.array(utils.range_with_slice(slc, maxlen=maxlen), dtype=float)

                if offsets is not None:
                    line = _offseted(line, offsets[0])

                index = line[i]

                if use_si:  # SI単位系を用いる場合
                    title_format = title + "({} {})"
                    axisunit = self.objs[0].axisunits[ax]
                    _title = title_format.format(axisunit.reverse(index), axisunit.unit)

                else:  # EMSES単位系を用いる場合
                    title_format = title + "({})"
                    _title = title_format.format(index)

            if offsets is not None:
                offsets2d = offsets[1:]
            else:
                offsets2d = None

            val.plot(
                title=_title,
                use_si=use_si,
                offsets=offsets2d,
                **kwargs,
            )

        if title is None:
            title = self.name

        if fig is None:
            fig = plt.figure()

        ani = animation.FuncAnimation(
            fig,
            _update,
            interval=interval,
            frames=self.objs[0].shape[axis],
            repeat=repeat,
        )

        if to_html:
            from IPython.display import HTML

            return HTML(ani.to_jshtml())
        elif savefilename is not None:
            ani.save(savefilename, writer="quantized-pillow")
        elif show:
            plt.show()
        else:
            return fig, ani

    def plot(
        self,
        *args,
        **kwargs,
    ):
        if self.x_data.ndim == 2:
            self.plot2d(
                *args,
                **kwargs,
            )

    def plot2d(
        self,
        mode="stream",
        axes="auto",
        show=False,
        use_si=True,
        offsets=None,
        **kwargs,
    ):
        """2次元データをプロットする.

        Parameters
        ----------
        mode : str
            プロットの種類('vec': quiver plot, 'stream': streamline plot), by default 'stream'
        axes : str, optional
            プロットする軸('xy', 'zx', etc), by default 'auto'
        show : bool
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default False
        offsets : (float or str, float or str, float or str)
            プロットのx,y,z軸のオフセット('left': 最初を0, 'center': 中心を0, 'right': 最後尾を0, float: 値だけずらす), by default None
        mesh : (numpy.ndarray, numpy.ndarray), optional
            メッシュ, by default None
        savefilename : str, optional
            保存するファイル名(Noneの場合保存しない), by default None
        cmap : matplotlib.Colormap or str or None, optional
            カラーマップ, by default cm.coolwarm
        vmin : float, optional
            最小値, by default None
        vmax : float, optional
            最大値, by default None
        figsize : (float, float), optional
            図のサイズ, by default None
        xlabel : str, optional
            x軸のラベル, by default None
        ylabel : str, optional
            y軸のラベル, by default None
        title : str, optional
            タイトル, by default None
        interpolation : str, optional
            用いる補間方法, by default 'bilinear'
        dpi : int, optional
            解像度(figsizeが指定された場合は無視される), by default 10

        Returns
        -------
        AxesImage or None
            プロットしたimageデータ(保存またはshowした場合None)

        Raises
        ------
        Exception
            プロットする軸のパラメータが間違っている場合の例外
        Exception
            プロットする軸がデータにない場合の例外
        Exception
            データの次元が2でない場合の例外
        """
        if self.valunit is None:
            use_si = False

        if axes == "auto":
            axes = "".join(sorted(self.objs[0].use_axes))

        if not re.match(r"x[yzt]|y[xzt]|z[xyt]|t[xyz]", axes):
            raise Exception(
                'Error: axes "{axes}" cannot be used with Data2d'.format(axes=axes)
            )
        if axes[0] not in self.objs[0].use_axes or axes[1] not in self.objs[0].use_axes:
            raise Exception(
                'Error: axes "{axes}" cannot be used because {axes}-axis does not exist in this data.'.format(
                    axes=axes
                )
            )
        if len(self.objs[0].shape) != 2:
            raise Exception(
                'Error: axes "{axes}" cannot be used because data is not 2dim shape.'.format(
                    axes=axes
                )
            )

        # x: 3, y: 2, z:1 t:0
        axis1 = self.objs[0].slice_axes[self.objs[0].use_axes.index(axes[0])]
        axis2 = self.objs[0].slice_axes[self.objs[0].use_axes.index(axes[1])]

        x = np.arange(*utils.slice2tuple(self.objs[0].slices[axis1]), dtype=float)
        y = np.arange(*utils.slice2tuple(self.objs[0].slices[axis2]), dtype=float)

        if use_si:
            xunit = self.objs[0].axisunits[axis1]
            yunit = self.objs[0].axisunits[axis2]
            valunit = self.objs[0].valunit

            x = xunit.reverse(x)
            y = yunit.reverse(y)

            _xlabel = "{} [{}]".format(axes[0], xunit.unit)
            _ylabel = "{} [{}]".format(axes[1], yunit.unit)
            _title = "{} [{}]".format(self.name, valunit.unit)

            x_data = self.x_data.val_si
            y_data = self.y_data.val_si
        else:
            _xlabel = axes[0]
            _ylabel = axes[1]
            _title = self.name

            x_data = self.x_data
            y_data = self.y_data

        def _offseted(line, offset):
            line = line.astype(float)
            if offset == "left":
                line -= line[0]
            elif offset == "center":
                line -= line[len(line) // 2]
            elif offset == "right":
                line -= line[-1]
            else:
                line += offset
            return line

        if offsets is not None:
            x = _offseted(x, offsets[0])
            y = _offseted(y, offsets[1])

        kwargs["xlabel"] = kwargs.get("xlabel", None) or _xlabel
        kwargs["ylabel"] = kwargs.get("ylabel", None) or _ylabel
        kwargs["title"] = kwargs.get("title", None) or _title

        mesh = np.meshgrid(x, y)
        if "vec" in mode:
            img = emplt.plot_2d_vector(x_data, y_data, mesh=mesh, **kwargs)
        elif "stream" in mode:
            img = emplt.plot_2d_streamline(x_data, y_data, mesh=mesh, **kwargs)

        if show:
            plt.show()
            return None
        else:
            return img

VectorData2d = VectorData
