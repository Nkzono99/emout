"""Generic grouping container with element-wise attribute delegation."""

import math


class Group:
    """Group クラス。
    """

    def __init__(self, objs, attrs=None):
        """グループを生成する.

        Parameters
        ----------
        objs : list
            オブジェクトのリスト
        """
        self.__dict__ = dict()
        self.objs = objs
        self.attrs = attrs

    def __binary_operator(self, callable, other):
        """各要素に二項演算を適用して新しい Group を返す。
        
        Parameters
        ----------
        callable : object
            各要素に適用する関数です。
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        if isinstance(other, Group):
            others = other
            new_objs = [callable(obj, other) for obj, other in zip(self.objs, others)]
        else:
            new_objs = [callable(obj, other) for obj in self.objs]
        return type(self)(new_objs, attrs=self.attrs)

    def __check_and_return_iterable(self, arg):
        """入力を Group 長に合わせて展開する。
        
        Parameters
        ----------
        arg : object
            `Group` もしくは単一値です。`Group` を渡した場合は
            要素数が `self` と一致することを確認し、単一値を渡した場合は
            全要素に同じ値を適用できる形へ展開します。
        Returns
        -------
        object
            `self` と同じ長さの反復可能オブジェクトです。
        """
        if isinstance(arg, Group):
            if len(self) != len(arg):
                raise ValueError(
                    f"group size mismatch: self has {len(self)} elements, arg has {len(arg)}"
                )
            args = arg
        else:
            args = [arg] * len(self.objs)
        return args

    def map(self, callable):
        """各要素へ関数を適用した Group を返す。
        
        Parameters
        ----------
        callable : object
            各要素に適用する関数です。
        Returns
        -------
        object
            処理結果です。
        """
        new_objs = list(map(callable, self.objs))
        return type(self)(new_objs, attrs=self.attrs)

    def filter(self, predicate):
        """条件を満たす要素のみを含む Group を返す。
        
        Parameters
        ----------
        predicate : object
            要素を採用するか判定する関数です。
        Returns
        -------
        object
            処理結果です。
        """
        new_objs = list(filter(predicate, self.objs))
        return type(self)(new_objs, attrs=self.attrs)

    def foreach(self, callable):
        """各要素へ関数を順に適用する。
        
        Parameters
        ----------
        callable : object
            各要素に適用する関数です。
        Returns
        -------
        None
            戻り値はありません。
        """
        for obj in self.objs:
            callable(obj)

    def __str__(self):
        """文字列表現を返す。
        
        Returns
        -------
        str
            文字列表現です。
        """
        return "Group({})".format(self.objs)

    def __repr__(self):
        """文字列表現を返す。
        
        Returns
        -------
        str
            文字列表現。
        """
        return str(self)

    def __format__(self, format_spec):
        """フォーマット済み文字列を返す。
        
        Parameters
        ----------
        format_spec : object
            Python のフォーマット指定子です。
            現状の実装では値にかかわらず `str(self)` を返します。
        Returns
        -------
        object
            `self` の文字列表現です。
        """
        return str(self)

    def __len__(self):
        """要素数を返す。
        
        Returns
        -------
        int
            要素数。
        """
        return len(self.objs)

    def __iter__(self):
        """イテレータを返す。
        
        Returns
        -------
        Iterator
            イテレータ。
        """
        return iter(self.objs)

    def __contains__(self, obj):
        """要素の包含判定を行う。
        
        Parameters
        ----------
        obj : object
            対象オブジェクトです。
        Returns
        -------
        object
            処理結果です。
        """
        return obj in self.objs

    def __pos__(self):
        """単項プラス演算を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(lambda obj: -obj)

    def __neg__(self):
        """単項マイナス演算を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(lambda obj: -obj)

    def __invert__(self):
        """ビット反転演算を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(lambda obj: ~obj)

    def __add__(self, other):
        """加算演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj + other, other)

    def __radd_(self, other):
        """右辺加算演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other + obj, other)

    def __sub__(self, other):
        """減算演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj - other, other)

    def __rsub(self, other):
        """右辺減算演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other - obj, other)

    def __mul__(self, other):
        """乗算演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj * other, other)

    def __rmul__(self, other):
        """右辺乗算演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other * obj, other)

    def __truediv__(self, other):
        """除算演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj / other, other)

    def __rtruediv__(self, other):
        """右辺除算演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other / obj, other)

    def __floordiv__(self, other):
        """切り捨て除算演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj // other, other)

    def __rfloordiv__(self, other):
        """右辺切り捨て除算演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other // obj, other)

    def __mod__(self, other):
        """剰余演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj % other, other)

    def __rmod__(self, other):
        """右辺剰余演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other % obj, other)

    def __divmod__(self, other):
        """divmod 演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: divmod(obj, other), other)

    def __rdivmod__(self, other):
        """右辺 divmod 演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: divmod(other, obj), other)

    def __pow__(self, other):
        """べき乗演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj**other, other)

    def __rpow__(self, other):
        """右辺べき乗演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other**obj, other)

    def __lshift__(self, other):
        """左シフト演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj << other, other)

    def __rlshift__(self, other):
        """右辺左シフト演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other << obj, other)

    def __rshift__(self, other):
        """右シフト演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj >> other, other)

    def __rrshift__(self, other):
        """右辺右シフト演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other >> obj, other)

    def __and__(self, other):
        """ビット積演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj & other, other)

    def __rand__(self, other):
        """右辺ビット積演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other & obj, other)

    def __or__(self, other):
        """ビット和演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj | other, other)

    def __ror__(self, other):
        """右辺ビット和演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other | obj, other)

    def __xor__(self, other):
        """排他的論理和演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj ^ other, other)

    def __rxor__(self, other):
        """右辺排他的論理和演算を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: other ^ obj, other)

    def __abs__(self):
        """絶対値演算を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(abs)

    def __eq__(self, other):
        """等価比較を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj == other, other)

    def __ne__(self, other):
        """非等価比較を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj != other, other)

    def __le__(self, other):
        """以下比較を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj <= other, other)

    def __ge__(self, other):
        """以上比較を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj >= other, other)

    def __lt__(self, other):
        """未満比較を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj < other, other)

    def __gt__(self, other):
        """超過比較を適用する。
        
        Parameters
        ----------
        other : object
            演算または比較の相手となる値です。
        Returns
        -------
        object
            処理結果です。
        """
        return self.__binary_operator(lambda obj, other: obj > other, other)

    def __int__(self):
        """整数変換を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(int)

    def __float__(self):
        """浮動小数変換を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(float)

    def __complex__(self):
        """複素数変換を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(complex)

    def __bool__(self):
        """真偽値変換を適用する。
        
        Returns
        -------
        bool
            条件判定結果です。
        """
        return self.map(bool)

    def __bytes__(self):
        """バイト列変換を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(bytes)

    def __hash__(self):
        """ハッシュ値を計算する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(hash)

    def __slice_to_iterable(self, slc):
        """Group を含む slice 指定を要素ごとの slice に展開する。
        
        Parameters
        ----------
        slc : object
            `slice` オブジェクトです。`start`/`stop`/`step` に `Group` や
            単一値を指定できます。
        Returns
        -------
        object
            各要素に対応する `slice` のリストです。
        """
        start = slc.start
        stop = slc.stop
        step = slc.step
        starts = self.__check_and_return_iterable(start)
        stops = self.__check_and_return_iterable(stop)
        steps = self.__check_and_return_iterable(step)
        slices = []
        for start, stop, step in zip(starts, stops, steps):
            slices.append(slice(start, stop, step))
        return slices

    def __expand_key(self, key):
        """インデックスキーを要素ごとのキー列に展開する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        Returns
        -------
        object
            処理結果です。
        """
        if not isinstance(key, tuple):
            key = (key,)

        slices = []
        for k in key:
            if isinstance(k, slice):
                slices.append(self.__slice_to_iterable(k))
            else:
                it = self.__check_and_return_iterable(k)
                slices.append(it)

        keys = []
        for key in zip(*slices):
            if len(key) == 1:
                keys.append(key[0])
            else:
                keys.append(tuple(key))

        return keys

    def __getitem__(self, key):
        """要素を取得する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        Returns
        -------
        object
            処理結果です。
        """
        keys = self.__expand_key(key)

        new_objs = [obj[key] for obj, key in zip(self.objs, keys)]

        return type(self)(new_objs, attrs=self.attrs)

    def __setitem__(self, key, value):
        """要素を設定する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        value : object
            値。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        keys = self.__expand_key(key)
        values = self.__check_and_return_iterable(value)

        for obj, key, value in zip(self.objs, keys, values):
            obj[key] = value

    def __delitem__(self, key):
        """要素を削除する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        Returns
        -------
        None
            戻り値はありません。
        """
        keys = self.__expand_key(key)

        for obj, key in zip(self.objs, keys):
            del obj[key]

    def __getattr__(self, key):
        """属性アクセスを解決する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        Returns
        -------
        object
            処理結果です。
        """
        keys = self.__check_and_return_iterable(key)
        new_objs = [getattr(obj, key) for obj, key in zip(self.objs, keys)]
        return type(self)(new_objs, attrs=self.attrs)

    def __setattr__(self, key, value):
        """属性を設定する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        value : object
            値。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        if key in ("objs", "__dict__", "attrs"):
            self.__dict__[key] = value
            return

        keys = self.__check_and_return_iterable(key)
        values = self.__check_and_return_iterable(value)

        for obj, key, value in zip(self.objs, keys, values):
            setattr(obj, key, value)

    def __call__(self, *args, **kwargs):
        """呼び出し可能オブジェクトとして実行する。
        
        Parameters
        ----------
        *args : tuple
            追加の位置引数。内部で呼び出す関数へ渡されます。
        **kwargs : dict
            追加のキーワード引数。内部で呼び出す関数へ渡されます。
        
        Returns
        -------
        object
            処理結果です。
        """
        new_argss = [list() for i in range(len(self.objs))]
        for arg in args:
            arg = self.__check_and_return_iterable(arg)
            for i, a in enumerate(arg):
                new_argss[i].append(a)

        new_kwargss = [dict() for i in range(len(self.objs))]
        for key, value in kwargs.items():
            keys = self.__check_and_return_iterable(key)
            values = self.__check_and_return_iterable(value)
            for i, (key, value) in enumerate(zip(keys, values)):
                new_kwargss[i][key] = value

        new_objs = [
            obj(*new_args, **new_kwargs)
            for obj, new_args, new_kwargs in zip(self.objs, new_argss, new_kwargss)
        ]

        return type(self)(new_objs, attrs=self.attrs)

    def __round__(self):
        """丸め演算を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(math.round)

    def __trunc__(self):
        """切り捨て演算を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(math.trunc)

    def __floor__(self):
        """床関数演算を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(math.floor)

    def __ceil__(self):
        """天井関数演算を適用する。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.map(math.ceil)
