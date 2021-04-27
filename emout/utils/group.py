import math


class Group:
    """複数のオブジェクトをまとめて処理するクラス.

    Attributes
    ----------
    objs : list[Any]
        オブジェクトのリスト

    Notes
    -----
    以下の関数以外の組み込み関数は管理するオブジェクトに対して実行する.
    - repr : return 'Group([objs])
    - str : return 'Group([objs])'
    - format : return 'Group([objs])'
    - len : return len(objs)
    - iter : return iter(objs)
    - in : return other in objs 

    Examples
    --------
    以下のように複数のオブジェクトをまとめて処理することができる.

    >>> group = Group([[1, 2], [3, 4]])
    >>> print(group)
    Group([[1, 2], [3, 4]])
    >>> print(group[0])
    Group([1, 3])
    >>> group.append(5)
    >>> print(group)
    Group([[1, 2, 5], [3, 4, 5]])
    """

    def __init__(self, objs):
        """グループを生成する.

        Parameters
        ----------
        objs : list
            オブジェクトのリスト
        """
        self.__dict__ = dict()
        self.objs = objs

    def __binary_operator(self, callable, other):
        """グループに対して二項演算を行う.

        Parameters
        ----------
        callable : Callable[[Any, Any], Any]
            二項演算
        other : Any or Group
            対象(Groupの場合は管理するオブジェクトそれぞれに対して演算を適用する)

        Returns
        -------
        Group
            二項演算結果のグループ
        """
        if isinstance(other, Group):
            others = other
            new_objs = [callable(obj, other)
                        for obj, other in zip(self.objs, others)]
        else:
            new_objs = [callable(obj, other) for obj in self.objs]
        return type(self)(new_objs)

    def __check_and_return_iterable(self, arg):
        """対象がGroupオブジェクトか判定し、対象をself.objs長にブロードキャストする.

        対象がGroupの場合:
            サイズが一致するかどうか検証し、そのままの対象を返す.
        対象がGroupでない場合:
            対象を複製し、self.objs長のリストにして返す.

        Parameters
        ----------
        arg : Any or Group
            チェックする対象

        Returns
        -------
        iterable (Group or list)
            self.objs長のイテラブルオブジェクト

        Raises
        ------
        Exception
            対象がGroupオブジェクトかつそのサイズが一致しない場合の例外
        """
        if isinstance(arg, Group):
            if len(self) != len(arg):
                raise Exception(
                    'Error: group size is not match (self:{}, arg:{})'.format(len(self), len(arg)))
            args = arg
        else:
            args = [arg] * len(self.objs)
        return args

    def map(self, callable):
        """すべてのオブジェクトに対して関数を適用し、新しいグループを生成する.

        Parameters
        ----------
        callable : Callable[Any, Any]
            適用する関数

        Returns
        -------
        Group
            新しいグループ

        Examples
        --------
        >>> group = Group([1, 2, 3])
        >>> group.map(lambda n: n * 2)
        Group([2, 4, 6])
        """
        new_objs = list(map(callable, self.objs))
        return type(self)(new_objs)

    def filter(self, predicate):
        """オブジェクトのうち関数が真を返すもののみで新しいグループを生成する.

        Parameters
        ----------
        predicate : Callable[Any, bool]
            フィルタ関数

        Returns
        -------
        Group
            新しいグループ

        Examples
        --------
        >>> group = Group([1, 2, 3])
        >>> group.filter(lambda n: n <= 2)
        Group([1, 2])
        """
        new_objs = list(filter(predicate, self.objs))
        return type(self)(new_objs)

    def foreach(self, callable):
        """すべてのオブジェクトに対して関数を適用する.

        Parameters
        ----------
        callable : Callable[Any, ]
            適用する関数

        Examples
        --------
        >>> group = Group([1, 2, 3])
        >>> group.foreach(print)
        1
        2
        3
        """
        for obj in self.objs:
            callable(obj)

    def __str__(self):
        return 'Group({})'.format(self.objs)

    def __repr__(self):
        return str(self)

    def __format__(self, format_spec):
        return str(self)

    def __len__(self):
        return len(self.objs)

    def __iter__(self):
        return iter(self.objs)

    def __contains__(self, obj):
        return obj in self.objs

    def __pos__(self):
        return self.map(lambda obj: -obj)

    def __neg__(self):
        return self.map(lambda obj: -obj)

    def __invert__(self):
        return self.map(lambda obj: ~obj)

    def __add__(self, other):
        return self.__binary_operator(lambda obj, other: obj + other, other)

    def __radd_(self, other):
        return self.__binary_operator(lambda obj, other: other + obj, other)

    def __sub__(self, other):
        return self.__binary_operator(lambda obj, other: obj - other, other)

    def __rsub(self, other):
        return self.__binary_operator(lambda obj, other: other - obj, other)

    def __mul__(self, other):
        return self.__binary_operator(lambda obj, other: obj * other, other)

    def __rmul__(self, other):
        return self.__binary_operator(lambda obj, other: other * obj, other)

    def __truediv__(self, other):
        return self.__binary_operator(lambda obj, other: obj / other, other)

    def __rtruediv__(self, other):
        return self.__binary_operator(lambda obj, other: other / obj, other)

    def __floordiv__(self, other):
        return self.__binary_operator(lambda obj, other: obj // other, other)

    def __rfloordiv__(self, other):
        return self.__binary_operator(lambda obj, other: other // obj, other)

    def __mod__(self, other):
        return self.__binary_operator(lambda obj, other: obj % other, other)

    def __rmod__(self, other):
        return self.__binary_operator(lambda obj, other: other % obj, other)

    def __divmod__(self, other):
        return self.__binary_operator(lambda obj, other: divmod(obj, other), other)

    def __rdivmod__(self, other):
        return self.__binary_operator(lambda obj, other: divmod(other, obj), other)

    def __pow__(self, other):
        return self.__binary_operator(lambda obj, other: obj ** other, other)

    def __rpow__(self, other):
        return self.__binary_operator(lambda obj, other: other ** obj, other)

    def __lshift__(self, other):
        return self.__binary_operator(lambda obj, other: obj << other, other)

    def __rlshift__(self, other):
        return self.__binary_operator(lambda obj, other: other << obj, other)

    def __rshift__(self, other):
        return self.__binary_operator(lambda obj, other: obj >> other, other)

    def __rrshift__(self, other):
        return self.__binary_operator(lambda obj, other: other >> obj, other)

    def __and__(self, other):
        return self.__binary_operator(lambda obj, other: obj & other, other)

    def __rand__(self, other):
        return self.__binary_operator(lambda obj, other: other & obj, other)

    def __or__(self, other):
        return self.__binary_operator(lambda obj, other: obj | other, other)

    def __ror__(self, other):
        return self.__binary_operator(lambda obj, other: other | obj, other)

    def __xor__(self, other):
        return self.__binary_operator(lambda obj, other: obj ^ other, other)

    def __rxor__(self, other):
        return self.__binary_operator(lambda obj, other: other ^ obj, other)

    def __abs__(self):
        return self.map(abs)

    def __eq__(self, other):
        return self.__binary_operator(lambda obj, other: obj == other, other)

    def __ne__(self, other):
        return self.__binary_operator(lambda obj, other: obj != other, other)

    def __le__(self, other):
        return self.__binary_operator(lambda obj, other: obj <= other, other)

    def __ge__(self, other):
        return self.__binary_operator(lambda obj, other: obj >= other, other)

    def __lt__(self, other):
        return self.__binary_operator(lambda obj, other: obj < other, other)

    def __gt__(self, other):
        return self.__binary_operator(lambda obj, other: obj > other, other)

    def __int__(self):
        return self.map(int)

    def __float__(self):
        return self.map(float)

    def __complex__(self):
        return self.map(complex)

    def __bool__(self):
        return self.map(bool)

    def __bytes__(self):
        return self.map(bytes)

    def __hash__(self):
        return self.map(hash)

    def __slice_to_iterable(self, slc):
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
        if not isinstance(key, tuple):
            key = (key, )

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
        keys = self.__expand_key(key)

        new_objs = [obj[key] for obj, key in zip(self.objs, keys)]

        return type(self)(new_objs)

    def __setitem__(self, key, value):
        keys = self.__expand_key(key)
        values = self.__check_and_return_iterable(value)

        for obj, key, value in zip(self.objs, keys, values):
            obj[key] = value

    def __delitem__(self, key):
        keys = self.__expand_key(key)

        for obj, key in zip(self.objs, keys):
            del obj[key]

    def __getattr__(self, key):
        keys = self.__check_and_return_iterable(key)
        new_objs = [getattr(obj, key) for obj, key in zip(self.objs, keys)]
        return type(self)(new_objs)

    def __setattr__(self, key, value):
        if key in ('objs', '__dict__'):
            self.__dict__[key] = value
            return

        keys = self.__check_and_return_iterable(key)
        values = self.__check_and_return_iterable(value)

        for obj, key, value in zip(self.objs, keys, values):
            setattr(obj, key, value)

    def __call__(self, *args, **kwargs):
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

        new_objs = [obj(*new_args, **new_kwargs)
                    for obj, new_args, new_kwargs in zip(self.objs, new_argss, new_kwargss)]

        return type(self)(new_objs)

    def __round__(self):
        return self.map(math.round)

    def __trunc__(self):
        return self.map(math.trunc)

    def __floor__(self):
        return self.map(math.floor)

    def __ceil__(self):
        return self.map(math.ceil)
