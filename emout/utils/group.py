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
        return Group(new_objs)

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
        return Group(new_objs)

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

    def __add__(self, other):
        return self.map(lambda obj: obj + other)

    def __sub__(self, other):
        return self.map(lambda obj: obj - other)

    def __mul__(self, other):
        return self.map(lambda obj: obj * other)

    def __truediv__(self, other):
        return self.map(lambda obj: obj / other)

    def __floordiv__(self, other):
        return self.map(lambda obj: obj // other)

    def __mod__(self, other):
        return self.map(lambda obj: obj % other)

    def __divmod__(self, other):
        return self.map(lambda obj: divmod(obj, other))

    def __pow__(self, other):
        return self.map(lambda obj: obj ** other)

    def __lshift__(self, other):
        return self.map(lambda obj: obj << other)

    def __rshift__(self, other):
        return self.map(lambda obj: obj >> other)

    def __and__(self, other):
        return self.map(lambda obj: obj & other)

    def __or__(self, other):
        return self.map(lambda obj: obj | other)

    def __xor__(self, other):
        return self.map(lambda obj: obj ^ other)

    def __abs__(self):
        return self.map(abs)

    def __eq__(self, other):
        return self.map(lambda obj: obj == other)

    def __ne__(self, other):
        return self.map(lambda obj: obj != other)

    def __le__(self, other):
        return self.map(lambda obj: obj <= other)

    def __ge__(self, other):
        return self.map(lambda obj: obj >= other)

    def __lt__(self, other):
        return self.map(lambda obj: obj < other)

    def __gt__(self, other):
        return self.map(lambda obj: obj > other)

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

    def __getitem__(self, key):
        return self.map(lambda obj: obj[key])

    def __setitem__(self, key, value):
        for obj in self.objs:
            obj[key] = value

    def __delitem__(self, key):
        def _func(obj):
            del obj[key]
        self.foreach(_func)

    def __getattr__(self, key):
        return self.map(lambda obj: getattr(obj, key))

    def __setattr__(self, key, value):
        if key in ('objs', '__dict__'):
            self.__dict__[key] = value
            return
        for obj in self.objs:
            setattr(obj, key, value)

    def __call__(self, *args, **kwargs):
        return self.map(lambda obj: obj(*args, **kwargs))

    def __round__(self):
        return self.map(math.round)

    def __trunc__(self):
        return self.map(math.trunc)

    def __floor__(self):
        return self.map(math.floor)

    def __ceil__(self):
        return self.map(math.ceil)
