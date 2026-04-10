"""Generic grouping container with element-wise attribute delegation."""

import math


class Group:
    """Container that applies operations element-wise to a list of objects."""

    def __init__(self, objs, attrs=None):
        """Create a group from the given objects.

        Parameters
        ----------
        objs : list
            List of objects to group.
        """
        self.__dict__ = dict()
        self.objs = objs
        self.attrs = attrs

    def __binary_operator(self, callable, other):
        """Apply a binary operation to each element and return a new Group.

        Parameters
        ----------
        callable : object
            Function to apply to each element.
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        if isinstance(other, Group):
            others = other
            new_objs = [callable(obj, other) for obj, other in zip(self.objs, others)]
        else:
            new_objs = [callable(obj, other) for obj in self.objs]
        return type(self)(new_objs, attrs=self.attrs)

    def __check_and_return_iterable(self, arg):
        """Expand the input to match the Group length.

        Parameters
        ----------
        arg : object
            A ``Group`` or a single value. If a ``Group`` is given, its
            length must match ``self``. A single value is broadcast to
            all elements.
        Returns
        -------
        object
            Iterable with the same length as ``self``.
        """
        if isinstance(arg, Group):
            if len(self) != len(arg):
                raise ValueError(f"group size mismatch: self has {len(self)} elements, arg has {len(arg)}")
            args = arg
        else:
            args = [arg] * len(self.objs)
        return args

    def map(self, callable):
        """Apply a function to each element and return a new Group.

        Parameters
        ----------
        callable : object
            Function to apply to each element.
        Returns
        -------
        object
            New Group containing the results.
        """
        new_objs = list(map(callable, self.objs))
        return type(self)(new_objs, attrs=self.attrs)

    def filter(self, predicate):
        """Return a new Group containing only elements that satisfy the predicate.

        Parameters
        ----------
        predicate : object
            Function that decides whether to keep an element.
        Returns
        -------
        object
            New Group containing the filtered elements.
        """
        new_objs = list(filter(predicate, self.objs))
        return type(self)(new_objs, attrs=self.attrs)

    def foreach(self, callable):
        """Apply a function to each element for its side effects.

        Parameters
        ----------
        callable : object
            Function to apply to each element.
        Returns
        -------
        None
            No return value.
        """
        for obj in self.objs:
            callable(obj)

    def __str__(self):
        """Return the string representation.

        Returns
        -------
        str
            String representation of the Group.
        """
        return "Group({})".format(self.objs)

    def __repr__(self):
        """Return the string representation.

        Returns
        -------
        str
            String representation of the Group.
        """
        return str(self)

    def __format__(self, format_spec):
        """Return the formatted string representation.

        Parameters
        ----------
        format_spec : object
            Python format specification. The current implementation
            returns ``str(self)`` regardless of the value.
        Returns
        -------
        object
            String representation of ``self``.
        """
        return str(self)

    def __len__(self):
        """Return the number of elements.

        Returns
        -------
        int
            Number of elements in the Group.
        """
        return len(self.objs)

    def __iter__(self):
        """Return an iterator over the group elements.

        Returns
        -------
        Iterator
            Iterator over the group elements.
        """
        return iter(self.objs)

    def __contains__(self, obj):
        """Check whether the Group contains the given object.

        Parameters
        ----------
        obj : object
            Object to look for.
        Returns
        -------
        object
            ``True`` if the object is found.
        """
        return obj in self.objs

    def __pos__(self):
        """Apply the unary positive operator.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(lambda obj: -obj)

    def __neg__(self):
        """Apply the unary negation operator.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(lambda obj: -obj)

    def __invert__(self):
        """Apply the bitwise inversion operator.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(lambda obj: ~obj)

    def __add__(self, other):
        """Apply the addition operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj + other, other)

    def __radd_(self, other):
        """Apply the reflected addition operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other + obj, other)

    def __sub__(self, other):
        """Apply the subtraction operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj - other, other)

    def __rsub(self, other):
        """Apply the reflected subtraction operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other - obj, other)

    def __mul__(self, other):
        """Apply the multiplication operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj * other, other)

    def __rmul__(self, other):
        """Apply the reflected multiplication operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other * obj, other)

    def __truediv__(self, other):
        """Apply the true division operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj / other, other)

    def __rtruediv__(self, other):
        """Apply the reflected true division operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other / obj, other)

    def __floordiv__(self, other):
        """Apply the floor division operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj // other, other)

    def __rfloordiv__(self, other):
        """Apply the reflected floor division operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other // obj, other)

    def __mod__(self, other):
        """Apply the modulo operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj % other, other)

    def __rmod__(self, other):
        """Apply the reflected modulo operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other % obj, other)

    def __divmod__(self, other):
        """Apply the divmod operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: divmod(obj, other), other)

    def __rdivmod__(self, other):
        """Apply the reflected divmod operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: divmod(other, obj), other)

    def __pow__(self, other):
        """Apply the power operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj**other, other)

    def __rpow__(self, other):
        """Apply the reflected power operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other**obj, other)

    def __lshift__(self, other):
        """Apply the left shift operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj << other, other)

    def __rlshift__(self, other):
        """Apply the reflected left shift operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other << obj, other)

    def __rshift__(self, other):
        """Apply the right shift operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj >> other, other)

    def __rrshift__(self, other):
        """Apply the reflected right shift operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other >> obj, other)

    def __and__(self, other):
        """Apply the bitwise AND operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj & other, other)

    def __rand__(self, other):
        """Apply the reflected bitwise AND operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other & obj, other)

    def __or__(self, other):
        """Apply the bitwise OR operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj | other, other)

    def __ror__(self, other):
        """Apply the reflected bitwise OR operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other | obj, other)

    def __xor__(self, other):
        """Apply the bitwise XOR operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj ^ other, other)

    def __rxor__(self, other):
        """Apply the reflected bitwise XOR operator.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: other ^ obj, other)

    def __abs__(self):
        """Apply the absolute value operation.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(abs)

    def __eq__(self, other):
        """Apply the equality comparison.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj == other, other)

    def __ne__(self, other):
        """Apply the inequality comparison.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj != other, other)

    def __le__(self, other):
        """Apply the less-than-or-equal comparison.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj <= other, other)

    def __ge__(self, other):
        """Apply the greater-than-or-equal comparison.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj >= other, other)

    def __lt__(self, other):
        """Apply the less-than comparison.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj < other, other)

    def __gt__(self, other):
        """Apply the greater-than comparison.

        Parameters
        ----------
        other : object
            Right-hand operand for the operation or comparison.
        Returns
        -------
        object
            New Group containing the results.
        """
        return self.__binary_operator(lambda obj, other: obj > other, other)

    def __int__(self):
        """Apply integer conversion.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(int)

    def __float__(self):
        """Apply float conversion.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(float)

    def __complex__(self):
        """Apply complex number conversion.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(complex)

    def __bool__(self):
        """Apply boolean conversion.

        Returns
        -------
        bool
            Boolean evaluation result.
        """
        return self.map(bool)

    def __bytes__(self):
        """Apply bytes conversion.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(bytes)

    def __hash__(self):
        """Compute the hash value.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(hash)

    def __slice_to_iterable(self, slc):
        """Expand a slice containing Group values into per-element slices.

        Parameters
        ----------
        slc : object
            A ``slice`` object whose ``start``/``stop``/``step`` may be
            ``Group`` instances or single values.
        Returns
        -------
        object
            List of ``slice`` objects, one per element.
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
        """Expand an index key into per-element keys.

        Parameters
        ----------
        key : object
            Index key for getting or setting items.
        Returns
        -------
        object
            List of per-element keys.
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
        """Retrieve items by key.

        Parameters
        ----------
        key : object
            Index key for getting items.
        Returns
        -------
        object
            New Group containing the retrieved items.
        """
        keys = self.__expand_key(key)

        new_objs = [obj[key] for obj, key in zip(self.objs, keys)]

        return type(self)(new_objs, attrs=self.attrs)

    def __setitem__(self, key, value):
        """Set items by key.

        Parameters
        ----------
        key : object
            Index key for setting items.
        value : object
            Value to assign.

        Returns
        -------
        None
            No return value.
        """
        keys = self.__expand_key(key)
        values = self.__check_and_return_iterable(value)

        for obj, key, value in zip(self.objs, keys, values):
            obj[key] = value

    def __delitem__(self, key):
        """Delete items by key.

        Parameters
        ----------
        key : object
            Index key for deleting items.
        Returns
        -------
        None
            No return value.
        """
        keys = self.__expand_key(key)

        for obj, key in zip(self.objs, keys):
            del obj[key]

    def __getattr__(self, key):
        """Resolve attribute access across all elements.

        Parameters
        ----------
        key : object
            Attribute name to retrieve.
        Returns
        -------
        object
            New Group containing the attribute values.
        """
        keys = self.__check_and_return_iterable(key)
        new_objs = [getattr(obj, key) for obj, key in zip(self.objs, keys)]
        return type(self)(new_objs, attrs=self.attrs)

    def __setattr__(self, key, value):
        """Set an attribute on all elements.

        Parameters
        ----------
        key : object
            Attribute name to set.
        value : object
            Value to assign.

        Returns
        -------
        None
            No return value.
        """
        if key in ("objs", "__dict__", "attrs"):
            self.__dict__[key] = value
            return

        keys = self.__check_and_return_iterable(key)
        values = self.__check_and_return_iterable(value)

        for obj, key, value in zip(self.objs, keys, values):
            setattr(obj, key, value)

    def __call__(self, *args, **kwargs):
        """Call each element as a callable.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to each element's call.
        **kwargs : dict
            Keyword arguments forwarded to each element's call.

        Returns
        -------
        object
            New Group containing the call results.
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
            obj(*new_args, **new_kwargs) for obj, new_args, new_kwargs in zip(self.objs, new_argss, new_kwargss)
        ]

        return type(self)(new_objs, attrs=self.attrs)

    def __round__(self):
        """Apply the rounding operation.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(math.round)

    def __trunc__(self):
        """Apply the truncation operation.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(math.trunc)

    def __floor__(self):
        """Apply the floor operation.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(math.floor)

    def __ceil__(self):
        """Apply the ceiling operation.

        Returns
        -------
        object
            New Group containing the results.
        """
        return self.map(math.ceil)
