"""Parser for EMSES ``plasma.inp`` namelist files.

:class:`InpFile` reads Fortran namelist parameters via :mod:`f90nml` and
exposes them as attributes, handling sparse-array indexing and the
``!!key`` unit-conversion header.
"""

import re

import f90nml

from emout.utils.units import Units


class UnitConversionKey:
    """Unit conversion key.

    Attributes
    ----------
    dx : float
        Grid width [m].
    to_c : float
        Speed of light in EMSES units.
    """

    def __init__(self, dx, to_c):
        """Create a unit conversion key.

        Parameters
        ----------
        dx : float
            Grid width [m].
        to_c : float
            Speed of light in EMSES units.
        """
        self.dx = dx
        self.to_c = to_c

    @classmethod
    def load(cls, filename):
        """Load a unit conversion key from a file.

        Read ``dx`` and ``to_c`` from the first line of the file if it
        matches the format ``!!key dx=[1.0],to_c=[10000.0]``.

        Parameters
        ----------
        filename : str or Path
            File containing the unit conversion key.

        Returns
        -------
        UnitConversionKey or None
            Unit conversion key, or ``None`` if the header is absent.
        """
        with open(filename, "r", encoding="utf-8") as f:
            line = f.readline()

        if not line.startswith("!!key"):
            return None

        # "!!key dx=[1.0],to_c=[10000.0]"
        text = line[6:].strip()
        pattern = r"dx=\[([+-]?\d+(?:\.\d+)?)\],to_c=\[([+-]?\d+(?:\.\d+)?)\]"
        m = re.match(pattern, text)
        dx = float(m.group(1))
        to_c = float(m.group(2))
        return UnitConversionKey(dx, to_c)

    @property
    def keytext(self):
        """Return the string representation of the conversion key.

        Returns
        -------
        str
            String representation of the conversion key.
        """
        return "dx=[{}],to_c=[{}]".format(self.dx, self.to_c)


class AttrDict(dict):
    """Dictionary subclass that supports attribute-style access.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the instance.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to ``dict``.
        **kwargs : dict
            Keyword arguments forwarded to ``dict``.
        """
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        """Resolve attribute access by delegating to item lookup.

        Parameters
        ----------
        key : object
            Key to retrieve.
        Returns
        -------
        object
            Value associated with the key.
        """
        return self[key]


class InpFile:
    """Manage an EMSES parameter file.

    nml : f90nml.Namelist
        Namelist object.
    """

    def __init__(self, filename=None, convkey=None):
        """Initialize the instance.

        Parameters
        ----------
        filename : object, optional
            Path to the namelist file to read.
        convkey : object, optional
            ``UnitConversionKey`` for unit conversion. If a ``!!key``
            header is found in *filename*, it takes precedence.
        """
        if filename:
            self.nml = f90nml.read(filename)
            self.convkey = UnitConversionKey.load(filename) or convkey
        else:
            self.nml = f90nml.Namelist()
            self.convkey = convkey

    def __contains__(self, key):
        """Check whether the namelist contains the given key.

        Parameters
        ----------
        key : object
            Group or parameter name to look up.
        Returns
        -------
        object
            ``True`` if the key is found.
        """
        if key in self.nml.keys():
            return True
        else:
            for group in self.nml.keys():
                if key in self.nml[group].keys():
                    return True
        return False

    def __getitem__(self, key):
        """Retrieve a group or parameter by key.

        Parameters
        ----------
        key : object
            Group or parameter name to retrieve.
        Returns
        -------
        object
            Namelist group or parameter value.
        """
        if key in self.nml.keys():
            return self.nml[key]
        else:
            for group in self.nml.keys():
                if key in self.nml[group].keys():
                    return self.nml[group][key]
        raise KeyError()

    def __setitem__(self, key, item):
        """Set a group or parameter by key.

        Parameters
        ----------
        key : object
            Group or parameter name to set.
        item : object
            Value to assign.
        Returns
        -------
        None
            No return value.
        """
        if key in self.nml.keys():
            self.nml[key] = item
            return
        else:
            for group in self.nml.keys():
                if key in self.nml[group].keys():
                    self.nml[group][key] = item
                    return
        raise KeyError()

    def __getattr__(self, key):
        """Resolve attribute access by delegating to item lookup.

        Parameters
        ----------
        key : object
            Key to retrieve.
        Returns
        -------
        object
            Namelist group (as ``AttrDict``) or parameter value.
        """
        item = self[key]
        if isinstance(item, dict):
            return AttrDict(item)
        return item

    def __setattr__(self, key, item):
        """Set an attribute, delegating to item assignment for namelist keys.

        Parameters
        ----------
        key : object
            Attribute or parameter name to set.
        item : object
            Value to assign.
        Returns
        -------
        None
            No return value.
        """
        if key in ["nml", "convkey"]:
            super().__setattr__(key, item)
            return

        if key not in self:
            raise KeyError()

        self[key] = item

    def remove(self, key, index=None):
        """Remove a parameter.

        Parameters
        ----------
        key : str
            Group name (``&groupname``) or parameter name.
        index : int, optional
            If given, remove only the element at this index.
        """
        if key in self.nml.keys():
            del self.nml[key]
        else:
            for group in self.nml.keys():
                if key in self.nml[group].keys():
                    try:
                        if index is None:
                            del self.nml[group][key]
                            del self.nml[group].start_index[key]
                        else:
                            (start_index,) = self.nml[group].start_index[key]
                            del self.nml[group][key][index - start_index]

                            if index == start_index:
                                self.nml[group].start_index[key] = [start_index + 1]

                            if len(list(self.nml[group][key])) == 0:
                                del self.nml[group][key]
                                del self.nml[group].start_index[key]
                    except (KeyError, IndexError):
                        pass
                    return

    def setlist(self, group, name, value, start_index=1):
        """Set a list-type parameter.

        Parameters
        ----------
        group : str
            Group name (``&groupname``).
        name : str
            Parameter name.
        value : type or list(type)
            Value(s) to set.
        start_index : int, optional
            Starting index for the list, by default 1.
        """
        if not isinstance(value, list):
            value = [value]

        if name in self.nml[group].start_index:
            end_index = start_index + len(value)

            (start_index_init,) = self.nml[group].start_index[name]
            end_index_init = start_index_init + len(self.nml[group][name])

            min_start_index = min(start_index, start_index_init)
            max_end_index = max(end_index, end_index_init)

            new_list = [None] * (max_end_index - min_start_index)
            for i, index in enumerate(
                range(
                    start_index_init - min_start_index, end_index_init - min_start_index
                )
            ):
                new_list[index] = self.nml[group][name][i]
            for i, index in enumerate(
                range(start_index - min_start_index, end_index - min_start_index)
            ):
                new_list[index] = value[i]

            self.nml[group].start_index[name] = [min_start_index]
            self.nml[group][name] = new_list
        else:
            self.nml[group].start_index[name] = [start_index]
            self.nml[group][name] = value

    def save(self, filename, convkey=None):
        """Save the parameters to a file.

        Parameters
        ----------
        filename : str or Path
            Output file path.
        convkey : UnitConversionKey, optional
            Unit conversion key to write, by default None.
        """
        convkey = convkey or self.convkey
        with open(filename, "wt", encoding="utf-8") as f:
            if convkey is not None:
                f.write("!!key {}\n".format(convkey.keytext))
            f90nml.write(self.nml, f, force=True)

    def __str__(self):
        """Return the string representation.

        Returns
        -------
        str
            String representation of the namelist.
        """
        return str(self.nml)

    def __repr__(self):
        """Return the string representation.

        Returns
        -------
        str
            String representation of the namelist.
        """
        return str(self)

    def conversion(self, unit_from: Units, unit_to: Units):
        """Apply unit conversion to relevant parameters.

        Parameters
        ----------
        unit_from : Units
            Source unit system.
        unit_to : Units
            Target unit system.
        Returns
        -------
        None
            No return value.
        """
        def conv(group, name, unit_name):
            """Convert a scalar parameter and update it in place.

            Parameters
            ----------
            group : object
                Namelist group name.
            name : object
                Parameter name.
            unit_name : object
                Unit translator attribute name.
            Returns
            -------
            None
                No return value.
            """
            value_from = self[group][name]

            value = getattr(unit_from, unit_name).reverse(value_from)
            value_to = getattr(unit_to, unit_name).trans(value)

            self[group][name] = value_to

        def conv1d(group, name, unit_name):
            """Convert a 1-D array parameter and update it in place.

            Parameters
            ----------
            group : object
                Namelist group name.
            name : object
                Parameter name.
            unit_name : object
                Unit translator attribute name.
            Returns
            -------
            None
                No return value.
            """
            if group not in self:
                return
            if name not in self[group]:
                return

            values_from = self[group][name]

            values_to = []
            for value_from in values_from:
                if value_from is None:
                    value_to = None
                else:
                    value = getattr(unit_from, unit_name).reverse(value_from)
                    value_to = getattr(unit_to, unit_name).trans(value)
                values_to.append(value_to)

            self[group][name] = values_to

        conv1d("plasma", "wp", "f")

        conv1d("intp", "path", "v")
        conv1d("intp", "peth", "v")
        conv1d("intp", "vdri", "v")

        conv1d("emissn", "curf", "J")
        conv1d("emissn", "curfs", "J")

        self.convkey = UnitConversionKey(unit_to.dx, unit_to.to_c)

    @property
    def dx(self) -> float:
        """Return the conversion key ``dx``.

        Returns
        -------
        float
            Grid width from the conversion key.
        """
        return self.convkey.dx

    @property
    def to_c(self) -> float:
        """Return the conversion key ``to_c``.

        Returns
        -------
        float
            Speed of light in EMSES units from the conversion key.
        """
        return self.convkey.to_c
