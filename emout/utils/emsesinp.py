import re

import f90nml


class UnitConversionKey:
    def __init__(self, dx, to_c):
        self.dx = dx
        self.to_c = to_c

    @classmethod
    def load(cls, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            line = f.readline()

        if not line.startswith('!!'):
            return None

        text = line[6:].strip()
        pattern = r'dx=\[([+-]?\d+(?:\.\d+)?)\],to_c=\[([+-]?\d+(?:\.\d+)?)\]'
        m = re.match(pattern, text)
        dx = float(m.group(1))
        to_c = float(m.group(2))
        return UnitConversionKey(dx, to_c)

    @property
    def keytext(self):
        return 'dx=[{}],to_c=[{}]'.format(self.dx, self.to_c)


class Plasmainp:
    def __init__(self, filename):
        self.nml = f90nml.read(filename)

    def __getitem__(self, key):
        if key in self.nml.keys():
            return self.nml[key]
        else:
            for group in self.nml.keys():
                if key in self.nml[group].keys():
                    return self.nml[group][key]
        raise KeyError()

    def remove(self, key, index=None):
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
                            start_index, = self.nml[group].start_index[key]
                            del self.nml[group][key][index - start_index]

                            if index == start_index:
                                self.nml[group].start_index[key] = [
                                    start_index + 1]

                            if len(list(self.nml[group][key])) == 0:
                                del self.nml[group][key]
                                del self.nml[group].start_index[key]
                    except (KeyError, IndexError):
                        pass
                    return

    def setlist(self, group, name, value, start_index=1):
        if not isinstance(value, list):
            value = [value]

        if name in self.nml[group].start_index:
            end_index = start_index + len(value)

            start_index_init, = self.nml[group].start_index[name]
            end_index_init = start_index_init + len(self.nml[group][name])

            min_start_index = min(start_index, start_index_init)
            max_end_index = max(end_index, end_index_init)

            new_list = [None] * (max_end_index - min_start_index)
            for i, index in enumerate(range(start_index_init-min_start_index, end_index_init-min_start_index)):
                new_list[index] = self.nml[group][name][i]
            for i, index in enumerate(range(start_index-min_start_index, end_index-min_start_index)):
                new_list[index] = value[i]

            self.nml[group].start_index[name] = [min_start_index]
            self.nml[group][name] = new_list
        else:
            self.nml[group].start_index[name] = [start_index]
            self.nml[group][name] = value

    def save(self, filename, convkey=None):
        with open(filename, 'wt', encoding='utf-8') as f:
            if convkey is not None:
                f.write('!!key {}\n'.format(convkey.keytext))
            f90nml.write(self.nml, f, force=True)
