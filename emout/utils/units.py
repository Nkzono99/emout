import inspect


class UnitTranslator:
    def __init__(self, from_unit, to_unit, name="None"):
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.ratio = to_unit / from_unit
        self.name = name

    def set_name(self, name):
        self.name = name
        return self

    def trans(self, value, reverse=False):
        if reverse:
            return value / self.ratio
        else:
            return value * self.ratio

    def reverse(self, value):
        return self.trans(value, reverse=True)

    def __mul__(self, other):
        from_unit = self.from_unit * other.from_unit
        to_unit = self.to_unit * other.to_unit
        return UnitTranslator(from_unit, to_unit)

    def __rmul__(self, other):
        other = UnitTranslator(other, other)
        return other * self

    def __truediv__(self, other):
        from_unit = self.from_unit / other.from_unit
        to_unit = self.to_unit / other.to_unit
        return UnitTranslator(from_unit, to_unit)

    def __rtruediv__(self, other):
        other = UnitTranslator(other, other)
        return other / self

    def __pow__(self, other):
        from_unit = self.from_unit ** other
        to_unit = self.to_unit ** other
        return UnitTranslator(from_unit, to_unit)

    def __str__(self):
        return '{}({:.4})'.format(self.name, self.ratio)

    def __repr__(self):
        return self.__str__()


class Units:
    def __init__(self,
                 dx=0.001,
                 to_c=10000):
        self.dx = dx
        self.to_c = to_c
        from_c = 299792458
        to_e0 = 1
        pi = UnitTranslator(3.141592654, 3.141592654, name='Circular constant')
        e = UnitTranslator(2.718281828, 2.718281828, name='Napiers constant')

        c = UnitTranslator(from_c, to_c, name='Light Speed')
        v = (1 * c).set_name('Velocity')

        _m0 = 4 * pi.from_unit * 1E-7
        e0 = UnitTranslator(1 / (_m0 * c.from_unit ** 2),
                            to_e0).set_name('FS-Permttivity')
        eps = (1 * e0).set_name('Permittivity')
        mu = (1 / eps / v**2).set_name('Permiability')
        m0 = UnitTranslator(_m0, mu.trans(_m0), name='FS-Permeablity')

        kB = UnitTranslator(1.38065052E-23, 1.38065052E-23,
                            'Boltzmann constant')

        length = UnitTranslator(dx, 1, name='Sim-to-Real length ratio')
        t = (length / v).set_name('Time')
        f = (1 / t).set_name('Frequency')
        n = (1 / (length ** 3)).set_name('Number density')
        N = (v * n).set_name('Flux')

        _qe = 1.6021765E-19
        _me = 9.1093819E-31
        _mi = 1.67261E-27
        qe_me = UnitTranslator(-_qe / _me, -1,
                               name='Electron charge-to-mass ratio')
        q_m = (1 * qe_me).set_name('Charge-to-mass ratio')

        q = (e0 / q_m * length * v**2).set_name('Charge')
        m = (q / q_m).set_name('Mass')

        qe = UnitTranslator(_qe, q.trans(_qe), name='Elementary charge')
        me = UnitTranslator(_me, m.trans(_me), name='Electron mass')
        mi = UnitTranslator(_mi, m.trans(_mi), 'Proton mass')
        rho = (q / length**3).set_name('Charge density')

        F = (m * length / t**2).set_name('Force')
        P = (F * v).set_name('Power')
        W = (F * length).set_name('Energy')
        w = (W / (length**3)).set_name('Energy density')

        i = (q / length * v).set_name('Current')
        J = (i / length**2).set_name('Current density')
        phi = (v**2 / q_m).set_name('Potential')
        E = (phi / length).set_name('Electric field')
        C = (eps * length).set_name('Capacitance')
        R = (phi / i).set_name('Resistance')
        G = (1 / R).set_name('Conductance')

        B = (v / length / q_m).set_name('Magnetic flux density')
        L = (mu * length).set_name('Inductance')
        T = (W / kB).set_name('Temperature')

        self.pi = pi
        self.e = e

        self.c = c
        self.e0 = e0
        self.m0 = m0
        self.qe = qe
        self.me = me
        self.mi = mi
        self.qe_me = qe_me
        self.kB = kB
        self.length = length

        self.m = m
        self.t = t
        self.f = f
        self.v = v
        self.n = n
        self.N = N
        self.F = F
        self.P = P
        self.W = W
        self.w = w
        self.eps = eps
        self.q = q
        self.rho = rho
        self.q_m = q_m
        self.i = i
        self.J = J
        self.phi = phi
        self.E = E
        self.C = C
        self.R = R
        self.G = G
        self.mu = mu
        self.B = B
        self.L = L
        self.T = T
    
    @property
    def translators(self):
        translators = inspect.getmembers(self, lambda x: isinstance(x, UnitTranslator))
        return list(map(lambda x: x[1], translators))
