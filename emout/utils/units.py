import inspect


class UnitTranslator:
    """単位変換器.

    Attributes
    ----------
    from_unit : float
        変換前の値
    to_unit : float
        変換後の値
    ratio : float
        変換係数 (変換後 = 変換係数 * 変換前)
    name : str
        単位の名前
    """

    def __init__(self, from_unit, to_unit, name="None"):
        """単位変換器を生成する.

        Parameters
        ----------
        from_unit : float
            変換前の値
        to_unit : float
            変換後の値
        name : str
            単位の名前, by default "None"
        """
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.ratio = to_unit / from_unit
        self.name = name

    def set_name(self, name):
        """名前を設定する.

        Parameters
        ----------
        name : str
            名前

        Returns
        -------
        UnitTranslator
            self
        """
        self.name = name
        return self

    def trans(self, value, reverse=False):
        """単位変換を行う.

        Parameters
        ----------
        value : float
            変換前の値(reverse=Trueの場合変換後の値)
        reverse : bool, optional
            逆変換を行う場合True, by default False

        Returns
        -------
        float
            変換後の値(reverse=Trueの場合変換前の値)
        """
        if reverse:
            return value / self.ratio
        else:
            return value * self.ratio

    def reverse(self, value):
        """単位逆変換を行う.

        Parameters
        ----------
        value : float
            変換後の値

        Returns
        -------
        float
            変換前の値
        """
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
    """EMSES用の単位変換器を管理する.

    SI単位系からEMSES単位系への変換を行う.

    Attributes
    ----------
    dx : float
        Grid length [m]
    to_c : float
        Light Speed in EMSES
    pi : UnitTranslator
        Circular constant
    e : UnitTranslator
        Napiers constant
    c : UnitTranslator
        Light Speed
    e0 : UnitTranslator
        Velocity
    m0 : UnitTranslator
        FS-Permeablity
    qe : UnitTranslator
        Elementary charge
    me : UnitTranslator
        Electron mass
    mi : UnitTranslator
        Proton mass
    qe_me : UnitTranslator
        Electron charge-to-mass ratio
    kB : UnitTranslator
        Boltzmann constant
    length : UnitTranslator
        Sim-to-Real length ratio
    m : UnitTranslator
        Mass
    t : UnitTranslator
        Time
    f : UnitTranslator
        Frequency
    v : UnitTranslator
        Velocity
    n : UnitTranslator
        Number density
    N : UnitTranslator
        Flux
    F : UnitTranslator
        Force
    P : UnitTranslator
        Power
    W : UnitTranslator
        Energy
    w : UnitTranslator]
        Energy density
    eps : UnitTranslator
        Permittivity
    q : UnitTranslator
        Charge
    rho : UnitTranslator
        Charge density
    q_m : UnitTranslator
        Charge-to-mass ratio
    i : UnitTranslator
        Current
    J : UnitTranslator
        Current density
    phi : UnitTranslator
        Potential
    E : UnitTranslator
        Electric field
    C : UnitTranslator
        Capacitance
    R : UnitTranslator
        Resistance
    G : UnitTranslator
        Conductance
    mu : UnitTranslator
        Permiability
    B : UnitTranslator
        Magnetic flux density
    L : UnitTranslator
        Inductance
    T : UnitTranslator
        Temperature
    """

    def __init__(self, dx, to_c):
        """EMSES用の単位変換器を生成する.

        Parameters
        ----------
        dx : float, optional 
            Grid length [m]
        to_c : float
            Light Speed in EMSES
        """
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
        """変換器のリストを返す.

        Returns
        -------
        list(UnitTranslator)
            変換器のリスト
        """
        translators = inspect.getmembers(
            self, lambda x: isinstance(x, UnitTranslator))
        return list(map(lambda x: x[1], translators))
