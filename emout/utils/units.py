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
    name : str or None
        単位の名前(例: "Mass", "Frequency")
    unit : str or None
        単位(例: "kg", "Hz")
    """

    def __init__(self, from_unit, to_unit, name=None, unit=None):
        """単位変換器を生成する.

        Parameters
        ----------
        from_unit : float
            変換前の値
        to_unit : float
            変換後の値
        name : str or None
            単位の名前(例: "Mass", "Frequency"), by default None
        unit : str or None
            単位(例: "kg", "Hz"), by default None
        """
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.ratio = to_unit / from_unit
        self.name = name
        self.unit = unit

    def set_name(self, name, unit=None):
        """名前を設定する.

        Parameters
        ----------
        name : str
            名前
        unit : str
            単位

        Returns
        -------
        UnitTranslator
            self
        """
        self.name = name
        self.unit = unit
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
        return '{}({:.4}, [{}])'.format(self.name, self.ratio, self.unit)

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
        Circular constant []
    e : UnitTranslator
        Napiers constant []
    c : UnitTranslator
        Light Speed [m/s]
    e0 : UnitTranslator
        FS-Permttivity [F/m]
    m0 : UnitTranslator
        FS-Permeablity [N/A^2]
    qe : UnitTranslator
        Elementary charge [C]
    me : UnitTranslator
        Electron mass [kg]
    mi : UnitTranslator
        Proton mass [kg]
    qe_me : UnitTranslator
        Electron charge-to-mass ratio [C/kg]
    kB : UnitTranslator
        Boltzmann constant [J/K]
    length : UnitTranslator
        Sim-to-Real length ratio [m]
    m : UnitTranslator
        Mass [kg]
    t : UnitTranslator
        Time [s]
    f : UnitTranslator
        Frequency [Hz]
    v : UnitTranslator
        Velocity [m/s]
    n : UnitTranslator
        Number density [/m^3]
    N : UnitTranslator
        Flux [/m^2s]
    F : UnitTranslator
        Force [N]
    P : UnitTranslator
        Power [W]
    W : UnitTranslator
        Energy [J]
    w : UnitTranslator]
        Energy density [J/m^3]
    eps : UnitTranslator
        Permittivity  [F/m]
    q : UnitTranslator
        Charge [C]
    rho : UnitTranslator
        Charge density [C/m^3]
    q_m : UnitTranslator
        Charge-to-mass ratio [C/kg]
    i : UnitTranslator
        Current [A]
    J : UnitTranslator
        Current density [A/m^2]
    phi : UnitTranslator
        Potential [V]
    E : UnitTranslator
        Electric field [V/m]
    H : UnitTranslator
        Magnetic field [A/m]
    C : UnitTranslator
        Capacitance [F]
    R : UnitTranslator
        Resistance [Ω]
    G : UnitTranslator
        Conductance [S]
    mu : UnitTranslator
        Permiability [H/m]
    B : UnitTranslator
        Magnetic flux density [T]
    L : UnitTranslator
        Inductance [H]
    T : UnitTranslator
        Temperature [K]
    a : UnitTranslator
        Acceleration [m/s^2]
    """

    def __init__(self, dx: float, to_c: float):
        """EMSES用の単位変換器を生成する.

        Parameters
        ----------
        dx : float, optional 
            Grid length [m]
        to_c : float
            Light speed in EMSES unit
        """
        self.dx = dx
        """Grid length [m]"""
        self.to_c = to_c
        """Light speed in EMSES"""

        from_c = 299792458
        to_e0 = 1
        pi = UnitTranslator(3.141592654, 3.141592654)
        e = UnitTranslator(2.718281828, 2.718281828)

        c = UnitTranslator(from_c, to_c)
        v = (1 * c)

        _m0 = 4 * pi.from_unit * 1E-7
        e0 = UnitTranslator(1 / (_m0 * c.from_unit ** 2), to_e0)
        eps = (1 * e0)
        mu = (1 / eps / v**2)
        m0 = UnitTranslator(_m0, mu.trans(_m0))

        kB = UnitTranslator(1.38065052E-23, 1.38065052E-23)

        length = UnitTranslator(dx, 1)
        t = (length / v)
        f = (1 / t)
        n = (1 / (length ** 3))
        N = (v * n)

        _qe = 1.6021765E-19
        _me = 9.1093819E-31
        _mi = 1.67261E-27
        qe_me = UnitTranslator(-_qe / _me, -1)
        q_m = (1 * qe_me)

        q = (e0 / q_m * length * v**2)
        m = (q / q_m)

        qe = UnitTranslator(_qe, q.trans(_qe))
        me = UnitTranslator(_me, m.trans(_me))
        mi = UnitTranslator(_mi, m.trans(_mi))
        rho = (q / length**3)

        F = (m * length / t**2)
        P = (F * v)
        W = (F * length)
        w = (W / (length**3))

        i = (q / length * v)
        J = (i / length**2)
        phi = (v**2 / q_m)
        E = (phi / length)
        H = (i / length)
        C = (eps * length)
        R = (phi / i)
        G = (1 / R)

        B = (v / length / q_m)
        L = (mu * length)
        T = (W / kB)

        a = (v / t)

        self.pi = pi.set_name('Circular constant', unit='')
        """Unit translator for Circular constant []"""

        self.e = e.set_name('Napiers constant', unit='')
        """Unit translator for Napiers constant []"""

        self.c = c.set_name('Light Speed', unit='m/s')
        """Unit translator for Light Speed [m/s]"""

        self.e0 = e0.set_name('FS-Permttivity', unit='F/m')
        """Unit translator for FS-Permttivity [F/m]"""

        self.m0 = m0.set_name('FS-Permeablity', unit='N/A^2')
        """Unit translator for FS-Permeablity [N/A^2]"""

        self.qe = qe.set_name('Elementary charge', unit='C')
        """Unit translator for Elementary charge [C]"""

        self.me = me.set_name('Electron mass', unit='kg')
        """Unit translator for Electron mass [kg]"""

        self.mi = mi.set_name('Proton mass', unit='kg')
        """Unit translator for Proton mass [kg]"""

        self.qe_me = qe_me.set_name(
            'Electron charge-to-mass ratio', unit='C/kg')
        """Unit translator for Electron charge-to-mass ratio [C/kg]"""

        self.kB = kB.set_name('Boltzmann constant', unit='J/K')
        """Unit translator for Boltzmann constant [J/K]"""

        self.length = length.set_name('Sim-to-Real length ratio', unit='m')
        """Unit translator for Sim-to-Real length ratio [m]"""

        self.m = m.set_name('Mass', unit='kg')
        """Unit translator for Mass [kg]"""

        self.t = t.set_name('Time', unit='s')
        """Unit translator for Time [s]"""

        self.f = f.set_name('Frequency', unit='Hz')
        """Unit translator for Frequency [Hz]"""

        self.v = v.set_name('Velocity', unit='m/s')
        """Unit translator for Velocity [m/s]"""

        self.n = n.set_name('Number density', unit='/m^3')
        """Unit translator for Number density [/m^3]"""

        self.N = N.set_name('Flux', unit='/m^2s')
        """Unit translator for Flux [/m^2s]"""

        self.F = F.set_name('Force', unit='N')
        """Unit translator for Force [N]"""

        self.P = P.set_name('Power', unit='W')
        """Unit translator for Power [W]"""

        self.W = W.set_name('Energy', unit='J')
        """Unit translator for Energy [J]"""

        self.w = w.set_name('Energy density', unit='J/m^3')
        """Unit translator for Energy density [J/m^3]"""

        self.eps = eps.set_name('Permittivity', unit='F/m')
        """Unit translator for Permittivity  [F/m]"""

        self.q = q.set_name('Charge', unit='C')
        """Unit translator for Charge [C]"""

        self.rho = rho.set_name('Charge density', unit='C/m^3')
        """Unit translator for Charge density [C/m^3]"""

        self.q_m = q_m.set_name('Charge-to-mass ratio', unit='C/kg')
        """Unit translator for Charge-to-mass ratio [C/kg]"""

        self.i = i.set_name('Current', unit='A')
        """Unit translator for Current [A]"""

        self.J = J.set_name('Current density', unit='A/m^2')
        """Unit translator for Current density [A/m^2]"""

        self.phi = phi.set_name('Potential', unit='V')
        """Unit translator for Potential [V]"""

        self.E = E.set_name('Electric field', unit='V/m')
        """Unit translator for Electric field [V/m]"""

        self.H = H.set_name('Magnetic field', unit='A/m')
        """Unit translator for Magnetic field [A/m]"""

        self.C = C.set_name('Capacitance', unit='F')
        """Unit translator for Capacitance [F]"""

        self.R = R.set_name('Resistance', unit='Ω')
        """Unit translator for Resistance [Ω]"""

        self.G = G.set_name('Conductance', unit='S')
        """Unit translator for Conductance [S]"""

        self.mu = mu.set_name('Permiability', unit='H/m')
        """Unit translator for Permiability [H/m]"""

        self.B = B.set_name('Magnetic flux density', unit='T')
        """Unit translator for Magnetic flux density [T]"""

        self.L = L.set_name('Inductance', unit='H')
        """Unit translator for Inductance [H]"""

        self.T = T.set_name('Temperature', unit='K')
        """Unit translator for Temperature [K]"""

        self.a = a.set_name('Acceleration', unit='m/s^2')
        """Unit translator for Acceleration [m/s^2]"""

    def translators(self):
        """変換器のリストを返す.

        Returns
        -------
        list(UnitTranslator)
            変換器のリスト
        """
        return
        # translators = inspect.getmembers(
        #     self, lambda x: isinstance(x, UnitTranslator))
        # return list(map(lambda x: x[1], translators))
