from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ============================================================
# Surfaces (Implicit / SDF-like)
# ============================================================


class Surface3D(ABC):
    """Implicit surface / solid via signed function.

    Convention:
      sdf(x,y,z) <= 0  : inside
      sdf(x,y,z) >  0  : outside

    This does NOT need to be an exact signed *distance*.
    For boolean operations, any continuous implicit function with correct sign works.
    """

    @abstractmethod
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : np.ndarray
            x 座標または x 成分。
        y : np.ndarray
            y 座標または y 成分。
        z : np.ndarray
            z 座標または z 成分。
        
        Returns
        -------
        np.ndarray
            処理結果です。
        """
        raise NotImplementedError

    # ---- composition operators ----
    def __or__(self, other: "Surface3D") -> "Surface3D":
        """ビット和演算を適用する。
        
        Parameters
        ----------
        other : "Surface3D"
            演算または比較の相手となる値です。
        Returns
        -------
        "Surface3D"
            処理結果です。
        """
        return UnionSurface(self, other)

    def __and__(self, other: "Surface3D") -> "Surface3D":
        """ビット積演算を適用する。
        
        Parameters
        ----------
        other : "Surface3D"
            演算または比較の相手となる値です。
        Returns
        -------
        "Surface3D"
            処理結果です。
        """
        return IntersectionSurface(self, other)

    def __sub__(self, other: "Surface3D") -> "Surface3D":
        """減算演算を適用する。
        
        Parameters
        ----------
        other : "Surface3D"
            演算または比較の相手となる値です。
        Returns
        -------
        "Surface3D"
            処理結果です。
        """
        return DifferenceSurface(self, other)

    def __invert__(self) -> "Surface3D":
        """ビット反転演算を適用する。
        
        Returns
        -------
        "Surface3D"
            処理結果です。
        """
        return ComplementSurface(self)

    def __xor__(self, other: "Surface3D") -> "Surface3D":
        """排他的論理和演算を適用する。
        
        Parameters
        ----------
        other : "Surface3D"
            演算または比較の相手となる値です。
        Returns
        -------
        "Surface3D"
            処理結果です。
        """
        return XorSurface(self, other)

    # ---- convenience transforms ----
    def offset(self, delta: float) -> "Surface3D":
        """Offset surface: inside expands if delta>0 (approx). Implemented as sdf - delta."""
        return OffsetSurface(self, delta)

    def translated(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Surface3D":
        """平行移動した曲面オブジェクトを返す。
        
        Parameters
        ----------
        dx : float, optional
            x 方向の格子間隔です。
        dy : float, optional
            y 方向の格子間隔です。
        dz : float, optional
            z 方向の格子間隔です。
        Returns
        -------
        "Surface3D"
            処理結果です。
        """
        return ShiftSurface(self, dx=dx, dy=dy, dz=dz)

    def rotated(
        self,
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        *,
        degrees: bool = True,
        order: str = "xyz",
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "Surface3D":
        """回転変換した曲面オブジェクトを返す。
        
        Parameters
        ----------
        rx : float, optional
            x 軸周りの回転角です。
        ry : float, optional
            y 軸周りの回転角です。
        rz : float, optional
            z 軸周りの回転角です。
        degrees : bool, optional
            `True` なら角度を度数法、`False` ならラジアンとして解釈します。
        order : str, optional
            回転適用順序（例: `xyz`）です。
        origin : Tuple[float, float, float], optional
            回転・平行移動の基準点です。
        Returns
        -------
        "Surface3D"
            処理結果です。
        """
        return RotateSurface(self, rx, ry, rz, degrees=degrees, order=order, origin=origin)


# ============================================================
# Primitives
# ============================================================


class HeightFieldSurface(Surface3D):
    """Single-valued surface z = z_s(x,y) representing a half-space below the surface.

    inside: z <= z_s(x,y)  -> sdf = z - z_s(x,y) <= 0
    """

    def __init__(self, z_of_xy: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        z_of_xy : Callable[[np.ndarray, np.ndarray], np.ndarray]
            コールバック関数です。
        """
        self.z_of_xy = z_of_xy

    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : np.ndarray
            x 座標または x 成分。
        y : np.ndarray
            y 座標または y 成分。
        z : np.ndarray
            z 座標または z 成分。
        
        Returns
        -------
        np.ndarray
            処理結果です。
        """
        return np.asarray(z) - np.asarray(self.z_of_xy(x, y))


class DEMHeightFieldSurface(HeightFieldSurface):
    """Height field from a DEM grid with SciPy interpolation.

    dem_z_yx: shape (len(y_dem), len(x_dem)) at coordinates (y_dem, x_dem)
    """

    def __init__(
        self,
        x_dem: np.ndarray,
        y_dem: np.ndarray,
        dem_z_yx: np.ndarray,
        *,
        bounds_error: bool = False,
        fill_value: float = np.nan,
        method: str = "linear",
    ):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        x_dem : np.ndarray
            DEM の x 座標軸（昇順 1 次元配列）です。
        y_dem : np.ndarray
            DEM の y 座標軸（昇順 1 次元配列）です。
        dem_z_yx : np.ndarray
            標高値配列です。形状は `(len(y_dem), len(x_dem))` を想定します。
        bounds_error : bool, optional
            `True` の場合、補間点が DEM 範囲外だと例外を送出します。
        fill_value : float, optional
            範囲外補間時に返す値です（`bounds_error=False` の場合）。
        method : str, optional
            補間方法です。`scipy.interpolate.RegularGridInterpolator` に渡します。
        """
        x_dem = np.asarray(x_dem, dtype=np.float64)
        y_dem = np.asarray(y_dem, dtype=np.float64)
        dem_z_yx = np.asarray(dem_z_yx, dtype=np.float64)
        if dem_z_yx.shape != (y_dem.size, x_dem.size):
            raise ValueError("dem_z_yx must have shape (len(y_dem), len(x_dem))")

        interp2 = RegularGridInterpolator(
            (y_dem, x_dem),  # (y, x)
            dem_z_yx,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )

        def z_of_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """`(x, y)` 座標での高さを補間して返す。
            
            Parameters
            ----------
            x : np.ndarray
                x 座標または x 成分。
            y : np.ndarray
                y 座標または y 成分。
            
            Returns
            -------
            np.ndarray
                処理結果です。
            """
            x = np.asarray(x)
            y = np.asarray(y)
            pts = np.stack([y.ravel(), x.ravel()], axis=-1)
            return interp2(pts).reshape(np.broadcast(x, y).shape)

        super().__init__(z_of_xy)


class PlaneSurface(Surface3D):
    """Plane: n·(p - p0) = 0

    inside: n·(p - p0) <= 0

    Notes
    -----
    - normal does not have to be unit length.
    - the sign convention is important when you build a "box" from 6 planes.
    """

    def __init__(self, normal: Tuple[float, float, float], point: Tuple[float, float, float]):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        normal : Tuple[float, float, float]
            平面の法線ベクトルです。ゼロベクトルは指定できません。
        point : Tuple[float, float, float]
            平面が通る 1 点 `(x0, y0, z0)` です。
        """
        n = np.asarray(normal, dtype=np.float64)
        if np.linalg.norm(n) == 0:
            raise ValueError("normal must be non-zero")
        self.n = n
        self.p0 = np.asarray(point, dtype=np.float64)

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        p = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=0)
        return (
            self.n[0] * (p[0] - self.p0[0])
            + self.n[1] * (p[1] - self.p0[1])
            + self.n[2] * (p[2] - self.p0[2])
        )


class BoxSurface(Surface3D):
    """Axis-aligned box (AABB) with a proper signed-distance-like function.

    inside: xmin<=x<=xmax, ymin<=y<=ymax, zmin<=z<=zmax

    This one is numerically stable for booleans and for marching cubes.
    """

    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        xmin : float
            x 方向最小座標です。
        xmax : float
            x 方向最大座標です。
        ymin : float
            y 方向最小座標です。
        ymax : float
            y 方向最大座標です。
        zmin : float
            z 方向最小座標です。
        zmax : float
            z 方向最大座標です。
        """
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.zmin = float(zmin)
        self.zmax = float(zmax)
        if not (self.xmin <= self.xmax and self.ymin <= self.ymax and self.zmin <= self.zmax):
            raise ValueError("Require xmin<=xmax, ymin<=ymax, zmin<=zmax")

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        cx = 0.5 * (self.xmin + self.xmax)
        cy = 0.5 * (self.ymin + self.ymax)
        cz = 0.5 * (self.zmin + self.zmax)
        hx = 0.5 * (self.xmax - self.xmin)
        hy = 0.5 * (self.ymax - self.ymin)
        hz = 0.5 * (self.zmax - self.zmin)

        qx = np.abs(x - cx) - hx
        qy = np.abs(y - cy) - hy
        qz = np.abs(z - cz) - hz

        ox = np.maximum(qx, 0.0)
        oy = np.maximum(qy, 0.0)
        oz = np.maximum(qz, 0.0)
        outside = np.sqrt(ox * ox + oy * oy + oz * oz)

        inside = np.minimum(np.maximum.reduce([qx, qy, qz]), 0.0)
        return outside + inside


class PlaneBoxSurface(Surface3D):
    """Axis-aligned box constructed as the intersection of 6 PlaneSurface half-spaces.

    This matches the "PlaneSurface*6" mental model (and is convenient if you later
    want to swap some planes or use different half-spaces). For most use-cases,
    BoxSurface (AABB SDF) is the more numerically stable primitive.
    """

    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        xmin : float
            x 方向最小座標です。
        xmax : float
            x 方向最大座標です。
        ymin : float
            y 方向最小座標です。
        ymax : float
            y 方向最大座標です。
        zmin : float
            z 方向最小座標です。
        zmax : float
            z 方向最大座標です。
        """
        xmin, xmax, ymin, ymax, zmin, zmax = map(float, (xmin, xmax, ymin, ymax, zmin, zmax))
        if not (xmin <= xmax and ymin <= ymax and zmin <= zmax):
            raise ValueError("Require xmin<=xmax, ymin<=ymax, zmin<=zmax")

        # x <= xmax
        pxu = PlaneSurface(normal=(+1.0, 0.0, 0.0), point=(xmax, 0.0, 0.0))
        # x >= xmin
        pxl = PlaneSurface(normal=(-1.0, 0.0, 0.0), point=(xmin, 0.0, 0.0))
        # y <= ymax
        pyu = PlaneSurface(normal=(0.0, +1.0, 0.0), point=(0.0, ymax, 0.0))
        # y >= ymin
        pyl = PlaneSurface(normal=(0.0, -1.0, 0.0), point=(0.0, ymin, 0.0))
        # z <= zmax
        pzu = PlaneSurface(normal=(0.0, 0.0, +1.0), point=(0.0, 0.0, zmax))
        # z >= zmin
        pzl = PlaneSurface(normal=(0.0, 0.0, -1.0), point=(0.0, 0.0, zmin))

        self._planes = (pxu, pxl, pyu, pyl, pzu, pzl)

    def sdf(self, x, y, z):
        # intersection of half-spaces => max
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        v = self._planes[0].sdf(x, y, z)
        for p in self._planes[1:]:
            v = np.maximum(v, p.sdf(x, y, z))
        return v


class SphereSurface(Surface3D):
    """Sphere solid: inside: |p-c| - r <= 0."""

    def __init__(self, center: Tuple[float, float, float], radius: float):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        center : Tuple[float, float, float]
            中心座標です。
        radius : float
            球の半径です。`0` より大きい値を指定します。
        """
        self.c = np.asarray(center, dtype=np.float64)
        self.r = float(radius)

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        dx = np.asarray(x) - self.c[0]
        dy = np.asarray(y) - self.c[1]
        dz = np.asarray(z) - self.c[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz) - self.r


AxisSpec = Union[str, Tuple[float, float, float], np.ndarray]


def _as_3vec(v, *, name: str) -> np.ndarray:
    """入力を 3 要素ベクトルへ正規化する。
    
    Parameters
    ----------
    v : object
        3 要素ベクトルとして解釈する入力値です。
    name : str
        対象データ名またはキー名です。
    Returns
    -------
    np.ndarray
        shape `(3,)` の `float64` ベクトルです。
    """
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError(f"{name} must be a 3-vector, got shape {a.shape}")
    return a


def _axis_to_unit(axis: AxisSpec) -> np.ndarray:
    """軸指定を正規化済み単位ベクトルへ変換する。
    
    Parameters
    ----------
    axis : AxisSpec
        対象軸。
    
    Returns
    -------
    np.ndarray
        処理結果です。
    """
    if isinstance(axis, str):
        s = axis.lower().strip()
        if s == "x":
            a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif s == "y":
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif s == "z":
            a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            raise ValueError("axis must be one of 'x','y','z' or a 3-vector.")
    else:
        a = _as_3vec(axis, name="axis")

    n = np.linalg.norm(a)
    if n == 0.0 or not np.isfinite(n):
        raise ValueError("axis must be a finite non-zero vector.")
    return a / n


class CylinderSurface(Surface3D):
    """Cylinder (optionally finite/capped) defined by axis direction and a point on axis.

    Infinite cylinder:
        sdf = dist_to_axis - r

    Finite capped cylinder (solid):
        the axial coordinate is constrained to [tmin, tmax] (relative to center)

    Parameters
    ----------
    center : (cx,cy,cz) or (cx,cy)
        A point on the cylinder axis. If 2-tuple, cz is assumed 0.
    axis : 'x'|'y'|'z' or 3-vector
        Axis direction.
    radius : float
        Radius.
    length : float, optional
        If given, makes a finite cylinder with symmetric interval [-L/2, +L/2].
    tmin, tmax : float, optional
        If given, makes a finite cylinder with interval [tmin, tmax].
        Use this when you want asymmetric extents along the axis.
    """

    def __init__(
        self,
        center: Union[Tuple[float, float], Tuple[float, float, float]],
        axis: AxisSpec,
        radius: float,
        *,
        length: Optional[float] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
    ):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        center : Union[Tuple[float, float], Tuple[float, float, float]]
            中心座標です。
        axis : AxisSpec
            対象軸。
        radius : float
            円柱半径です。`0` より大きい値を指定します。
        length : Optional[float], optional
            有限円柱にする場合の軸方向長さです。
            指定時は `[-length/2, +length/2]` を使用します。
        tmin : Optional[float], optional
            有限円柱の軸方向下限です。`tmax` と対で指定します。
        tmax : Optional[float], optional
            有限円柱の軸方向上限です。`tmin < tmax` が必要です。
        """
        if len(center) == 2:
            cx, cy = center
            cz = 0.0
            self.c = np.array([float(cx), float(cy), float(cz)], dtype=np.float64)
        elif len(center) == 3:
            self.c = np.array([float(center[0]), float(center[1]), float(center[2])], dtype=np.float64)
        else:
            raise ValueError("center must be a 2-tuple (cx,cy) or 3-tuple (cx,cy,cz).")

        self.a = _axis_to_unit(axis)
        self.r = float(radius)
        if self.r <= 0:
            raise ValueError("radius must be > 0")

        if length is not None and (tmin is not None or tmax is not None):
            raise ValueError("Specify either length OR (tmin,tmax), not both.")
        if length is not None:
            L = float(length)
            if L <= 0:
                raise ValueError("length must be > 0")
            self.tmin = -0.5 * L
            self.tmax = +0.5 * L
        elif (tmin is None) != (tmax is None):
            raise ValueError("Specify both tmin and tmax for finite cylinder.")
        elif tmin is not None and tmax is not None:
            self.tmin = float(tmin)
            self.tmax = float(tmax)
            if not (self.tmin < self.tmax):
                raise ValueError("Require tmin < tmax")
        else:
            self.tmin = None
            self.tmax = None

    @property
    def is_finite(self) -> bool:
        """円柱が有限長かどうかを返す。
        
        Returns
        -------
        bool
            条件判定結果です。
        """
        return self.tmin is not None

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        dx = x - self.c[0]
        dy = y - self.c[1]
        dz = z - self.c[2]

        t = dx * self.a[0] + dy * self.a[1] + dz * self.a[2]

        rx = dx - t * self.a[0]
        ry = dy - t * self.a[1]
        rz = dz - t * self.a[2]
        radial = np.sqrt(rx * rx + ry * ry + rz * rz)

        d1 = radial - self.r

        if self.tmin is None:
            return d1

        mid = 0.5 * (self.tmin + self.tmax)
        half = 0.5 * (self.tmax - self.tmin)
        d2 = np.abs(t - mid) - half

        o1 = np.maximum(d1, 0.0)
        o2 = np.maximum(d2, 0.0)
        outside = np.sqrt(o1 * o1 + o2 * o2)
        inside = np.minimum(np.maximum(d1, d2), 0.0)
        return outside + inside


# ============================================================
# Boolean composition
# ============================================================


class UnionSurface(Surface3D):
    """UnionSurface クラス。
    """
    def __init__(self, a: Surface3D, b: Surface3D):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        a : Surface3D
            始点側の値です。
        b : Surface3D
            終点側の値です。
        """
        self.a = a
        self.b = b

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        return np.minimum(self.a.sdf(x, y, z), self.b.sdf(x, y, z))


class IntersectionSurface(Surface3D):
    """IntersectionSurface クラス。
    """
    def __init__(self, a: Surface3D, b: Surface3D):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        a : Surface3D
            始点側の値です。
        b : Surface3D
            終点側の値です。
        """
        self.a = a
        self.b = b

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        return np.maximum(self.a.sdf(x, y, z), self.b.sdf(x, y, z))


class DifferenceSurface(Surface3D):
    """A \ B

    inside: inside(A) AND outside(B)
    sdf = max(sdfA, -sdfB)
    """

    def __init__(self, a: Surface3D, b: Surface3D):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        a : Surface3D
            始点側の値です。
        b : Surface3D
            終点側の値です。
        """
        self.a = a
        self.b = b

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        return np.maximum(self.a.sdf(x, y, z), -self.b.sdf(x, y, z))


class ComplementSurface(Surface3D):
    """ComplementSurface クラス。
    """
    def __init__(self, a: Surface3D):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        a : Surface3D
            始点側の値です。
        """
        self.a = a

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        return -self.a.sdf(x, y, z)


class XorSurface(Surface3D):
    """Symmetric difference: inside = (A \ B) ∪ (B \ A).

    SDF form:
      union = min(sa, sb)
      inter = max(sa, sb)
      xor   = (A∪B)\(A∩B)  => max(union, -inter)
    """

    def __init__(self, a: Surface3D, b: Surface3D):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        a : Surface3D
            始点側の値です。
        b : Surface3D
            終点側の値です。
        """
        self.a = a
        self.b = b

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        sa = self.a.sdf(x, y, z)
        sb = self.b.sdf(x, y, z)
        u = np.minimum(sa, sb)
        i = np.maximum(sa, sb)
        return np.maximum(u, -i)


def xor(a: Surface3D, b: Surface3D) -> Surface3D:
    """2 つの曲面の XOR 合成曲面を生成する。
    
    Parameters
    ----------
    a : Surface3D
        始点側の値です。
    b : Surface3D
        終点側の値です。
    Returns
    -------
    Surface3D
        処理結果です。
    """
    return XorSurface(a, b)


# ============================================================
# Transforms (offset / shift / rotation)
# ============================================================


class OffsetSurface(Surface3D):
    """Geometric offset of the implicit function.

    sdf_offset = sdf_base - delta

    delta>0 makes the inside region larger (heuristically) if base.sdf behaves
    like a signed distance.
    """

    def __init__(self, base: Surface3D, delta: float):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        base : Surface3D
            基準となる値です。
        delta : float
            オフセット量です。`base.sdf - delta` として適用されます。
        """
        self.base = base
        self.delta = float(delta)

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self.base.sdf(x, y, z) - self.delta


class TransformSurface(Surface3D):
    """Coordinate transform wrapper.

    Provide inverse transform: (x,y,z) -> (x',y',z') in base-surface space.
    """

    def __init__(
        self,
        base: Surface3D,
        inv_transform: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        base : Surface3D
            基準となる値です。
        inv_transform : Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
            コールバック関数です。
        """
        self.base = base
        self.inv_transform = inv_transform

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        xp, yp, zp = self.inv_transform(np.asarray(x), np.asarray(y), np.asarray(z))
        return self.base.sdf(xp, yp, zp)


class ShiftSurface(Surface3D):
    """Translate (shift) a surface by (dx,dy,dz).

    sdf_shifted(p) = sdf_base(p - d)
    """

    def __init__(self, base: Surface3D, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        base : Surface3D
            基準となる値です。
        dx : float, optional
            x 方向の格子間隔です。
        dy : float, optional
            y 方向の格子間隔です。
        dz : float, optional
            z 方向の格子間隔です。
        """
        self.base = base
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        x = np.asarray(x, dtype=np.float64) - self.dx
        y = np.asarray(y, dtype=np.float64) - self.dy
        z = np.asarray(z, dtype=np.float64) - self.dz
        return self.base.sdf(x, y, z)


class RotateSurface(Surface3D):
    """Rotate a surface around an origin by Euler angles (rx,ry,rz).

    For SDF evaluation, apply inverse rotation to query point:
        sdf_rot(p) = sdf_base(R^{-1} (p - origin) + origin)

    order="xyz" means extrinsic rotations about x then y then z.
    """

    def __init__(
        self,
        base: Surface3D,
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        *,
        degrees: bool = True,
        order: str = "xyz",
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        base : Surface3D
            基準となる値です。
        rx : float, optional
            x 軸周りの回転角です。
        ry : float, optional
            y 軸周りの回転角です。
        rz : float, optional
            z 軸周りの回転角です。
        degrees : bool, optional
            `True` なら角度を度数法、`False` ならラジアンとして解釈します。
        order : str, optional
            回転適用順序（例: `xyz`）です。
        origin : Tuple[float, float, float], optional
            回転・平行移動の基準点です。
        """
        self.base = base
        self.origin = np.asarray(origin, dtype=np.float64)

        if degrees:
            rx, ry, rz = np.deg2rad([rx, ry, rz])

        order = order.lower().strip()
        if set(order) != set("xyz") or len(order) != 3:
            raise ValueError("order must be a permutation of 'xyz' (e.g. 'xyz', 'zxy').")
        self.order = order

        def Rx(a):
            """x 軸回転行列を生成する。
            
            Parameters
            ----------
            a : object
                始点側の値です。
            Returns
            -------
            object
                処理結果です。
            """
            c, s = np.cos(a), np.sin(a)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

        def Ry(a):
            """y 軸回転行列を生成する。
            
            Parameters
            ----------
            a : object
                始点側の値です。
            Returns
            -------
            object
                処理結果です。
            """
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

        def Rz(a):
            """z 軸回転行列を生成する。
            
            Parameters
            ----------
            a : object
                始点側の値です。
            Returns
            -------
            object
                処理結果です。
            """
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

        ang = {"x": float(rx), "y": float(ry), "z": float(rz)}
        rot = {"x": Rx, "y": Ry, "z": Rz}

        R = np.eye(3, dtype=np.float64)
        for ax in order:
            R = rot[ax](ang[ax]) @ R

        self.R = R
        self.Rinv = R.T

    def sdf(self, x, y, z):
        """符号付き距離関数の値を計算する。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        y : object
            y 座標または y 成分。
        z : object
            z 座標または z 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        X = x - self.origin[0]
        Y = y - self.origin[1]
        Z = z - self.origin[2]

        x0 = self.Rinv[0, 0] * X + self.Rinv[0, 1] * Y + self.Rinv[0, 2] * Z
        y0 = self.Rinv[1, 0] * X + self.Rinv[1, 1] * Y + self.Rinv[1, 2] * Z
        z0 = self.Rinv[2, 0] * X + self.Rinv[2, 1] * Y + self.Rinv[2, 2] * Z

        x0 += self.origin[0]
        y0 += self.origin[1]
        z0 += self.origin[2]

        return self.base.sdf(x0, y0, z0)
