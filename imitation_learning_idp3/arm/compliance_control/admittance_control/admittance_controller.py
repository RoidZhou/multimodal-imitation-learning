import numpy as np


class AdmittanceController:
    def __init__(self, m: float, c: float, k: float, dt=0.001) -> None:
        super().__init__()

        self._m = m
        self._c = c
        self._k = k

        self._dt = dt

        self._x = np.zeros(6)  # [x, y, z, dx, dy, dz]
        self._dx = np.zeros(6)
        self._f0 = np.zeros(6)

    def _mdl_derivatives(self, u: np.ndarray):
        self._dx[:3] = self._x[3: 6]
        self._dx[3: 6] = (u[0: 3] - self._f0[0: 3] - self._c * self._x[3: 6] - self._k * self._x[:3]) / self._m

    def _mdl_integral(self):
        self._x += self._dx * self._dt

    def get_position(self, u: np.ndarray) -> np.ndarray:
        self._mdl_derivatives(u)
        self._mdl_integral()
        return self._x[:3].copy()

    def get_state(self) -> np.ndarray:
        return self._x[:3].copy()

    @property
    def f0(self):
        return self._f0

    @f0.setter
    def f0(self, f0: np.ndarray):
        self._f0[:] = f0
