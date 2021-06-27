import numpy as np
import os, sys
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar


class NonLinPowerSpecCDM:
    def __init__(self, Omega_m=0.3):
        self.Om_m = Omega_m

    def set_Del2L(self, Del2L):
        self.Del2L = Del2L
    def set_Del2L_interpolate(self, k, pLk):
        self.Del2L = interp1d(k, k**3*pLk/ (2*np.pi**2), bounds_error=False, fill_value=0)
        self.k_range = np.array([min(k), max(k)])

    def _get_integ_sig(self, R):
        return lambda lnk: self.Del2L(np.exp(lnk)) * np.exp(-np.exp(2*lnk)*R**2)

    # def _integ_sig_2(R):
    #         k = np.exp(lnk) 
    #         return self.Del2L(k)* np.exp(-k**2*R**2)

    def sig2(self, R):
        return integrate.quad(self._get_integ_sig(R), *np.log(self.k_range))[0]

    def solve_k_sig(self):
        self.k_sig = root_scalar(lambda k: self.sig2(1/k)-1, bracket=self.k_range.tolist()).root

    def compute_neff(self):
        R = self.k_sig**-1
        def integ_sig_1(lnk):
            k = np.exp(lnk)
            return self.Del2L(np.exp(lnk)) * np.exp(-np.exp(2*lnk)*R**2) * (-2)*(k*R)**2

        self.neff = - integrate.quad(integ_sig_1, *np.log(self.k_range))[0] - 3

    # def C_fn(self,R):

    def compute_C(self):
        R = 1/self.k_sig
        def integ_sig_2(lnk):
            k = np.exp(lnk)
            return self.Del2L(np.exp(lnk)) * np.exp(-np.exp(2*lnk)*R**2) * 4*((k*R)**4 - (k*R)**2)

        # d2sig2_dlnR2 = integrate.quad(integ_sig_2, -np.inf, np.inf)
        self.C = - integrate.quad(integ_sig_2, *np.log(self.k_range))[0] + (self.neff+3)**2

    def compute_params(s, param_from='takahashi'):
        s.solve_k_sig()
        s.compute_neff()
        s.compute_C()

        if param_from=='takahashi':
            s.a_n = 10**(1.5222 + 2.8553*s.neff + 2.3706*s.neff**2 + 0.9903*s.neff**3 + 0.2250*s.neff**4 - 0.6038*s.C)
            s.b_n = 10**(-0.5642 + 0.5864*s.neff + 0.5716*s.neff**2 - 1.5474*s.C)
            s.c_n = 10**(0.3698 + 2.0404*s.neff + 0.8161*s.neff**2 + 0.5869*s.C)

            s.gamma_n = 0.1971 - 0.0843*s.neff + 0.8460*s.C
            s.alpha_n = abs(6.0835 + 1.3373*s.neff - 0.1959*s.neff**2 - 5.5274*s.C)
            s.beta_n = 2.0379 - 0.7354*s.neff + 0.3157*s.neff**2 + 1.2490*s.neff**3 + 0.3980*s.neff**4 - 0.1682*s.C 

            s.mu_n = 0
            s.nu_n = 10**(5.2105 + 3.6902*s.neff)

        elif param_from=='smith':
            s.a_n = 10**(1.4861 + 1.8369*s.neff + 1.6762*s.neff**2 + 0.7940*s.neff**3 + 0.1670*s.neff**4 - 0.6206*s.C)
            s.b_n = 10**(0.9463 + 0.9466*s.neff + 0.3084*s.neff**2 - 0.9400*s.C)
            s.c_n = 10**(-0.2807 + 0.6669*s.neff + 0.3214*s.neff**2 - 0.0793*s.C)

            s.gamma_n = 0.8649 + 0.2989*s.neff + 0.1631*s.C
            s.alpha_n = 1.3884 + 0.3700*s.neff - 0.1452*s.neff**2
            s.beta_n = 0.8291 + 0.9854*s.neff + 0.3401*s.neff**2 

            s.mu_n = 10**(-3.5442 + 0.1908*s.neff)
            s.nu_n = 10**(0.9589 + 1.2857*s.neff)

    def f(self, y): return y/4 + y**2/8

    def f1(self, om): return om**-0.0307
    def f2(self, om): return om**-0.0585
    def f3(self, om): return om**0.0743

    def Del2Q(s, k):
        y = k/s.k_sig
        frac = (1 + s.Del2L(k) )**s.beta_n / (1 + s.alpha_n*s.Del2L(k) )
        return s.Del2L(k) * frac * np.exp(-s.f(y))

    def Del2pH(s, y):
        # y = k/s.k_sig
        return s.a_n * y**(3*s.f1(s.Om_m)) / (1 + s.b_n * y**s.f2(s.Om_m) + (s.c_n*s.f3(s.Om_m)*y)**(3-s.gamma_n) )

    def Del2H(s, k):
        y = k/s.k_sig
        return s.Del2pH(y)/ (1 + s.mu_n * y**-1 + s.nu_n*y**-2)

    def Del2(self, k): return self.Del2Q(k) + self.Del2H(k)

    def P(self, k): return self.Del2(k)/k**3 * (2*np.pi**2)

    



    




