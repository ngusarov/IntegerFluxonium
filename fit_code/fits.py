import sys
import warnings
import traceback
import numpy as np
from scipy.optimize import curve_fit, minimize, least_squares


class Transform(object):
    """comment by Taketo
    Class for transform the complex array data.
    Assuming all the data points are on the same line on IQplane.
    When "transform_type"='optimal', all the data will be rotated to real axis.
    """
    def __init__(self, time, data, transform_type='optimal',
            params=None, deg=True, fixed_phi=None):
        self._time = time
        self._data = data

        transform_type = str(transform_type).lower()
        if transform_type.startswith('op'):
            self._transform_type = 'optm'
        elif transform_type.startswith('pr'):
            self._transform_type = 'optm'
        elif transform_type.startswith('translation+rotation'):
            self._transform_type = 'trrt'
        elif transform_type.startswith('re'):
            self._transform_type = 'real'
        elif transform_type.startswith('qu'):
            self._transform_type = 'real'
        elif transform_type.startswith('im'):
            self._transform_type = 'imag'
        elif transform_type.startswith('in'):
            self._transform_type = 'imag'
        elif transform_type.startswith('am'):
            self._transform_type = 'ampl'
        elif transform_type.startswith('ph'):
            self._transform_type = 'angl'
        elif transform_type.startswith('an'):
            self._transform_type = 'angl'

        if self._transform_type == 'optm' and params is None and fixed_phi == None:
            params = self._opt_transform(data)
        elif self._transform_type == 'optm' and params is None and fixed_phi != None:
            params = self._opt_transform(data)
            params[2] = fixed_phi
        self._params = params

        if self._transform_type in ['optm', 'trrt']:
            transformed_data = self._transform(data, *params)
            self._tranformed = transformed_data.real
            self._residual = transformed_data.imag
        elif self._transform_type == 'real':
            self._tranformed = data.real
            self._residual = data.imag
        elif self._transform_type == 'imag':
            self._tranformed = data.imag
            self._residual = data.real
        elif self._transform_type == 'ampl':
            self._tranformed = np.abs(data)
            self._residual = np.unwrap(np.angle(data))
            if deg:
                self._residual = 180. * self._residual / np.pi
        elif self._transform_type == 'angl':
            self._tranformed = np.unwrap(np.angle(data))
            self._residual = np.abs(data)
            if deg:
                self._tranformed = 180. * self._tranformed / np.pi

    @staticmethod
    def _transform(data, x0, y0, phi):
        return (data - x0 - 1.j * y0) * np.exp(1.j * phi)

    @staticmethod
    def _inv_transform(data, x0, y0, phi):
        return data * np.exp(-1.j * phi) + x0 + 1.j * y0

    def _opt_transform(self, data):
        def _transform_err(x):
            return np.sum((self._transform(data, x[0], x[1], x[2]).imag)**2.)
        res = minimize(fun=_transform_err,
                method='Nelder-Mead',
                x0=[np.mean(data.real), np.mean(data.imag),
                    -np.arctan2(np.std(data.imag), np.std(data.real))],
                options={'maxiter': 1000})

        params = res.x
        transformed_data = self._transform(data, *res.x)
        if transformed_data[0] < transformed_data[-1]:
            params[2] += np.pi
        return params

    # def _opt_transform(self, data):
        # """
        # find angle by PCO analysis which is mathmatically equal to diagonalization
        # """
        # covariance_matrix = np.cov(np.real(data), np.imag(data))
        # eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # first_vector = eigenvectors[list(eigenvalues).index(max(eigenvalues))]
        # angle = np.angle(first_vector[0]+1j*first_vector[1])
        # x_0=np.mean(np.real(data))
        # y_0=np.mean(np.imag(data))
        # params=[x_0, y_0, -1*angle]
        # return params
        
    def params(self):
        return self._params
        
    def data(self):
        return self._tranformed
    
    def residual(self):
        return self._residual
        
    def inverse(self, data):
        if self._transform_type in ['optm', 'trrt']:
            return self._inv_transform(data, *self._params)
        raise NotImplementedError()


class DecayingExponent(object):
    """Fit to the decaying exponent function."""
    def __init__(self, time, data):
        self._time = time
        self._data = data
        self._decaying_exp_fit()

    @staticmethod
    def _decaying_exp(x, a, b, c):
        return a * np.exp(-x / b) + c

    def _decaying_exp_fit(self):
        x = self._time
        y = self._data
        
        max_y = np.max(y)
        min_y = np.min(y)
        half = .5 * (max_y + min_y)
        if y[0] > y[-1]:
            b0 = x[np.argmax(y < half)]
        else:
            b0 = x[np.argmax(y > half)]
        if b0 == 0:
            b0 = .5 * (x[0] + x[-1])
        p0 = [y[0] - y[-1], b0, y[-1]]

        span_y = max_y - min_y
        c0_min = min_y - 100. * span_y
        c0_max = max_y + 100. * span_y
        bounds = ([-100. * span_y, 0., c0_min], 
                  [100. * span_y, 100. * (np.max(x) - np.min(x)), c0_max])

        popt, pcov = curve_fit(self._decaying_exp, x, y, p0, bounds=bounds)
        perr = np.sqrt(np.abs(np.diag(pcov)))
       
        self._popt = popt
        self._perr = perr
        
    def time_constant(self):
        return self._popt[1]
        
    def absolute_error(self):
        return self._perr[1]
    
    def relative_error(self):
        return self._perr[1] / self._popt[1]
    
    def time(self):
        return self._time
        
    def data(self):
        return self._data
        
    def amplitude(self):
        return self._popt[0]
        
    def offset(self):
        return self._popt[2]
        
    def fit(self, time=None):
        if time is None:
           time = self._time
        return self._decaying_exp(time, *self._popt)


class DoubleExponent(object):
    """Fit to the decaying exponent function."""
    def __init__(self, time, data):
        self._time = time
        self._data = data
        self._double_exp_fit()

    @staticmethod
    def _double_exp(x, a, b, c, d, e):
        return (a * np.exp(np.abs(e) * (np.exp(-x / d) - 1))
                  * np.exp(-x / b)) + c

    def _double_exp_fit(self):
        x = self._time
        y = self._data
        
        fit_single_exp = DecayingExponent(x, y)
        
        a = fit_single_exp.amplitude()
        T1 = fit_single_exp.time_constant()
        c = fit_single_exp.offset() 

        p0 = [a * np.exp(1.), 2. * T1, c, .1 * T1, 1.]
        bounds = ([-20. * np.abs(a), 1.e-1 * T1, -10. * np.abs(c),
                   1.e-4 * T1, 0.],
                  [ 20. * np.abs(a), 1.e3 * T1, 10. * np.abs(c),
                   10. * T1, 1.e3])

        popt, pcov = curve_fit(self._double_exp, x, y, p0, bounds=bounds,
                maxfev=10000, ftol=1e-11)
        perr = np.sqrt(np.abs(np.diag(pcov)))

        self._popt = popt
        self._perr = perr

    def T1R(self):
        return self._popt[1]

    def T1R_absolute_error(self):
        return self._perr[1]

    def T1R_relative_error(self):
        return self._perr[1] / self._popt[1]

    def T1QP(self):
        return self._popt[3]

    def T1QP_absolute_error(self):
        return self._perr[3]

    def T1QP_relative_error(self):
        return self._perr[3] / self._popt[3]

    def nQP(self):
        return np.abs(self._popt[4])

    def nQP_absolute_error(self):
        return self._perr[4]

    def nQP_relative_error(self):
        return self._perr[4] / np.abs(self._popt[4])

    def time(self):
        return self._time

    def data(self):
        return self._data

    def fit(self, time=None):
        if time is None:
           time = self._time
        return self._double_exp(time, *self._popt)


class DecayingOscillations(object):
    """Fit to the decaying oscillations."""
    def __init__(self, time, data):
        self._time = time
        self._data = data
        self._decaying_oscs_fit()

    @staticmethod
    def _decaying_oscs(x, a, b, c, d, e):
        return a * np.exp(-x / b) * np.cos(2. * np.pi * (x - d) / e) + c

    def _decaying_oscs_fit(self):
        x = self._time
        y = self._data

        min_y = np.min(y)
        max_y = np.max(y)
        half = .5 * (max_y + min_y)
        period = 0.5#2. * np.abs(x[np.argmax(y)] - x[np.argmin(y)])
        span_x = np.max(x) - np.min(x)
        
        perr_best = np.array([np.nan] * 5)
        popt_best = np.array([np.nan] * 5)
        for d0 in np.linspace(0., np.pi * period, 10):
            for factor in [y[-1], np.mean(y)]: 
                p0 = [y[0] - y[-1], span_x, factor, d0, period]

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        popt, pcov = curve_fit(self._decaying_oscs, x, y, p0)
                    perr = np.sqrt(np.abs(np.diag(pcov)))
                    if (not np.isfinite(perr_best[1])
                            or perr_best[1] / np.abs(popt_best[1])
                                  > perr[1] / np.abs(popt[1])):
                        popt_best = popt
                        perr_best = perr
                except:
                    if not np.isfinite(perr_best[1]):
                        # exc_type, exc_value, exc_traceback = sys.exc_info()
                        # traceback.print_exception(exc_type, exc_value,
                                # exc_traceback, file=sys.stdout)

                        def _decaying_oscs_res(p, x, y):
                            (a, b, c, d, e) = p
                            return self._decaying_oscs(x, a, b, c, d, e) - y
                        popt_best = least_squares(_decaying_oscs_res, p0,
                                loss='soft_l1', f_scale=0.1, args=(x, y)).x

        pi_opt = .5 * popt_best[4] + popt_best[3] # half period + offset phase
        while pi_opt > .75 * np.abs(popt_best[4]):
            pi_opt -= .5 * np.abs(popt_best[4])
        while pi_opt < .25 * np.abs(popt_best[4]):
            pi_opt += .5 * np.abs(popt_best[4])
        pi_err = np.abs(np.hypot(perr_best[4] / 2., perr_best[3]))
        
        self._pi_opt = pi_opt
        self._pi_err = pi_err 
       
        self._popt = popt_best
        self._perr = perr_best
    def rabi_amp(self): #include + or -!!!!!
        return self._popt[0]

    def rabi_amp_absolute_error(self):
        return self._perr[0]

    def rabi_amp_relative_error(self):
        return self._perr[0] / self._popt[0]
        
    def time_constant(self):
        return self._popt[1]

    def absolute_error(self):
        return self._perr[1]

    def relative_error(self):
        return self._perr[1] / self._popt[1]
        
    def time_pi(self):
        return self._pi_opt
        
    def time_pi_absolute_error(self):
        return self._pi_err
        
    def time_pi_relative_error(self):
        return self._pi_err / self._pi_opt
        
    def frequency(self):
        return 1. / self._popt[4]
        
    def frequency_absolute_error(self):
        return self._perr[4] / self._popt[4]**2.
        
    def frequency_relative_error(self):
        return self._perr[4] / self._popt[4]

    def time(self):
        return self._time

    def data(self):
        return self._data

    def fit(self, time=None):
        if time is None:
           time = self._time
        return self._decaying_oscs(time, *self._popt)
        

class Lorentzian(object):
    """Fit to the decaying exponent function."""
    def __init__(self, frequency, data):
        self._freq = frequency
        self._data = data
        self._lorentzian_fit()

    @staticmethod
    def _lorentzian(x, a, b, c, d):
        return a * (np.abs(c) / 2.) / ((x - b)**2. + c**2. / 4.) + d

    def _lorentzian_fit(self):
        x = self._freq
        y = self._data
        
        median_y = np.median(y)
        max_x = np.max(x)
        min_x = np.min(x)
        max_y = np.max(y)
        min_y = np.min(y)
        if max_y - median_y >= median_y - min_y:
            d0 = min_y
            idx = np.argmax(y)
            a0 = 1 / (max_y - median_y)
        else:
            d0 = max_y
            idx = np.argmin(y)
            a0 = 1 / (min_y - median_y)
        b0 = x[idx]
        half = d0 + a0 / 2.
        dx = np.abs(np.diff(x[np.argsort(np.abs(y - half))]))
        dx_min = np.abs(np.diff(x))
        dx = dx[dx >= 2. * dx_min]
        if dx.size:
            c0 = dx[0] / 2.
        else:
            c0 = dx_min
        p0 = [a0, b0, c0, d0]

        bounds=([-5. * np.abs(a0), min_x, c0 / 100., min_y], 
                [ 5. * np.abs(a0), max_x, 10. * c0, max_y])
        try:
            popt, pcov = curve_fit(self._lorentzian, x, y, p0=p0, ) #bounds=bounds) #edit mnmuelle
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError:
            popt = p0
            perr = p0

        self._popt = popt
        self._perr = perr
        
    def resonance_frequency(self):
        return self._popt[1]
        
    def resonance_frequency_error(self):
        return self._perr[1]
    
    def resonance_relative_error(self):
        return self._perr[1] / self._popt[1]
        
    def fwhm(self):
        return self._popt[2]
        
    def fwhm_error(self):
        return self._perr[2]
    
    def frequency(self):
        return self._frequency
        
    def data(self):
        return self._data
        
    def fit(self, frequency=None):
        if frequency is None:
           frequency = self._frequency
        return self._lorentzian(frequency, *self._popt)





class DecayingOscillations_fixed_freq(object):
    """Fit to the decaying oscillations."""
    def __init__(self, time, data, time_const, xoffset, frequency):
        self._time = time
        self._data = data
        
        self.time_const= time_const
        self.xoffset= xoffset
        self.frequency= frequency
        
        self._decaying_oscs_fit()

    @staticmethod
    def _decaying_oscs(x, a, b, c, d, e):
        return a * np.exp(-x / b) * np.cos(2. * np.pi * (x - d) / e) + c
    

    def _decaying_oscs_fit(self):
        x = self._time
        y = self._data
        min_y = np.min(y)
        max_y = np.max(y)
        half = .5 * (max_y + min_y)
        
        period = 1/self.frequency
        time_const=self.time_const
        xoffset= self.xoffset
        
        span_x = np.max(x) - np.min(x)
        perr_best = np.array([np.nan] * 5)
        popt_best = np.array([np.nan] * 5)
        for factor in [y[-1], np.mean(y)]: 
            
            # p0 = [y[0] - y[-1], span_x, factor, d0, period]
            p0 = [y[0] - y[-1], time_const, factor, xoffset, period]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    
                    fix_decaying_osc= lambda x,a,c: self._decaying_oscs(x, a, time_const, c, xoffset, period)
                    
                    popt, pcov = curve_fit(fix_decaying_osc, x, y, p0)
                
                perr = np.sqrt(np.abs(np.diag(pcov)))
                if (not np.isfinite(perr_best[1])
                        or perr_best[1] / np.abs(popt_best[1])
                              > perr[1] / np.abs(popt[1])):
                    popt_best = popt
                    perr_best = perr

            except:
                if not np.isfinite(perr_best[1]):
                    # exc_type, exc_value, exc_traceback = sys.exc_info()
                    # traceback.print_exception(exc_type, exc_value,
                            # exc_traceback, file=sys.stdout)

                    def _decaying_oscs_res(p, x, y):
                        (a, b, c, d, e) = p
                        return fix_decaying_osc(x, a, c) - y
                    popt_best = least_squares(_decaying_oscs_res, p0,
                            loss='soft_l1', f_scale=0.1, args=(x, y)).x


        pi_opt = .5 * popt_best[4] + popt_best[3] # half period + offset phase
        while pi_opt > .75 * np.abs(popt_best[4]):
            pi_opt -= .5 * np.abs(popt_best[4])
        while pi_opt < .25 * np.abs(popt_best[4]):
            pi_opt += .5 * np.abs(popt_best[4])
        pi_err = np.abs(np.hypot(perr_best[4] / 2., perr_best[3]))
        
        self._pi_opt = pi_opt
        self._pi_err = pi_err 
       
        self._popt = popt_best
        self._perr = perr_best

    def rabi_amp(self): #include + or -!!!!!
        return self._popt[0]

    def rabi_amp_absolute_error(self):
        return self._perr[0]

    def rabi_amp_relative_error(self):
        return self._perr[0] / self._popt[0]
        
    def time_constant(self):
        return self._popt[1]

    def absolute_error(self):
        return self._perr[1]

    def relative_error(self):
        return self._perr[1] / self._popt[1]
        
    def time_pi(self):
        return self._pi_opt
        
    def time_pi_absolute_error(self):
        return self._pi_err
        
    def time_pi_relative_error(self):
        return self._pi_err / self._pi_opt
        
    def frequency(self):
        return 1. / self._popt[4]
        
    def frequency_absolute_error(self):
        return self._perr[4] / self._popt[4]**2.
        
    def frequency_relative_error(self):
        return self._perr[4] / self._popt[4]

    def time(self):
        return self._time

    def data(self):
        return self._data

    def fit(self, time=None):
        if time is None:
           time = self._time
        return self._decaying_oscs(time, *self._popt)