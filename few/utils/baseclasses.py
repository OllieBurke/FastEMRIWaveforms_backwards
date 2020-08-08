from abc import ABC
import warnings

try:
    import cupy as xp
except:
    import numpy as xp

import numpy as np
from scipy.interpolate import CubicSpline
from scipy import constants as ct

from few.utils.constants import *
from few.utils.citations import *


class SchwarzschildEccentric(ABC):
    """Base class for Schwarzschild eccentric waveforms.

    This class creates shared traits between different implementations of the
    same model. Particularly, this class includes descriptive traits as well as
    the sanity check class method that should be used in all implementations of
    this model. This method can be overwritten if necessary. Here we describe
    the overall qualities of this base class.


    In this limit, Eq. :eq:`emri_wave_eq` is reduced to the equatortial plane
    with no spin. Therefore, we only concerned with :math:`(l,m,n)` indices and
    the parameters :math:`(p,e)` because :math:`k=a=\iota=0`. Therefore, in this
    model we calculate :math:`A_{lmn}` and :math:`\Phi_{mn}=m\Phi_\phi+n\Phi_r`.
    This also allows us to use -2 spin-weighted spherical harmonics
    (:math:`(s=-2)Y_{l,m}`) in place of the more generic angular function from
    Eq. :eq:`emri_wave_eq`.

    :math:`l` ranges from 2 to 10; :math:`m` from :math:`-l` to :math:`l`;
    and :math:`n` from -30 to 30. This is for Schwarzschild eccentric.
    The model validity ranges from :math:`0.1 \leq e_0 \leq 0.7` and
    :math:`10 \leq p_0 \leq 16 + 2*e_0`. The user can start at any :math:`p` and
    :math:`e` combination that exists under the :math:`p_0=10, e_0=0.7`
    trajectory within those bounds (and is outside of the separatrix). **Important Note**: if the trajectory is
    within the bounds listed, but it is above :math:`p_0=10, e_0=0.7` trajectory.,
    the user may not receive an error. See the documentation introduction for
    more information on this.

    args:
        use_gpu (bool, optional): If True, will allocate arrays on the GPU.
            Default is False.

    """

    def attributes_SchwarzschildEccentric(self):
        """
        attributes:
            xp (module): numpy or cupy based on hardware chosen.
            background (str): Spacetime background for this model.
            descriptor (str): Short description for model validity.
            num_modes, num_teuk_modes (int): Total number of Tuekolsky modes
                in the model.
            lmax, nmax (int): Maximum :math:`l`, :math:`n`  values
            ndim (int): Dimensionality in terms of orbital parameters and phases.
            m0sort (1D int xp.ndarray): array of indices to sort accoring to
                :math:`(m=0)` parts first and then :math:`m>0` parts.
            m0mask (1D bool xp.ndarray): Masks values with :math:`m==0`.
            m_zero_up_mask (1D bool xp.ndarray): Masks values with :math:`m<1`.
            l_arr, m_arr, n_arr (1D int xp.ndarray): :math:`(l,m,n)` arrays
                containing indices for each mode.
            lmn_indices (dict): Dictionary mapping a tuple of :math:`(l,m,n)` to
                the respective index in l_arr, m_arr, and n_arr.
            num_m_zero_up (int): Number of modes with :math:`m\geq0`.
            num_m0 (int): Number of modes with :math:`m=0`.
            num_m_1_up (int): Number of modes with :math:`m\geq1`.
            unique_l, unique_m (1D int xp.ndarray): Arrays of unique :math:`l` and
                :math:`m` values.
            inverse_lm (1D int xp.ndarray): Array of indices that expands unique
                :math:`(l, m)` values to the full array of :math:`(l,m,n)` values.
            index_map (dict): Dictionary mapping the location of the `(l,m,n)`
                indices back to there spot in l_arr, m_arr, n_arr.
            special_index_map (dict): Dictionary mapping the location of the `(l,m,n)`
                indices back to there spot in l_arr, m_arr, n_arr. However, this
                maps locations of -m values to +m values.

        """
        pass

    def __init__(self, use_gpu=False, **kwargs):

        self.use_gpu = use_gpu
        if use_gpu is True:
            self.xp = xp
        else:
            self.xp = np
        self.background = "Schwarzschild"
        self.descriptor = "eccentric"

        self.lmax = 10
        self.nmax = 30

        self.ndim = 2

        md = []

        for l in range(2, self.lmax + 1):
            for m in range(0, l + 1):
                for n in range(-self.nmax, self.nmax + 1):
                    md.append([l, m, n])

        self.num_modes = self.num_teuk_modes = len(md)

        m0mask = self.xp.array(
            [
                m == 0
                for l in range(2, 10 + 1)
                for m in range(0, l + 1)
                for n in range(-30, 30 + 1)
            ]
        )

        self.m0sort = m0sort = self.xp.concatenate(
            [
                self.xp.arange(self.num_teuk_modes)[m0mask],
                self.xp.arange(self.num_teuk_modes)[~m0mask],
            ]
        )

        md = self.xp.asarray(md).T[:, m0sort].astype(self.xp.int32)

        self.l_arr, self.m_arr, self.n_arr = md[0], md[1], md[2]

        try:
            self.lmn_indices = {tuple(md_i): i for i, md_i in enumerate(md.T.get())}

        except AttributeError:
            self.lmn_indices = {tuple(md_i): i for i, md_i in enumerate(md.T)}

        self.m0mask = self.m_arr != 0
        self.num_m_zero_up = len(self.m_arr)
        self.num_m0 = len(self.xp.arange(self.num_teuk_modes)[m0mask])

        self.num_m_1_up = self.num_m_zero_up - self.num_m0
        self.l_arr = self.xp.concatenate([self.l_arr, self.l_arr[self.m0mask]])
        self.m_arr = self.xp.concatenate([self.m_arr, -self.m_arr[self.m0mask]])
        self.n_arr = self.xp.concatenate([self.n_arr, self.n_arr[self.m0mask]])

        self.m_zero_up_mask = self.m_arr >= 0

        try:
            temp, self.inverse_lm = np.unique(
                np.asarray([self.l_arr.get(), self.m_arr.get()]).T,
                axis=0,
                return_inverse=True,
            )

        except AttributeError:
            temp, self.inverse_lm = np.unique(
                np.asarray([self.l_arr, self.m_arr]).T, axis=0, return_inverse=True
            )

        self.unique_l, self.unique_m = self.xp.asarray(temp).T
        self.num_unique_lm = len(self.unique_l)

        self.index_map = {}
        self.special_index_map = {}  # maps the minus m values to positive m
        for i, (l, m, n) in enumerate(zip(self.l_arr, self.m_arr, self.n_arr)):

            try:
                l = l.item()
                m = m.item()
                n = n.item()

            except AttributeError:
                pass

            self.index_map[(l, m, n)] = i
            self.special_index_map[(l, m, n)] = (
                i if i < self.num_modes else i - self.num_m_1_up
            )

    @property
    def citation(self):
        return few_citation

    def sanity_check_viewing_angles(self, theta, phi):
        """Sanity check on viewing angles.

        Make sure parameters are within allowable ranges.

        args:
            theta (double): Polar viewing angle.
            phi (double): Azimuthal viewing angle.

        Returns:
            tuple: (theta, phi). Phi is wrapped.

        Raises:
            ValueError: If any of the trajectory points are not allowed.

        """
        if theta < 0.0 or theta > np.pi:
            raise ValueError("theta must be between 0 and pi.")

        phi = phi % (2 * np.pi)
        return (theta, phi)

    def sanity_check_traj(self, p, e):
        """Sanity check on parameters output from thte trajectory module.

        Make sure parameters are within allowable ranges.

        args:
            p (1D np.ndarray): Array of semi-latus rectum values produced by
                the trajectory module.
            e (1D np.ndarray): Array of eccentricity values produced by
                the trajectory module.

        Raises:
            ValueError: If any of the trajectory points are not allowed.
            warn: If any points in the trajectory are allowable,
                but outside calibration region.

        """

        if np.any(e < 0.0):
            raise ValueError("Members of e array are less than zero.")

        if np.any(p < 0.0):
            raise ValueError("Members of p array are less than zero.")

    def sanity_check_init(self, M, mu, p0, e0):
        """Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            M (double): Massive black hole mass in solar masses.
            mu (double): compact object mass in solar masses.
            p0 (double): Initial semilatus rectum (dimensionless)
                :math:`(10\leq p_0\leq 16 + 2e_0)`. See the documentation for
                more information on :math:`p_0 \leq 10.0`.
            e0 (double): Initial eccentricity :math:`(0\leq e_0\leq0.7)`.

        Raises:
            ValueError: If any of the parameters are not allowed.

        """

        for val, key in [[M, "M"], [p0, "p0"], [e0, "e0"], [mu, "mu"]]:
            test = val < 0.0
            if test:
                raise ValueError("{} is negative. It must be positive.".format(key))

        if e0 > 0.7:
            raise ValueError(
                "Initial eccentricity above 0.7 not allowed. (e0={})".format(e0)
            )

        if e0 < 0.0:
            raise ValueError(
                "Initial eccentricity below 0.0 not physical. (e0={})".format(e0)
            )

        if mu / M > 1e-4:
            warnings.warn(
                "Mass ratio is outside of generally accepted range for an extreme mass ratio (1e-4). (q={})".format(
                    mu / M
                )
            )

        if p0 < 10.0:
            warnings.warn(
                "With p0 = {} < 10, the user must be careful the trajectory does not lie above the p0=10 e0=0.7 curve.".format(
                    p0
                )
            )

        if p0 > 16.0 + 2 * e0:
            raise ValueError(
                "Initial p0 is too large (p0={}). Must be 10 <= p0 <= 16 + 2 * e.".format(
                    p0
                )
            )


class TrajectoryBase(ABC):
    """Base class used for trajectory modules.

    This class provides a flexible interface to various trajectory
    implementations. Specific arguments to each trajectory can be found with
    each associated trajectory module discussed below.

    """

    def __init__(self, *args, **kwargs):
        pass

    @property
    def citation(self):
        return few_citation

    @classmethod
    def get_inspiral(self, *args, **kwargs):
        """Inspiral Generator

        @classmethod that requires a child class to have a get_inspiral method.

        returns:
            2D double np.ndarray: t, p, e, Phi_phi, Phi_r, flux with shape:
                (params, traj length).

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    def __call__(
        self,
        *args,
        in_coordinate_time=True,
        dt=10.0,
        T=1.0,
        new_t=None,
        spline_kwargs={},
        DENSE_STEPPING=0,
        max_init_len=1000,
        upsample=False,
        err=1e-10,
        use_rk4=False,
        fix_t=False,
        **kwargs
    ):
        """Call function for trajectory interface.

        This is the function for calling the creation of the
        trajectory. Inputs define the output time spacing.

        args:
            *args (list): Input of variable number of arguments specific to the
                inspiral model (see the trajectory class' `get_inspiral` method).
                **Important Note**: M must be the first parameter of any model
                that uses this base class.
            in_coordinate_time (bool, optional): If True, the trajectory will be
                outputted in coordinate time. If False, the trajectory will be
                outputted in units of M. Default is True.
            dt (double, optional): Time step for output waveform in seconds. Also sets
                initial step for integrator. Default is 10.0.
            T (double, optional): Total observation time in years. Sets the maximum time
                for the integrator to run. Default is 1.0.
            new_t (1D np.ndarray, optional): If given, this represents the final
                time array at which the trajectory is analyzed. This is
                performed by using a cubic spline on the integrator output.
                Default is None.
            spline_kwargs (dict, optional): If using upsampling, spline_kwargs
                provides the kwargs desired for scipy.interpolate.CubicSpline.
                Default is {}.
            DENSE_STEPPING (int, optional): If 1, the trajectory used in the
                integrator will be densely stepped at steps of :obj:`dt`. If 0,
                the integrator will determine its stepping. Default is 0.
            max_init_len (int, optional): Sets the allocation of memory for
                trajectory parameters. This should be the maximum length
                expected for a trajectory. Trajectories with default settings
                will be ~100 points. Default is 1000.
            upsample (bool, optional): If True, upsample, with a cubic spline,
                the trajectories from 0 to T in steps of dt. Default is False.
            err (double, optional): Tolerance for integrator. Default is 1e-10.
                Decreasing this parameter will give more steps over the
                trajectory, but if it is too small, memory issues will occur as
                the trajectory length will blow up. We recommend not adjusting
                this parameter.
            fix_T (bool, optional): If upsampling, this will affect excess
                points in the t array. If True, it will shave any excess on the
                trajectories arrays where the time is greater than the overall
                time of the trajectory requested.
            use_rk4 (bool, optional): If True, use rk4 integrator from gsl.
                If False, use rk8. Default is False (rk8).
            **kwargs (dict, optional): kwargs passed to trajectory module.
                Default is {}.

        Returns:
            tuple: Tuple of (t, p, e, Phi_phi, Phi_r, flux_norm).

        Raises:
            ValueError: If input parameters are not allowed in this model.

        """

        kwargs["dt"] = dt
        kwargs["T"] = T
        kwargs["max_init_len"] = max_init_len
        kwargs["err"] = err
        kwargs["DENSE_STEPPING"] = DENSE_STEPPING
        kwargs["use_rk4"] = use_rk4

        T = T * ct.Julian_year

        out = self.get_inspiral(*args, **kwargs)

        t = out[0]
        params = out[1:]

        if in_coordinate_time is False:
            M = args[0]
            Msec = M * MTSUN_SI
            t = t / Msec

        if not upsample:
            return (t,) + params

        splines = [CubicSpline(t, temp, **spline_kwargs) for temp in list(params)]

        if new_t is not None:
            if isinstance(new_t, np.ndarray) is False:
                raise ValueError("new_t parameter, if provided, must be numpy array.")

        else:
            new_t = np.arange(0.0, T + dt, dt)

        if new_t[-1] > t[-1]:
            if fix_t:
                new_t = new_t[new_t <= t[-1]]
            else:
                warnings.warn(
                    "new_t array goes beyond generated t array. If you want to cut the t array at the end of the trajectory, set fix_t to True."
                )

        out = tuple([spl(new_t) * (new_t < t[-1]) for spl in splines])
        return (new_t,) + out


class SummationBase(ABC):
    """Base class used for summation modules.

    This class provides a common flexible interface to various summation
    implementations. Specific arguments to each summation module can be found
    with each associated module discussed below.

    args:
        pad_output (bool, optional): Add zero padding to the waveform for time
            between plunge and observation time. Default is False.

    """

    def __init__(self, *args, pad_output=False, **kwargs):
        self.pad_output = pad_output

    def attributes_SummationBase(self):
        """
        attributes:
            waveform (1D complex128 np.ndarray): Complex waveform given by
                :math:`h_+ + i*h_x`.
        """
        pass

    @property
    def citation(self):
        return few_citation

    @classmethod
    def sum(self, *args, **kwargs):
        """Sum Generator

        @classmethod that requires a child class to have a sum method.

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    def __call__(self, t, teuk_modes, ylms, dt, T, *args, **kwargs):
        """Common call function for summation modules.

        Provides a common interface for summation modules. It can adjust for
        more dimensions in a model.

        args:
            t (1D double xp.ndarray): Array of t values.
            teuk_modes (2D double xp.array): Array of complex amplitudes.
                Shape: (len(t), num_teuk_modes).
            ylms (1D complex128 xp.ndarray): Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0 first then) m>0 then m<0.
            dt (double): Time spacing between observations in seconds (inverse of sampling
                rate).
            T (double): Maximum observing time in years.
            *args (list): This should be a tuple of phases combined with mode
                index arrays. For equatorial, :math:`(\Phi_\phi, \Phi_r, m, n)`.
                For generic, :math:`(\Phi_\phi, \Phi_\Theta, \Phi_r, m, k, n)`.
            **kwargs (dict, placeholder): Added for future flexibility.

        """

        if T < t[-1].item():
            num_pts = int((T - t[0]) / dt) + 1
            num_pts_pad = 0

        else:
            num_pts = int((t[-1] - t[0]) / dt) + 1
            if self.pad_output:
                num_pts_pad = int((T - t[0]) / dt) + 1 - num_pts
            else:
                num_pts_pad = 0

        self.num_pts, self.num_pts_pad = num_pts, num_pts_pad
        self.dt = dt
        init_len = len(t)
        num_teuk_modes = teuk_modes.shape[1]

        self.waveform = self.xp.zeros(
            (self.num_pts + self.num_pts_pad,), dtype=self.xp.complex128
        )

        args_in = (t, teuk_modes, ylms) + args + (init_len, num_pts, num_teuk_modes, dt)
        self.sum(*args_in)

        return self.waveform


class AmplitudeBase(ABC):
    """Base class used for amplitude modules.

    This class provides a common flexible interface to various amplitude
    implementations. Specific arguments to each amplitude module can be found
    with each associated module discussed below.

    args:
        pad_output (bool, optional): Add zero padding to the waveform for time
            between plunge and observation time. Default is False.

    """

    def __init__(self, **kwargs):
        pass

    @classmethod
    def get_amplitudes(self, *args, **kwargs):
        """Amplitude Generator

        @classmethod that requires a child class to have a get_amplitudes method.

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    @property
    def citation(self):
        return few_citation

    def __call__(self, p, e, *args, specific_modes=None, **kwargs):
        """Common call for Teukolsky amplitudes

        This function takes the inputs the trajectory in :math:`(p,e)` as arrays
        and returns the complex amplitude of all modes to adiabatic order at
        each step of the trajectory.

        args:
            p (1D double numpy.ndarray): Array containing the trajectory for values of
                the semi-latus rectum.
            e (1D double numpy.ndarray): Array containing the trajectory for values of
                the eccentricity.
            *args (tuple, placeholder): Added to create future flexibility when calling different
                amplitude modules.
            specific_modes (list, optional): List of tuples for (l, m, n) values
                desired modes. Default is None.
            **kwargs (dict, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.

        returns:
            2D array (double): If specific_modes is None, Teukolsky modes in shape (number of trajectory points, number of modes)
            dict: Dictionary with requested modes.


        """

        return self.get_amplitudes(p, e, *args, specific_modes=specific_modes, **kwargs)
