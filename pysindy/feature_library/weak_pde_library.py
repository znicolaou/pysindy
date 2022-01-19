from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r
from itertools import product as iproduct

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import hyp2f1
from scipy.special import poch
from sklearn import __version__
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary
from pysindy.differentiation import FiniteDifference

# from scipy.integrate import trapezoid


class WeakPDELibrary(BaseFeatureLibrary):
    """Generate a weak formulation library with custom functions and,
       optionally, any spatial derivatives in arbitrary dimensions.

    Parameters
    ----------
    library_functions : list of mathematical functions, optional (default None)
        Functions to include in the library. Each function will be
        applied to each input variable (but not their derivatives)

    derivative_order : int, optional (default 0)
        Order of derivative to take on each input variable,
        can be arbitrary non-negative integer.

    spatiotemporal_grid : np.ndarray (default None)
        The spatiotemporal grid for computing derivatives.
        This variable must be specified with
        at least one dimension corresponding to a temporal grid, so that
        integration by parts can be done in the weak formulation.

    function_names : list of functions, optional (default None)
        List of functions used to generate feature names for each library
        function. Each name function must take a string input (representing
        a variable name), and output a string depiction of the respective
        mathematical function applied to that variable. For example, if the
        first library function is sine, the name function might return
        :math:`\\sin(x)` given :math:`x` as input. The function_names list
        must be the same length as library_functions.
        If no list of function names is provided, defaults to using
        :math:`[ f_0(x),f_1(x), f_2(x), \\ldots ]`.

    interaction_only : boolean, optional (default True)
        Whether to omit self-interaction terms.
        If True, function evaulations of the form :math:`f(x,x)`
        and :math:`f(x,y,x)` will be omitted, but those of the form
        :math:`f(x,y)` and :math:`f(x,y,z)` will be included.
        If False, all combinations will be included.

    include_bias : boolean, optional (default False)
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
        This is hard to do with just lambda functions, because if the system
        is not 1D, lambdas will generate duplicates.

    include_interaction : boolean, optional (default True)
        This is a different than the use for the PolynomialLibrary. If true,
        it generates all the mixed derivative terms. If false, the library
        will consist of only pure no-derivative terms and pure derivative
        terms, with no mixed terms.

    is_uniform : boolean, optional (default True)
        If True, assume the grid is uniform in all spatial directions, so
        can use uniform grid spacing for the derivative calculations.

    K : int, optional (default 100)
        Number of domain centers, corresponding to subdomain squares of length
        Hxt. If K is not
        specified, defaults to 100.

    H_xt : array of floats, optional (default None)
        Half of the length of the square subdomains in each spatiotemporal
        direction. If H_xt is not specified, defaults to H_xt = L_xt / 20,
        where L_xt is the length of the full domain in each spatiotemporal
        direction. If H_xt is specified as a scalar, this value will be applied
        to all dimensions of the subdomains.

    p : int, optional (default 4)
        Positive integer to define the polynomial degree of the spatial weights
        used for weak/integral SINDy.

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library.

    Attributes
    ----------
    functions : list of functions
        Mathematical library functions to be applied to each input feature.

    function_names : list of functions
        Functions for generating string representations of each library
        function.

    n_input_features_ : int
        The total number of input features.
        WARNING: This is deprecated in scikit-learn version 1.0 and higher so
        we check the sklearn.__version__ and switch to n_features_in if needed.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the number of library functions and the number of
        input features.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import WeakPDELibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib = WeakPDELibrary(library_functions=functions).fit(x)
    >>> lib.transform(x)
    array([[ 1.        ,  0.36787944, -0.84147098],
           [ 2.71828183,  1.        ,  0.84147098],
           [ 7.3890561 ,  0.36787944,  0.84147098]])
    >>> lib.get_feature_names()
    ['f0(x0)', 'f0(x1)', 'f1(x0,x1)']
    """

    def __init__(
        self,
        library_functions=[],
        derivative_order=0,
        spatiotemporal_grid=None,
        function_names=None,
        interaction_only=True,
        include_bias=False,
        include_interaction=True,
        is_uniform=False,
        K=100,
        num_pts_per_domain=100,
        H_xt=None,
        p=4,
        library_ensemble=False,
        ensemble_indices=[0],
        periodic=False,
    ):
        super(WeakPDELibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.functions = library_functions
        self.derivative_order = derivative_order
        self.function_names = function_names
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.is_uniform = is_uniform
        self.K = K
        self.num_pts_per_domain = num_pts_per_domain
        self.H_xt = H_xt
        self.p = p
        self.periodic = periodic
        self.num_trajectories = 1

        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        if library_functions is None and derivative_order == 0:
            raise ValueError(
                "No library functions were specified, and no "
                "derivatives were asked for. The library is empty."
            )
        if spatiotemporal_grid is None:
            raise ValueError(
                "Spatiotemporal grid was not passed, and at least a 1D"
                " grid is required, corresponding to the time base."
            )

        # list of integrals
        indices = ()
        if np.array(spatiotemporal_grid).ndim == 1:
            spatiotemporal_grid = np.reshape(
                spatiotemporal_grid, (len(spatiotemporal_grid), 1)
            )
        dims = spatiotemporal_grid.shape[:-1]
        self.grid_dims = dims
        self.grid_ndim = len(dims)

        # if want to include temporal terms -> range(len(dims))
        for i in range(len(dims) - 1):
            indices = indices + (range(derivative_order + 1),)

        multiindices = []
        for ind in iproduct(*indices):
            current = np.array(ind)
            if np.sum(ind) > 0 and np.sum(ind) <= derivative_order:
                multiindices.append(current)
        multiindices = np.array(multiindices)
        num_derivatives = len(multiindices)

        self.num_derivatives = num_derivatives
        self.multiindices = multiindices
        self.spatiotemporal_grid = spatiotemporal_grid

        # Weak form checks and setup
        self._weak_form_setup()

    def _weak_form_setup(self):
        xt1, xt2 = self._get_spatial_endpoints()
        L_xt = xt2 - xt1
        if self.H_xt is not None:
            if np.isscalar(self.H_xt):
                self.H_xt = np.array(self.grid_ndim * [self.H_xt])
            if self.grid_ndim != len(self.H_xt):
                raise ValueError(
                    "The user-defined grid (spatiotemporal_grid) and "
                    "the user-defined sizes of the subdomains for the "
                    "weak form, do not have the same # of spatiotemporal "
                    "dimensions. For instance, if spatiotemporal_grid is 4D, "
                    "then H_xt should be a 4D list of the subdomain lengths."
                )
            if any(self.H_xt <= np.zeros(len(self.H_xt))):
                raise ValueError("Values in H_xt must be a positive float")
            elif any(self.H_xt >= L_xt / 2.0):
                raise ValueError(
                    "2 * H_xt in some dimension is larger than the "
                    "corresponding grid dimension."
                )
        else:
            self.H_xt = L_xt / 20.0

        if self.spatiotemporal_grid is not None:
            if self.p < 0:
                raise ValueError("Poly degree of the spatial weights must be > 0")
            if self.p < self.derivative_order:
                self.p = self.derivative_order
        if self.K <= 0:
            raise ValueError("The number of subdomains must be > 0")

        self._set_up_grids()

    def _get_spatial_endpoints(self):
        x1 = np.zeros(self.grid_ndim)
        x2 = np.zeros(self.grid_ndim)
        for i in range(self.grid_ndim):
            inds = [slice(None)] * (self.grid_ndim + 1)
            for j in range(self.grid_ndim):
                inds[j] = 0
            x1[i] = self.spatiotemporal_grid[tuple(inds)][i]
            inds[i] = -1
            x2[i] = self.spatiotemporal_grid[tuple(inds)][i]
        return x1, x2

    def _set_up_grids(self):
        dims = self.spatiotemporal_grid.shape[:-1]
        self.grid_dims = dims

        xt1, xt2 = self._get_spatial_endpoints()
        self.domain_centers = np.zeros((self.K, self.grid_ndim))
        for i in range(self.grid_ndim):
            self.domain_centers[:, i] = np.random.uniform(
                xt1[i] + self.H_xt[i], xt2[i] - self.H_xt[i], size=self.K
            )

        self.inds_k = []
        for k in range(self.K):
            # indices corresponding to the subdomain
            inds = []
            for i in range(self.grid_ndim):
                s = [0] * (self.grid_ndim + 1)
                s[i] = slice(None)
                s[-1] = i
                newinds = np.intersect1d(
                    np.where(
                        self.spatiotemporal_grid[tuple(s)]
                        >= self.domain_centers[k][i] - self.H_xt[i]
                    ),
                    np.where(
                        self.spatiotemporal_grid[tuple(s)]
                        <= self.domain_centers[k][i] + self.H_xt[i]
                    ),
                )
                if len(newinds) == 0:
                    for i in range(self.grid_ndim):
                        self.domain_centers[k, i] = np.random.uniform(
                            xt1[i] + self.H_xt[i], xt2[i] - self.H_xt[i], size=1
                        )
                    k -= 1
                    break
                else:
                    inds = inds + [newinds]
            self.inds_k = self.inds_k + [inds]

        self.XT_k = [
            self.spatiotemporal_grid[np.ix_(*self.inds_k[k])] for k in range(self.K)
        ]

        self.XT_interp_k = []
        for k in range(self.K):
            self.XT_interp_k = self.XT_interp_k + [self.XT_k[k].copy()]
            for axis in range(self.grid_ndim):
                s = [slice(None)] * self.grid_ndim
                s[axis] = 0
                dims = np.array(self.XT_interp_k[k].shape)
                dims[axis] = 1
                left = np.reshape(self.XT_interp_k[k][tuple(s)].copy(), dims)
                right = np.reshape(self.XT_interp_k[k][tuple(s)].copy(), dims)
                left[..., axis] = self.domain_centers[k][axis] - self.H_xt[axis]
                right[..., axis] = self.domain_centers[k][axis] + self.H_xt[axis]
                self.XT_interp_k[k] = np.concatenate(
                    [left, self.XT_interp_k[k], right], axis=axis
                )
        self.xtilde_k = [
            (self.XT_interp_k[k] - self.domain_centers[k]) / self.H_xt
            for k in range(self.K)
        ]
        self.H_xt_k = np.zeros((self.K, self.grid_ndim))
        self.H_xt_k[:] = np.reshape(self.H_xt, (1, self.grid_ndim))

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """Get the combinations of features to be passed to a library function."""
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

    def convert_u_dot_integral(self, u):
        """
        Takes a full set of spatiotemporal fields u(x, t) and finds the weak
        form of u_dot using a pre-defined weak pde library.
        """
        K = self.K
        gdim = self.grid_ndim
        u_dot_integral = np.zeros((K, u.shape[-1]))
        deriv_orders = np.zeros(gdim)
        deriv_orders[-1] = 1

        # Interpolate the input onto the boundary of each domain
        grids = []
        for axis in range(gdim):
            s = [0] * (gdim + 1)
            s[axis] = slice(None)
            s[-1] = axis
            grids = grids + [self.spatiotemporal_grid[tuple(s)]]

        dims = np.array(self.spatiotemporal_grid.shape)
        dims[-1] = u.shape[-1]
        x_interpolator = RegularGridInterpolator(grids, np.reshape(u, dims))

        for k in range(K):
            weights = self.weights2(
                self.xtilde_k[k], deriv_orders, self.p
            ) * np.product(self.H_xt_k[k] ** (1.0 - deriv_orders))

            for j in range(u.shape[-1]):
                u_dot_integral[k, j] = -np.sum(
                    weights * x_interpolator(self.XT_interp_k[k])[..., j]
                )

        return u_dot_integral

    def _poly_derivative(self, xt, d_xt):
        """Compute analytic derivatives instead of relying on finite diffs"""
        return np.prod(
            (2 * xt) ** d_xt
            * (xt ** 2 - 1) ** (self.p - d_xt)
            * hyp2f1((1 - d_xt) / 2.0, -d_xt / 2.0, self.p + 1 - d_xt, 1 - 1 / xt ** 2)
            * poch(self.p + 1 - d_xt, d_xt),
            axis=-1,
        )

    def phi(self, x, d, p):
        return (
            (2 * x) ** d
            * (x ** 2 - 1) ** (p - d)
            * hyp2f1((1 - d) / 2.0, -d / 2.0, p + 1 - d, 1 - 1 / x ** 2)
            * poch(p + 1 - d, d)
        )

    def w(self, x, d, p):
        if d == 0:
            return (-1) ** p * x * hyp2f1(0.5, -p, 1.5, x ** 2)
        else:
            return self.phi(x, d - 1, p)

    def z(self, x, d, p):
        if d == 0:
            return (x ** 2 - 1) ** (p + 1) / (2 * (p + 1))
        elif d == 1:
            return (
                -(2.0 / 3.0)
                * (-1) ** p
                * p
                * x ** 3
                * hyp2f1(3.0 / 2.0, 1 - p, 5.0 / 2.0, x ** 2)
            )
        else:
            return x * self.phi(x, d - 1, p) - self.phi(x, d - 2, p)

    def linear_weights(self, x, d, p):
        ws = self.w(x, d, p)
        zs = self.z(x, d, p)
        return np.concatenate(
            [
                [
                    x[1] / (x[1] - x[0]) * (ws[1] - ws[0])
                    - 1 / (x[1] - x[0]) * (zs[1] - zs[0])
                ],
                x[2:] / (x[2:] - x[1:-1]) * (ws[2:] - ws[1:-1])
                - x[:-2] / (x[1:-1] - x[:-2]) * (ws[1:-1] - ws[:-2])
                + 1 / (x[1:-1] - x[:-2]) * (zs[1:-1] - zs[:-2])
                - 1 / (x[2:] - x[1:-1]) * (zs[2:] - zs[1:-1]),
                [
                    -x[-2] / (x[-1] - x[-2]) * (ws[-1] - ws[-2])
                    + 1 / (x[-1] - x[-2]) * (zs[-1] - zs[-2])
                ],
            ]
        )

    def weights2(self, xt, derivs, p):
        ret = np.ones(xt.shape[:-1])

        for i in range(self.grid_ndim):
            s = [0] * (self.grid_ndim + 1)
            s[i] = slice(None, None, None)
            s[-1] = i
            dims = np.ones(self.grid_ndim, dtype=int)
            dims[i] = xt.shape[i]
            ret = ret * np.reshape(
                self.linear_weights(xt[tuple(s)], derivs[i], p), dims
            )

        return ret

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        check_is_fitted(self)
        if float(__version__[:3]) >= 1.0:
            n_features = self.n_features_in_
        else:
            n_features = self.n_input_features
        if input_features is None:
            input_features = ["x%d" % i for i in range(n_features)]
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(n_features),
                )
            )
        feature_names = []

        # Include constant term
        if self.include_bias:
            feature_names.append("1")

        # Include any non-derivative terms
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )

        if self.grid_ndim != 0:

            def derivative_string(multiindex):
                ret = ""
                for axis in range(self.grid_ndim - 1):
                    for i in range(multiindex[axis]):
                        ret = ret + str(axis + 1)
                return ret

            # Include integral terms
            for k in range(self.num_derivatives):
                for j in range(n_features):
                    feature_names.append(
                        input_features[j]
                        + "_"
                        + derivative_string(self.multiindices[k])
                    )
            # Include mixed non-derivative + integral terms
            if self.include_interaction:
                for k in range(self.num_derivatives):
                    for i, f in enumerate(self.functions):
                        for c in self._combinations(
                            n_features,
                            f.__code__.co_argcount,
                            self.interaction_only,
                        ):
                            for jj in range(n_features):
                                feature_names.append(
                                    self.function_names[i](
                                        *[input_features[j] for j in c]
                                    )
                                    + input_features[jj]
                                    + "_"
                                    + derivative_string(self.multiindices[k])
                                )
        return feature_names

    def fit(self, x, y=None):
        """Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Measurement data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = check_array(x).shape
        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features

        n_output_features = 0

        # Count the number of non-derivative terms
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )

        if self.grid_ndim != 0:
            # Add the mixed derivative library_terms
            if self.include_interaction:
                n_output_features += (
                    n_output_features * n_features * self.num_derivatives
                )
            # Add the pure derivative library terms
            n_output_features += n_features * self.num_derivatives

        # If there is a constant term, add 1 to n_output_features
        if self.include_bias:
            n_output_features += 1

        self.n_output_features_ = n_output_features

        # required to generate the function names
        self.get_feature_names()

        return self

    def transform(self, x):
        """Transform data to custom features

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of
            features generated from applying the custom functions
            to the inputs.
        """
        check_is_fitted(self)

        x = check_array(x)

        n_samples_original_full, n_features = x.shape
        n_samples_original = n_samples_original_full // self.num_trajectories

        if float(__version__[:3]) >= 1.0:
            if n_features != self.n_features_in_:
                raise ValueError("x shape does not match training shape")
        else:
            if n_features != self.n_input_features_:
                raise ValueError("x shape does not match training shape")

        if self.spatiotemporal_grid is not None:
            n_samples = self.K
            n_samples_full = self.K * self.num_trajectories

        xp_full = np.empty(
            (self.num_trajectories, n_samples, self.n_output_features_), dtype=x.dtype
        )
        x_full = np.reshape(
            x, np.concatenate([[self.num_trajectories], self.grid_dims, [n_features]])
        )

        # Loop over each trajectory
        for trajectory_ind in range(self.num_trajectories):
            x = np.reshape(x_full[trajectory_ind], (n_samples_original, n_features))
            xp = np.empty((n_samples, self.n_output_features_), dtype=x.dtype)

            # Interpolate the input onto the boundary of each domain
            grids = []
            for axis in range(self.grid_ndim):
                s = [0] * (self.grid_ndim + 1)
                s[axis] = slice(None)
                s[-1] = axis
                grids = grids + [self.spatiotemporal_grid[tuple(s)]]

            dims = np.array(self.spatiotemporal_grid.shape)
            dims[-1] = n_features
            # If we remove the domain boundary interpolation,
            # things run a little faster but are less robust
            x_interpolator = RegularGridInterpolator(grids, np.reshape(x, dims))
            self.x_interp_k = [
                x_interpolator(self.XT_interp_k[k]) for k in range(self.K)
            ]

            # library function terms
            n_library_terms = 0
            for f in self.functions:
                for c in self._combinations(
                    n_features, f.__code__.co_argcount, self.interaction_only
                ):
                    n_library_terms += 1

            library_functions = np.empty((n_samples, n_library_terms), dtype=x.dtype)
            library_idx = 0

            # funcs=np.zeros(dims)
            # func_idx=0
            # for f in self.functions:
            #     for c in self._combinations(
            #         n_features, f.__code__.co_argcount, self.interaction_only
            #     ):
            #         funcs[...,func_idx]=f(*[x_shaped[..., l] for l in c])
            #         func_idx += 1

            for k in range(self.K):
                weights = self.weights2(
                    self.xtilde_k[k], np.zeros(self.grid_ndim), self.p
                ) * np.product(self.H_xt_k[k])

                library_idx = 0

                for f in self.functions:
                    for c in self._combinations(
                        n_features, f.__code__.co_argcount, self.interaction_only
                    ):
                        # integral of product over subdomain
                        # Would it be better to calculate f(x) on all grid points
                        # and interpolate it onto self.x_inter_k?
                        # That's probably less accurate...
                        func = f(*[self.x_interp_k[k][..., j] for j in c])

                        library_functions[k, library_idx] = np.sum(func * weights)
                        library_idx += 1

            if self.derivative_order != 0:
                # pure integral terms, need to differentiate the weight functions
                library_integrals = np.empty(
                    (n_samples, n_features * self.num_derivatives), dtype=x.dtype
                )
                library_idx = 0

                for k in range(self.K):
                    library_idx = 0
                    for j in range(self.num_derivatives):
                        deriv = np.concatenate([self.multiindices[j], [0]])
                        weights = self.weights2(
                            self.xtilde_k[k], deriv, self.p
                        ) * np.product(self.H_xt_k[k] ** (1.0 - deriv))

                        for n in range(n_features):
                            # integral of product over subdomain
                            library_integrals[k, library_idx] = (-1) ** (
                                np.sum(deriv) % 2
                            ) * np.sum(weights * self.x_interp_k[k][..., n])
                            library_idx += 1

                # Mixed derivative/non-derivative terms
                if self.include_interaction:
                    # mixed integral terms
                    library_mixed_integrals = np.empty(
                        (
                            n_samples,
                            n_library_terms * n_features * self.num_derivatives,
                        ),
                        dtype=x.dtype,
                    )

                    x_shaped = np.reshape(
                        x,
                        np.concatenate(
                            [self.spatiotemporal_grid.shape[:-1], [x.shape[-1]]]
                        ),
                    )

                    dims = np.array(x_shaped.shape)
                    dims[-1] = n_library_terms
                    funcs = np.zeros(dims)
                    func_idx = 0
                    for f in self.functions:
                        for c in self._combinations(
                            n_features, f.__code__.co_argcount, self.interaction_only
                        ):
                            funcs[..., func_idx] = f(*[x_shaped[..., j] for j in c])
                            func_idx += 1

                    funcs_derivs = np.zeros(
                        np.concatenate([[self.num_derivatives + 1], funcs.shape])
                    )
                    x_derivs = np.zeros(
                        np.concatenate([[self.num_derivatives + 1], x_shaped.shape])
                    )
                    funcs_derivs[0] = funcs
                    x_derivs[0] = x_shaped
                    self.dx_interp_k_j = []
                    self.dfx_interp_k_j = []
                    for j in range(self.num_derivatives):
                        for axis in range(self.grid_ndim - 1):
                            s = [0] * (self.grid_ndim + 1)
                            s[axis] = slice(None, None, None)
                            s[-1] = axis

                            if self.multiindices[j][axis] > 0 and self.multiindices[j][
                                axis
                            ] <= (self.derivative_order // 2):
                                funcs_derivs[j + 1] = FiniteDifference(
                                    d=self.multiindices[j][axis],
                                    axis=axis,
                                    is_uniform=self.is_uniform,
                                )._differentiate(
                                    funcs, self.spatiotemporal_grid[tuple(s)]
                                )
                            if self.multiindices[j][axis] > 0 and self.multiindices[j][
                                axis
                            ] <= (self.derivative_order - (self.derivative_order // 2)):
                                x_derivs[j + 1] = FiniteDifference(
                                    d=self.multiindices[j][axis],
                                    axis=axis,
                                    is_uniform=self.is_uniform,
                                )._differentiate(
                                    x_shaped, self.spatiotemporal_grid[tuple(s)]
                                )

                    dx_interpolator = RegularGridInterpolator(
                        grids,
                        np.transpose(x_derivs, np.roll(np.arange(x_derivs.ndim), -1)),
                    )
                    dfx_interpolator = RegularGridInterpolator(
                        grids,
                        np.transpose(
                            funcs_derivs, np.roll(np.arange(x_derivs.ndim), -1)
                        ),
                    )
                    self.dx_interp_k_j = [
                        dx_interpolator(self.XT_interp_k[k]) for k in range(self.K)
                    ]
                    self.dfx_interp_k_j = [
                        dfx_interpolator(self.XT_interp_k[k]) for k in range(self.K)
                    ]

                    library_idx = 0
                    for j in range(self.num_derivatives):
                        integral = np.zeros((self.K, n_library_terms, n_features))
                        derivs_mixed = self.multiindices[j] // 2
                        derivs_pure = self.multiindices[j] - derivs_mixed
                        derivs = np.concatenate(
                            [
                                [np.zeros(self.grid_ndim - 1, dtype=int)],
                                self.multiindices,
                            ],
                            axis=0,
                        )
                        for deriv in derivs[
                            np.where(np.all(derivs <= derivs_mixed, axis=1))[0]
                        ]:
                            for k in range(self.K):
                                weights = self.weights2(
                                    self.xtilde_k[k],
                                    np.concatenate([deriv, [0]]),
                                    self.p,
                                ) * np.product(
                                    self.H_xt_k[k]
                                    ** (1.0 - np.concatenate([deriv, [0]]))
                                )
                                for m in range(n_library_terms):
                                    for n in range(n_features):
                                        j1 = np.where(
                                            np.all(
                                                derivs == derivs_mixed - deriv, axis=1
                                            )
                                        )[0][0]
                                        j2 = np.where(
                                            np.all(derivs == derivs_pure, axis=1)
                                        )[0][0]
                                        integral[k, m, n] = integral[
                                            k, m, n
                                        ] + np.product((-1) ** derivs_mixed) * np.sum(
                                            weights
                                            * self.dfx_interp_k_j[k][..., m, j1]
                                            * self.dx_interp_k_j[k][..., n, j2]
                                        )
                        for n in range(n_features):
                            for m in range(n_library_terms):
                                library_mixed_integrals[:, library_idx] = integral[
                                    :, m, n
                                ]
                                library_idx += 1

            library_idx = 0
            constants_final = np.zeros(self.K)
            # Constant term
            if self.include_bias:
                for k in range(self.K):
                    weights = self.weights2(
                        self.xtilde_k[k], np.zeros(self.grid_ndim), self.p
                    ) * np.product(self.H_xt_k[k])
                    constants_final[k] = np.sum(weights)
                xp[:, library_idx] = constants_final
                library_idx += 1

            # library function terms
            xp[:, library_idx : library_idx + n_library_terms] = library_functions
            library_idx += n_library_terms

            if self.derivative_order != 0:
                # pure integral terms
                xp[
                    :, library_idx : library_idx + self.num_derivatives * n_features
                ] = library_integrals
                library_idx += self.num_derivatives * n_features

                # mixed function integral terms
                if self.include_interaction:
                    xp[
                        :,
                        library_idx : library_idx
                        + n_library_terms * self.num_derivatives * n_features,
                    ] = library_mixed_integrals
                    library_idx += n_library_terms * self.num_derivatives * n_features

            xp_full[trajectory_ind] = xp

        # If library bagging, return xp missing the terms at ensemble_indices
        # return self._ensemble(xp)
        return self._ensemble(
            np.reshape(xp_full, (n_samples_full, self.n_output_features_))
        )
