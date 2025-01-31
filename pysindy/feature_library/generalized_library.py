import numpy as np
from sklearn import __version__
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary
from .weak_pde_library import WeakPDELibrary


class GeneralizedLibrary(BaseFeatureLibrary):
    """Put multiple libraries into one library. All settings
    provided to individual libraries will be applied. Note that this class
    allows one to specifically choose which input variables are used for
    each library, and take tensor products of any pair of libraries. Tensored
    libraries inherit the same input variables specified for the individual
    libraries.

    Parameters
    ----------
    libraries : list of libraries
        Library instances to be applied to the input matrix.

    tensor_array : 2D list of booleans, optional, (default None)
        Default is to not tensor any of the libraries together. Shape
        equal to the # of tensor libraries and the # feature libraries.
        Indicates which pairs of libraries to tensor product together and
        add to the overall library. For instance if you have 5 libraries,
        and want to do two tensor products, you could use the list
        [[1, 0, 0, 1, 0], [0, 1, 0, 1, 1]] to indicate that you want two
        tensored libraries from tensoring libraries 0 and 3 and libraries
        1, 3, and 4.

    inputs_per_library : 2D np.ndarray, optional (default None)
        Shape should be equal to # feature libraries by # variable input.
        Can be used to specify a subset of the variables to use to generate
        a feature library. If number of feature libraries > 1, then can be
        used to generate a large number of libraries, each using their own
        subsets of the input variables. Note that this must be specified for
        all the individual feature libraries.

    exclude_libraries : list of ints, optional (default [])

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library).

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library. For instance, if
        ensemble_indices = [0], it chops off the first column of the library.

    Attributes
    ----------
    libraries_ : list of libraries
        Library instances to be applied to the input matrix.

    tensor_array_ : 2D list of booleans (default None)
        Indicates which pairs of libraries to tensor product together and
        add to the overall library. For instance if you have 5 libraries,
        and want to do two tensor products, you could use the list
        [[1, 0, 0, 1, 0], [0, 1, 0, 1, 1]] to indicate that you want two
        tensored libraries from tensoring libraries 0 and 3 and libraries
        1, 3, and 4. Shape equal to # of tensor libraries to make
        by the # feature libraries.

    inputs_per_library_ : 2D np.ndarray, (default None)
        Default is that all inputs are used for every library.
        Can be used to specify a subset of the variables to use to generate
        a feature library. If number of feature libraries > 1, then can be
        use to generate a large number of libraries, each using their own
        subsets of the input variables. Note that this must be specified for
        all the individual feature libraries. The shape is equal to
        # feature libraries, # variable inputs.

    n_input_features_ : int
        The total number of input features.
        WARNING: This is deprecated in scikit-learn version 1.0 and higher so
        we check the sklearn.__version__ and switch to n_features_in if needed.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the sum of the numbers of output features for each of the
        concatenated libraries.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import FourierLibrary, CustomLibrary
    >>> from pysindy.feature_library import GeneralizedLibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib_custom = CustomLibrary(library_functions=functions)
    >>> lib_fourier = FourierLibrary()
    >>> lib_generalized = GeneralizedLibrary([lib_custom, lib_fourier])
    >>> lib_generalized.fit(x)
    >>> lib_generalized.transform(x)
    """

    def __init__(
        self,
        libraries: list,
        tensor_array=None,
        inputs_per_library=None,
        exclude_libraries=[],
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        super(GeneralizedLibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        if len(libraries) > 0:
            self.libraries_ = libraries
            weak_libraries = False
            nonweak_libraries = False
            for lib in self.libraries_:
                if isinstance(lib, WeakPDELibrary):
                    weak_libraries = True
                else:
                    nonweak_libraries = True
            if weak_libraries and nonweak_libraries:
                raise ValueError(
                    "At least one of the libraries is a weak form library, "
                    "and at least one of the libraries is not, which will "
                    "result in a nonsensical optimization problem. Please use "
                    "all weak form libraries or no weak form libraries."
                )
        else:
            raise ValueError(
                "Empty or nonsensical library list passed to this library."
            )
        if inputs_per_library is not None:
            if inputs_per_library.ndim != 2:
                raise ValueError("Input libraries array should form a 2D numpy array.")
            if inputs_per_library.shape[0] != len(libraries):
                raise ValueError(
                    "If specifying different inputs for each library, then "
                    "first dimension of inputs_per_library must be equal to "
                    "the number of libraries being used."
                )
            if np.any(inputs_per_library < 0):
                raise ValueError(
                    "The inputs_per_library parameter must be a numpy array "
                    "of integers with values between 0 and "
                    "len(input_variables) - 1."
                )

        if tensor_array is not None:
            if np.asarray(tensor_array).ndim != 2:
                raise ValueError("Tensor product array should be 2D list.")
            if np.asarray(tensor_array).shape[-1] != len(libraries):
                raise ValueError(
                    "If specifying tensor products between libraries, then "
                    "last dimension of tensor_array must be equal to the "
                    "number of libraries being used."
                )
            if np.any(np.ravel(tensor_array) > 1) or np.any(np.ravel(tensor_array) < 0):
                raise ValueError(
                    "The tensor_array parameter must be a numpy array "
                    "of booleans, so values must be either 0 or 1."
                )
            for i in range(len(tensor_array)):
                if np.sum(tensor_array[i]) < 2:
                    raise ValueError(
                        "If specifying libraries to tensor together, must "
                        "specify at least two libraries (there should be at "
                        "least two entries with value 1 in the tensor_array)."
                    )
        self.tensor_array_ = tensor_array
        self.inputs_per_library_ = inputs_per_library
        self.libraries_full_ = self.libraries_
        self.exclude_libs_ = exclude_libraries
    def fit(self, x, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        _, n_features = check_array(x).shape

        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features

        # If parameter is not set, use all the inputs
        if self.inputs_per_library_ is None:
            temp_inputs = np.tile(range(n_features), len(self.libraries_))
            self.inputs_per_library_ = np.reshape(
                temp_inputs, (len(self.libraries_), n_features)
            )
        else:
            # Check that the numbers in inputs_per_library are sensible
            if np.any(self.inputs_per_library_ >= n_features):
                raise ValueError(
                    "Each row in inputs_per_library must consist of integers "
                    "between 0 and the number of total input features - 1. "
                )

        # First fit all libraries separately below, with subset of the inputs
        fitted_libs = [
            lib.fit(x[:, np.unique(self.inputs_per_library_[i, :])], y)
            for i, lib in enumerate(self.libraries_)
        ]

        # Next, tensor some libraries and append them to the list
        if self.tensor_array_ is not None:
            num_tensor_prods = np.shape(self.tensor_array_)[0]
            for i in range(num_tensor_prods):
                lib_inds = np.ravel(np.where(self.tensor_array_[i]))
                library_subset = np.asarray(fitted_libs)[lib_inds]
                library_full = library_subset[0]
                n_output_features = library_subset[0].n_output_features_
                for j in range(1, len(library_subset)):
                    library_full = library_full * library_subset[j]
                    n_output_features = (
                        n_output_features * library_subset[j].n_output_features_
                    )
                library_full._set_inputs_per_library(
                    self.inputs_per_library_[lib_inds, :]
                )
                library_full.fit(x, y)
                fitted_libs.append(library_full)

        # Calculate the sum of output features
        # self.n_output_features_ = sum([lib.n_output_features_ for lib in fitted_libs])
        # exclude libraries in exclude_libs_ 
        self.n_output_features_ = sum([fitted_libs[i].n_output_features_ for i in np.setdiff1d(np.arange(len(fitted_libs)), self.exclude_libs_)])

        # Save fitted libs
        self.libraries_full_ = fitted_libs

        return self

    def transform(self, x):
        """Transform data with libs provided below.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of features
            generated from applying the custom functions to the inputs.

        """
        for lib in self.libraries_full_:
            check_is_fitted(lib)

        n_samples, n_features = x.shape

        if isinstance(self.libraries_[0], WeakPDELibrary):
            n_samples = self.libraries_[0].K * self.libraries_[0].num_trajectories

        if float(__version__[:3]) >= 1.0:
            n_input_features = self.n_features_in_
        else:
            n_input_features = self.n_input_features_
        if n_features != n_input_features:
            raise ValueError("x shape does not match training shape")

        # preallocate matrix
        xp = np.zeros((n_samples, self.n_output_features_))

        current_feat = 0
        for i, lib in enumerate(self.libraries_full_):
            if not i in self.exclude_libs_:

                # retrieve num output features from lib
                lib_n_output_features = lib.n_output_features_

                start_feature_index = current_feat
                end_feature_index = start_feature_index + lib_n_output_features

                if i < self.inputs_per_library_.shape[0]:
                    xp[:, start_feature_index:end_feature_index] = lib.transform(
                        x[:, np.unique(self.inputs_per_library_[i, :])]
                    )
                else:
                    xp[:, start_feature_index:end_feature_index] = lib.transform(x)

                current_feat += lib_n_output_features

        # If library bagging, return xp missing the terms at ensemble_indices
        return self._ensemble(xp)

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
        feature_names = list()
        for i, lib in enumerate(self.libraries_full_):
            if not i in self.exclude_libs_:
                if i < self.inputs_per_library_.shape[0]:
                    if input_features is None:
                        input_features_i = [
                            "x%d" % k for k in np.unique(self.inputs_per_library_[i, :])
                        ]
                    else:
                        input_features_i = np.asarray(input_features)[
                            np.unique(self.inputs_per_library_[i, :])
                        ].tolist()
                else:
                    # Tensor libraries need all the inputs and then internally
                    # handle the subsampling of the input variables
                    if input_features is None:
                        input_features_i = [
                            "x%d" % k for k in range(self.inputs_per_library_.shape[1])
                        ]
                    else:
                        input_features_i = input_features
                feature_names += lib.get_feature_names(input_features_i)
        return feature_names
