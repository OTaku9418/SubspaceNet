"""Subspace-Net 
Details
----------
    Name: methods.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 03/06/23

Description:
-----------
This script contains the implementation of there subspace Doa estimation algorithms:
1. MUSIC (Multiple Signal Classification)
2. RootMUSIC
3. Esprit
and the implementation of the MVDR beamformer.
These algorithms are used to estimate the directions from which signals arrive at an array of sensors.

Purpose:
--------
The purpose of this script is to provide implementations of the MUSIC, RootMUSIC, Esprit and MVDR algorithms for DoA estimation.
These algorithms utilize subspace methods and covariance matrices to estimate the angles of arrival of signals in both broadband and narrowband scenarios.

Classes:
--------
- SubspaceMethod: A base class providing common functionality and methods for the subspace-based DoA estimation algorithms.
- MUSIC: A class representing the model-based MUSIC algorithm for DoA estimation.
- RootMUSIC: A class representing the model-based RootMUSIC algorithm for DoA estimation.
- Esprit: A class representing the model-based Esprit algorithm for DoA estimation.
- MVDR: A class representing the model-based MVDR algorithm for DoA estimation.

Methods:
--------
- SubspaceMethod:
  - calculate_covariance: Calculates the covariance matrix based on the given input samples.
  - subspace_separation: Performs subspace separation to obtain signal and noise subspaces.
  - spectrum_calculation: Calculates the MUSIC spectrum based on the noise subspace and steering vectors.
- MUSIC:
  - broadband: Implements the model-based MUSIC algorithm in the broadband scenario.
  - narrowband: Implements the model-based MUSIC algorithm in the narrowband scenario.
- RootMUSIC:
  - narrowband: Implements the model-based narrow-band RootMUSIC algorithm.
  - extract_angels_from_roots: Extracts the DoA predictions from the roots of the polynomial.
- Esprit:
  - narrowband: Implements the model-based narrow-band Esprit algorithm.
- MVDR:
  - narrowband: Implements the narrow-band MVDR beamformer algorithm with spatial smoothing.

"""

# Imports
import numpy as np
import scipy
from src.models import SubspaceNet
from src.system_model import SystemModel
from src.utils import sum_of_diag, find_roots, R2D


class SubspaceMethod(object):
    """
    An abstract class representing subspace-based method.

    Attributes:
        system_model: An instance of the SystemModel class representing the system model.

    Methods:
        __init__(self, system_model):
            Constructor for the SubspaceMethod class.

        calculate_covariance(self, X, mode="sample", model=None):
            Calculates the covariance matrix based on the specified mode.

        subspace_separation(self, covariance_mat, M):
            Performs subspace separation to obtain the noise and signal subspaces.
    """

    def __init__(self, system_model: SystemModel):
        """
        Constructor for the SubspaceMethod class.

        Args:
            system_model: An instance of the SystemModel class.

        Attributes:
            system_model: An instance of the SystemModel class.
        """
        self.system_model = system_model

    def calculate_covariance(
        self, X: np.ndarray, mode: str = "sample", model: SubspaceNet = None
    ):
        """
        Calculates the covariance matrix based on the specified mode.

        Args:
        -----
            X (np.ndarray): Input samples matrix.
            mode (str): Covariance calculation mode. Options: "spatial_smoothing", "SubspaceNet", "sample".
            model: Optional model used for SubspaceNet covariance calculation.

        Returns:
        --------
            covariance_mat (np.ndarray): Covariance matrix.

        Raises:
        -------
            Exception: If the given model for covariance calculation is not from SubspaceNet type.
            Exception: If the covariance calculation mode is not defined.
        """

        def spatial_smoothing_covariance(X: np.ndarray):
            """
            Calculates the covariance matrix using spatial smoothing technique.

            Args:
            -----
                X (np.ndarray): Input samples matrix.

            Returns:
            --------
                covariance_mat (np.ndarray): Covariance matrix.
            """
            # Define the sub-arrays size
            sub_array_size = int(self.system_model.params.N / 2) + 1
            # Define the number of sub-arrays
            number_of_sub_arrays = self.system_model.params.N - sub_array_size + 1
            # Initialize covariance matrix
            covariance_mat = np.zeros((sub_array_size, sub_array_size)) + 1j * np.zeros(
                (sub_array_size, sub_array_size)
            )
            for j in range(number_of_sub_arrays):
                # Run over all sub-arrays
                x_sub = X[j : j + sub_array_size, :]
                # Calculate sample covariance matrix for each sub-array
                sub_covariance = np.cov(x_sub)
                # Aggregate sub-arrays covariances
                covariance_mat += sub_covariance
            # Divide overall matrix by the number of sources
            covariance_mat /= number_of_sub_arrays
            return covariance_mat

        def subspacnet_covariance(X: np.ndarray, subspacenet_model: SubspaceNet):
            """
            Calculates the covariance matrix using the SubspaceNet model.

            Args:
            -----
                X (np.ndarray): Input samples matrix.
                subspacenet_model (SubspaceNet): Model used for covariance calculation.

            Returns:
            --------
                covariance_mat (np.ndarray): Covariance matrix.

            Raises:
            -------
                Exception: If the given model for covariance calculation is not from SubspaceNet type.
            """
            # Validate model type
            if not isinstance(model, SubspaceNet):
                raise Exception(
                    (
                        "SubspaceMethod.subspacenet_covariance: given model for covariance\
                    calculation isn't from SubspaceNet type"
                    )
                )
            # Predict the covariance matrix using the SubspaceNet model
            subspacenet_model.eval()
            covariance_mat = subspacenet_model(X)[-1]
            # Convert to np.array type
            covariance_mat = covariance_mat.detach().cpu().squeeze().numpy()
            return covariance_mat

        if mode.startswith("spatial_smoothing"):
            return spatial_smoothing_covariance(X)
        elif mode.startswith("SubspaceNet"):
            return subspacnet_covariance(X, model)
        elif mode.startswith("sample"):
            return np.cov(X)
        else:
            raise Exception(
                (
                    f"SubspaceMethod.subspacnet_covariance: {mode} type for covariance\
                calculation is not defined"
                )
            )

    def subspace_separation(self, covariance_mat: np.ndarray, M: int):
        """
        Performs subspace separation to obtain the noise and signal subspaces.

        Args:
            covariance_mat (np.ndarray): Covariance matrix.
            M (int): Number of sources.

        Returns:
            noise_subspace (np.ndarray): Noise subspace.
            signal_subspace (np.ndarray): Signal subspace.
        """
        # Find eigenvalues and eigenvectors (EVD)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)
        # Sort eigenvectors based on eigenvalues order
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        # Assign signal subspace as the eigenvectors associated with M greatest eigenvalues
        signal_subspace = eigenvectors[:, 0:M]
        # Assign noise subspace as the eigenvectors associated with M lowest eigenvalues
        noise_subspace = eigenvectors[:, M:]
        return noise_subspace, signal_subspace


class MUSIC(SubspaceMethod):
    """
    A class representing the model-based MUSIC algorithm for DoA estimation.
    Inherits from the SubspaceMethod class.

    Attributes:
        system_model: An instance of the SystemModel class representing the system model.
        angels: An array of angles for direction of arrival (DOA) estimation.

    Methods:
    --------
        __init__(self, system_model): Constructor for the MUSIC class.

        broadband(self, X, known_num_of_sources=True):
            Implementation of the model-based non-coherent broadband MUSIC algorithm.

        narrowband(self, X, known_num_of_sources=True, mode: str="sample", model: SubspaceNet=None):
            Implementation of the model-based narrowband MUSIC algorithm.

        spectrum_calculation(self, Un, f =1, array_form="ULA"):
            Calculates the MUSIC spectrum and the core equation for DOA estimation.

        get_spectrum_peaks(self, spectrum: np.ndarray):
            Calculates the peaks of the MUSIC spectrum.

    """

    def __init__(self, system_model: SystemModel):
        """
        Constructor for the MUSIC class.

        Args:
            system_model: An instance of the SystemModel class.
        """
        super().__init__(system_model)
        # angle axis for representation of the MUSIC spectrum
        self._angels = np.linspace(-1 * np.pi / 2, np.pi / 2, 18000, endpoint=False)

    def spectrum_calculation(
        self, Un: np.ndarray, f: float = 1, array_form: str = "ULA"
    ):
        """
        Calculates the MUSIC spectrum and the core equation for DOA estimation,
        and calculate the peaks of the spectrum.

        Args:
        -----
            Un (np.ndarray): Noise subspace matrix.
            f (float, optional): Frequency. Defaults to 1.
            array_form (str, optional): Array form. Defaults to "ULA".

        Returns:
        --------
            spectrum (np.ndarray): MUSIC spectrum.
            core_equation (np.ndarray): Core equation.
        """
        # 批量版: 一次算出整张网格的导向矢量矩阵 A (G, N), 再把二次型
        # a^H (Un Un^H) a 的**对角元**一次算完. 等价于下面被注释掉的逐角度循环,
        # 但快 30x 以上 (网格 18000 点: ~175 ms -> ~6 ms); 精度差异在 1e-15 量级,
        # 见 verify_music_batch.py 的等价性回归.
        A = self.system_model.steering_vec_batch(
            self._angels, f=f, array_form=array_form, nominal=True
        )
        A = A[:, : Un.shape[0]]
        Un_UnH = Un @ np.conj(Un).T
        core_equation = np.real(np.einsum("gn,nm,gm->g", np.conj(A), Un_UnH, A))
        # 逐角度循环的原始实现 (保留为参考):
        #   for angle in self._angels:
        #       a = self.system_model.steering_vec(
        #           theta=angle, f=f, array_form=array_form, nominal=True)[: Un.shape[0]]
        #       core_equation.append(np.conj(a).T @ Un @ np.conj(Un).T @ a)
        # Convert core equation to complex np.ndarray form
        core_equation = np.asarray(core_equation, dtype=complex)
        # MUSIC spectrum as the inverse core equation
        spectrum = 1 / core_equation
        return spectrum, core_equation

    def get_spectrum_peaks(self, spectrum: np.ndarray):
        """
        Calculates the peaks of the MUSIC spectrum.

        Args:
        -----
            spectrum (np.ndarray): MUSIC spectrum.

        Returns:
        --------
            peaks (list): the spectrum ordered peaks.
        """
        # Find spectrum peaks
        peaks = list(scipy.signal.find_peaks(spectrum)[0])
        # Sort the peak by their amplitude
        peaks.sort(key=lambda x: spectrum[x], reverse=True)
        return peaks

    def broadband(self, X: np.ndarray, known_num_of_sources: bool = True):
        """
        Implementation of the model-based non-coherent broadband MUSIC algorithm.

        Args:
        -----
            X (np.ndarray): Input samples vector.
            known_num_of_sources (bool, optional): Flag indicating if the number of sources is known. Defaults to True.
        Returns:
        --------
            doa_predictions (list): Predicted DOAs.
            spectrum (np.ndarray): MUSIC spectrum.
            M (int): Number of sources.
        """
        # Number of frequency bins calculation
        number_of_bins = int(self.system_model.max_freq["Broadband"] / 10)
        # Whether the number of sources is given
        if known_num_of_sources:
            M = self.system_model.params.M
        else:
            # clustering technique
            pass
        # Convert samples to frequency domain
        X = np.fft.fft(X, axis=1, n=self.system_model.f_sampling["Broadband"])
        # Calculate general params for calculations
        num_of_samples = len(self._angels)
        # Initialize spectrum array
        spectrum = np.zeros((number_of_bins, num_of_samples))
        # Calculate narrow-band spectrum for every frequency bin
        for i in range(number_of_bins):
            # Find the index of relevant bin
            ind = (
                int(self.system_model.min_freq["Broadband"])
                + i * len(self.system_model.f_rng["Broadband"]) // number_of_bins
            )
            # Calculate sample covariance matrix for measurements in the bin range
            covariance_mat = self.calculate_covariance(
                X=X[
                    :,
                    ind : ind
                    + len(self.system_model.f_rng["Broadband"]) // number_of_bins,
                ],
                mode="sample",
            )
            # Get noise subspace
            Un, _ = self.subspace_separation(covariance_mat=covariance_mat, M=M)
            # Calculate narrow-band music spectrum
            spectrum[i], _ = self.spectrum_calculation(
                Un,
                f=ind + len(self.system_model.f_rng["Broadband"]) // number_of_bins - 1,
            )
        # Sum all bins spectrums contributions to unified music spectrum
        spectrum = np.sum(spectrum, axis=0)
        # Find spectrum peaks
        doa_predictions = self.get_spectrum_peaks(spectrum)
        # Associate predictions to angels
        predictions = self._angels[doa_predictions] * R2D
        # Take first M predictions
        predictions = predictions[:M][::-1]
        return predictions, spectrum, M

    def narrowband(
        self,
        X: np.ndarray,
        known_num_of_sources: bool = True,
        mode: str = "sample",
        model: SubspaceNet = None,
    ):
        """
        Implementation of the model-based narrow-band MUSIC algorithm.

        Args:
        -----
            X (np.ndarray): Input samples matrix.
            known_num_of_sources (bool, optional): Flag indicating if the number of sources is known. Defaults to True.
            mode (str, optional): Covariance calculation mode. Defaults to 'sample'.
            model: Optional model used for SubspaceNet covariance calculation.

        Returns:
        --------
            doa_predictions (list): Predicted DOAs.
            spectrum (np.ndarray): MUSIC spectrum.
            M (int): Number of sources.
        """
        # Whether the number of sources is given
        if known_num_of_sources:
            M = self.system_model.params.M
        else:
            # clustering technique
            pass
        # Calculate covariance matrix
        covariance_mat = self.calculate_covariance(X=X, mode=mode, model=model)
        # Get noise subspace
        Un, _ = self.subspace_separation(covariance_mat=covariance_mat, M=M)
        # TODO: Check if this condition is hold after the change
        # Assign the frequency for steering vector calculation (multiplied in self.dist to get dist = 1/2)
        f = self.system_model.max_freq[self.system_model.params.signal_type]
        # Generate the MUSIC spectrum
        spectrum, _ = self.spectrum_calculation(Un, f=f)
        # Find spectrum peaks
        doa_predictions = self.get_spectrum_peaks(spectrum)
        # Associate predictions to angels
        predictions = self._angels[doa_predictions] * R2D
        # Take first M predictions
        predictions = predictions[:M][::-1]
        return predictions, spectrum, M


class RootMUSIC(SubspaceMethod):
    """
    A class representing the model-based Root MUSIC algorithm for DoA estimation.
    Inherits from the SubspaceMethod class.

    Attributes:
    -----------
        system_model: An instance of the SystemModel class representing the system model.

    Methods:
    --------
        narrowband(X: np.ndarray, known_num_of_sources: bool = True, mode: str = "sample"):
            Implementation of the model-based narrow-band RootMUSIC algorithm.

        extract_predictions_from_roots(self, roots: np.ndarray):
            Extracts the DoA predictions  from the roots of the polynomial.
    """

    def __init__(self, system_model: SystemModel):
        """
        Constructor for the RootMUSIC class.

        Args:
            system_model: An instance of the SystemModel class.
        """
        super().__init__(system_model)

    def narrowband(
        self,
        X: np.ndarray,
        known_num_of_sources: bool = True,
        mode: str = "sample",
        model: SubspaceNet = None,
    ):
        """
        Implementation of the model-based narrow-band RootMUSIC algorithm.

        Args:
        -----
            X (np.ndarray): Input samples vector.
            known_num_of_sources (bool, optional): Flag indicating if the number of sources is known. Defaults to True.
            mode (str, optional): Covariance calculation mode. Defaults to "sample".
            model: Optional model used for SubspaceNet covariance calculation.

        Returns:
        --------
            doa_predictions (np.ndarray): Predicted DOAs.
            roots (np.ndarray): Roots of the polynomial defined by F matrix diagonals.
            doa_predictions_all (np.ndarray): All predicted angles.
            roots_angels_all (np.ndarray): Phase components of all the roots.
            M (int): Number of sources.

        """
        # Whether the number of sources is given
        if known_num_of_sources:
            M = self.system_model.params.M
        else:
            # clustering technique
            pass
        # Calculate covariance matrix
        covariance_mat = self.calculate_covariance(X=X, mode=mode, model=model)
        # Get noise subspace
        Un, _ = self.subspace_separation(covariance_mat=covariance_mat, M=M)
        # Generate hermitian noise subspace matrix
        F = Un @ np.conj(Un).T
        # Calculates the sum of F matrix diagonals
        coefficients = sum_of_diag(F)
        # Calculates the roots of the polynomial defined by F matrix diagonals
        roots = list(find_roots(coefficients))
        # Sort roots by their distance from the unit circle
        roots.sort(key=lambda x: abs(abs(x) - 1))
        # Calculate all predicted angels for spectrum presentation
        doa_predictions_all, roots_angels_all = self.extract_predictions_from_roots(
            roots
        )
        # Take only roots which inside the unit circle
        roots_inside = [root for root in roots if ((abs(root) - 1) < 0)][:M]
        # Calculate DoA out of the roots inside the unit circle
        doa_predictions, _ = self.extract_predictions_from_roots(roots_inside)
        return doa_predictions, roots, doa_predictions_all, roots_angels_all, M

    def extract_predictions_from_roots(self, roots: np.ndarray):
        """
        Extracts the DoA predictions  from the roots of the polynomial.

        Args:
        -----
            roots (np.ndarray): Roots of the polynomial.

        Returns:
        --------
            doa_predictions (np.ndarray): Extracted DOAs.
            roots_angels (np.ndarray): Phase components of the roots.

        """
        # Calculate the phase component of the roots
        roots_angels = np.angle(roots)
        # Calculate the DoA out of the phase component
        doa_predictions = np.arcsin((1 / np.pi) * roots_angels) * R2D
        return doa_predictions, roots_angels


class Esprit(RootMUSIC):
    """
    A class representing the model-based Esprit algorithm for DoA estimation.
    Inherits from the RootMUSIC class.

    Attributes:
    -----------
        system_model: An instance of the SystemModel class representing the system model.

    Methods:
    --------
        narrowband(X: np.ndarray, known_num_of_sources: bool = True, mode: str = "sample"):
            Implementation of the model-based narrow-band RootMUSIC algorithm.

        extract_predictions_from_roots(self, roots: np.ndarray):
            Extracts the DoA predictions  from the roots of the polynomial.
    """

    def __init__(self, system_model: SystemModel):
        """
        Constructor for the RootMUSIC class.

        Args:
            system_model: An instance of the SystemModel class.
        """
        super().__init__(system_model)

    def narrowband(
        self,
        X: np.ndarray,
        known_num_of_sources: bool = True,
        mode: str = "sample",
        model: SubspaceNet = None,
    ):
        """
        Implementation of the model-based narrow-band Esprit algorithm.

        Args:
        -----
            X (np.ndarray): Input samples matrix.
            known_num_of_sources (bool, optional): Flag indicating if the number of sources is known. Defaults to True.
            mode (str, optional): Covariance calculation mode. Defaults to 'sample'.
            model: Optional model used for SubspaceNet covariance calculation.

        Returns:
        --------
            doa_predictions (list): Predicted DOAs.
            M (int): Number of sources.
        """
        # Whether the number of sources is given
        if known_num_of_sources:
            M = self.system_model.params.M
        else:
            # clustering technique
            pass
        # Calculate covariance matrix
        covariance_mat = self.calculate_covariance(X=X, mode=mode, model=model)
        # Get noise subspace
        _, Us = self.subspace_separation(covariance_mat=covariance_mat, M=M)
        # Separate the signal subspace into 2 overlapping subspaces
        Us_upper, Us_lower = (
            Us[0 : covariance_mat.shape[0] - 1],
            Us[1 : covariance_mat.shape[0]],
        )
        # Generate Phi matrix
        phi = np.linalg.pinv(Us_upper) @ Us_lower
        # Find eigenvalues and eigenvectors (EVD) of Phi
        phi_eigenvalues, _ = np.linalg.eig(phi)
        # Calculate DoA out of the eigenvalues of Phi
        doa_predictions = -1 * self.extract_predictions_from_roots(phi_eigenvalues)[0]
        return doa_predictions, M


class MVDR(MUSIC):
    """
    A class representing the model-based MVDR algorithm for DoA estimation.
    Inherits from the MUSIC class.

    Attributes:
        system_model: An instance of the SystemModel class representing the system model.

    Methods:
    narrowband(self, X:np.ndarray, mode: str="sample", model: SubspaceNet=None, eps:float = 1):
        Implementation of the narrow-band Minimum Variance beamformer algorithm.

    """

    def __init__(self, system_model: SystemModel):
        """
        Constructor for the MVDR class.

        Args:
            system_model: An instance of the SystemModel class.
        """
        super().__init__(system_model)

    def narrowband(
        self,
        X: np.ndarray,
        mode: str = "sample",
        model: SubspaceNet = None,
        eps: float = 1,
    ):
        """
        Implementation of the narrow-band Minimum Variance beamformer algorithm.

        Args:
        -----
            X (np.ndarray): Input samples vector.
            mode (str, optional): Covariance calculation mode. Defaults to "sample".
            model (SubspaceNet, optional): SubspaceNet model. Defaults to None.
            eps (float, optional): Diagonal loading factor. Defaults to 1.

        Returns:
        --------
            response_curve (np.ndarray): The response curve of the MVDR beamformer.

        """
        response_curve = []
        # Calculate covariance matrix
        covariance_mat = self.calculate_covariance(X=X, mode=mode, model=model)
        # Diagonal Loading
        diagonal_loaded_covariance = covariance_mat + eps * np.trace(
            covariance_mat
        ) * np.identity(covariance_mat.shape[0])
        # BeamForming response calculation
        inv_covariance = np.linalg.inv(diagonal_loaded_covariance)
        # TODO: Check if this condition is hold after the change
        # Assign the frequency for steering vector calculation (multiplied in self.dist to get dist = 1/2)
        f = self.system_model.max_freq[self.system_model.params.signal_type]
        # 批量版: 整张网格一次算完 (等价于下面的逐角度循环, 18000 点约 250 ms -> ~6 ms).
        # 推导: 原循环里 a 是 **列向量** 且 a[m] = exp(+j*2*pi*d*m*sin(theta)),
        # 而 steering_vec_batch 返回的行是 A[g,m] = exp(-j*2*pi*d*m*sin(theta)) = conj(a[m]).
        # 因 invC Hermitian: a^T @ invC = (invC @ conj(a))^T, 故取 conj(A) 作为行向量.
        # (直接用 A 会算成"共轭导向矢量"的响应, 各角度相对缩放不同, 相对偏差 3.581e-01.)
        # 记 d_g = a_g^H invC a_g (正实数), w_g = invC a_g / d_g, 则
        #   response_g = w_g^H C_loaded w_g = a_g^H invC C_loaded invC a_g / d_g^2
        # 再用 invC C_loaded invC = invC (因 C_loaded 可逆且 invC 是它的逆) 化简为 1 / d_g.
        # 必须用化简式: 直接算 num/d**2 时要先形成 ill-conditioned 的
        # invC @ dl @ invC (cond ~ 1.3e3), 残差达 1.4e-01; 化简后与循环版相差 7.2e-16.
        A = np.conj(self.system_model.steering_vec_batch(
            self._angels, f=f, array_form="ULA", nominal=True
        ))                                                    # (G, N) = a_g^T
        C_A = A @ inv_covariance                              # (G, N) = (invC a_g)^T
        denominator = np.real(np.einsum("gn,gn->g", np.conj(A), C_A))
        response_curve = 1.0 / denominator
        # 逐角度循环的原始实现 (保留为参考):
        #   for angle in self._angels:
        #       a = self.system_model.steering_vec(
        #           theta=angle, f=f, array_form="ULA", nominal=True).reshape((N, 1))
        #       optimal_weights = (inv_covariance @ a) / (np.conj(a).T @ inv_covariance @ a)
        #       response_curve.append(
        #           (np.conj(optimal_weights).T @ diagonal_loaded_covariance
        #            @ optimal_weights).reshape((1)).item())
        response_curve = np.array(response_curve, dtype=complex)
        predictions = None
        return predictions, response_curve
