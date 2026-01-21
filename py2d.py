import os
import numpy as np
import torch
import scipy.io
import glob, pdb, shutil

from torch.utils.data import Dataset

class NSdataset(Dataset):
    def __init__(self, get_UV=False, get_Psi=False, normalize=False, extra_index = None, eval=False, validation=False):
        """
        Args:
            get_UV (bool): Whether to include U and V in the dataset.
            get_Psi (bool): Whether to include Psi in the dataset.
            normalize (bool): Whether to normalize the data. # Not implemented yet
        Returns:
            torch.utils.data.Dataset: Dataset with the [Channel, Y, X] shape.
        """
        self.extra_index = extra_index

        eval_dir = (os.getcwd().split('/')[:-1])
        eval_dir.append('eval')
        validation_dir = (os.getcwd().split('/')[:-1])
        validation_dir.append('validation')
        train_dir = (os.getcwd().split('/')[:-1])
        train_dir.append('train') 

        self.eval_dir = '/' + os.path.join(*eval_dir)
        self.validation_dir = '/' + os.path.join(*validation_dir)
        self.train_dir = '/' + os.path.join(*train_dir)

        if eval:
            file_list = glob.glob(f'{self.eval_dir}/*.mat')
            self.pth = self.eval_dir
        elif validation:
            file_list = glob.glob(f'{self.validation_dir}/*.mat')
            self.pth = self.validation_dir
        else:
            file_list = glob.glob(f'{self.train_dir}/*.mat')
            self.pth = self.train_dir
        assert len(file_list) > 0

        ################# increase the time_step to create a more difficult task
        time_step = 5
        self.file_list = []

        for file in file_list:
            i_t = file.split('/')[-1].split('.mat')[0]
            if int(i_t) % time_step == 0:
                self.file_list.append(file)

        it_list = []
        print(len(self.file_list))
        for file in self.file_list:
            i_t = file.split('/')[-1].split('.mat')[0]
            it_list.append(int(i_t))
        it_list.sort()
        self.it_list = torch.tensor(it_list)
        assert (np.arange(it_list[0], it_list[-1]+time_step, time_step) - np.array(it_list)).mean() == 0

        self.get_UV = get_UV
        self.get_Psi = get_Psi
        self.normalize = normalize

        # Determine which variables to include
        self.variables = ['Omega']
        if self.get_UV:
            self.variables.extend(['U', 'V'])
        if self.get_Psi:
            self.variables.append('Psi')

        # Load one file to get the shape
        sample_file = self.file_list[0]
        mat_data = scipy.io.loadmat(sample_file)
        Omega_sample = mat_data['Omega']
        nx, ny = Omega_sample.shape

        # Precompute wavenumbers
        Lx, Ly = 2 * np.pi, 2 * np.pi
        Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij')
        self.Kx = Kx
        self.Ky = Ky
        self.invKsq = invKsq

        # Load mean and std if normalize is True
        if self.normalize:
            pass
        else:
            pass

    def __len__(self):
        if self.extra_index is not None:
            return self.extra_index.shape[0]
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the file to load.
        Returns:
            torch.Tensor: Data loaded from the .mat file.
        """
        if self.extra_index is not None:
            idx = self.extra_index[idx]
        else:
            if torch.is_tensor(idx):
                idx = idx.tolist()

        file_path = f'{self.pth}/{self.it_list[idx]}.mat' 
        try:
            mat_data = scipy.io.loadmat(file_path)
        except:
            print(file_path)
        Omega = mat_data['Omega']

        if not self.get_UV:
            return Omega
        else:
            # Initialize a dictionary to store computed variables
            computed_vars = {'Omega': Omega}
            # Compute Psi if needed
            Psi = Omega2Psi(Omega, self.invKsq)
            computed_vars['Psi'] = Psi

            U, V = Psi2UV(computed_vars['Psi'], self.Kx, self.Ky)
            computed_vars['U'] = U
            computed_vars['V'] = V

            # Collect the required variables
            data_list = [computed_vars[var] for var in self.variables]

            # Stack data into a tensor
            data_array = np.stack(data_list, axis=0)  # Shape: (num_variables, ny, nx)
            data_tensor = torch.from_numpy(data_array).float() 

            if self.normalize:
                # Normalize data (Needs to be implemented)
                mean_tensor = np.zeros_like(data_tensor)
                std_tensor = np.ones_like(data_tensor)
                data_tensor = (data_tensor - mean_tensor) / std_tensor

            return data_tensor # Shape: (Channels, Y, X)

############ Functions for calculating stream function and velocity components ############

def Omega2Psi(Omega, invKsq, spectral=False):
    """
    Calculate the stream function from vorticity.

    This function calculates the stream function (Psi) from vorticity (Omega) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    both physical and spectral space calculations.

    Parameters:
    -----------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical or spectral space, depending on the 'spectral' flag.
    invKsq : numpy.ndarray
        2D array of the inverse of square of the wavenumber magnitudes.
    spectral : bool, optional
        If True, assumes input vorticity is in spectral space and returns stream function in
        spectral space. If False (default), assumes input vorticity is in physical space and
        returns stream function in physical space.

    Returns:
    --------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.

    Notes:
    ------
    Ksq[0,0] is set to a large value to avoid division by zero.
    Ksq[0,0] maybe me set to 0, causing division by zero.
    Psi[0,0] can be set to 0, to avoid any nan values.
    """
    # Check if the 'spectral' flag is set to False. If it is, transform the vorticity from physical space to spectral space using a 2D Fast Fourier Transform.
    if not spectral:
        Psi = Omega2Psi_physical(Omega, invKsq)
        return Psi
    # If the 'spectral' flag is set to True, assume that the input vorticity is already in spectral space.
    else:
        Omega_hat = Omega
        Psi_hat = Omega2Psi_spectral(Omega_hat, invKsq)
        return Psi_hat

def Psi2UV(Psi, Kx, Ky, spectral = False):
    """
    Calculate the velocity components U and V from the stream function.

    This function calculates the velocity components U and V from the stream function (Psi) 
    using the relationships:
        U = d(Psi)/dy
        V = -d(Psi)/dx
    Depending on the 'spectral' flag, the function can handle both physical and spectral space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    spectral : bool, optional
        If True, assumes input stream function is in spectral space and returns velocity components
        in spectral space. If False (default), assumes input stream function is in physical space and
        returns velocity components in physical space.

    Returns:
    --------
    U, V : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in physical or spectral space, depending on the 'spectral' flag.

    """

    # If the 'spectral' flag is False, perform a 2D Fast Fourier Transform on the input stream function
    # to transform it into spectral space
    if not spectral:
        U, V = Psi2UV_physical(Psi, Kx, Ky)
        return U, V
    else:
        Psi_hat = Psi
        U_hat, V_hat = Psi2UV_spectral(Psi_hat, Kx, Ky)
        return U_hat, V_hat

def Omega2Psi_spectral(Omega_hat, invKsq):
    """
    Calculate the stream function from vorticity in spectral space

    This function calculates the stream function (Psi) from vorticity (Omega) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    spectral space calculations.

    Parameters:
    -----------
    Omega_hat : numpy.ndarray
        Vorticity spectral space.
    invKsq : numpy.ndarray
        2D array of the inverse of square of the wavenumber magnitudes.

    Returns:
    --------
    Psi_hat : numpy.ndarray
        spectral space

    Notes:
    ------
    Ksq[0,0] is set to a large value to avoid division by zero.
    Ksq[0,0] maybe me set to 0, causing division by zero.
    Psi[0,0] can be set to 0, to avoid any nan values.
    """

    # Compute the Laplacian of the stream function in spectral space by taking the negative of the vorticity in spectral space.
    lap_Psi_hat = -Omega_hat
    # Divide the Laplacian of the stream function by the negative of the square of the wavenumber magnitudes (1/Ksq = inKsq) to compute the stream function in spectral space.
    Psi_hat = lap_Psi_hat * (-invKsq)

    # Return the stream function in spectral space.
    return Psi_hat

def Omega2Psi_physical(Omega, invKsq):
    """
    Calculate the stream function from vorticity in physical space

    This function calculates the stream function (Psi) from vorticity (Omega) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle physical space calculations.

    Parameters:
    -----------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical space
    invKsq : numpy.ndarray
        2D array of the inverse of square of the wavenumber magnitudes.

    Returns:
    --------
    Psi : numpy.ndarray
        Stream function (2D array) in physical space.

    """
    Nx, Ny = Omega.shape

    # Transform the vorticity from physical space to spectral space using a 2D Fast Fourier Transform.
    Omega_hat = np.fft.rfft2(Omega)

    # Compute the stream function in spectral space using the Omega2Psi_spectral function.
    Psi_hat = Omega2Psi_spectral(Omega_hat, invKsq)

    # Transform the stream function from spectral space back to physical space using an inverse 2D Fast Fourier Transform before returning it.
    return np.fft.irfft2(Psi_hat, s=[Nx,Ny])

def Psi2UV_spectral(Psi_hat, Kx, Ky):
    """
    Calculate the velocity components U and V from the stream function in spectral space

    This function calculates the velocity components U and V from the stream function (Psi) 
    using the relationships:
        U = d(Psi)/dy
        V = -d(Psi)/dx
    The function can handle spectral space calculations.

    Parameters:
    -----------
    Psi_hat : numpy.ndarray
        Stream function (2D array) in spectral space
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.

    Returns:
    --------
    U_hat, V_hat : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in spectral space.

    """

    # Calculate the Fourier coefficient of U (velocity in y-direction)
    # using the relationship U = d(Psi)/dy
    # In Fourier space, differentiation corresponds to multiplication by an imaginary unit and the wavenumber
    U_hat = (1.j) * Ky * Psi_hat

    # Calculate the Fourier coefficient of V (velocity in x-direction)
    # using the relationship V = -d(Psi)/dx
    # In Fourier space, differentiation corresponds to multiplication by an imaginary unit and the wavenumber
    V_hat = -(1.j) * Kx * Psi_hat

    return U_hat, V_hat

def Psi2UV_physical(Psi, Kx, Ky):
    """
    Calculate the velocity components U and V from the stream function in physical space

    This function calculates the velocity components U and V from the stream function (Psi) 
    using the relationships:
        U = d(Psi)/dy
        V = -d(Psi)/dx
    The function can handle both physical space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical space.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.

    Returns:
    --------
    U, V : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in physical space.

    """
    Nx, Ny = Psi.shape

    # Perform a 2D Fast Fourier Transform on the input stream function to transform it into spectral space
    Psi_hat = np.fft.rfft2(Psi)

    # Calculate the Fourier coefficients of the velocity components U and V in spectral space using the Psi2UV_spectral function
    U_hat, V_hat = Psi2UV_spectral(Psi_hat, Kx, Ky)

    # Perform an inverse 2D Fast Fourier Transform on the Fourier coefficients of the velocity components to transform them into physical space
    return np.fft.irfft2(U_hat, s=[Nx,Ny]), np.fft.irfft2(V_hat, s=[Nx,Ny])

############ Functions for Calculating Wavenumbers required for derivatives ############

def initialize_wavenumbers_fft2(nx, ny, Lx, Ly, INDEXING='ij'):
    '''
    Initialize the wavenumbers for 2D Forced Homogeneous Isotropic Turbulence (2D-FHIT).
    
    Parameters:
    -----------
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    Lx : float
        Length of the domain in the x-direction.
    Ly : float
        Length of the domain in the y-direction.

    Returns:
    --------
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Kabs : numpy.ndarray
        2D array of the absolute values of the wavenumbers.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.
    invKsq : numpy.ndarray
        2D array of the inverse of the square of the wavenumber magnitudes.
    

    Notes:
    ------
    inKsq[0,0] = 0 to avoid numerical errors, since invKsq[0.0] = 1/0 = inf
    '''

    # Create an array of the discrete Fourier Transform sample frequencies in x-direction
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)

    # Create an array of the discrete Fourier Transform sample frequencies in y-direction
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)

    # Return coordinate grids (2D arrays) for the x and y wavenumbers
    (Kx, Ky) = np.meshgrid(kx, ky, indexing=INDEXING)

    # Compute the squared magnitudes of the 2D wavenumbers (Kx and Ky)
    Ksq = Kx ** 2 + Ky ** 2

    # Compute the absolute value of the wavenumbers
    Kabs = np.sqrt(Ksq)

    # To avoid division by zero, set the zero wavenumber to a large value 
    Ksq[0,0] = 1e16

    # Compute the inverse of the squared wavenumbers
    invKsq = 1.0 / Ksq
    # Set the inverse of the zero wavenumber to zero
    invKsq[0,0] = 0.0

    # Set the zero wavenumber back to zero
    Ksq[0,0] = 0.0

    return Kx, Ky, Kabs, Ksq, invKsq

def initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij'):

    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_fft2(nx, ny, Lx, Ly, INDEXING=INDEXING)

    return fft2_to_rfft2(Kx), fft2_to_rfft2(Ky), fft2_to_rfft2(Kabs), fft2_to_rfft2(Ksq), fft2_to_rfft2(invKsq)

# Function to convert fft2 outputs to rfft2 outputs
def fft2_to_rfft2(a_hat_fft):
    if a_hat_fft.shape[0] % 2 == 0:
        # Shape of matrix is even
        return a_hat_fft[:,:a_hat_fft.shape[1]//2+1]
    else:
        # Shape of matrix is odd
        return a_hat_fft[:,:(a_hat_fft.shape[1]-1)//2+1]