import sys
sys.path.insert(0, "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP")


from qmlip.descriptors.fitsnap import qACE
from mpi4py import MPI
import pandas
from qmlip.trainers.lstsq import SVD
from settings import ACE_settings
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

def load_files(file_name_structures, file_name_energies):
    df_structures = pandas.read_hdf(file_name_structures)
    df_structures.sort_index(inplace=True)
    
    df_energies = pandas.read_hdf(file_name_energies)
    df_energies.sort_values(by=["index"], inplace=True)

    df_structures = df_structures[df_structures.index.isin(df_energies["index"].values)]
    #df_energies = df_energies[df_energies.index.isin(df_structures.index)]

    print("structures shape: ", df_structures.shape)
    print("energies shape: ", df_energies.shape)

    return df_structures, df_energies


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


file_name_structures = "Be_large_subset_structures.h5"
file_name_energies = "Be_high_3.hdf"
atoms_clmn = "ASEatoms_rescale"
df_structures, df_energies = load_files(file_name_structures, file_name_energies)

qace = qACE(comm, ACE_settings, df_structures[atoms_clmn], df_energies)
qace.calculate_descriptors()

comm.Barrier()
if rank == 0:
    print("Shape of aw:", qace.aw.shape)
    print("Size of aw (bytes):", qace.aw.nbytes)
    print("Shape of bw:", qace.bw.shape)
    print("Size of bw (bytes):", qace.bw.nbytes)

    print(qace.aw.shape[0])
    print(qace.bw.shape[0])
    print(qace.energy_selector.shape[0])
    print(qace.force_selector.shape[0])

    # Assertion checks to ensure consistency
    assert qace.aw.shape[0] == qace.bw.shape[0], "Mismatch: Number of rows in aw and bw don't match"
    assert qace.energy_selector.shape[0] == qace.bw.shape[0], "Mismatch: Number of rows in energy_selector and bw don't match"
    assert qace.force_selector.shape[0] == qace.bw.shape[0], "Mismatch: Number of rows in force_selector and bw don't match"

    # Save the aw, bw, energy_selector, and force_selector
    np.save('/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/aw.npy', qace.aw)
    np.save('/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/bw.npy', qace.bw)

    # Saving selectors
    print("Shape of energy_selector:", qace.energy_selector.shape)
    print("Shape of force_selector:", qace.force_selector.shape)
    np.save('/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/energy_selector.npy', qace.energy_selector)  
    np.save('/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/force_selector.npy', qace.force_selector)  
