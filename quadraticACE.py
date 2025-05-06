from mpi4py import MPI
import pandas
from qmlip.descriptors.fitsnap import qACE
from qmlip.trainers.lstsq import SVD
from settings import ACE_settings


def load_files(file_name_structures, file_name_energies):
    df_structures = pandas.read_hdf(file_name_structures)
    df_structures.sort_index(inplace=True)

    df_energies = pandas.read_hdf(file_name_energies)
    df_energies.sort_values(by=["index"], inplace=True)

    df_structures = df_structures[df_structures.index.isin(df_energies["index"].values)]
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
if rank == 0:
    qace.train()

