import os, random, time, resource
from ase.geometry import get_distances
from copy import deepcopy
import numpy as np
from mpi4py import MPI
from scipy.linalg import lstsq
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import get_apre
import pandas
from sklearn.linear_model import Ridge
from settings import snap_settings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
import psutil
import os

def print_memory(rank, msg=""):
    """
    Prints the memory usage of the current process and the available system memory.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / (1024**2)  # in MB
    avail_mem = psutil.virtual_memory().available / (1024**2)
    print(f"Rank {rank} {msg} - Process RSS: {mem_info:.2f} MB | System available: {avail_mem:.2f} MB")



# os.chdir("lanl/W/")

def ase_scraper(snap, frames, energies, forces, stresses):
    """
    Function to organize groups and allocate shared arrays used in Calculator. For now when using 
    ASE frames, we don't have groups.

    Args:
        s: fitsnap instance.
        data: List of ASE frames or dictionary group table containing frames.

    Returns a list of data dictionaries suitable for fitsnap descriptor calculator.
    If running in parallel, this list will be distributed over procs, so that each proc will have a 
    portion of the list.
    """

    snap.data = [collate_data(snap, indx, len(frames), a, e, f, s) for indx, (a,e,f,s) in enumerate(zip(frames, energies, forces, stresses))]
    # Simply collate data from Atoms objects if we have a list of Atoms objecst.
    # if type(frames) == list:
        # s.data = [collate_data(atoms) for atoms in data]
    # If we have a dictionary, assume we are dealing with groups.
    # elif type(data) == dict:
    #     assign_validation(data)
    #     snap.data = []
    #     for name in data:
    #         frames = data[name]["frames"]
    #         # Extend the fitsnap data list with this group.
    #         snap.data.extend([collate_data(atoms, name, data[name]) for atoms in frames])
    # else:
    #     raise Exception("Argument must be list or dictionary for ASE scraper.")

def collate_data(s, indx, size, atoms, energy, forces, stresses):
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args: 
        atoms: ASE atoms object for a single configuration of atoms.
        name: Optional name of this configuration.
        group_dict: Optional dictionary containing group information.

    Returns a data dictionary for a single configuration.
    """

    # Transform ASE cell to be appropriate for LAMMPS.
    apre = get_apre(cell=atoms.cell)
    R = np.dot(np.linalg.inv(atoms.cell), apre)
    positions = np.matmul(atoms.get_positions(), R)
    cell = apre.T

    # Make a data dictionary for this config.

    data = {}
    data['PositionsStyle'] = 'angstrom'
    data['AtomTypeStyle'] = 'chemicalsymbol'
    data['StressStyle'] = 'bar'
    data['LatticeStyle'] = 'angstrom'
    data['EnergyStyle'] = 'electronvolt'
    data['ForcesStyle'] = 'electronvoltperangstrom'
    data['Group'] = 'All'
    data['File'] = None
    data['Stress'] = stresses
    data['Positions'] = positions
    data['Energy'] = energy
    data['AtomTypes'] = atoms.get_chemical_symbols()
    data['NumAtoms'] = len(atoms)
    data['Forces'] = forces
    data['QMLattice'] = cell
    data['test_bool'] = indx>=s.config.sections["GROUPS"].group_table["All"]["training_size"]*size
    data['Lattice'] = cell
    data['Rotation'] = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data['Translation'] = np.zeros((len(atoms), 3))
    data['eweight'] = s.config.sections["GROUPS"].group_table["All"]["eweight"]
    data['fweight'] = s.config.sections["GROUPS"].group_table["All"]["fweight"]
    data['vweight'] = s.config.sections["GROUPS"].group_table["All"]["vweight"]

    return data

def load_files(file_name_structures, file_name_energies):
    df_structures = pandas.read_hdf(file_name_structures)
    print("df_structures: ", (len(df_structures)))
    df_structures.sort_index(inplace=True)
    #df_structures = df_structures.iloc[:len(df_structures)]
    #print(df_structures.head(10))

    df_energies = pandas.read_hdf(file_name_energies)
    print("df_energies: ", (len(df_energies)))
    df_energies.sort_values(by=["index"], inplace=True)
    #df_energies = df_energies.iloc[:len(df_energies)]
    #print(df_energies.head(10))

    df_structures = df_structures[df_structures.index.isin(df_energies["index"].values)]
    
    return df_structures, df_energies

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

fs_instance1 = FitSnap(snap_settings, comm=comm, arglist=["--overwrite"])
file_name_structures = "Be_large_subset_structures.h5"
file_name_energies = "Be_high_3.hdf"
atoms_clmn = "ASEatoms_rescale"

start_time_load = time.time()
df_structures, df_energies = load_files(file_name_structures, file_name_energies)

print("Files are loadedd in ", time.time()-start_time_load, "sec")
print_memory(rank, "- after file load")


configs_num = df_structures[atoms_clmn].shape[0]
ratio = configs_num//size
rem = configs_num%size
a1 = rank*ratio + min(rank,rem)
a2 = (rank+1)*ratio + min(rank,rem-1) + 1

print(f"Rank {rank} is processing indexes from {a1} to {a2} ")

print("Scraping process starts!")
start_time_scrape = time.time()
ase_scraper(fs_instance1, df_structures[atoms_clmn].values[a1:a2], df_energies['energy'].values[a1:a2], df_energies["forces"].values[a1:a2], df_energies["stress"].values[a1:a2])
# Now `fs_instance1.data` is a list of dictionaries containing configuration/structural info.
print("Scraping finished in", time.time()-start_time_scrape, "sec")
print_memory(rank, "After scraping")

print("length ",len(fs_instance1.data))

start_time_scrape = time.time()
fs_instance1.process_configs(allgather=True)


fs_instance1.pt.all_barrier()
# Perform a fit using data in the shared arrays.

start_time_train = time.time()
fs_instance1.perform_fit()

print("Training finished in", time.time()-start_time_train, "sec")

# Analyze error metrics.
fs_instance1.solver.error_analysis()

# Write error metric and LAMMPS files.
fs_instance1.output.output(fs_instance1.solver.fit, fs_instance1.solver.errors)

# Dataframe of detailed errors per group.
print(fs_instance1.solver.errors)

# Can also access the fitsnap dataframe here:
# print(snap.solver.df)
# WriteLAMMPS potential files and error analysis.
fs_instance1.write_output()