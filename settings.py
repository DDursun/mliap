ACE_settings = \
{
"ACE":
    {
    "numTypes": 1,
    "ranks": "1 2",
    "lmin": "0 0",
    "lmax": "0 3",
    #"nmax": "8 4",
    "nmax": "10 6",
    #"nmax": "10 3 1",
    "rcutfac": 6.0,
    "lambda": 3.059235105,
    #"rfac0": 0.99363,
    #"rmin0": 0.0,
    #"wj": 1.0,
    #"radelem": 0.5,
    "type": "Be",
    #"wselfallflag": 0,
    #"chemflag": 0,
    "bzeroflag": 0,
    "bikflag" : 1,
    "dgradflag" : 1
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSPACE",
    "energy": 1,
    "per_atom_energy": 1,
    "force": 1,
    "stress" : 0,
    "nonlinear" : 1
    },
"ESHIFT":
    {
    "Be" : 0.0
    },
"PYTORCH":
    {
    "layer_sizes" : "num_desc 32 32 1",
    "learning_rate" : 5e-5,
    "num_epochs" : 1000,
    "batch_size" : 32, # 363 configs in entire set
    "save_state_output" : "W_Pytorch.pt",
    "energy_weight" : 150.0,
    "force_weight" : 1.0,
    # "training_fraction" : 0.5,
    },
"GROUPS":
    {
    # name size eweight fweight vweight
    "group_sections" : "name training_size testing_size eweight fweight vweight",
    "group_types" : "str float float float float float",
    "smartweights" : 0,
    "random_sampling" : 0,
    "All" :  "0.875    0.125    150.0      1.0  0.0"
    },
"OUTFILE":
    {
    "output_style" : "PACE",
    "metrics" : "Be_metrics_final-loop1000-lr-5e-5-batch32.md",
    "potential" : "Be_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "pair_style": "zero 10.0",
    "pair_coeff": "* *",
    },
"SOLVER":
    {
    "solver": "PYTORCH",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"EXTRAS":
    {
    "dump_descriptors": 0,
    "dump_truth": 0,
    "dump_weights": 0,
    "dump_dataframe": 0
    },
"MEMORY":
    {
    "override": 0
    }
}

linear_settings = \
{
"ACE":
    {
    "numTypes": 1,
    "ranks": "1 2",
    "lmin": "0 0",
    "lmax": "0 3",
    "nmax": "8 4",
    #"nmax": "10 3 1",
    "rcutfac": 6.0,
    "lambda": 3.059235105,
    #"rfac0": 0.99363,
    #"rmin0": 0.0,
    #"wj": 1.0,
    #"radelem": 0.5,
    "type": "Be",
    #"wselfallflag": 0,
    #"chemflag": 0,
    "bzeroflag": 0,
    #"bikflag" : 0,
    #"dgradflag" : 1
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSPACE",
    "energy": 1,
    "per_atom_energy": 0,
    "force": 1,
    "stress" : 0,
    "nonlinear" : 0
    },
"ESHIFT":
    {
    "Be" : 0.0
    },

"GROUPS":
    {
    # name size eweight fweight vweight
    "group_sections" : "name training_size testing_size eweight fweight vweight",
    "group_types" : "str float float float float float",
    "smartweights" : 0,
    "random_sampling" : 0,
    "All" :  "0.875    0.125    150.0      1.0  0.0"
    },
"OUTFILE":
    {
    "output_style" : "PACE",
    "metrics" : "Be_metrics-linear.md",
    "potential" : "Be_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "pair_style": "zero 10.0",
    "pair_coeff": "* *",
    },
"SOLVER":
    {
    "solver": "SVD",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"EXTRAS":
    {
    "dump_descriptors": 0,
    "dump_truth": 0,
    "dump_weights": 0,
    "dump_dataframe": 0
    },
"MEMORY":
    {
    "override": 0
    }
}


snap_settings = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax" : 8,
    "rcutfac": 6.0,
    #"lambda": 3.059235105,
    #"rfac0": 0.99363,
    #"rmin0": 0.0,
    #"wj": 1.0,
    #"radelem": 0.5,
    "type": "Be",
    #"wselfallflag": 0,
    #"chemflag": 0,
    "bzeroflag": 1,
    "bikflag" : 1,
    "dgradflag" : 1
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "per_atom_energy": 1,
    "force": 1,
    "stress" : 0,
    "nonlinear" : 1
    },
"ESHIFT":
    {
    "Be" : 0.0
    },

"GROUPS":
    {
    # name size eweight fweight vweight
    "group_sections" : "name training_size testing_size eweight fweight vweight",
    "group_types" : "str float float float float float",
    "smartweights" : 0,
    "random_sampling" : 0,
    "All" :  "0.875    0.125    150.0      1.0  0.0"
    },
"OUTFILE":
    {
    "output_style" : "PACE",
    "metrics" : "Be-SNAP-lr5e4-500-16b-64.md",
    "potential" : "Be_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "pair_style": "zero 10.0",
    "pair_coeff": "* *",
    },
"SOLVER":
    {
    "solver": "PYTORCH",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
    "PYTORCH":
    {
    "layer_sizes" : "num_desc 64 64 1",
    "learning_rate" : 5e-4,
    "num_epochs" : 500,
    "batch_size" : 16, # 363 configs in entire set
    "save_state_output" : "W_Pytorch.pt",
    "energy_weight" : 150.0,
    "force_weight" : 1.0,
    # "training_fraction" : 0.5,
    },
"EXTRAS":
    {
    "dump_descriptors": 0,
    "dump_truth": 0,
    "dump_weights": 0,
    "dump_dataframe": 0
    },
"MEMORY":
    {
    "override": 0
    }
}