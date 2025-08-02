from Bio.PDB import PDBParser
import numpy as np

CST_STDERR = 0.5

def extract_dist_cst_from_pdb(pdb_in, lig_tr_atms, bsite_res=''):
    """
    pdb_in      : Path to the PDB file
    lig_tr_atms : List of atom names of interest in the ligand, e.g., ['NE1','O', ...]
    bsite_res   : If '', use all protein CA residues; otherwise, a comma-separated string of residue numbers like '10,15,27'
    return      : ['AtomPair <lig_atm> <het_resno> CA <closest_resno> HARMONIC <dist> 0.5', ...]
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("strc", pdb_in)

    # Assume a single chain/model (add loop if needed)
    model = next(structure.get_models())

    # bsite residue set
    if bsite_res == '':
        # All protein CA residue numbers
        bsite_res_s = {res.get_id()[1] for res in model.get_residues()
                       if res.get_id()[0] == ' ' and 'CA' in res}  # ' ' => ATOM
    else:
        bsite_res_s = {int(x) for x in bsite_res.split(',')}

    # Last protein residue number (for estimating ligand resno)
    last_prot_resno = max(res.get_id()[1] for res in model.get_residues()
                          if res.get_id()[0] == ' ')

    # CA coordinates dict
    bsite_CA_R = {}
    for res in model.get_residues():
        if res.get_id()[0] == ' ' and res.get_id()[1] in bsite_res_s and 'CA' in res:
            bsite_CA_R[res.get_id()[1]] = res['CA'].get_coord()

    # Ligand atom coordinates dict
    het_atm_R = {}
    for res in model.get_residues():
        if res.get_id()[0].startswith('H'):  # HETATM
            for atom in res:
                if atom.get_name() in lig_tr_atms:
                    het_atm_R[atom.get_name()] = atom.get_coord()

    het_resno = last_prot_resno + 1  # Corresponding discontinuous number for diffusion, etc.

    cst_s = []
    bsite_resnos = np.array(list(bsite_CA_R.keys()))
    bsite_coords = np.array([bsite_CA_R[r] for r in bsite_resnos])

    for atm_name in lig_tr_atms:
        het_coord = het_atm_R[atm_name]

        # All CA distances
        diff = bsite_coords - het_coord
        dists = np.sqrt((diff ** 2).sum(axis=1))

        # Closest residue
        idx = np.argmin(dists)
        closest_resno = int(bsite_resnos[idx])
        closest_d = float(dists[idx])

        cst_line = f'AtomPair {atm_name} {het_resno} CA {closest_resno} HARMONIC {closest_d} {CST_STDERR}'
        cst_s.append(cst_line)

    return cst_s
