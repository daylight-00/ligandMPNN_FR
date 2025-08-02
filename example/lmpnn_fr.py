import sys
sys.path.append('/home/hwjang/aipd/LigandMPNN')
sys.path.append('/home/hwjang/aipd/ligandMPNN_FR')
from ligandmpnn_fastrelax_complete import parse_arguments, main
from tqdm import tqdm
import os
import pandas as pd

df_path = sys.argv[1]
df = pd.read_parquet(df_path)

if len(sys.argv) > 1:
    job_idx = int(sys.argv[2])
    N = int(sys.argv[3])
    df = df[df.index % N == job_idx].reset_index(drop=True)

mapping = {
    # 'ligand_params_path': {
    #     "pht": "/home/hwjang/aipd/250621/12_pht_rscore/PHT.params",
    #     "cbz": "/home/hwjang/aipd/250710/0_cbz_params/out_gen/CBZ.params"
    #     },
    'target_atm_for_cst': {
        "af3": {
            "pht": "O4,O5,N1,N2",
            "cbz": "O4,N1,N2"
        },
        "boltz": {
            "pht": "O17,O18,N37,N38",
            "cbz": "O17,N37,N38"
        },
        "lmpnn": {
            "pht": "O1,O2,N1,N2",
            "cbz": "O2,N3,N4"
        }
    }
}

for idx, row in tqdm(df.iterrows(), total=len(df)):
    pdb_path = row[('link', 'path')]
    tag = row[('link', 'tag')]
    batch_name = row[('diffusion', 'batch')]
    lig_name = batch_name.split('_')[0]
    if 'af3' in tag: model = 'af3'
    elif 'boltz' in tag: model = 'boltz'
    # ligand_params_path = mapping['ligand_params_path'][model][lig_name]
    ligand_params_path = f'/home/hwjang/aipd/250729/5_cif_to_mol2/params/{tag}.params'
    target_atm_for_cst = mapping['target_atm_for_cst'][model][lig_name]
    out_folder = f"{lig_name}_fastrelax"
    cmd = (
        f"python /home/hwjang/aipd/ligandMPNN_FR/ligandmpnn_fastrelax_complete.py "
        f"--path_to_model_weights /home/hwjang/aipd/LigandMPNN/model_params "
        f"--pdb_path {pdb_path} "
        f"--ligand_params_path {ligand_params_path} "
        f"--out_folder {out_folder} "
        "--num_seq_per_target 16 "
        "--num_processes 16 "
        "--pyrosetta_threads 6 "
        "--n_cycles 32 "
        "--pack_side_chains "
        "--temperature 0.1 "
        "--save_stats "
        "--seed 42 "
        "--repackable_res '' "
        f"--target_atm_for_cst {target_atm_for_cst} "
        "--selection_metric ddg "
        "--selection_order ascending"
    )
    os.system(cmd)
