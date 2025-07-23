from Bio.PDB import PDBParser
import numpy as np

CST_STDERR = 0.5

def extract_dist_cst_from_pdb(pdb_in, lig_tr_atms, bsite_res=''):
    """
    pdb_in      : PDB 파일 경로
    lig_tr_atms : ['NE1','O', ...] 등 리간드에서 관심있는 atom 이름 리스트
    bsite_res   : '' 이면 모든 단백질 CA 사용, 아니면 '10,15,27' 처럼 콤마로 구분된 residue 번호 문자열
    return      : ['AtomPair <lig_atm> <het_resno> CA <closest_resno> HARMONIC <dist> 0.5', ...]
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("strc", pdb_in)

    # 체인/모델은 하나라고 가정(필요시 loop 추가)
    model = next(structure.get_models())

    # bsite residue set
    if bsite_res == '':
        # 모든 단백질 CA residue 번호
        bsite_res_s = {res.get_id()[1] for res in model.get_residues()
                       if res.get_id()[0] == ' ' and 'CA' in res}  # ' ' => ATOM
    else:
        bsite_res_s = {int(x) for x in bsite_res.split(',')}

    # 단백질 마지막 residue 번호(리간드 resno 추정용)
    last_prot_resno = max(res.get_id()[1] for res in model.get_residues()
                          if res.get_id()[0] == ' ')

    # CA 좌표 dict
    bsite_CA_R = {}
    for res in model.get_residues():
        if res.get_id()[0] == ' ' and res.get_id()[1] in bsite_res_s and 'CA' in res:
            bsite_CA_R[res.get_id()[1]] = res['CA'].get_coord()

    # 리간드 atom 좌표 dict
    het_atm_R = {}
    for res in model.get_residues():
        if res.get_id()[0].startswith('H'):  # HETATM
            for atom in res:
                if atom.get_name() in lig_tr_atms:
                    het_atm_R[atom.get_name()] = atom.get_coord()

    het_resno = last_prot_resno + 1  # diffusion 등 불연속 번호 대응

    # 거리 계산 (벡터화)
    cst_s = []
    bsite_resnos = np.array(list(bsite_CA_R.keys()))
    bsite_coords = np.array([bsite_CA_R[r] for r in bsite_resnos])

    for atm_name in lig_tr_atms:
        het_coord = het_atm_R[atm_name]

        # 모든 CA와의 거리
        diff = bsite_coords - het_coord
        dists = np.sqrt((diff ** 2).sum(axis=1))

        # 최단 거리 residue
        idx = np.argmin(dists)
        closest_resno = int(bsite_resnos[idx])
        closest_d = float(dists[idx])

        cst_line = f'AtomPair {atm_name} {het_resno} CA {closest_resno} HARMONIC {closest_d} {CST_STDERR}'
        cst_s.append(cst_line)

    return cst_s
