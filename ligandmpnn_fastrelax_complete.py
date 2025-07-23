#!/usr/bin/env python3
"""
LigandMPNN with Fast Relax Integration - Complete Version (spawn start method)
Updated to use multiprocessing.spawn to ensure fresh PyRosetta threading

Author: Updated version based on successful debugging session
Date: 2025
"""

import os
import sys
import json
import random
import argparse
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

# Import XML templates and constraint generation
from xml_relax_after_ligMPNN import XML_BSITE_FASTRELAX
from gen_prot_lig_dist_cst import extract_dist_cst_from_pdb

# Global function for multiprocessing - must be at module level
def fast_relax_worker(input_data):
    """
    Worker function for multiprocessing fast relax with simplified XML-based protocol
    Each process gets its own PyRosetta instance and uses minimal threading
    Extracts key metrics: ddg, cms, totalscore
    """
    import os
    import tempfile
    
    pdb_path, output_path, ligand_params_path, use_genpot, verbose, pyrosetta_threads, repackable_res, target_atm_for_cst = input_data
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
        import tempfile
        import os

        # Improved init flags - use flexible threading
        init_flags = ['-beta', '-mute all']  # Always mute to reduce log spam
        
        # Use configured number of threads per process (now allowing more than 1)
        init_flags.extend([
            f'-multithreading:total_threads {pyrosetta_threads}',
            f'-multithreading:interaction_graph_threads {pyrosetta_threads}'
        ])
        
        # Add ligand parameters
        if use_genpot:
            init_flags.extend(['-gen_potential', f'-extra_res_fa {ligand_params_path}'])
        else:
            init_flags.append(f'-extra_res_fa {ligand_params_path}')
            
        # Add native structure for coordinate constraints
        init_flags.append(f'-in:file:native {pdb_path}')

        # Initialize PyRosetta
        pyrosetta.init(' '.join(init_flags))

        # Load pose
        pose = pyrosetta.pose_from_pdb(pdb_path)
        if pose.total_residue() == 0:
            raise ValueError("Empty pose loaded")

        # Generate distance constraints if target atoms specified
        cst_file_path = None
        if target_atm_for_cst:
            try:
                target_atoms = target_atm_for_cst.split(',') if isinstance(target_atm_for_cst, str) else target_atm_for_cst
                if target_atoms and target_atoms != ['']:
                    # Generate constraints
                    if verbose:
                        print(f"Generating constraints for atoms: {target_atoms}")
                    csts = extract_dist_cst_from_pdb(pdb_path, target_atoms, bsite_res=repackable_res)
                    if csts:
                        # Create temporary constraint file
                        cst_fd, cst_file_path = tempfile.mkstemp(suffix='.cst', text=True)
                        with os.fdopen(cst_fd, 'w') as f:
                            f.write('\n'.join(csts) + '\n')
                        if verbose:
                            print(f"Created constraint file with {len(csts)} constraints")
                    else:
                        if verbose:
                            print("No constraints generated")
                else:
                    if verbose:
                        print("Empty target atoms, skipping constraint generation")
            except Exception as e:
                if verbose:
                    print(f"Error generating constraints: {e}")
                    print(f"target_atm_for_cst: {repr(target_atm_for_cst)}")
                    print(f"target_atoms: {repr(target_atoms) if 'target_atoms' in locals() else 'not defined'}")
                # Continue with empty constraint file
                pass

        # If no constraint file, create empty temp file
        if cst_file_path is None:
            cst_fd, cst_file_path = tempfile.mkstemp(suffix='.cst', text=True)
            with os.fdopen(cst_fd, 'w') as f:
                f.write('')

        # Prepare simplified XML script with repackable residues
        if not repackable_res:
            # If no repackable residues specified, get all protein residues
            repack_res_str = ','.join([str(i) for i in range(1, pose.total_residue())])
        else:
            repack_res_str = repackable_res

        # Use the complete XML template with full DDG calculations
        complete_xml = XML_BSITE_FASTRELAX.format(repack_res_str, cst_file_path)
        
        # Create temporary XML file
        xml_fd, xml_file_path = tempfile.mkstemp(suffix='.xml', text=True)
        with os.fdopen(xml_fd, 'w') as f:
            f.write(complete_xml)

        metrics = {}
        try:
            # Parse and run XML script
            objs = XmlObjects.create_from_file(xml_file_path)
            
            # Apply the protocol
            protocol = objs.get_mover('ParsedProtocol')
            protocol.apply(pose)
            
            # Extract comprehensive metrics
            try:
                # Get contact molecular surface
                cms_filter = objs.get_filter('cms') 
                if cms_filter:
                    metrics['cms'] = cms_filter.report_sm(pose)
                
                # Get total score
                totalscore_filter = objs.get_filter('totalscore')
                if totalscore_filter:
                    metrics['totalscore'] = totalscore_filter.report_sm(pose)
                
                # Get residue-normalized total score
                res_totalscore_filter = objs.get_filter('res_totalscore')
                if res_totalscore_filter:
                    metrics['res_totalscore'] = res_totalscore_filter.report_sm(pose)
                
                # Get DDG after relaxation with constraints
                ddg_after_relax_filter = objs.get_filter('ddg_after_relax_cst')
                if ddg_after_relax_filter:
                    metrics['ddg_after_relax_cst'] = ddg_after_relax_filter.report_sm(pose)
                
                # Get DDG without constraints (more accurate binding energy)
                ddg_filter = objs.get_filter('ddg')
                if ddg_filter:
                    metrics['ddg'] = ddg_filter.report_sm(pose)
                else:
                    # Fallback to DDG with constraints
                    metrics['ddg'] = metrics.get('ddg_after_relax_cst', metrics.get('totalscore', 0.0))
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not extract some metrics: {e}")
                # Set fallback values for missing metrics
                from pyrosetta.rosetta.core.scoring import get_score_function
                scorefxn = get_score_function()
                total_energy = scorefxn(pose)
                metrics['totalscore'] = total_energy
                metrics['ddg'] = total_energy  # Use total score as ddg fallback
                metrics['cms'] = 0.0
                metrics['res_totalscore'] = total_energy / pose.total_residue()
                metrics['ddg_after_relax_cst'] = total_energy
                    
        finally:
            # Clean up temporary files
            try:
                os.unlink(xml_file_path)
                os.unlink(cst_file_path)
            except Exception:
                pass

        # Save output
        pose.dump_pdb(output_path)
        if not os.path.exists(output_path):
            raise RuntimeError("Output PDB file was not created")

        return {
            'success': True, 
            'input_path': pdb_path, 
            'output_path': output_path, 
            'error': None,
            'metrics': metrics
        }
        
    except Exception as e:
        return {
            'success': False, 
            'input_path': pdb_path, 
            'output_path': output_path, 
            'error': str(e),
            'metrics': {}
        }


def get_worker_config(num_structures, args):
    available_cores = cpu_count()
    num_processes = min(
        getattr(args, 'num_processes', 1),
        num_structures,
        available_cores
    )
    pyrosetta_threads = getattr(args, 'pyrosetta_threads', 2)
    total_threads = num_processes * pyrosetta_threads
    if total_threads > available_cores:
        print(f"Warning: total threads ({total_threads}) exceed available CPU cores ({available_cores}). "
              "This may cause oversubscription and reduce performance.")
    return {
        'num_processes': num_processes,
        'pyrosetta_threads': pyrosetta_threads
    }


def setup_ligandmpnn():
    """Setup LigandMPNN model and dependencies"""
    # Add LigandMPNN to path
    ligandmpnn_path = '/home/hwjang/aipd/LigandMPNN'
    if ligandmpnn_path not in sys.path:
        sys.path.insert(0, ligandmpnn_path)
    
    try:
        from data_utils import (
            alphabet,
            featurize,
            get_score,
            parse_PDB,
            restype_int_to_str,
            restype_str_to_int,
            restype_1to3,
            write_full_PDB,
        )
        from model_utils import ProteinMPNN
        from sc_utils import Packer, pack_side_chains
        from prody import writePDB
        return True, (parse_PDB, featurize, get_score, alphabet, restype_int_to_str, restype_str_to_int, restype_1to3, ProteinMPNN, write_full_PDB, Packer, pack_side_chains, writePDB)
    except ImportError as e:
        print(f"Error importing LigandMPNN modules: {e}")
        return False, None

def setup_pyrosetta(ligand_params_path, native_pdb_path, use_genpot=False, verbose=True):
    """Setup PyRosetta with appropriate flags"""
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.scoring import get_score_function
        from pyrosetta import toolbox
        
        # Build initialization flags
        init_flags = ['-beta']
        if not verbose:
            init_flags.append('-mute all')
        
        # Improved multithreading configuration for main process
        # Use environment variables to determine thread count
        pyrosetta_threads = int(os.environ.get('ROSETTA_NUM_THREADS', '4'))
        init_flags.extend([
            f'-multithreading:total_threads {pyrosetta_threads}',
            f'-multithreading:interaction_graph_threads {pyrosetta_threads}'
        ])
        
        if use_genpot:
            init_flags.extend([
                '-gen_potential',
                f'-extra_res_fa {ligand_params_path}',
                f'-in:file:native {native_pdb_path}'
            ])
        else:
            init_flags.extend([
                f'-extra_res_fa {ligand_params_path}',
                f'-in:file:native {native_pdb_path}'
            ])
        
        if verbose:
            print("Initializing PyRosetta...")
        pyrosetta.init(' '.join(init_flags))
        
        return True, (pyrosetta, FastRelax, get_score_function, toolbox)
    
    except ImportError as e:
        print(f"Error importing PyRosetta: {e}")
        return False, None


class LigandMPNNFastRelax:
    """
    Class for iterative LigandMPNN design with PyRosetta fast relax
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup LigandMPNN
        success, modules = setup_ligandmpnn()
        if not success:
            raise RuntimeError("Failed to setup LigandMPNN")
        
        self.parse_PDB, self.featurize, self.get_score, self.alphabet, self.restype_int_to_str, self.restype_str_to_int, self.restype_1to3, self.ProteinMPNN, self.write_full_PDB, self.Packer, self.pack_side_chains, self.writePDB = modules
        
        # Load model
        self.model = self._load_ligandmpnn_model()
        
        # Load side chain packing model if enabled
        self.model_sc = None
        if getattr(args, 'pack_side_chains', False):
            try:
                if args.verbose:
                    print("Loading side chain packing model...")
                self.model_sc = self._load_packer_model()
                if self.model_sc is not None:
                    if args.verbose:
                        print("Side chain packing model loaded successfully!")
                else:
                    if args.verbose:
                        print("Side chain packing model failed to load!")
            except Exception as e:
                if args.verbose:
                    print(f"Warning: Could not load side chain packer: {e}")
                    print("Side chain packing will be disabled")
                self.model_sc = None
        else:
            if args.verbose:
                print("Side chain packing is disabled (pack_side_chains=False)")
        
        # Setup PyRosetta
        success, modules = setup_pyrosetta(
            args.ligand_params_path, 
            args.pdb_path, 
            args.use_genpot,
            args.verbose
        )
        if not success:
            raise RuntimeError("Failed to setup PyRosetta")
        
        self.pyrosetta, self.FastRelax, self.get_score_function, self.toolbox = modules
        
        # Setup output directories
        self._setup_output_dirs()
        
    def _load_ligandmpnn_model(self):
        """Load LigandMPNN model (robust path handling like run.py)"""
        checkpoint_path = self.args.checkpoint_path
        if not checkpoint_path:
            # Handle path_to_model_weights properly
            if hasattr(self.args, 'path_to_model_weights') and self.args.path_to_model_weights:
                model_folder = self.args.path_to_model_weights
                if model_folder[-1] != '/':
                    model_folder += '/'
                # If the path doesn't end with model_params/, add it
                if not model_folder.endswith('model_params/'):
                    model_folder += 'model_params/'
            else:
                model_folder = './model_params/'
            checkpoint_path = f"{model_folder}{getattr(self.args, 'model_name', 'ligandmpnn_v_32_010_25')}.pt"
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("Could not determine checkpoint_path for LigandMPNN model.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        if self.args.verbose:
            print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        atom_context_num = checkpoint.get("atom_context_num", 1)
        k_neighbors = checkpoint["num_edges"]
        model = self.ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=self.device,
            atom_context_num=atom_context_num,
            model_type="ligand_mpnn",
            ligand_mpnn_use_side_chain_context=self.args.ligand_mpnn_use_side_chain_context,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model
        
    def _load_packer_model(self):
        """Load LigandMPNN side chain packer model following run.py pattern"""
        # Set default checkpoint path if not provided
        checkpoint_path_sc = getattr(self.args, 'checkpoint_path_sc', None)
        
        if not checkpoint_path_sc:
            # Use default path structure like run.py
            model_folder = getattr(self.args, 'path_to_model_weights', '/home/hwjang/aipd/LigandMPNN/')
            if model_folder[-1] != '/':
                model_folder += '/'
            checkpoint_path_sc = f"{model_folder}ligandmpnn_sc_v_32_002_16.pt"
        
        # Validate checkpoint path
        if not checkpoint_path_sc or not isinstance(checkpoint_path_sc, str):
            if self.args.verbose:
                print("Invalid checkpoint path for side chain packer. Packing disabled.")
            return None
            
        if not os.path.exists(checkpoint_path_sc):
            if self.args.verbose:
                print(f"Side chain packer model not found: {checkpoint_path_sc}")
                print("Side chain packing will be disabled")
            return None
            
        if self.args.verbose:
            print(f"Loading side chain packer from: {checkpoint_path_sc}")
            
        try:
            checkpoint_sc = torch.load(checkpoint_path_sc, map_location=self.device, weights_only=True)
            
            # Create side chain packer model following run.py exactly
            model_sc = self.Packer(
                node_features=128,
                edge_features=128,
                num_positional_embeddings=16,
                num_chain_embeddings=16,
                num_rbf=16,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                atom_context_num=16,
                lower_bound=0.0,
                upper_bound=20.0,
                top_k=32,
                dropout=0.0,
                augment_eps=0.0,
                atom37_order=False,
                device=self.device,
                num_mix=3,
            )
            
            model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
            model_sc.to(self.device)
            model_sc.eval()
            
            return model_sc
            
        except Exception as e:
            if self.args.verbose:
                print(f"Error loading side chain packer: {e}")
                print("Side chain packing will be disabled")
            return None
        
    def _setup_output_dirs(self):
        """Create output directories"""
        self.base_folder = self.args.out_folder
        if self.base_folder[-1] != "/":
            self.base_folder += "/"
            
        os.makedirs(self.base_folder, exist_ok=True)
        os.makedirs(f"{self.base_folder}seqs", exist_ok=True)
        os.makedirs(f"{self.base_folder}backbones", exist_ok=True)
        os.makedirs(f"{self.base_folder}relaxed", exist_ok=True)
        
        if self.args.save_stats:
            os.makedirs(f"{self.base_folder}stats", exist_ok=True)
            
    def generate_sequences(self, pdb_path, num_sequences=1, temperature=0.1,
                          fixed_residues=None, redesigned_residues=None):
        """
        Generate sequences using LigandMPNN with batch processing
        
        Args:
            pdb_path: Path to input PDB file
            num_sequences: Number of sequences to generate
            temperature: Sampling temperature
            fixed_residues: List of residues to keep fixed
            redesigned_residues: List of residues to redesign
            
        Returns:
            List of generated sequences with scores
        """
        if self.args.verbose:
            print(f"Generating {num_sequences} sequence(s) for: {pdb_path}")
        
        # Parse PDB structure
        protein_dict, backbone, other_atoms, icodes, _ = self.parse_PDB(
            pdb_path,
            device=self.device,
            chains=[],  # Parse all chains
            parse_all_atoms=self.args.ligand_mpnn_use_side_chain_context or (
                self.args.pack_side_chains and not getattr(self.args, 'repack_everything', False)
            ),
            parse_atoms_with_zero_occupancy=False,
        )
        
        # Create residue encoding
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters_list = list(protein_dict["chain_letters"])
        
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            chain_letter = chain_letters_list[i]
            icode = icodes[i] if i < len(icodes) else ""
            encoded_residue = f"{chain_letter}{R_idx_item}{icode}"
            encoded_residues.append(encoded_residue)
            
        # Set up masks for fixed/redesigned residues
        if fixed_residues is None:
            fixed_residues = []
        if redesigned_residues is None:
            redesigned_residues = []
            
        # Create chain mask
        if redesigned_residues:
            chain_mask = torch.tensor(
                [int(item in redesigned_residues) for item in encoded_residues],
                device=self.device,
            )
        elif fixed_residues:
            chain_mask = torch.tensor(
                [int(item not in fixed_residues) for item in encoded_residues],
                device=self.device,
            )
        else:
            # Design all residues
            chain_mask = torch.ones(len(encoded_residues), device=self.device)
            
        # Add chain_mask to protein_dict
        protein_dict["chain_mask"] = chain_mask
        
        # Featurize the structure (using single dict, not list)
        feature_dict = self.featurize(
            protein_dict,
            cutoff_for_score=8.0,
            use_atom_context=getattr(self.args, 'ligand_mpnn_use_atom_context', True),
            number_of_ligand_atoms=16,
            model_type="ligand_mpnn",
        )
        
        # Calculate optimal batch size
        max_batch_size = getattr(self.args, 'max_batch_size', 8)  # Adjustable max batch
        batch_size = min(num_sequences, max_batch_size)
        num_batches = (num_sequences + batch_size - 1) // batch_size  # Ceiling division
        
        if self.args.verbose:
            print(f"Processing {num_sequences} sequences in {num_batches} batches of size {batch_size}")
        
        B, L, _, _ = feature_dict["X"].shape

        # Set bias in feature_dict as done in successful simple version
        omit_AA = torch.tensor(
            np.array([AA in self.args.omit_AAs for AA in self.alphabet]).astype(np.float32),
            device=self.device
        )
        bias_AA = torch.zeros(len(self.alphabet), device=self.device)

        # Create per-residue omit and bias (zeros for now)
        omit_AA_per_residue = torch.zeros([L, 21], device=self.device)
        bias_AA_per_residue = torch.zeros([L, 21], device=self.device)

        # Set bias in feature_dict following successful pattern
        feature_dict["bias"] = (
            (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
            + bias_AA_per_residue[None]
            - 1e8 * omit_AA_per_residue[None]
        )

        # Add symmetry information (empty lists)
        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]
        
        generated_sequences = []
        
        # Process in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_sequences)
            current_batch_size = end_idx - start_idx
            
            # Set batch size and temperature for this batch
            feature_dict["batch_size"] = current_batch_size
            feature_dict["temperature"] = temperature
            
            # Add randn for sampling
            feature_dict["randn"] = torch.randn([current_batch_size, L], device=self.device)
            
            with torch.no_grad():
                # Sample sequences using batch processing
                output_dict = self.model.sample(feature_dict)
                
                # Get sequences and scores
                S_samples = output_dict["S"]  # Shape: [batch_size, L]
                log_probs = output_dict.get("log_probs", None)
                
                # Process each sequence in the batch
                for i in range(current_batch_size):
                    seq = self._S_to_seq(S_samples[i])
                    
                    # Calculate score if log_probs available
                    score = 0.0
                    if log_probs is not None:
                        mask = feature_dict["mask"] * feature_dict["chain_mask"]
                        # For batch processing, we need to select the right sequence
                        S_single = S_samples[i:i+1]  # Keep batch dimension
                        log_probs_single = log_probs[i:i+1]
                        score, _ = self.get_score(S_single, log_probs_single, mask)
                        score = score.item()
                        
                    generated_sequences.append({
                        'sequence': seq,
                        'score': score,
                        'S_sample': S_samples[i],
                        'temperature': temperature,
                        'index': start_idx + i,
                        'batch_idx': batch_idx
                    })
                    
                    if self.args.verbose and (start_idx + i + 1) % 5 == 0:
                        print(f"Generated sequence {start_idx + i + 1}/{num_sequences}: score {score:.4f}")
                
        if self.args.verbose:
            avg_score = np.mean([seq['score'] for seq in generated_sequences])
            print(f"Generated {len(generated_sequences)} sequences, average score: {avg_score:.4f}")
                
        return generated_sequences
        
    def _S_to_seq(self, S):
        """Convert sequence tensor to string"""
        return ''.join([self.restype_int_to_str[s.item()] for s in S])
        
    def create_structure_with_sequence(self, pdb_path, sequence, S_sample, output_path):
        """
        Create structure with new sequence using LigandMPNN's write_full_PDB
        Following run.py pattern closely for proper side chain packing
        """
        if self.args.verbose:
            print("Creating structure with new sequence using LigandMPNN...")
        
        # Parse PDB structure - use different parsing modes based on side chain packing
        parse_all_atoms_flag = (
            self.args.ligand_mpnn_use_side_chain_context or (
                self.args.pack_side_chains and not getattr(self.args, 'repack_everything', False)
            )
        )
        
        protein_dict, backbone, other_atoms, icodes, _ = self.parse_PDB(
            pdb_path, device=self.device, chains=[],
            parse_all_atoms=parse_all_atoms_flag,
            parse_atoms_with_zero_occupancy=False
        )
        
        # Set the sequence in protein_dict
        protein_dict["S"] = S_sample.squeeze() if S_sample.dim() > 1 else S_sample
        if "chain_mask" not in protein_dict:
            protein_dict["chain_mask"] = torch.ones(len(protein_dict["R_idx"]), device=self.device)

        # Check if side chain packing is enabled and model is loaded
        if self.args.pack_side_chains and self.model_sc is not None:
            if self.args.verbose:
                print("Packing side chains...")
            
            # Featurize for side chain packing following run.py pattern
            feature_dict_ = self.featurize(
                protein_dict,
                cutoff_for_score=8.0,
                use_atom_context=getattr(self.args, 'pack_with_ligand_context', True),
                number_of_ligand_atoms=16,
                model_type="ligand_mpnn",
            )
            
            # Prepare feature dict for side chain packing (following run.py pattern exactly)
            import copy
            sc_feature_dict = copy.deepcopy(feature_dict_)
            B = 1  # batch size for single sequence
            
            # Repeat tensors for batch processing (following run.py pattern)
            for k, v in sc_feature_dict.items():
                if k != "S":
                    try:
                        num_dim = len(v.shape)
                        if num_dim == 2:
                            sc_feature_dict[k] = v.repeat(B, 1)
                        elif num_dim == 3:
                            sc_feature_dict[k] = v.repeat(B, 1, 1)
                        elif num_dim == 4:
                            sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                        elif num_dim == 5:
                            sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
                    except:
                        pass
            
            # Set the sequence for side chain packing
            # S_sample should be unsqueezed to add batch dimension
            S_for_packing = S_sample.squeeze() if S_sample.dim() > 1 else S_sample
            sc_feature_dict["S"] = S_for_packing.unsqueeze(0)  # Add batch dimension
            
            if self.args.verbose:
                print(f"Side chain packing input: S shape={sc_feature_dict['S'].shape}")
            
            # Pack side chains using LigandMPNN's pack_side_chains function
            try:
                sc_dict = self.pack_side_chains(
                    sc_feature_dict,
                    self.model_sc,
                    getattr(self.args, 'sc_num_denoising_steps', 3),
                    getattr(self.args, 'sc_num_samples', 16),
                    getattr(self.args, 'repack_everything', False),
                )
                
                if sc_dict is not None:
                    if self.args.verbose:
                        print("Side chain packing completed successfully")
                        print(f"Side chain packing output: X shape={sc_dict['X'].shape}, X_m shape={sc_dict['X_m'].shape}")
                    
                    # Extract packed coordinates
                    X_packed = sc_dict["X"]
                    X_m_packed = sc_dict["X_m"]
                    
                    # Remove batch dimension if present
                    if X_packed.dim() > 3:  # Expected: [L, 14, 3], but might be [1, L, 14, 3]
                        X_packed = X_packed.squeeze(0)
                    if X_m_packed.dim() > 2:  # Expected: [L, 14], but might be [1, L, 14]
                        X_m_packed = X_m_packed.squeeze(0)
                    
                    if self.args.verbose:
                        print(f"After squeeze: X shape={X_packed.shape}, X_m shape={X_m_packed.shape}")
                    
                    # Handle b_factors
                    if "b_factors" in sc_dict:
                        b_factors = sc_dict["b_factors"]
                        if hasattr(b_factors, 'dim') and b_factors.dim() > 2:
                            b_factors = b_factors.squeeze(0)
                        if hasattr(b_factors, 'detach'):
                            b_factors = b_factors.detach().cpu().numpy()
                        elif hasattr(b_factors, 'cpu'):
                            b_factors = b_factors.cpu().numpy()
                    else:
                        b_factors = np.ones_like(X_m_packed.detach().cpu().numpy())
                    
                    # Write full PDB using LigandMPNN's function with packed coordinates
                    self.write_full_PDB(
                        save_path=output_path,
                        X=X_packed.detach().cpu().numpy(),
                        X_m=X_m_packed.detach().cpu().numpy(),
                        b_factors=b_factors,
                        R_idx=protein_dict["R_idx"].cpu().numpy(),
                        chain_letters=protein_dict["chain_letters"],
                        S=protein_dict["S"].cpu().numpy(),
                        other_atoms=other_atoms,
                        icodes=icodes,
                        force_hetatm=getattr(self.args, 'force_hetatm', False)
                    )
                    
                    if self.args.verbose:
                        print(f"Structure with packed side chains saved: {output_path}")
                    return output_path
                else:
                    if self.args.verbose:
                        print("Side chain packing failed, falling back to backbone-only structure...")
                    # Fall through to backbone-only writing
                    
            except Exception as e:
                if self.args.verbose:
                    print(f"Warning: Side chain packing failed: {e}")
                    print("Falling back to backbone-only structure...")
                # Fall through to backbone-only writing
        else:
            # No side chain packing - use backbone coordinates like original run.py
            if self.args.verbose:
                if not self.args.pack_side_chains:
                    print("Side chain packing disabled, using backbone coordinates...")
                elif self.model_sc is None:
                    print("Side chain packer model not loaded, using backbone coordinates...")
                else:
                    print("No side chain packing, using backbone coordinates...")
        
        # Backbone-only structure creation (following run.py pattern)
        try:
            # Convert sequence to prody format following run.py pattern
            seq_prody = np.array([self.restype_1to3[AA] for AA in list(sequence)])[None,].repeat(4, 1)
            
            # Set residue names in backbone
            backbone.setResnames(seq_prody)
            
            # Set B-factors to 1.0 for all atoms
            backbone.setBetas(np.ones_like(backbone.getBetas()))
            
            # Write PDB using prody (following run.py pattern)
            if other_atoms:
                self.writePDB(output_path, backbone + other_atoms)
            else:
                self.writePDB(output_path, backbone)
            
            if self.args.verbose:
                print(f"Structure with new sequence saved: {output_path}")
            return output_path
            
        except Exception as e:
            if self.args.verbose:
                print(f"Warning: Backbone writing failed: {e}")
    
    def fast_relax_parallel(self, structure_paths):
        worker_config = get_worker_config(len(structure_paths), self.args)
        num_processes = worker_config['num_processes']
        pyrosetta_threads = worker_config['pyrosetta_threads']
        if self.args.verbose:
            print(f"Fast relax: {len(structure_paths)} structures, {num_processes} processes, {pyrosetta_threads} threads/process")
        
        # Get repackable residues and target atoms for constraints
        repackable_res = getattr(self.args, 'repackable_res', '')
        target_atm_for_cst = getattr(self.args, 'target_atm_for_cst', '')
        
        worker_inputs = []
        for pdb_path in structure_paths:
            relaxed_path = pdb_path.replace('/backbones/', '/relaxed/').replace('.pdb', '_relaxed.pdb')
            os.makedirs(os.path.dirname(relaxed_path), exist_ok=True)
            worker_inputs.append((
                pdb_path, relaxed_path,
                self.args.ligand_params_path,
                self.args.use_genpot,
                self.args.verbose,
                pyrosetta_threads,
                repackable_res,
                target_atm_for_cst
            ))

        successful_paths = []
        all_metrics = []
        failed_count = 0
        try:
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=num_processes, mp_context=ctx) as executor:
                futures = {executor.submit(fast_relax_worker, inp): inp for inp in worker_inputs}
                for future in as_completed(futures):
                    result = future.result()
                    if result['success']:
                        successful_paths.append(result['output_path'])
                        all_metrics.append(result['metrics'])
                    else:
                        failed_count += 1
                        if self.args.verbose:
                            print(f"Failed {result['input_path']}: {result['error']}")
                        import shutil
                        shutil.copy2(result['input_path'], result['output_path'])
                        successful_paths.append(result['output_path'])
                        all_metrics.append({})  # Empty metrics for failed relaxation
            if self.args.verbose and failed_count > 0:
                print(f"Fast relax completed: {len(successful_paths)}/{len(structure_paths)} successful")
        except Exception as e:
            if self.args.verbose:
                print(f"Error during fast relax: {e}")
            raise e
        return successful_paths, all_metrics

    def run_mpnn_fastrelax_cycle(self, pdb_path, cycle_num=1):
        """
        Run one cycle of LigandMPNN design + fast relax with parallel processing
        
        Args:
            pdb_path: Input PDB path
            cycle_num: Current cycle number
            
        Returns:
            Path to best relaxed PDB file
        """
        if self.args.verbose:
            print(f"\n{'='*50}")
            print(f"Starting cycle {cycle_num}")
            print(f"{'='*50}")
            
        # Generate sequences
        sequences = self.generate_sequences(
            pdb_path,
            num_sequences=self.args.num_seq_per_target,
            temperature=self.args.temperature,
            fixed_residues=getattr(self.args, 'fixed_residues', []),
            redesigned_residues=getattr(self.args, 'redesigned_residues', [])
        )
        
        if self.args.verbose:
            print(f"Generated {len(sequences)} sequences for cycle {cycle_num}")
            scores = [seq['score'] for seq in sequences]
            print(f"Score range: {min(scores):.4f} to {max(scores):.4f}")
        
        # Create structures for all sequences (if multiple sequences generated)
        header = os.path.basename(pdb_path).replace('.pdb', '')
        structure_paths = []
        # Multiple sequences - create structures for all and parallel relax
        for i, seq_data in enumerate(sequences):
            threaded_path = f"{self.base_folder}backbones/{header}_cycle_{cycle_num}_seq_{i}.pdb"
            
            self.create_structure_with_sequence(
                pdb_path, 
                seq_data['sequence'], 
                seq_data['S_sample'], 
                threaded_path
            )
            structure_paths.append(threaded_path)
        
        # Perform fast relax (parallel if multiple structures)
        # Multiple structures - use parallel relaxation
        if self.args.verbose:
            print(f"Performing parallel fast relax on {len(structure_paths)} structures")
            
        relaxed_paths, all_metrics = self.fast_relax_parallel(
            structure_paths, 
            # max_workers=getattr(self.args, 'max_relax_workers', None)
        )
        
        # Evaluate relaxed structures and select best
        best_sequence, best_relaxed_path = self._select_best_relaxed_structure(
            sequences, relaxed_paths, all_metrics, cycle_num
        )
        
        # Move best structure to final location
        final_relaxed_path = f"{self.base_folder}relaxed/{header}_cycle_{cycle_num}_relaxed.pdb"
        if best_relaxed_path != final_relaxed_path:
            import shutil
            shutil.copy2(best_relaxed_path, final_relaxed_path)
        relaxed_paths = [final_relaxed_path]
        
        # Save sequence information
        seq_path = f"{self.base_folder}seqs/{header}_cycle_{cycle_num}.fa"
        with open(seq_path, 'w') as f:
            f.write(f">cycle_{cycle_num}_score_{best_sequence['score']:.4f}\n")
            f.write(f"{best_sequence['sequence']}\n")
            
        # Save statistics if requested
        if self.args.save_stats:
            stats_path = f"{self.base_folder}stats/{header}_cycle_{cycle_num}.json"
            stats = {
                'cycle': cycle_num,
                'input_pdb': pdb_path,
                'relaxed_pdb': relaxed_paths[0],
                'best_sequence': best_sequence['sequence'],
                'best_score': best_sequence['score'],
                'temperature': best_sequence['temperature'],
                'num_sequences_generated': len(sequences),
                'sequence_diversity': len(set([seq['sequence'] for seq in sequences])),
                # Simplified selection method
                'selection_metric': self.args.selection_metric,
                'selection_order': self.args.selection_order,
                'selection_value': best_sequence.get('selection_value', None),
                # Individual scores for best structure
                'best_mpnn_score': best_sequence.get('mpnn', best_sequence['score']),
                # All available metrics for best structure
                'best_metrics': best_sequence.get('metrics', {}),
                # All scores for analysis
                'all_mpnn_scores': [seq['score'] for seq in sequences],
                'all_detailed_scores': best_sequence.get('all_scores', []),
            }
                
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
                
        if self.args.verbose:
            print(f"Cycle {cycle_num} completed")
            print(f"Best sequence: {best_sequence['sequence']}")
            print(f"Best MPNN score: {best_sequence.get('mpnn', best_sequence['score']):.4f}")
            print(f"Selection metric ({self.args.selection_metric}): {best_sequence.get('selection_value', 'N/A'):.4f}")
            
            # Show additional XML metrics if available
            if 'metrics' in best_sequence and best_sequence['metrics']:
                metrics = best_sequence['metrics']
                metric_items = []
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if key in ['ddg', 'totalscore', 'ddg_after_relax_cst']:
                            metric_items.append(f"{key}={value:.2f}")
                        elif key == 'cms':
                            metric_items.append(f"{key}={value:.4f}")
                        elif key == 'res_totalscore':
                            metric_items.append(f"{key}={value:.3f}")
                        else:
                            metric_items.append(f"{key}={value:.2f}")
                if metric_items:
                    print(f"All metrics: {', '.join(metric_items)}")
            print(f"Final relaxed structure: {relaxed_paths[0]}")
            
        return best_sequence, relaxed_paths[0]
    
    def _select_best_relaxed_structure(self, sequences, relaxed_paths, all_metrics, cycle_num):
        """
        Select best relaxed structure based on single metric (simplified from combined scoring)
        """
        if self.args.verbose:
            print(f"Evaluating {len(sequences)} relaxed structures to select best...")
            print(f"Using selection metric: {self.args.selection_metric} ({self.args.selection_order})")
            
        best_score = float('inf') if self.args.selection_order == 'ascending' else float('-inf')
        best_sequence = None
        best_path = None
        
        # Store individual scores for statistics
        all_scores = []
        
        # Extract scores from sequences and metrics
        mpnn_scores = [seq_data['score'] for seq_data in sequences]
        
        if self.args.verbose:
            print(f"MPNN scores range: {min(mpnn_scores):.4f} to {max(mpnn_scores):.4f}")
            
        # Extract all available metrics for display
        metric_values = {}
        for metric in ['ddg', 'totalscore', 'res_totalscore', 'cms', 'ddg_after_relax_cst']:
            values = [metrics.get(metric, float('inf')) for metrics in all_metrics]
            metric_values[metric] = values
            if any(v != float('inf') for v in values):
                print(f"{metric.upper()} range: {min(v for v in values if v != float('inf')):.3f} to {max(v for v in values if v != float('inf')):.3f}")
        
        # Select best structure based on chosen metric
        for i, (seq_data, relaxed_path) in enumerate(zip(sequences, relaxed_paths)):
            mpnn_score = mpnn_scores[i]
            structure_metrics = all_metrics[i]
            
            # Get the value for the selection metric
            if self.args.selection_metric == 'mpnn':
                selection_value = mpnn_score
            else:
                selection_value = structure_metrics.get(self.args.selection_metric, float('inf'))
            
            # Store all scores for this structure
            score_components = {
                'mpnn': mpnn_score,
                'selection_metric': self.args.selection_metric,
                'selection_value': selection_value,
                'metrics': structure_metrics
            }
            
            all_scores.append(score_components)
            
            # Track best structure based on selection order
            is_better = False
            if self.args.selection_order == 'ascending':
                is_better = selection_value < best_score
            else:  # descending
                is_better = selection_value > best_score
                
            if is_better:
                best_score = selection_value
                best_sequence = seq_data.copy()
                best_sequence.update(score_components)
                best_path = relaxed_path
                
            if self.args.verbose:
                print(f"Structure {i}: MPNN={mpnn_score:.4f}, {self.args.selection_metric.upper()}={selection_value:.3f}")
                if structure_metrics:
                    # Enhanced metric display with all available values
                    metric_items = []
                    for k, v in structure_metrics.items():
                        if isinstance(v, (int, float)):
                            if k in ['totalscore', 'ddg', 'ddg_after_relax_cst']:
                                metric_items.append(f"{k}={v:.2f}")
                            elif k == 'cms':
                                metric_items.append(f"{k}={v:.4f}")
                            elif k == 'res_totalscore':
                                metric_items.append(f"{k}={v:.3f}")
                            else:
                                metric_items.append(f"{k}={v:.2f}")
                    if metric_items:
                        print(f"  All metrics: {', '.join(metric_items)}")
        
        # Handle case where no valid structures found
        if best_sequence is None:
            print("ERROR: All fast relax jobs failed! No valid structures to select from.")
            if sequences:
                best_sequence = sequences[0].copy()
                best_sequence.update({
                    'selection_metric': self.args.selection_metric,
                    'selection_value': float('inf'),
                    'metrics': {}
                })
                best_path = relaxed_paths[0] if relaxed_paths else None
            else:
                raise ValueError("No sequences available for selection")
        
        # Store all scores in best sequence for statistics
        best_sequence['all_scores'] = all_scores
        
        if self.args.verbose:
            print("\nBest structure selected:")
            print(f"  Selection metric ({self.args.selection_metric}): {best_score:.4f}")
            print(f"  MPNN score: {best_sequence.get('mpnn', 'N/A'):.4f}")
            if 'metrics' in best_sequence and best_sequence['metrics']:
                metrics = best_sequence['metrics']
                print(f"  DDG: {metrics.get('ddg', 'N/A'):.2f}")
                print(f"  Total score: {metrics.get('totalscore', 'N/A'):.2f}")
                print(f"  Contact surface: {metrics.get('cms', 'N/A'):.4f}")
                
        return best_sequence, best_path
        
    def run_iterative_design(self):
        """Run iterative LigandMPNN + FastRelax cycles"""
        current_pdb = self.args.pdb_path
        all_cycle_results = []
        
        start_time = time.time()
        
        for cycle in range(1, self.args.n_cycles + 1):
            # Run MPNN + FastRelax cycle
            best_sequence, relaxed_pdb = self.run_mpnn_fastrelax_cycle(current_pdb, cycle)
            
            # Store cycle results
            all_cycle_results.append({
                'cycle': cycle,
                'best_sequence': best_sequence,
                'relaxed_pdb': relaxed_pdb
            })
            
            # Use relaxed structure as input for next cycle
            current_pdb = relaxed_pdb
                
        end_time = time.time()
        
        if self.args.verbose:
            print("\nIterative design completed!")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            print(f"Final structure: {current_pdb}")
            
        return all_cycle_results

def parse_arguments(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='LigandMPNN FastRelax - Complete Version')
    
    # Input/Output
    parser.add_argument('--pdb_path', type=str, required=True,
                        help='Path to input PDB file')
    parser.add_argument('--ligand_params_path', type=str, required=True,
                        help='Path to ligand parameters file')
    parser.add_argument('--out_folder', type=str, default='./ligmpnn_fr_output',
                        help='Output folder path')
    
    # Model settings
    parser.add_argument('--path_to_model_weights', type=str, default='/home/hwjang/aipd/LigandMPNN/',
                        help='Path to model weights')
    parser.add_argument('--model_name', type=str, default='ligandmpnn_v_32_010_25',
                        help='Model name')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Direct path to checkpoint file')
    
    # Sequence generation
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')
    parser.add_argument('--num_seq_per_target', type=int, default=1,
                        help='Number of sequences per target')
    
    # Iterative design
    parser.add_argument('--n_cycles', type=int, default=3,
                        help='Number of iterative design cycles')
    
    # Constraints and design control
    parser.add_argument('--fixed_residues', type=str, nargs='*', default=[],
                        help='List of residues to keep fixed')
    parser.add_argument('--redesigned_residues', type=str, nargs='*', default=[],
                        help='List of residues to redesign')
    parser.add_argument('--omit_AAs', type=str, default='X',
                        help='Amino acids to omit')
    
    # PyRosetta settings
    parser.add_argument('--use_genpot', action='store_true',
                        help='Use genpot for fast relax')
    
    # XML relaxation settings (from original ligMPNN_FR)
    parser.add_argument('--repackable_res', type=str, default='',
                        help='Repackable residue numbers concatenated with comma (e.g., "10,15,27")')
    parser.add_argument('--target_atm_for_cst', type=str, default='',
                        help='Target ligand atom names to extract distance constraints from input design (e.g., "O,N")')
    
    # LigandMPNN settings
    parser.add_argument('--use_side_chain_context', action='store_true',
                        help='Use side chain context in LigandMPNN')
    parser.add_argument('--ligand_mpnn_use_side_chain_context', type=int, default=0,
                        help='Use side chain context in LigandMPNN')
    parser.add_argument('--ligand_mpnn_use_atom_context', type=int, default=1,
                        help='Use atom context in LigandMPNN')
    parser.add_argument('--pack_side_chains', action='store_true', default=False,
                        help='Use LigandMPNN side chain packing')
    parser.add_argument('--checkpoint_path_sc', type=str, default=None,
                        help='Path to side chain packer model')
    parser.add_argument('--pack_with_ligand_context', type=int, default=1,
                        help='Use ligand context during side chain packing')
    
    # Batch processing and parallelization
    parser.add_argument('--max_batch_size', type=int, default=1,
                        help='Maximum batch size for sequence generation (default: 1)')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of processes for fast relax (default: 1)')
    parser.add_argument('--pyrosetta_threads', type=int, default=1,
                        help='Number of threads per PyRosetta process (default: 1)')

    # Side chain packing parameters
    parser.add_argument('--number_of_packs_per_design', type=int, default=1,
                        help='Number of side chain packing samples per design')
    parser.add_argument('--sc_num_denoising_steps', type=int, default=3,
                        help='Number of denoising steps for side chain packing')
    parser.add_argument('--sc_num_samples', type=int, default=16,
                        help='Number of samples for side chain packing')
    parser.add_argument('--repack_everything', action='store_true',
                        help='Repack all residues (not just designed ones)')
    # parser.add_argument('--pack_with_ligand_context', action='store_true', default=True,
    #                     help='Use ligand context during side chain packing')
    parser.add_argument('--force_hetatm', action='store_true',
                        help='Force ligand atoms to be written as HETATM')
    parser.add_argument('--zero_indexed', action='store_true',
                        help='Use zero-indexed residue numbering')
    
    # Filtering and selection parameters - simplified to single metric
    parser.add_argument('--selection_metric', type=str, default='ddg', 
                        choices=['mpnn', 'ddg', 'totalscore', 'res_totalscore', 'cms', 'ddg_after_relax_cst'],
                        help='Metric to use for selecting best structure')
    parser.add_argument('--selection_order', type=str, default='ascending', 
                        choices=['ascending', 'descending'],
                        help='Order for selection: ascending (lower is better) or descending (higher is better)')
    
    # Other settings
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--save_stats', action='store_true',
                        help='Save detailed statistics')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = parse_arguments()
    
    # Map the --use_side_chain_context flag to the ligand_mpnn_use_side_chain_context attribute
    if hasattr(args, 'use_side_chain_context') and args.use_side_chain_context:
        args.ligand_mpnn_use_side_chain_context = 1
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    try:
        mp.freeze_support()
        mp.set_start_method('spawn', force=True)
        mpnn_fr = LigandMPNNFastRelax(args)
        cycle_results = mpnn_fr.run_iterative_design()
        
        # Get final structure from last cycle
        final_structure = cycle_results[-1]['relaxed_pdb'] if cycle_results else None
        
        print("\nIterative design completed successfully!")
        if final_structure:
            print(f"Final structure: {final_structure}")
            
            # Print summary of all cycles
            print(f"\nSummary of {len(cycle_results)} design cycles:")
            for result in cycle_results:
                cycle = result['cycle']
                seq_info = result['best_sequence']
                if 'final' in seq_info:
                    print(f"  Cycle {cycle}: Final score = {seq_info['final']:.4f}")
                    if 'mpnn' in seq_info and 'rosetta' in seq_info:
                        print(f"    MPNN: {seq_info['mpnn']:.4f}, Rosetta: {seq_info['rosetta']:.2f}")
                else:
                    print(f"  Cycle {cycle}: Score = {seq_info.get('score', 'N/A')}")
        else:
            print("No cycles completed successfully.")
        
    except Exception as e:
        print(f"Error during iterative design: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
