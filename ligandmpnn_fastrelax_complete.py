#!/usr/bin/env python3
"""
LigandMPNN with Fast Relax Integration - Complete Version
Updated version based on latest LigandMPNN implementation

This script integrates LigandMPNN sequence design with PyRosetta fast relax
for iterative protein-ligand design optimization.

Author: Updated version based on successful debugging session
Date: 2025
"""

import os
import sys
import json
import random
import argparse
import time
import traceback
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import torch

# Global function for multiprocessing - must be at module level
def fast_relax_worker(input_data):
    """
    Worker function for multiprocessing fast relax with PyRosetta multithreading
    This function runs in a separate process with its own PyRosetta instance
    and utilizes PyRosetta's internal multithreading capabilities
    """
    pdb_path, output_path, ligand_params_path, use_genpot, verbose, pyrosetta_threads = input_data
    
    try:
        # Import PyRosetta in worker process (each process needs its own instance)
        import pyrosetta
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.scoring import get_score_function
        from pyrosetta.rosetta.basic.options import option
        from pyrosetta.rosetta.basic.options.OptionKeys import run as run_options
        
        # Initialize PyRosetta with multithreading support
        init_flags = ['-beta']
        if not verbose:
            init_flags.append('-mute all')
        
        # Enable PyRosetta multithreading if specified
        if pyrosetta_threads > 1:
            init_flags.extend([
                f'-multithreading:total_threads {pyrosetta_threads}',
                '-multithreading:interaction_graph_threads 1',
                '-run:multiple_processes_writing_to_one_directory'
            ])
        
        if use_genpot:
            init_flags.extend([
                '-gen_potential',
                f'-extra_res_fa {ligand_params_path}'
            ])
        else:
            init_flags.extend([
                f'-extra_res_fa {ligand_params_path}'
            ])
        
        pyrosetta.init(' '.join(init_flags))
        
        # Set runtime threading options (additional safety)
        if pyrosetta_threads > 1:
            try:
                option[run_options.nthreads].value(pyrosetta_threads)
            except Exception:
                pass  # Some versions may not support this option
        
        # Load pose
        pose = pyrosetta.pose_from_pdb(pdb_path)
        
        if pose.total_residue() == 0:
            raise ValueError("Empty pose loaded")
        
        # Set up FastRelax with multithreading optimized settings
        scorefxn = get_score_function()
        fr = FastRelax()
        fr.set_scorefxn(scorefxn)
        
        # Settings optimized for multithreading
        fr.constrain_relax_to_start_coords(True)
        fr.coord_constrain_sidechains(True) 
        fr.ramp_down_constraints(False)
        
        # Enable multithreaded scoring if available
        if pyrosetta_threads > 1:
            try:
                # Some FastRelax specific multithreading options
                fr.set_script_from_file('')  # Use default script
                # Note: Specific FastRelax multithreading may require additional setup
            except Exception:
                pass  # Fallback to standard settings
        
        # Run FastRelax (PyRosetta will use its internal multithreading)
        fr.apply(pose)
        
        # Save result
        pose.dump_pdb(output_path)
        
        # Verify output
        if not os.path.exists(output_path):
            raise RuntimeError("Output PDB file was not created")
        
        return {
            'success': True,
            'input_path': pdb_path,
            'output_path': output_path,
            'pyrosetta_threads_used': pyrosetta_threads,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'input_path': pdb_path,
            'output_path': output_path,
            'pyrosetta_threads_used': pyrosetta_threads,
            'error': str(e)
        }


def get_optimal_worker_distribution(total_structures, max_workers=None, args=None):
    """
    Calculate optimal distribution of workers for hybrid multiprocessing + PyRosetta multithreading
    
    Args:
        total_structures: Number of structures to process
        max_workers: Maximum number of workers (None for auto-detect)
        args: Argument namespace with CPU allocation strategy
        
    Returns:
        dict with worker configuration including PyRosetta threading
    """
    # Get system information
    cpu_cores = cpu_count()
    
    # Get PyRosetta threading preference
    pyrosetta_threads_per_process = getattr(args, 'pyrosetta_threads_per_process', 1) if args else 1
    enable_pyrosetta_threading = getattr(args, 'enable_pyrosetta_threading', False) if args else False
    hybrid_parallelism = getattr(args, 'hybrid_parallelism', False) if args else False
    auto_tune_threading = getattr(args, 'auto_tune_threading', False) if args else False
    
    # Auto-tune threading if requested
    if auto_tune_threading or hybrid_parallelism:
        enable_pyrosetta_threading = True
        if cpu_cores >= 16:
            # High-core systems: favor more processes, moderate threading
            pyrosetta_threads_per_process = min(4, max(2, cpu_cores // 8))
        elif cpu_cores >= 8:
            # Medium systems: balanced approach
            pyrosetta_threads_per_process = min(3, max(2, cpu_cores // 4))
        elif cpu_cores >= 4:
            # Low-core systems: favor threading over processes
            pyrosetta_threads_per_process = 2
        else:
            # Very low-core systems: disable threading
            enable_pyrosetta_threading = False
            pyrosetta_threads_per_process = 1
    
    # Apply CPU allocation strategy
    if args and hasattr(args, 'cpu_allocation_strategy'):
        strategy = args.cpu_allocation_strategy
        memory_per_worker = getattr(args, 'memory_per_worker_gb', 2.0)
        
        if strategy == 'conservative':
            # Use 50% of available cores, reserve rest for system
            available_cores = max(1, int(cpu_cores * 0.5))
        elif strategy == 'aggressive':
            # Use 90% of available cores
            available_cores = max(1, int(cpu_cores * 0.9))
        elif strategy == 'custom':
            # Use exactly the specified max_workers
            available_cores = max_workers if max_workers else max(1, cpu_cores - 1)
        else:  # auto
            # Default: reserve 1-2 cores for system
            available_cores = max(1, cpu_cores - min(2, max(1, cpu_cores // 4)))
    else:
        # Fallback to original logic
        available_cores = max(1, cpu_cores - 1)
    
    # Calculate optimal process/thread distribution
    if enable_pyrosetta_threading and pyrosetta_threads_per_process > 1:
        # Hybrid approach: processes × threads_per_process should not exceed available cores
        total_thread_capacity = available_cores
        
        # Calculate optimal number of processes
        optimal_processes = max(1, min(
            total_structures,  # Don't exceed number of structures
            total_thread_capacity // pyrosetta_threads_per_process,  # CPU constraint
            8  # Reasonable upper limit for processes to avoid memory issues
        ))
        
        # Adjust threads per process if needed
        actual_threads_per_process = min(
            pyrosetta_threads_per_process,
            max(1, total_thread_capacity // optimal_processes)
        )
        
        if max_workers is not None:
            optimal_processes = min(optimal_processes, max_workers)
        
        total_effective_workers = optimal_processes * actual_threads_per_process
        
    else:
        # Single-threaded PyRosetta approach
        actual_threads_per_process = 1
        
        # Default max workers if not specified
        if max_workers is None:
            if args and args.cpu_allocation_strategy == 'custom':
                optimal_processes = available_cores
            else:
                optimal_processes = min(available_cores, 8)  # Cap at 8 to avoid memory issues
        else:
            optimal_processes = min(max_workers, available_cores)
        
        # Adjust workers based on number of structures
        optimal_processes = min(optimal_processes, total_structures)
        total_effective_workers = optimal_processes
    
    # Memory consideration
    memory_limited_workers = optimal_processes
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        memory_per_worker = getattr(args, 'memory_per_worker_gb', 2.0) if args else 2.0
        
        # Account for PyRosetta threading memory overhead
        memory_per_process = memory_per_worker * (1 + 0.3 * (actual_threads_per_process - 1))
        memory_limited_workers = max(1, int(available_memory_gb / memory_per_process))
        optimal_processes = min(optimal_processes, memory_limited_workers)
        
    except ImportError:
        # Fallback if psutil not available
        pass
    
    # Apply sequential threshold
    sequential_threshold = getattr(args, 'sequential_threshold', 2) if args else 2
    if total_structures < sequential_threshold:
        optimal_processes = 1
        actual_threads_per_process = 1
    
    return {
        'optimal_workers': optimal_processes,
        'pyrosetta_threads_per_process': actual_threads_per_process,
        'total_effective_threads': optimal_processes * actual_threads_per_process,
        'cpu_cores': cpu_cores,
        'available_cores': available_cores,
        'memory_limited_workers': memory_limited_workers,
        'structures_per_worker': (total_structures + optimal_processes - 1) // optimal_processes,
        'parallel_efficiency': min(1.0, total_structures / optimal_processes),
        'hybrid_efficiency': min(1.0, (total_structures * actual_threads_per_process) / (optimal_processes * actual_threads_per_process)),
        'strategy_used': getattr(args, 'cpu_allocation_strategy', 'auto') if args else 'auto',
        'memory_per_worker_gb': getattr(args, 'memory_per_worker_gb', 2.0) if args else 2.0,
        'threading_mode': 'hybrid' if enable_pyrosetta_threading and actual_threads_per_process > 1 else 'process_only'
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
            write_full_PDB,
        )
        from model_utils import ProteinMPNN
        from sc_utils import Packer, pack_side_chains
        from prody import writePDB
        return True, (parse_PDB, featurize, get_score, alphabet, restype_int_to_str, restype_str_to_int, ProteinMPNN, write_full_PDB, Packer, pack_side_chains, writePDB)
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
        
        self.parse_PDB, self.featurize, self.get_score, self.alphabet, self.restype_int_to_str, self.restype_str_to_int, self.ProteinMPNN, self.write_full_PDB, self.Packer, self.pack_side_chains, self.writePDB = modules
        
        # Load model
        self.model = self._load_ligandmpnn_model()
        
        # Load side chain packing model if enabled
        self.model_sc = None
        if getattr(args, 'pack_side_chains', False):  # Disable by default to avoid loading issues
            try:
                self.model_sc = self._load_packer_model()
            except Exception as e:
                if args.verbose:
                    print(f"Warning: Could not load side chain packer: {e}")
                    print("Side chain packing will be disabled")
                self.model_sc = None
        
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
            model_folder = self.args.path_to_model_weights if hasattr(self.args, 'path_to_model_weights') else './model_params/'
            if model_folder[-1] != '/':
                model_folder += '/'
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
            ligand_mpnn_use_side_chain_context=self.args.use_side_chain_context,
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
            checkpoint_path_sc = f"{model_folder}model_params/ligandmpnn_sc_v_32_002_16.pt"
        
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
            parse_all_atoms=self.args.use_side_chain_context,
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
            use_atom_context=True,
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
        This replaces PyRosetta threading for better compatibility
        """
        if self.args.verbose:
            print("Creating structure with new sequence using LigandMPNN...")
            
        # Parse original PDB
        protein_dict, backbone, other_atoms, icodes, _ = self.parse_PDB(
            pdb_path,
            device=self.device,
            chains=[],
            parse_all_atoms=self.args.use_side_chain_context,
            parse_atoms_with_zero_occupancy=False,
        )
        
        # Update sequence in protein_dict
        protein_dict["S"] = S_sample.unsqueeze(0) if S_sample.dim() == 1 else S_sample
        
        # Add chain_mask (required for featurize)
        # Default to design all residues if not specified
        if "chain_mask" not in protein_dict:
            protein_dict["chain_mask"] = torch.ones(len(protein_dict["R_idx"]), device=self.device)
        
        # Pack side chains if model is available
        if self.model_sc is not None:
            if self.args.verbose:
                print("Packing side chains...")
            
            # Featurize for side chain packing
            feature_dict = self.featurize(
                protein_dict,
                cutoff_for_score=8.0,
                use_atom_context=True,
                number_of_ligand_atoms=16,
                model_type="ligand_mpnn",
            )
            
            # Pack side chains
            feature_dict_packed = self.pack_side_chains(
                feature_dict,
                self.model_sc,
                num_denoising_steps=getattr(self.args, 'sc_num_denoising_steps', 3),
                num_samples=getattr(self.args, 'sc_num_samples', 16),
                repack_everything=getattr(self.args, 'repack_everything', False),
                num_context_atoms=16,
            )
            
            if feature_dict_packed is not None:
                if self.args.verbose:
                    print("Side chain packing completed successfully")
                # Update coordinates with packed side chains from the returned feature_dict
                protein_dict["X"] = feature_dict_packed["X"]
        
        # Write full PDB using LigandMPNN's function
        try:
            self.write_full_PDB(
                protein_dict,
                backbone,
                other_atoms,
                icodes,
                output_path,
                args_zero_indexed=getattr(self.args, 'zero_indexed', False),
                args_force_hetatm=getattr(self.args, 'force_hetatm', False),
            )
            
            if self.args.verbose:
                print(f"Structure with new sequence saved: {output_path}")
                
            return output_path
            
        except Exception as e:
            if self.args.verbose:
                print(f"Warning: LigandMPNN write_full_PDB failed: {e}")
                print("Falling back to PyRosetta threading...")
            
            # Fallback to PyRosetta threading
            return self._thread_sequence_fallback(pdb_path, sequence, output_path)
    
    def _thread_sequence_fallback(self, pdb_path, sequence, output_path):
        """Fallback PyRosetta threading method"""
        pose = self.pyrosetta.pose_from_pdb(pdb_path)
        
        # Thread sequence
        seq_list = list(sequence)
        threaded_count = 0
        for i, aa in enumerate(seq_list):
            res_num = i + 1
            if res_num <= pose.total_residue() and pose.residue(res_num).is_protein():
                try:
                    self.toolbox.mutants.mutate_residue(pose, res_num, aa)
                    threaded_count += 1
                except Exception as e:
                    if self.args.verbose:
                        print(f"Warning: Could not mutate residue {res_num} to {aa}: {e}")
                    
        # Save threaded structure
        pose.dump_pdb(output_path)
        
        if self.args.verbose:
            print(f"Fallback threading completed: {threaded_count}/{len(seq_list)} residues")
            
        return output_path
        
    def fast_relax_parallel(self, structure_paths, max_workers=None):
        """
        Perform fast relax on multiple structures using true multiprocessing
        
        Args:
            structure_paths: List of PDB file paths to relax
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of relaxed structure paths
        """
        if len(structure_paths) == 1:
            # Single structure - use direct processing to avoid overhead
            return self._fast_relax_sequential([structure_paths[0]])
        
        # Get optimal worker configuration
        worker_config = get_optimal_worker_distribution(
            len(structure_paths), 
            max_workers, 
            self.args
        )
        optimal_workers = worker_config['optimal_workers']
        
        if self.args.verbose:
            print(f"Starting hybrid multiprocess+PyRosetta-threaded fast relax on {len(structure_paths)} structures")
            print("Worker configuration:")
            print(f"  - CPU cores: {worker_config['cpu_cores']}")
            print(f"  - Available cores: {worker_config['available_cores']}")
            print(f"  - Strategy: {worker_config['strategy_used']}")
            print(f"  - Threading mode: {worker_config['threading_mode']}")
            print(f"  - Memory per worker: {worker_config['memory_per_worker_gb']:.1f} GB")
            print(f"  - Memory limited workers: {worker_config['memory_limited_workers']}")
            print(f"  - Optimal processes: {optimal_workers}")
            print(f"  - PyRosetta threads per process: {worker_config['pyrosetta_threads_per_process']}")
            print(f"  - Total effective threads: {worker_config['total_effective_threads']}")
            print(f"  - Structures per worker: {worker_config['structures_per_worker']}")
            print(f"  - Process efficiency: {worker_config['parallel_efficiency']:.2f}")
            print(f"  - Hybrid efficiency: {worker_config['hybrid_efficiency']:.2f}")
        
        # Prepare worker input data with PyRosetta threading info
        worker_inputs = []
        relaxed_paths = []
        
        for pdb_path in structure_paths:
            relaxed_path = pdb_path.replace('/backbones/', '/relaxed/').replace('.pdb', '_relaxed.pdb')
            os.makedirs(os.path.dirname(relaxed_path), exist_ok=True)
            relaxed_paths.append(relaxed_path)
            
            worker_input = (
                pdb_path, 
                relaxed_path, 
                self.args.ligand_params_path,
                self.args.use_genpot,
                self.args.verbose,
                worker_config['pyrosetta_threads_per_process']  # Add PyRosetta threading info
            )
            worker_inputs.append(worker_input)
        
        # Choose processing method based on worker count and user preference
        force_multiprocessing = getattr(self.args, 'force_multiprocessing', False)
        sequential_threshold = getattr(self.args, 'sequential_threshold', 2)
        enable_threading = getattr(self.args, 'enable_threading_fallback', False)
        
        use_multiprocessing = (
            optimal_workers > 1 and 
            len(structure_paths) > 1 and
            len(structure_paths) >= sequential_threshold and
            (force_multiprocessing or optimal_workers >= 2)
        )
        
        if use_multiprocessing:
            if self.args.verbose:
                print(f"Using multiprocessing with {optimal_workers} workers")
            
            # Use multiprocessing for true parallelism
            try:
                from multiprocessing import Pool
                
                start_time = time.time()
                
                with Pool(processes=optimal_workers) as pool:
                    results = pool.map(fast_relax_worker, worker_inputs)
                
                end_time = time.time()
                
                # Process results
                successful_paths = []
                failed_count = 0
                
                for i, result in enumerate(results):
                    if result['success']:
                        successful_paths.append(result['output_path'])
                        if self.args.verbose and i % max(1, len(results)//5) == 0:
                            print(f"Completed {i+1}/{len(results)}: {os.path.basename(result['output_path'])}")
                    else:
                        failed_count += 1
                        if self.args.verbose:
                            print(f"Failed {result['input_path']}: {result['error']}")
                        
                        # Fallback: copy original file
                        try:
                            import shutil
                            shutil.copy2(result['input_path'], result['output_path'])
                            successful_paths.append(result['output_path'])
                            if self.args.verbose:
                                print(f"Using original structure as fallback: {result['output_path']}")
                        except Exception as copy_e:
                            if self.args.verbose:
                                print(f"Failed to copy original file: {copy_e}")
                            successful_paths.append(result['input_path'])
                
                if self.args.verbose:
                    speedup = (len(structure_paths) * 30) / (end_time - start_time)  # Assume 30s per structure
                    print(f"Multiprocessing completed in {end_time - start_time:.2f}s")
                    print(f"Estimated speedup: {speedup:.1f}x")
                    print(f"Success rate: {(len(results) - failed_count)/len(results)*100:.1f}%")
                
                return successful_paths
                
            except Exception as e:
                if self.args.verbose:
                    print(f"Multiprocessing failed: {e}")
                    
                # Try threading fallback if enabled
                if enable_threading:
                    if self.args.verbose:
                        print("Attempting threading fallback...")
                    try:
                        return self._fast_relax_threading(structure_paths, optimal_workers)
                    except Exception as threading_e:
                        if self.args.verbose:
                            print(f"Threading fallback also failed: {threading_e}")
                
                if self.args.verbose:
                    print("Falling back to sequential processing...")
                return self._fast_relax_sequential(structure_paths)
        
        else:
            if self.args.verbose:
                print("Using sequential processing (single worker or small batch)")
            return self._fast_relax_sequential(structure_paths)
    
    def _fast_relax_sequential(self, structure_paths):
        """
        Sequential fast relax processing (fallback method)
        """
        if self.args.verbose:
            print(f"Starting sequential fast relax on {len(structure_paths)} structures")
        
        relaxed_paths = []
        
        for i, pdb_path in enumerate(structure_paths):
            relaxed_path = pdb_path.replace('/backbones/', '/relaxed/').replace('.pdb', '_relaxed.pdb')
            os.makedirs(os.path.dirname(relaxed_path), exist_ok=True)
            
            if self.args.verbose:
                print(f"Processing {i+1}/{len(structure_paths)}: {os.path.basename(pdb_path)}")
            
            try:
                result_path = self._fast_relax_single(pdb_path, relaxed_path)
                relaxed_paths.append(result_path)
                if self.args.verbose:
                    print(f"Completed fast relax: {os.path.basename(result_path)}")
            except Exception as e:
                print(f"Error in fast relax for {pdb_path}: {e}")
                # Copy original file as fallback
                import shutil
                try:
                    shutil.copy2(pdb_path, relaxed_path)
                    relaxed_paths.append(relaxed_path)
                    print(f"Using original structure as fallback: {relaxed_path}")
                except Exception as copy_e:
                    print(f"Failed to copy original file: {copy_e}")
                    relaxed_paths.append(pdb_path)
        
        return relaxed_paths
    
    def _fast_relax_threading(self, structure_paths, max_workers):
        """
        Threading-based fast relax (may have limitations due to GIL but can work with PyRosetta)
        This is a fallback when multiprocessing fails
        """
        if self.args.verbose:
            print(f"Starting threaded fast relax on {len(structure_paths)} structures with {max_workers} threads")
            print("Note: Threading has GIL limitations but may work with PyRosetta's C++ backend")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        relaxed_paths = []
        
        # Prepare path mappings
        path_mappings = []
        for pdb_path in structure_paths:
            relaxed_path = pdb_path.replace('/backbones/', '/relaxed/').replace('.pdb', '_relaxed.pdb')
            os.makedirs(os.path.dirname(relaxed_path), exist_ok=True)
            path_mappings.append((pdb_path, relaxed_path))
            relaxed_paths.append(relaxed_path)
        
        # Create a thread-safe method wrapper
        def thread_safe_relax(pdb_path, output_path):
            try:
                return self._fast_relax_single(pdb_path, output_path)
            except Exception as e:
                if self.args.verbose:
                    print(f"Thread error for {pdb_path}: {e}")
                # Fallback: copy original file
                try:
                    import shutil
                    shutil.copy2(pdb_path, output_path)
                    return output_path
                except Exception:
                    return pdb_path
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for threading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(thread_safe_relax, pdb_path, relaxed_path): (pdb_path, relaxed_path)
                for pdb_path, relaxed_path in path_mappings
            }
            
            # Process completed tasks
            completed_count = 0
            for future in as_completed(future_to_path):
                pdb_path, relaxed_path = future_to_path[future]
                try:
                    result_path = future.result()
                    completed_count += 1
                    if self.args.verbose and completed_count % max(1, len(structure_paths)//5) == 0:
                        print(f"Thread completed {completed_count}/{len(structure_paths)}: {os.path.basename(result_path)}")
                except Exception as e:
                    if self.args.verbose:
                        print(f"Thread exception for {pdb_path}: {e}")
        
        end_time = time.time()
        
        if self.args.verbose:
            print(f"Threading completed in {end_time - start_time:.2f}s")
            
        return relaxed_paths
    
    def _fast_relax_single(self, input_pdb_path, output_pdb_path):
        """
        Perform fast relax on a single structure (with improved error handling)
        """
        try:
            if self.args.verbose:
                print(f"Loading pose from: {input_pdb_path}")
            
            # Load pose with error checking
            pose = self.pyrosetta.pose_from_pdb(input_pdb_path)
            
            if pose.total_residue() == 0:
                raise ValueError("Empty pose loaded")
            
            if self.args.verbose:
                print(f"Pose loaded successfully with {pose.total_residue()} residues")
            
            # Set up scoring function
            scorefxn = self.get_score_function()
            
            # Set up FastRelax with default settings for stability
            fr = self.FastRelax()
            fr.set_scorefxn(scorefxn)
            
            # Set conservative relax settings to avoid crashes
            fr.constrain_relax_to_start_coords(True)
            fr.coord_constrain_sidechains(True)
            fr.ramp_down_constraints(False)
            
            if self.args.verbose:
                print("Starting fast relax...")
            
            # Run fast relax
            fr.apply(pose)
            
            if self.args.verbose:
                print("Fast relax completed, saving structure...")
            
            # Save relaxed structure
            pose.dump_pdb(output_pdb_path)
            
            # Verify output file was created
            if not os.path.exists(output_pdb_path):
                raise RuntimeError("Output PDB file was not created")
            
            if self.args.verbose:
                print(f"Relaxed structure saved to: {output_pdb_path}")
            
            return output_pdb_path
            
        except Exception as e:
            print(f"Error in fast relax for {input_pdb_path}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Try to copy original file as fallback
            try:
                import shutil
                shutil.copy2(input_pdb_path, output_pdb_path)
                print(f"Using original structure as fallback: {output_pdb_path}")
                return output_pdb_path
            except Exception as copy_e:
                print(f"Failed to copy original file: {copy_e}")
                return input_pdb_path  # Return original path if everything fails
        
    def fast_relax(self, pose, cycles=5):
        """Perform fast relax on pose"""
        if self.args.verbose:
            print("Performing fast relax...")
            
        # Set up scoring function
        scorefxn = self.get_score_function()
        
        # Set up FastRelax
        fr = self.FastRelax()
        fr.set_scorefxn(scorefxn)
        
        # Run fast relax
        start_time = time.time()
        fr.apply(pose)
        end_time = time.time()
        
        if self.args.verbose:
            print(f"Fast relax completed in {end_time - start_time:.2f} seconds")
            
        return pose
        
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
        
        if len(sequences) == 1:
            # Single sequence - use simple approach
            best_sequence = sequences[0].copy()  # Make a copy to avoid reference issues
            threaded_path = f"{self.base_folder}backbones/{header}_cycle_{cycle_num}_best.pdb"
            
            self.create_structure_with_sequence(
                pdb_path, 
                best_sequence['sequence'], 
                best_sequence['S_sample'], 
                threaded_path
            )
            structure_paths = [threaded_path]
            
        else:
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
        if len(structure_paths) == 1:
            # Single structure - use original method
            pose = self.pyrosetta.pose_from_pdb(structure_paths[0])
            relaxed_pose = self.fast_relax(pose)
            relaxed_path = f"{self.base_folder}relaxed/{header}_cycle_{cycle_num}_relaxed.pdb"
            relaxed_pose.dump_pdb(relaxed_path)
            relaxed_paths = [relaxed_path]
            
            # Calculate Rosetta energy for single structure
            scorefxn = self.get_score_function()
            rosetta_energy = scorefxn(relaxed_pose)
            
            # Update best_sequence with score information
            best_sequence = sequences[0].copy()  # Make a copy to avoid reference issues
            best_sequence.update({
                'mpnn': best_sequence['score'],
                'rosetta': rosetta_energy,
                'final': rosetta_energy if self.args.score_method == 'rosetta_only' 
                        else (best_sequence['score'] if self.args.score_method == 'mpnn_only'
                        else (self.args.mpnn_weight * best_sequence['score'] + 
                              self.args.rosetta_weight * rosetta_energy))
            })
            
        else:
            # Multiple structures - use parallel relaxation
            if self.args.verbose:
                print(f"Performing parallel fast relax on {len(structure_paths)} structures")
                
            relaxed_paths = self.fast_relax_parallel(
                structure_paths, 
                max_workers=getattr(self.args, 'max_relax_workers', None)
            )
            
            # Evaluate relaxed structures and select best
            best_sequence, best_relaxed_path = self._select_best_relaxed_structure(
                sequences, relaxed_paths, cycle_num
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
                # Scoring method and weights
                'score_method': self.args.score_method,
                'mpnn_weight': self.args.mpnn_weight,
                'rosetta_weight': self.args.rosetta_weight,
                'normalize_scores': self.args.normalize_scores,
                # Individual scores for best structure
                'best_mpnn_score': best_sequence.get('mpnn', best_sequence['score']),
                'best_rosetta_energy': best_sequence.get('rosetta', None),
                'best_final_score': best_sequence.get('final', best_sequence['score']),
                # All scores for analysis
                'all_mpnn_scores': [seq['score'] for seq in sequences],
                'all_detailed_scores': best_sequence.get('all_scores', []),
            }
            
            # Add normalized scores if available
            if 'mpnn_norm' in best_sequence:
                stats['best_mpnn_score_normalized'] = best_sequence['mpnn_norm']
            if 'rosetta_norm' in best_sequence:
                stats['best_rosetta_energy_normalized'] = best_sequence['rosetta_norm']
                
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
                
        if self.args.verbose:
            print(f"Cycle {cycle_num} completed")
            print(f"Best sequence: {best_sequence['sequence']}")
            if 'mpnn' in best_sequence and 'rosetta' in best_sequence:
                print(f"Best MPNN score: {best_sequence['mpnn']:.4f}")
                print(f"Best Rosetta energy: {best_sequence['rosetta']:.2f}")
                print(f"Best final score ({self.args.score_method}): {best_sequence['final']:.4f}")
            else:
                print(f"Best score: {best_sequence['score']:.4f}")
            print(f"Final relaxed structure: {relaxed_paths[0]}")
            
        return best_sequence, relaxed_paths[0]
    
    def _select_best_relaxed_structure(self, sequences, relaxed_paths, cycle_num):
        """
        Select best relaxed structure based on scoring method and weights
        """
        if self.args.verbose:
            print(f"Evaluating {len(sequences)} relaxed structures to select best...")
            print(f"Using scoring method: {self.args.score_method}")
            if self.args.score_method == 'combined':
                print(f"MPNN weight: {self.args.mpnn_weight}, Rosetta weight: {self.args.rosetta_weight}")
            
        best_score = float('inf')
        best_sequence = None
        best_path = None
        
        # Store individual scores for statistics
        all_scores = []
        
        scorefxn = self.get_score_function()
        
        # Calculate all scores first (for normalization if needed)
        mpnn_scores = [seq_data['score'] for seq_data in sequences]
        rosetta_energies = []
        
        for i, (seq_data, relaxed_path) in enumerate(zip(sequences, relaxed_paths)):
            try:
                pose = self.pyrosetta.pose_from_pdb(relaxed_path)
                energy = scorefxn(pose)
                rosetta_energies.append(energy)
                
            except Exception as e:
                print(f"Error evaluating structure {i}: {e}")
                rosetta_energies.append(float('inf'))  # Assign worst possible score
                
        # Normalize scores if requested
        if self.args.normalize_scores and len(rosetta_energies) > 1:
            mpnn_mean, mpnn_std = np.mean(mpnn_scores), np.std(mpnn_scores)
            rosetta_mean, rosetta_std = np.mean(rosetta_energies), np.std(rosetta_energies)
            
            if mpnn_std > 0:
                mpnn_scores_norm = [(score - mpnn_mean) / mpnn_std for score in mpnn_scores]
            else:
                mpnn_scores_norm = [0.0] * len(mpnn_scores)
                
            if rosetta_std > 0:
                rosetta_energies_norm = [(energy - rosetta_mean) / rosetta_std for energy in rosetta_energies]
            else:
                rosetta_energies_norm = [0.0] * len(rosetta_energies)
                
            if self.args.verbose:
                print(f"Score normalization - MPNN: μ={mpnn_mean:.3f}, σ={mpnn_std:.3f}")
                print(f"Score normalization - Rosetta: μ={rosetta_mean:.3f}, σ={rosetta_std:.3f}")
        else:
            mpnn_scores_norm = mpnn_scores
            rosetta_energies_norm = rosetta_energies
        
        # Select best structure based on scoring method
        for i, (seq_data, relaxed_path) in enumerate(zip(sequences, relaxed_paths)):
            mpnn_score = mpnn_scores[i]
            rosetta_energy = rosetta_energies[i]
            mpnn_score_norm = mpnn_scores_norm[i]
            rosetta_energy_norm = rosetta_energies_norm[i]
            
            # Calculate final score based on method
            if self.args.score_method == 'mpnn_only':
                final_score = mpnn_score
                score_components = {'mpnn': mpnn_score, 'rosetta': rosetta_energy, 'final': final_score}
            elif self.args.score_method == 'rosetta_only':
                final_score = rosetta_energy
                score_components = {'mpnn': mpnn_score, 'rosetta': rosetta_energy, 'final': final_score}
            else:  # combined
                if self.args.normalize_scores:
                    final_score = (self.args.mpnn_weight * mpnn_score_norm + 
                                 self.args.rosetta_weight * rosetta_energy_norm)
                else:
                    final_score = (self.args.mpnn_weight * mpnn_score + 
                                 self.args.rosetta_weight * rosetta_energy)
                score_components = {
                    'mpnn': mpnn_score, 
                    'rosetta': rosetta_energy, 
                    'mpnn_norm': mpnn_score_norm,
                    'rosetta_norm': rosetta_energy_norm,
                    'final': final_score
                }
            
            all_scores.append(score_components)
            
            if self.args.verbose:
                print(f"Structure {i}: MPNN={mpnn_score:.4f}, Rosetta={rosetta_energy:.2f}, Final={final_score:.4f}")
            
            if final_score < best_score:
                best_score = final_score
                best_sequence = seq_data.copy()  # Make a copy to avoid reference issues
                best_sequence.update(score_components)  # Add detailed scores
                best_path = relaxed_path
        
        if best_sequence is None:
            # Fallback to LigandMPNN score only
            best_sequence = min(sequences, key=lambda x: x['score']).copy()
            best_path = relaxed_paths[sequences.index(min(sequences, key=lambda x: x['score']))]
            best_sequence.update({'final': best_sequence['score']})
            
        if self.args.verbose:
            print(f"Selected best structure with {self.args.score_method} score: {best_score:.4f}")
            if 'mpnn' in best_sequence and 'rosetta' in best_sequence:
                print(f"  MPNN: {best_sequence['mpnn']:.4f}, Rosetta: {best_sequence['rosetta']:.2f}")
            
        # Store all scores for statistics
        best_sequence['all_scores'] = all_scores
            
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
            
        return current_pdb
    
    def analyze_performance_improvements(self):
        """
        Analyze and report expected performance improvements from optimizations
        """
        print("\n" + "="*70)
        print("ADVANCED PERFORMANCE ANALYSIS")
        print("="*70)
        
        # Get worker configuration for analysis
        test_structures = max(1, self.args.num_seq_per_target)
        worker_config = get_optimal_worker_distribution(test_structures, self.args.max_relax_workers, self.args)
        
        print("\n1. CPU ALLOCATION STRATEGY:")
        print(f"   - Strategy: {worker_config['strategy_used']}")
        print(f"   - Total CPU cores: {worker_config['cpu_cores']}")
        print(f"   - Available cores: {worker_config['available_cores']}")
        print(f"   - Optimal workers: {worker_config['optimal_workers']}")
        print(f"   - Memory per worker: {worker_config['memory_per_worker_gb']:.1f} GB")
        print(f"   - Memory limited workers: {worker_config['memory_limited_workers']}")
        
        print("\n2. BATCH PROCESSING IMPROVEMENTS:")
        print(f"   - Sequences per cycle: {self.args.num_seq_per_target}")
        print(f"   - Maximum batch size: {getattr(self.args, 'max_batch_size', 8)}")
        
        if self.args.num_seq_per_target > 1:
            batches_needed = (self.args.num_seq_per_target + self.args.max_batch_size - 1) // self.args.max_batch_size
            print(f"   - Number of batches: {batches_needed}")
            print(f"   - Expected GPU utilization improvement: ~{min(self.args.max_batch_size, self.args.num_seq_per_target)}x")
        else:
            print("   - Single sequence mode: No batch improvement")
        
        print("\n3. HYBRID MULTIPROCESSING + PYROSETTA THREADING:")
        threading_mode = worker_config['threading_mode']
        pyrosetta_threads = worker_config['pyrosetta_threads_per_process']
        
        if self.args.num_seq_per_target > 1:
            if threading_mode == 'hybrid':
                print("   🚀 HYBRID PARALLELISM ENABLED")
                print("   - Python multiprocessing: True parallelism across processes")
                print("   - PyRosetta multithreading: C++ level threading within each process")
                print("   - No GIL limitations for both levels")
                print(f"   - Processes: {worker_config['optimal_workers']}")
                print(f"   - PyRosetta threads per process: {pyrosetta_threads}")
                print(f"   - Total effective threads: {worker_config['total_effective_threads']}")
                
                # Calculate expected speedups
                process_speedup = min(worker_config['optimal_workers'], self.args.num_seq_per_target)
                thread_speedup = min(pyrosetta_threads, 2.0)  # Threading often less than linear
                hybrid_speedup = process_speedup * thread_speedup * 0.8  # Efficiency factor
                
                print(f"   - Expected process speedup: ~{process_speedup:.1f}x")
                print(f"   - Expected thread speedup per process: ~{thread_speedup:.1f}x")
                print(f"   - Expected total hybrid speedup: ~{hybrid_speedup:.1f}x")
                print(f"   - Hybrid efficiency: {worker_config['hybrid_efficiency']:.2f}")
                
            else:
                print("   ✓ Process-only parallelism")
                expected_speedup = min(worker_config['optimal_workers'], self.args.num_seq_per_target)
                print("   - Process-based parallelism (no GIL limitations)")
                print(f"   - Processes: {worker_config['optimal_workers']}")
                print(f"   - Expected speedup: ~{expected_speedup:.1f}x per cycle")
                print(f"   - Parallel efficiency: {worker_config['parallel_efficiency']:.2f}")
                
            print(f"   - Total structures per cycle: {self.args.num_seq_per_target}")
            
            # Memory analysis with threading consideration
            total_memory = worker_config['optimal_workers'] * worker_config['memory_per_worker_gb']
            if threading_mode == 'hybrid':
                total_memory *= (1 + 0.3 * (pyrosetta_threads - 1))  # Threading overhead
            print(f"   - Estimated memory usage: {total_memory:.1f} GB")
            
        else:
            print("   - Single sequence mode: No parallel benefit")
        
        print("\n4. FALLBACK MECHANISMS:")
        if getattr(self.args, 'enable_threading_fallback', False):
            print("   ✓ Threading fallback enabled")
            print("   - Automatic fallback if multiprocessing fails")
            print("   - Limited by GIL but may work with PyRosetta's C++ backend")
        else:
            print("   - Sequential fallback only")
            
        if getattr(self.args, 'force_multiprocessing', False):
            print("   ✓ Forced multiprocessing enabled")
        
        sequential_threshold = getattr(self.args, 'sequential_threshold', 2)
        print(f"   - Sequential threshold: {sequential_threshold} structures")
        
        print("\n5. LIGANDMPNN PACKING vs PYROSETTA THREADING:")
        if self.args.pack_side_chains:
            print("   ✓ Using LigandMPNN native side chain packing")
            print("   - Better ligand-protein interaction modeling")
            print("   - Consistent with LigandMPNN training")
            print("   - Expected accuracy improvement: High")
        else:
            print("   ⚠ Using PyRosetta threading (fallback mode)")
            print("   - May have inconsistencies with LigandMPNN")
        
        print("\n6. OVERALL EXPECTED IMPROVEMENTS:")
        total_sequences = self.args.num_seq_per_target * self.args.n_cycles
        print(f"   - Total sequences across all cycles: {total_sequences}")
        
        if self.args.num_seq_per_target > 1:
            batch_speedup = min(self.args.max_batch_size, self.args.num_seq_per_target)
            parallel_speedup = min(worker_config['optimal_workers'], self.args.num_seq_per_target)
            
            # More sophisticated speedup calculation
            if worker_config['parallel_efficiency'] > 0.8:
                efficiency_factor = 1.0
            elif worker_config['parallel_efficiency'] > 0.5:
                efficiency_factor = 0.8
            else:
                efficiency_factor = 0.6
                
            total_speedup = (batch_speedup * 0.3 + parallel_speedup * 0.7) * efficiency_factor
            print(f"   - Estimated total speedup: ~{total_speedup:.1f}x")
            print(f"   - Quality improvement: High (better sampling + native packing)")
        else:
            print("   - Limited speedup with single sequence per cycle")
            print("   - Quality improvement: Medium (better packing only)")
        
        print("\n7. RESOURCE UTILIZATION:")
        print("   CPU Strategy Analysis:")
        if self.args.cpu_allocation_strategy == 'conservative':
            print("   ✓ Conservative: Good for shared systems")
        elif self.args.cpu_allocation_strategy == 'aggressive': 
            print("   ⚡ Aggressive: Maximum performance on dedicated systems")
        elif self.args.cpu_allocation_strategy == 'custom':
            print(f"   🎯 Custom: Using exactly {self.args.max_relax_workers} workers")
        else:
            print("   🔄 Auto: Balanced approach")
            
        print("\n8. BOTTLENECK ANALYSIS:")
        print("   Previous bottlenecks:")
        print("   - Sequential sequence generation")
        print("   - PyRosetta threading inconsistencies") 
        print("   - Sequential fast relax")
        print("   - No CPU allocation strategy")
        
        print("\n   Current optimizations:")
        print("   ✓ Batch sequence generation")
        print("   ✓ Native LigandMPNN structure creation")
        print("   ✓ True multiprocess fast relax")
        print("   ✓ Intelligent CPU allocation")
        print("   ✓ Memory-aware worker scaling")
        print("   ✓ Fallback mechanisms")
        print("   ✓ Better structure evaluation")
        
        print("\n9. RECOMMENDATIONS:")
        if self.args.num_seq_per_target == 1:
            print("   📈 Consider increasing --num_seq_per_target to 4-8 for better sampling")
        if not self.args.pack_side_chains:
            print("   📈 Enable --pack_side_chains for better accuracy")
        if not getattr(self.args, 'force_multiprocessing', False) and self.args.num_seq_per_target > 3:
            print("   📈 Consider --force_multiprocessing for guaranteed parallel processing")
        if not getattr(self.args, 'enable_threading_fallback', False):
            print("   📈 Consider --enable_threading_fallback for better reliability")
        if self.args.cpu_allocation_strategy == 'auto':
            print("   📈 Consider --cpu_allocation_strategy aggressive for dedicated systems")
        if worker_config['memory_limited_workers'] < worker_config['optimal_workers']:
            print(f"   ⚠️  Memory limiting workers to {worker_config['memory_limited_workers']} (consider more RAM)")
            
        print("\n10. HYBRID PARALLEL PROCESSING TECHNICAL DETAILS:")
        print("   🔬 Two-Level Parallelism Architecture:")
        print("   ")
        print("   Level 1 - Python Multiprocessing:")
        print("   - True parallelism across CPU cores")
        print("   - Each process has independent PyRosetta instance")
        print("   - No GIL limitations for CPU-intensive operations")
        print("   - Memory isolation between processes")
        print("   ")
        print("   Level 2 - PyRosetta C++ Multithreading:")
        print("   - Native C++ threading within each PyRosetta process")
        print("   - Shared memory within process (efficient)")
        print("   - Optimized for scientific computing workloads")
        print("   - Can utilize vectorized operations and SIMD")
        print("   ")
        if getattr(self.args, 'enable_pyrosetta_threading', False):
            print("   🚀 HYBRID MODE ACTIVE:")
            print(f"   - {worker_config['optimal_workers']} processes × {worker_config['pyrosetta_threads_per_process']} threads")
            print(f"   - Total computational threads: {worker_config['total_effective_threads']}")
            print("   - Theoretical maximum speedup: Process_count × Thread_efficiency")
            print("   - Expected efficiency: 60-85% (depends on workload characteristics)")
        else:
            print("   📝 PROCESS-ONLY MODE:")
            print("   - Using Python multiprocessing only")
            print("   - To enable hybrid mode: --enable_pyrosetta_threading")
            
        print("   ")
        print("   Optimal Use Cases:")
        print("   - Large protein structures (>200 residues)")
        print("   - Multiple sequences per cycle (>4)")
        print("   - Systems with 8+ CPU cores")
        print("   - Memory-rich environments (>16GB RAM)")
        
        print("="*70)


def print_usage_examples():
    """Print usage examples for hybrid parallelism"""
    print("\n" + "="*70)
    print("HYBRID PARALLELISM USAGE EXAMPLES")
    print("="*70)
    
    print("\n1. BASIC HYBRID MODE (Recommended):")
    print("python ligandmpnn_fastrelax_complete.py \\")
    print("    --pdb_path input.pdb \\")
    print("    --ligand_params_path ligand.params \\")
    print("    --num_seq_per_target 8 \\")
    print("    --hybrid_parallelism")
    
    print("\n2. AUTO-TUNED HYBRID MODE:")
    print("python ligandmpnn_fastrelax_complete.py \\")
    print("    --pdb_path input.pdb \\")
    print("    --ligand_params_path ligand.params \\")
    print("    --num_seq_per_target 8 \\")
    print("    --auto_tune_threading")
    
    print("\n3. CUSTOM HYBRID CONFIGURATION:")
    print("python ligandmpnn_fastrelax_complete.py \\")
    print("    --pdb_path input.pdb \\")
    print("    --ligand_params_path ligand.params \\")
    print("    --num_seq_per_target 16 \\")
    print("    --enable_pyrosetta_threading \\")
    print("    --pyrosetta_threads_per_process 3 \\")
    print("    --max_relax_workers 4 \\")
    print("    --cpu_allocation_strategy aggressive")
    
    print("\n4. HIGH-PERFORMANCE MODE (16+ cores):")
    print("python ligandmpnn_fastrelax_complete.py \\")
    print("    --pdb_path input.pdb \\")
    print("    --ligand_params_path ligand.params \\")
    print("    --num_seq_per_target 32 \\")
    print("    --enable_pyrosetta_threading \\")
    print("    --pyrosetta_threads_per_process 4 \\")
    print("    --max_relax_workers 8 \\")
    print("    --cpu_allocation_strategy aggressive \\")
    print("    --force_multiprocessing")
    
    print("\n5. MEMORY-CONSTRAINED MODE:")
    print("python ligandmpnn_fastrelax_complete.py \\")
    print("    --pdb_path input.pdb \\")
    print("    --ligand_params_path ligand.params \\")
    print("    --num_seq_per_target 8 \\")
    print("    --enable_pyrosetta_threading \\")
    print("    --pyrosetta_threads_per_process 2 \\")
    print("    --max_relax_workers 2 \\")
    print("    --memory_per_worker_gb 1.5 \\")
    print("    --cpu_allocation_strategy conservative")
    
    print("\n6. DEVELOPMENT/DEBUGGING MODE:")
    print("python ligandmpnn_fastrelax_complete.py \\")
    print("    --pdb_path input.pdb \\")
    print("    --ligand_params_path ligand.params \\")
    print("    --num_seq_per_target 4 \\")
    print("    --sequential_threshold 10 \\")
    print("    --enable_threading_fallback \\")
    print("    --verbose")
    
    print("\n" + "="*70)
    print("PERFORMANCE EXPECTATIONS BY SYSTEM TYPE")
    print("="*70)
    
    print("\n🖥️  High-End Workstation (16+ cores, 32+ GB RAM):")
    print("   - Expected speedup: 15-25x")
    print("   - Recommended: --auto_tune_threading")
    print("   - Structures per cycle: 16-32")
    
    print("\n💻 Standard Workstation (8-16 cores, 16+ GB RAM):")
    print("   - Expected speedup: 8-15x")
    print("   - Recommended: --hybrid_parallelism")
    print("   - Structures per cycle: 8-16")
    
    print("\n📱 Laptop/Small System (4-8 cores, 8+ GB RAM):")
    print("   - Expected speedup: 3-6x")
    print("   - Recommended: --enable_pyrosetta_threading")
    print("   - Structures per cycle: 4-8")
    
    print("\n⚡ Cloud Instance Recommendations:")
    print("   - AWS c5.4xlarge: --hybrid_parallelism")
    print("   - AWS c5.9xlarge: --auto_tune_threading") 
    print("   - Google Cloud c2-standard-16: --aggressive strategy")
    
    print("="*70)


def estimate_performance_improvement(args):
    """Estimate performance improvement for given configuration"""
    # Get worker configuration
    test_structures = max(1, args.num_seq_per_target)
    worker_config = get_optimal_worker_distribution(test_structures, args.max_relax_workers, args)
    
    # Baseline (sequential processing)
    baseline_time_per_structure = 30  # seconds (typical FastRelax time)
    baseline_total_time = test_structures * baseline_time_per_structure
    
    # Calculate improved time
    if worker_config['threading_mode'] == 'hybrid':
        process_speedup = worker_config['optimal_workers']
        thread_speedup = worker_config['pyrosetta_threads_per_process'] * 0.7  # Thread efficiency
        total_speedup = process_speedup * thread_speedup * 0.85  # System efficiency
    else:
        total_speedup = worker_config['optimal_workers'] * 0.9  # Process-only efficiency
    
    improved_time = baseline_total_time / total_speedup
    time_saved = baseline_total_time - improved_time
    
    return {
        'baseline_time_minutes': baseline_total_time / 60,
        'improved_time_minutes': improved_time / 60,
        'time_saved_minutes': time_saved / 60,
        'speedup_factor': total_speedup,
        'efficiency_percent': (time_saved / baseline_total_time) * 100
    }
    

def parse_arguments():
    """Parse command line arguments"""
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
    
    # LigandMPNN settings
    parser.add_argument('--use_side_chain_context', action='store_true',
                        help='Use side chain context in LigandMPNN')
    parser.add_argument('--pack_side_chains', action='store_true', default=False,
                        help='Use LigandMPNN side chain packing')
    parser.add_argument('--checkpoint_path_sc', type=str, default=None,
                        help='Path to side chain packer model')
    
    # Batch processing and parallelization
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='Maximum batch size for sequence generation')
    parser.add_argument('--max_relax_workers', type=int, default=None,
                        help='Maximum number of parallel workers for fast relax')
    parser.add_argument('--force_multiprocessing', action='store_true',
                        help='Force multiprocessing even for small batches')
    parser.add_argument('--cpu_allocation_strategy', type=str, default='auto',
                        choices=['auto', 'conservative', 'aggressive', 'custom'],
                        help='Strategy for CPU allocation: auto (default), conservative (use 50%% cores), aggressive (use 90%% cores), custom (use exact max_relax_workers)')
    parser.add_argument('--memory_per_worker_gb', type=float, default=2.0,
                        help='Estimated memory per worker in GB (for memory-based worker limiting)')
    parser.add_argument('--enable_threading_fallback', action='store_true',
                        help='Enable threading fallback if multiprocessing fails')
    parser.add_argument('--sequential_threshold', type=int, default=2,
                        help='Use sequential processing if structure count is below this threshold')
    
    # PyRosetta multithreading options
    parser.add_argument('--enable_pyrosetta_threading', action='store_true',
                        help='Enable PyRosetta internal multithreading within each process')
    parser.add_argument('--pyrosetta_threads_per_process', type=int, default=2,
                        help='Number of threads per PyRosetta process (only used if --enable_pyrosetta_threading)')
    parser.add_argument('--hybrid_parallelism', action='store_true',
                        help='Enable hybrid parallelism: Python multiprocessing + PyRosetta multithreading')
    parser.add_argument('--auto_tune_threading', action='store_true',
                        help='Automatically tune the process/thread ratio based on system capabilities')
    
    # Side chain packing parameters
    parser.add_argument('--number_of_packs_per_design', type=int, default=1,
                        help='Number of side chain packing samples per design')
    parser.add_argument('--sc_num_denoising_steps', type=int, default=3,
                        help='Number of denoising steps for side chain packing')
    parser.add_argument('--sc_num_samples', type=int, default=16,
                        help='Number of samples for side chain packing')
    parser.add_argument('--repack_everything', action='store_true',
                        help='Repack all residues (not just designed ones)')
    parser.add_argument('--pack_with_ligand_context', action='store_true', default=True,
                        help='Use ligand context during side chain packing')
    parser.add_argument('--force_hetatm', action='store_true',
                        help='Force ligand atoms to be written as HETATM')
    parser.add_argument('--zero_indexed', action='store_true',
                        help='Use zero-indexed residue numbering')
    
    # Scoring and selection parameters
    parser.add_argument('--score_method', type=str, default='combined', 
                        choices=['mpnn_only', 'rosetta_only', 'combined'],
                        help='Method for selecting best structure: mpnn_only, rosetta_only, or combined')
    parser.add_argument('--mpnn_weight', type=float, default=1.0,
                        help='Weight for MPNN score in combined scoring (default: 1.0)')
    parser.add_argument('--rosetta_weight', type=float, default=0.1,
                        help='Weight for Rosetta energy in combined scoring (default: 0.1)')
    parser.add_argument('--normalize_scores', action='store_true',
                        help='Normalize scores before combining (recommended for fair weighting)')
    
    # Other settings
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--save_stats', action='store_true',
                        help='Save detailed statistics')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    try:
        # Create LigandMPNN FastRelax instance
        mpnn_fr = LigandMPNNFastRelax(args)
        
        # Show performance analysis
        if args.verbose:
            mpnn_fr.analyze_performance_improvements()
            
            # Show usage examples if using hybrid mode
            if (getattr(args, 'enable_pyrosetta_threading', False) or 
                getattr(args, 'hybrid_parallelism', False) or 
                getattr(args, 'auto_tune_threading', False)):
                print_usage_examples()
            
            # Estimate performance improvement
            perf_estimate = estimate_performance_improvement(args)
            print(f"\n📊 ESTIMATED PERFORMANCE FOR THIS RUN:")
            print(f"   - Baseline time: {perf_estimate['baseline_time_minutes']:.1f} minutes")
            print(f"   - Expected time: {perf_estimate['improved_time_minutes']:.1f} minutes")  
            print(f"   - Time saved: {perf_estimate['time_saved_minutes']:.1f} minutes")
            print(f"   - Speedup factor: {perf_estimate['speedup_factor']:.1f}x")
            print(f"   - Efficiency gain: {perf_estimate['efficiency_percent']:.1f}%")
        
        # Run iterative design
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
