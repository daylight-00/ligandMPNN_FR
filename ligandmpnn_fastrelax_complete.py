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

import torch
import numpy as np

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
        )
        from model_utils import ProteinMPNN
        return True, (parse_PDB, featurize, get_score, alphabet, restype_int_to_str, restype_str_to_int, ProteinMPNN)
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
        
        self.parse_PDB, self.featurize, self.get_score, self.alphabet, self.restype_int_to_str, self.restype_str_to_int, self.ProteinMPNN = modules
        
        # Load model
        self.model = self._load_ligandmpnn_model()
        
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
        """Load LigandMPNN model"""
        if self.args.checkpoint_path:
            checkpoint_path = self.args.checkpoint_path
        else:
            model_folder = self.args.path_to_model_weights
            if model_folder[-1] != '/':
                model_folder += '/'
            checkpoint_path = f"{model_folder}{self.args.model_name}.pt"
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
        if self.args.verbose:
            print(f"Loading model from: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Get model parameters
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
        Generate sequences using LigandMPNN
        
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
        
        # Add required keys to feature_dict based on successful simple version
        B, L, _, _ = feature_dict["X"].shape

        # 반드시 batch_size를 명시적으로 추가 (run.py, simple 모두 동일)
        feature_dict["batch_size"] = 1
        feature_dict["temperature"] = temperature

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
        
        for i in range(num_sequences):
            # Add randn for sampling
            feature_dict["randn"] = torch.randn([1, L], device=self.device)
            
            with torch.no_grad():
                # Sample sequence using only feature_dict parameter
                output_dict = self.model.sample(feature_dict)
                
                # Get sequence and score
                S_sample = output_dict["S"]
                log_probs = output_dict.get("log_probs", None)
                
                # Convert to sequence string
                seq = self._S_to_seq(S_sample[0])
                
                # Calculate score if log_probs available
                score = 0.0
                if log_probs is not None:
                    mask = feature_dict["mask"] * feature_dict["chain_mask"]
                    score, _ = self.get_score(S_sample, log_probs, mask)
                    score = score.item()
                    
                generated_sequences.append({
                    'sequence': seq,
                    'score': score,
                    'S_sample': S_sample,
                    'temperature': temperature,
                    'index': i
                })
                
                if self.args.verbose:
                    print(f"Generated sequence {i+1}: {seq}")
                    print(f"Score: {score:.4f}")
                
        return generated_sequences
        
    def _S_to_seq(self, S):
        """Convert sequence tensor to string"""
        return ''.join([self.restype_int_to_str[s.item()] for s in S])
        
    def thread_sequence_to_pdb(self, pdb_path, sequence, output_path):
        """Thread new sequence onto PDB structure"""
        if self.args.verbose:
            print("Threading sequence onto structure...")
            
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
            print(f"Threaded structure saved: {output_path}")
            print(f"Successfully threaded {threaded_count}/{len(seq_list)} residues")
            
        return pose
        
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
        Run one cycle of LigandMPNN design + fast relax
        
        Args:
            pdb_path: Input PDB path
            cycle_num: Current cycle number
            
        Returns:
            Path to relaxed PDB file
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
        
        # Select best sequence
        best_sequence = min(sequences, key=lambda x: x['score'])
        
        if self.args.verbose:
            print(f"Best sequence score: {best_sequence['score']:.4f}")
            print(f"Best sequence: {best_sequence['sequence']}")
            
        # Thread sequence to structure
        header = os.path.basename(pdb_path).replace('.pdb', '')
        threaded_path = f"{self.base_folder}backbones/{header}_cycle_{cycle_num}_threaded.pdb"
        
        threaded_pose = self.thread_sequence_to_pdb(
            pdb_path, best_sequence['sequence'], threaded_path
        )
        
        # Perform fast relax
        relaxed_pose = self.fast_relax(threaded_pose)
        
        # Save relaxed structure
        relaxed_path = f"{self.base_folder}relaxed/{header}_cycle_{cycle_num}_relaxed.pdb"
        relaxed_pose.dump_pdb(relaxed_path)
        
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
                'threaded_pdb': threaded_path,
                'relaxed_pdb': relaxed_path,
                'sequence': best_sequence['sequence'],
                'score': best_sequence['score'],
                'temperature': best_sequence['temperature'],
                'num_sequences_generated': len(sequences),
                'all_scores': [seq['score'] for seq in sequences]
            }
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
                
        if self.args.verbose:
            print(f"Cycle {cycle_num} completed")
            print(f"Relaxed structure: {relaxed_path}")
            
        return relaxed_path
        
    def run_iterative_design(self):
        """Run iterative LigandMPNN + FastRelax cycles"""
        current_pdb = self.args.pdb_path
        
        start_time = time.time()
        
        for cycle in range(1, self.args.n_cycles + 1):
            # Run MPNN + FastRelax cycle
            relaxed_pdb = self.run_mpnn_fastrelax_cycle(current_pdb, cycle)
            
            # Use relaxed structure as input for next cycle
            current_pdb = relaxed_pdb
                
        end_time = time.time()
        
        if self.args.verbose:
            print("\nIterative design completed!")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            print(f"Final structure: {current_pdb}")
            
        return current_pdb


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
        
        # Run iterative design
        final_pdb = mpnn_fr.run_iterative_design()
        
        print("\nIterative design completed successfully!")
        print(f"Final structure: {final_pdb}")
        
    except Exception as e:
        print(f"Error during iterative design: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
