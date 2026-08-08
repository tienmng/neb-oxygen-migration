#!/usr/bin/env python3
"""
VASP Calculation Management Module
=================================
Handles VASP job submission, monitoring, and energy reading
"""

import os
import time
import subprocess
from typing import Dict, List, Optional, Tuple
from pymatgen.io.vasp import Outcar


class VASPManager:
    """Manages VASP calculations including submission and monitoring"""
    
    @staticmethod
    def check_calculation_complete(calc_dir: str) -> bool:
        """
        Check if VASP calculation is complete
        
        Args:
            calc_dir: Directory containing VASP calculation
            
        Returns:
            True if calculation is complete
        """
        outcar_path = os.path.join(calc_dir, "OUTCAR")
        
        if not os.path.exists(outcar_path):
            return False
        
        # Check file size
        if os.path.getsize(outcar_path) < 1000:
            return False
        
        # Check if calculation finished properly
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
                if "General timing and accounting informations for this job:" in content:
                    return True
                if "reached required accuracy" in content:
                    return True
        except Exception:
            pass
        
        return False
    
    @staticmethod
    def read_energy(calc_dir: str) -> Optional[float]:
        """
        Read energy from OUTCAR with multiple fallback methods
        
        Args:
            calc_dir: Directory containing VASP calculation
            
        Returns:
            Energy in eV or None if reading failed
        """
        outcar_path = os.path.join(calc_dir, "OUTCAR")
        
        if not os.path.exists(outcar_path):
            return None
        
        try:
            outcar = Outcar(outcar_path)
            
            # Try different methods to get energy
            if hasattr(outcar, 'final_energy') and outcar.final_energy is not None:
                return outcar.final_energy
            
            if hasattr(outcar, 'final_e0') and outcar.final_e0 is not None:
                return outcar.final_e0
            
            if hasattr(outcar, 'ionic_steps') and len(outcar.ionic_steps) > 0:
                last_step = outcar.ionic_steps[-1]
                if 'e_0_energy' in last_step:
                    return last_step['e_0_energy']
            
            # Manual parsing as last resort
            with open(outcar_path, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if "energy(sigma->0)" in line:
                        try:
                            energy = float(line.split()[-1])
                            return energy
                        except:
                            continue
                    if "free  energy   TOTEN" in line:
                        try:
                            energy = float(line.split()[4])
                            return energy
                        except:
                            continue
                            
        except Exception as e:
            print(f"Warning: Could not read energy from {calc_dir}: {e}")
        
        return None
    
    @staticmethod
    def wait_for_slurm_jobs(job_ids: List[str], check_interval: int = 30, 
                          status_interval: int = 300):
        """
        Wait for SLURM jobs to complete with robust monitoring
        
        Args:
            job_ids: List of SLURM job IDs
            check_interval: Seconds between status checks
            status_interval: Seconds between status updates
        """
        if not job_ids:
            return
        
        print(f"Waiting for job IDs: {job_ids}")
        
        start_time = time.time()
        last_status_time = 0
        
        while True:
            try:
                # Use squeue to check job status
                cmd = ['squeue', '-j', ','.join(map(str, job_ids)), '-h', '-o', '%i %T']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Parse output to get running jobs
                    running_jobs = []
                    pending_jobs = []
                    
                    if result.stdout.strip():
                        for line in result.stdout.strip().split('\n'):
                            if line.strip():
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    job_id, state = parts[0], parts[1]
                                    if state in ['RUNNING', 'R']:
                                        running_jobs.append(job_id)
                                    elif state in ['PENDING', 'PD']:
                                        pending_jobs.append(job_id)
                    
                    # All jobs finished if no running or pending jobs
                    if not running_jobs and not pending_jobs:
                        elapsed = time.time() - start_time
                        print(f"✓ All jobs completed after {elapsed/60:.1f} minutes")
                        break
                    
                    # Print status update every status_interval
                    current_time = time.time()
                    if current_time - last_status_time > status_interval:
                        elapsed = (current_time - start_time) / 60
                        print(f"  Status update ({elapsed:.1f} min): "
                              f"{len(running_jobs)} running, {len(pending_jobs)} pending")
                        last_status_time = current_time
                
                else:
                    # squeue failed, try sacct
                    print(f"squeue failed (return code {result.returncode}), trying sacct...")
                    
                    cmd = ['sacct', '-j', ','.join(map(str, job_ids)), '-n', '-o', 'JobID,State']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        # Check if any jobs are still running
                        still_running = False
                        for line in result.stdout.strip().split('\n'):
                            if line.strip() and any(state in line for state in ['RUNNING', 'PENDING']):
                                still_running = True
                                break
                        
                        if not still_running:
                            elapsed = time.time() - start_time
                            print(f"✓ All jobs completed after {elapsed/60:.1f} minutes")
                            break
                    else:
                        print("Both squeue and sacct failed. Jobs may have completed.")
                        break
            
            except subprocess.TimeoutExpired:
                print("SLURM command timed out, retrying...")
            except Exception as e:
                print(f"Error checking job status: {e}")
                print("Will check calculation directories instead...")
                break
            
            # Wait before next check
            time.sleep(check_interval)
        
        print("Job monitoring complete")
    
    @staticmethod
    def check_calculations_complete(job_data: List[Tuple[str, str, str]]) -> bool:
        """
        Fallback method to check if calculations are complete by examining directories
        
        Args:
            job_data: List of (job_id, unique_key, calc_dir) tuples
            
        Returns:
            True if all calculations are complete
        """
        print("Checking calculation directories for completion...")
        
        incomplete_jobs = []
        for job_id, unique_key, calc_dir in job_data:
            if not VASPManager.check_calculation_complete(calc_dir):
                incomplete_jobs.append((job_id, unique_key))
        
        if incomplete_jobs:
            print(f"⚠️  {len(incomplete_jobs)} calculations may still be running:")
            for job_id, unique_key in incomplete_jobs[:5]:  # Show first 5
                print(f"   {unique_key} (Job ID: {job_id})")
            if len(incomplete_jobs) > 5:
                print(f"   ... and {len(incomplete_jobs) - 5} more")
            print("You may want to check these manually with 'squeue' or 'sacct'")
        else:
            print("✓ All calculations appear to be complete")
        
        return len(incomplete_jobs) == 0
    
    @staticmethod
    def batch_read_energies(calc_dir: str, unique_structures: Dict[str, any]) -> Dict[str, float]:
        """
        Read energies from all completed calculations
        
        Args:
            calc_dir: Base calculation directory
            unique_structures: Dictionary of unique structure keys
            
        Returns:
            Dictionary mapping structure keys to energies
        """
        energies = {}
        successful = 0
        failed = 0
        
        print(f"Reading energies from {len(unique_structures)} calculations...")
        
        for i, unique_key in enumerate(sorted(unique_structures.keys())):
            struct_calc_dir = os.path.join(calc_dir, unique_key)
            
            if VASPManager.check_calculation_complete(struct_calc_dir):
                energy = VASPManager.read_energy(struct_calc_dir)
                if energy is not None:
                    energies[unique_key] = energy
                    successful += 1
                    if (i + 1) % 10 == 0:  # Progress update every 10
                        print(f"  Progress: {i+1}/{len(unique_structures)} calculations read")
                else:
                    failed += 1
            else:
                failed += 1
        
        print(f"\n✓ Energy reading complete:")
        print(f"  - Successful: {successful}")
        print(f"  - Failed: {failed}")
        
        return energies