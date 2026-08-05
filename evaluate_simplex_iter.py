import os
import glob
import re
import gc
import joblib
import pickle
import argparse
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
from matpowercaseframes import CaseFrames
import torch

# =============================================================================
# 1. Import from your local helper.py
# =============================================================================
from helper import (
    helper, 
    create_scenario_multivariate, 
    lpformulator_dc_body_primal, 
    update_injection_constraints
)

# =============================================================================
# 2. Core Functions
# =============================================================================

def pred_omega(case, model_primal, N, sigma_scaling, model, d_ref):
    """Generates load deviation scenarios and predicts the optimal basis."""
    omega_valid = create_scenario_multivariate(case, model_primal, N, sigma_scaling=sigma_scaling)
    d_valid = d_ref + omega_valid
    y_val = model.predict(d_valid)
    return y_val, d_valid, omega_valid

def iter_warmstart(case, omega_valid, y_val, num_cbasis):
    """Injects predicted basis into Gurobi and tracks Simplex iteration counts."""
    model_valid = gp.Model("DC_Formulation_Model_valid")
    model_valid.params.OutputFlag = 0
    
    # REQUIRED: Presolve must be 0 for exact basis mapping
    model_valid.setParam('Presolve', 0) 
    model_valid.setParam('Method', 1)
    model_valid.setParam('LPWarmStart', 1)
    
    lpformulator_dc_body_primal(case, model_valid)
    iter_valid = {}
    
    for i in tqdm(range(len(omega_valid)), desc="Evaluating Warm-Starts", leave=False):
        model_valid.reset()
        update_injection_constraints(case, model_valid, omega_valid[i])
        
        single_prediction = y_val[i]
        predicted_cbasis = single_prediction[:num_cbasis]
        predicted_vbasis = single_prediction[num_cbasis:]

        constrs = model_valid.getConstrs()
        for k, constr in enumerate(constrs):
            constr.CBasis = int(predicted_cbasis[k])
            
        vars = model_valid.getVars()
        for k, var in enumerate(vars):
            var.VBasis = int(predicted_vbasis[k])

        model_valid.update()
        model_valid.optimize()
        iter_valid[i] = model_valid.IterCount    
        
    return iter_valid

# =============================================================================
# 3. Automated Batch Evaluation Engine
# =============================================================================

def run_batch_evaluation(target_case, models_dir='./models', data_dir='./omega_cbasis', results_dir='./results', N=5000):
    os.makedirs(results_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Hardware context initialized: {device.upper()}")
    print(f"Targeting network case: {target_case}\n")

    # ---------------------------------------------------------
    # Initialize Case Framework ONCE using your helper.py
    # ---------------------------------------------------------
    print(f" [*] Initializing LP formulation for {target_case}...")
    case_path = f'../pglib-opf-21.07/{target_case}.m'
    case = CaseFrames(case_path)
    
    # Use your helper function to build H, M, and clean generators
    _, d_ref, _, _, _, _, _, _ = helper(case)

    # Initialize Gurobi environment
    model_primal = gp.Model(f"DC_Formulation_Model_primal_{target_case}")
    model_primal.params.OutputFlag = 0
    model_primal.setParam('Presolve', 0)
    model_primal.setParam('LPWarmStart', 0)
    model_primal.setParam('Method', 1)
    lpformulator_dc_body_primal(case, model_primal)
    
    print(" [+] Network initialization complete.\n")

    # ---------------------------------------------------------
    # Scan models and evaluate only those matching target_case
    # ---------------------------------------------------------
    model_files = glob.glob(os.path.join(models_dir, '*.joblib'))
    if not model_files:
        print(f"No models found in {models_dir}.")
        return
    
    pattern = re.compile(r"(?P<case_name>pglib_opf_[a-zA-Z0-9\_]+)_sigma(?P<sigma>[0-9\.]+)\_(?P<model_type>[a-zA-Z]+)\_seed(?P<seed>\d+)\.joblib")
    evaluation_results = []

    for model_path in sorted(model_files):
        filename = os.path.basename(model_path)
        match = pattern.match(filename)
        
        if not match:
            continue
            
        case_name = match.group('case_name')
        
        # Skip if this model file doesn't match the command-line flag
        if case_name != target_case:
            continue
            
        sigma_scaling = float(match.group('sigma'))
        model_type = match.group('model_type')
        seed = int(match.group('seed'))
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {filename}")
        print(f"Parse: Sigma={sigma_scaling} | Model={model_type.upper()} | Seed={seed}")
        print(f"{'='*70}")

        # Load target mapping to identify the size of the constraint basis
        pickle_path = os.path.join(data_dir, f'{case_name}_{sigma_scaling}.pkl')
        try:
            with open(pickle_path, 'rb') as f:
                historical_data = pickle.load(f)
                
            cbasis_loaded = historical_data['cbasis']
            num_cbasis = len(cbasis_loaded[0])
            
        except FileNotFoundError:
            print(f" [!] Missing target data {pickle_path}. Skipping.")
            continue

        # Execute Warm-Start Prediction and Evaluation
        try:
            loaded_model = joblib.load(model_path)
            
            print(" [*] Predicting basis structures...")
            y_val, d_valid, omega_valid = pred_omega(case, model_primal, N, sigma_scaling, loaded_model, d_ref)

            print(" [*] Injecting predictions to Gurobi...")
            iter_valid = iter_warmstart(case, omega_valid, y_val, num_cbasis)
            
            ml_avg_iter = np.average(list(iter_valid.values()))
            
            print(f"\n -> {model_type.upper()} Warm-Start Avg Iterations:  {ml_avg_iter:.2f}")
            
            evaluation_results.append({
                'case_name': case_name,
                'sigma_scaling': sigma_scaling,
                'model_type': model_type,
                'seed': seed,
                'ml_avg_iter': ml_avg_iter
            })

        except Exception as e:
            print(f" [!] Error evaluating {filename}: {e}")

        # Clear large objects from memory
        if 'loaded_model' in locals(): del loaded_model
        if 'y_val' in locals(): del y_val, d_valid, omega_valid, iter_valid
        gc.collect()

    # ---------------------------------------------------------
    # Save Consolidated Report for this Case
    # ---------------------------------------------------------
    if evaluation_results:
        results_df = pd.DataFrame(evaluation_results)
        csv_out = os.path.join(results_dir, f"{target_case}_model_evaluation_summary.csv")
        results_df.to_csv(csv_out, index=False)
        print(f"\nBatch evaluation complete for {target_case}. Summary saved to {csv_out}")
        
        print("\n--- Quick Pivot Summary (Average Iterations) ---")
        summary_pivot = results_df.pivot_table(
            index=['case_name', 'sigma_scaling'], 
            columns='model_type', 
            values='ml_avg_iter', 
            aggfunc='mean'
        ).round(2)
        print(summary_pivot)
    else:
        print(f"\nNo valid models were successfully evaluated for {target_case}.")

# =============================================================================
# Execute Engine
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ML Models for DCOPF Simplex Warm-Starts")
    parser.add_argument("--case", type=str, required=True, help="Name of the PGLib-OPF case (e.g., pglib_opf_case73_ieee_rts)")
    parser.add_argument("--models_dir", type=str, default="./models", help="Directory containing .joblib models")
    parser.add_argument("--data_dir", type=str, default="./omega_cbasis", help="Directory containing target pickle data")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save evaluation summaries")
    parser.add_argument("--N", type=int, default=5000, help="Number of scenarios to simulate")
    
    args = parser.parse_args()
    
    run_batch_evaluation(
        target_case=args.case, 
        models_dir=args.models_dir, 
        data_dir=args.data_dir, 
        results_dir=args.results_dir, 
        N=args.N
    )