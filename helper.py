import os
import json
import glob
import gurobipy as gp
from gurobipy import GRB
from matpowercaseframes import CaseFrames
import joblib
import numpy as np
from tqdm import tqdm
import pathlib, re
import pickle
import matplotlib.pyplot as plt

from sys import stderr
from numpy import zeros, arange, isscalar, dot, ix_, ones, r_, pi, flatnonzero as find
from scipy.sparse import csr_matrix as sparse
from pypower.idx_bus import BUS_TYPE, REF, BUS_I
from pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, SHIFT, BR_STATUS
from numpy.linalg import solve

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def makeBdc(baseMVA, bus, branch):
    """Builds the B matrices and phase shift injections for DC power flow.

    Returns the B matrices and phase shift injection vectors needed for a
    DC power flow.
    The bus real power injections are related to bus voltage angles by::
        P = Bbus * Va + PBusinj
    The real power flows at the from end the lines are related to the bus
    voltage angles by::
        Pf = Bf * Va + Pfinj
    Does appropriate conversions to p.u.
    @see: L{dcpf}
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## constants
    nb = bus.shape[0]          ## number of buses
    nl = branch.shape[0]       ## number of lines

    ## check that bus numbers are equal to indices to bus (one set of bus nums)
    if any(bus[:, BUS_I]-1 != list(range(nb))):
        stderr.write('makeBdc: buses must be numbered consecutively in '
                     'bus matrix\n')

    ## for each branch, compute the elements of the branch B matrix and the phase
    ## shift "quiescent" injections, where
    ##
    ##      | Pf |   | Bff  Bft |   | Vaf |   | Pfinj |
    ##      |    | = |          | * |     | + |       |
    ##      | Pt |   | Btf  Btt |   | Vat |   | Ptinj |
    ##
    stat = branch[:, BR_STATUS]               ## ones at in-service branches
    b = stat / branch[:, BR_X]                ## series susceptance
    tap = ones(nl)                            ## default tap ratio = 1
    i = find(branch[:, TAP])               ## indices of non-zero tap ratios
    tap[i] = branch[i, TAP]                   ## assign non-zero tap ratios
    b = b / tap

    ## build connection matrix Cft = Cf - Ct for line and from - to buses
    f = branch[:, F_BUS] -1                           ## list of "from" buses
    t = branch[:, T_BUS] -1                          ## list of "to" buses
    i = r_[range(nl), range(nl)]                   ## double set of row indices
    ## connection matrix
    Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))

    ## build Bf such that Bf * Va is the vector of real branch powers injected
    ## at each branch's "from" bus
    Bf = sparse((r_[b, -b], (i, r_[f, t])), shape = (nl, nb))## = spdiags(b, 0, nl, nl) * Cft

    ## build Bbus
    Bbus = Cft.T * Bf

    ## build phase shift injection vectors
    Pfinj = b * (-branch[:, SHIFT] * pi / 180)  ## injected at the from bus ...
    # Ptinj = -Pfinj                            ## and extracted at the to bus
    Pbusinj = Cft.T * Pfinj                ## Pbusinj = Cf * Pfinj + Ct * Ptinj

    return Bbus, Bf, Pbusinj, Pfinj

def makePTDF(baseMVA, bus, branch, slack=None):
    """Builds the DC PTDF matrix for a given choice of slack.

    Returns the DC PTDF matrix for a given choice of slack. The matrix is
    C{nbr x nb}, where C{nbr} is the number of branches and C{nb} is the
    number of buses. The C{slack} can be a scalar (single slack bus) or an
    C{nb x 1} column vector of weights specifying the proportion of the
    slack taken up at each bus. If the C{slack} is not specified the
    reference bus is used by default.

    For convenience, C{slack} can also be an C{nb x nb} matrix, where each
    column specifies how the slack should be handled for injections
    at that bus.

    @see: L{makeLODF}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## use reference bus for slack by default
    if slack is None:
        slack = find(bus[:, BUS_TYPE] == REF)
        slack = slack[0]

    ## set the slack bus to be used to compute initial PTDF
    if isscalar(slack):
        slack_bus = slack
    else:
        slack_bus = 0      ## use bus 1 for temp slack bus

    nb = bus.shape[0]
    nbr = branch.shape[0]
    noref = arange(1, nb)      ## use bus 1 for voltage angle reference
    noslack = find(arange(nb) != slack_bus)

    ## check that bus numbers are equal to indices to bus (one set of bus numbers)
    if any(bus[:, BUS_I]-1 != arange(nb)):
        stderr.write('makePTDF: buses must be numbered consecutively')

    ## compute PTDF for single slack_bus
    Bbus, Bf, _, _ = makeBdc(baseMVA, bus, branch)
    Bbus, Bf = Bbus.todense(), Bf.todense()
    H = zeros((nbr, nb))
    H[:, noslack] = solve( Bbus[ix_(noslack, noref)].T, Bf[:, noref].T ).T
    #             = Bf[:, noref] * inv(Bbus[ix_(noslack, noref)])

    ## distribute slack, if requested
    if not isscalar(slack):
        if len(slack.shape) == 1:  ## slack is a vector of weights
            slack = slack / sum(slack)   ## normalize weights

            ## conceptually, we want to do ...
            ##    H = H * (eye(nb, nb) - slack * ones((1, nb)))
            ## ... we just do it more efficiently
            v = dot(H, slack)
            for k in range(nb):
                H[:, k] = H[:, k] - v
        else:
            H = dot(H, slack)

    return H                

def create_scenario_multivariate(case, model, N, sigma_scaling = 0.03):
    buses = case.bus.values
    buses_cols = {col:num for num,col in enumerate(case.bus.columns.values)}
    base_demand = buses[:,buses_cols['PD']] / case.baseMVA
    sigma = (sigma_scaling * base_demand)
    mean = np.zeros(len(base_demand))
    covariance_matrix = np.diag(sigma**2)
    omega = np.random.multivariate_normal(mean, covariance_matrix, N)
    return omega

def helper(case):
    bus_to_idx = {bus: i+1 for i, bus in enumerate(case.bus.BUS_I.values)}
    bus_idx = [bus_to_idx[bus] for bus in case.bus.BUS_I.values]
    case.bus.BUS_I = case.bus.BUS_I.replace(bus_to_idx) # rename the bus for making PTDF
    case.gen.GEN_BUS = case.gen.GEN_BUS.replace(bus_to_idx)
    case.branch.F_BUS = case.branch.F_BUS.replace(bus_to_idx)
    case.branch.T_BUS = case.branch.T_BUS.replace(bus_to_idx)
    pmax = case.gen.PMAX.values/case.baseMVA
    pmin = case.gen.PMIN.values/case.baseMVA
    zero_gen_idx = []
    for num,i in enumerate(pmax):
        if i == 0 and pmin[num] == 0:
            zero_gen_idx.append(num+1)
    case.gen.drop(index=zero_gen_idx, inplace=True) # drop
    case.gencost.drop(index=zero_gen_idx, inplace=True) # drop
    pmax = np.delete(pmax, np.array(zero_gen_idx).astype(np.int32)-1)
    pmin = np.delete(pmin, np.array(zero_gen_idx).astype(np.int32)-1)
    case.M = makePTDF(case.baseMVA, case.bus.values, case.branch.values, slack=None)
    nbus = case.bus.shape[0]
    ngen = case.gen.shape[0]
    nbranch = case.branch.shape[0]
    case.H = np.zeros((nbus,ngen))
    for gen,bus in enumerate(case.gen.GEN_BUS.values):
        case.H[int(bus)-1][gen] = 1
    fmax = case.branch.RATE_A.values/case.baseMVA
    d0 = case.bus.PD.values/case.baseMVA
    c2 = case.gencost.C2.values * case.baseMVA**2
    c1 = case.gencost.C1.values * case.baseMVA
    c0 = case.gencost.C0.values
    c = c2 + c1 + c0
    c = np.hstack([c,np.zeros(ngen*2+nbranch*2)])

    return c, d0, fmax, nbranch, ngen, nbus, pmin, pmax

def lpformulator_dc_body_primal(case, model):
    _add_dc_gen_bus_variables(case, model)				
    set_gencost_objective_primal(case, model)			# (1a✓) 
    _add_dc_bus_balance_constraints(case, model)		# (1b✓)
    _add_generator_limit_constraints(case, model)       # (1c✓)
    _add_branch_limit_constraints(case, model)          # (1d✓)

def _add_dc_gen_bus_variables(case, model):  #(1b)
    gens = case.gen.values
    branches = case.branch.values
    buses = case.bus.values
    p = model.addMVar(shape=len(gens),lb=-GRB.INFINITY, ub=GRB.INFINITY , name='p')
    omega = model.addMVar(shape=len(buses), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='omega')
    case.p = p
    case.omega = omega

def _add_dc_bus_balance_constraints(case, model): # (1b)
    p = case.p
    omega = case.omega
    buses = case.bus.values
    buses_cols = {col:num for num,col in enumerate(case.bus.columns.values)}
    Pd = buses[:,buses_cols['PD']]/case.baseMVA
    pmin = case.gen.PMIN.values / case.baseMVA
    model.addConstr(p.sum() == Pd.sum()  + omega.sum(), name="(4b)")

def _add_generator_limit_constraints(case, model):
    gens = case.gen.values
    p = case.p
    gens_cols = {col:num for num,col in enumerate(case.gen.columns.values)}    
    Pmax = gens[:,gens_cols['PMAX']]/case.baseMVA
    Pmin = gens[:,gens_cols['PMIN']]/case.baseMVA    
    status = gens[:,gens_cols['GEN_STATUS']]
    model.addConstr(p <= Pmax, name="(4c_up)")
    model.addConstr(p >= Pmin, name="(4c_low)")

def _add_branch_limit_constraints(case, model): #(1d)
    branches = case.branch.values
    branches_cols = {col:num for num,col in enumerate(case.branch.columns.values)}
    buses = case.bus.values
    buses_cols = {col:num for num,col in enumerate(case.bus.columns.values)}    
    M, H = case.M, case.H
    p = case.p
    omega = case.omega           
    d = buses[:,buses_cols['PD']]/case.baseMVA
    limit = branches[:,branches_cols['RATE_A']]/case.baseMVA
    pmin = case.gen.PMIN.values / case.baseMVA
    model.addConstr(M@(H@p -d-omega) <= limit, name="(4d)")    
    model.addConstr(M@(H@p -d-omega) >= -limit,name="(4e)")

def set_gencost_objective_primal(case, model): # (1a)
    p = case.p
    pmin = case.gen.PMIN.values / case.baseMVA
    costvector = case.gencost[['C2', 'C1', 'C0']].values
    objective_quadratic = (costvector[:, -3]*case.baseMVA**2 ) @ p
    objective_linear = (costvector[:, -2]*case.baseMVA) @ p
    objective_constant = costvector[:, -1] @ p
    model.setObjective(
        objective_quadratic + objective_constant + objective_linear , sense=GRB.MINIMIZE
    )    

def update_injection_constraints(case, model, omega_bound):
    omega = case.omega
    omega.LB = omega_bound
    omega.UB = omega_bound
    model.update()  # Update the model to reflect these changes

class XGBMultiOutputWrapper:
    def __init__(self, n_estimators=100, eval_metric='logloss', random_state=None):
        
        # Dynamically detect hardware so it still works locally on your laptop
        import torch
        compute_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   -> [XGBoost] Hardware detected and assigned: {compute_device.upper()}")
        
        self.model = MultiOutputClassifier(
            XGBClassifier(
                n_estimators=n_estimators, 
                eval_metric=eval_metric, 
                random_state=random_state,
                tree_method='hist',           # REQUIRED for GPU acceleration
                device=compute_device,        # REQUIRED for GPU acceleration
                n_jobs=1                      # Prevents CPU thread crashing
            ),
            n_jobs=-1                         # <--- THE FIX: Trains targets concurrently
        )
        self.encoders = []
        
    def fit(self, X, y):
        # Create a separate LabelEncoder for every column in the target matrix y
        self.encoders = [LabelEncoder() for _ in range(y.shape[1])]
        y_encoded = np.zeros_like(y)
        
        # Fit encoder and transform data column by column
        for i in range(y.shape[1]):
            y_encoded[:, i] = self.encoders[i].fit_transform(y[:, i])
        
        # Train XGBoost on the 0-indexed encoded values
        self.model.fit(X, y_encoded)
        return self
        
    def predict(self, X):
        # Get raw predictions (which will be 0, 1, 2, etc.)
        y_pred_encoded = self.model.predict(X)
        y_pred = np.zeros_like(y_pred_encoded)
        
        # Decode predictions back to original values (-1, 0, etc.)
        for i in range(y_pred_encoded.shape[1]):
            y_pred[:, i] = self.encoders[i].inverse_transform(y_pred_encoded[:, i])
            
        return y_pred

# 1. Define the architecture globally so joblib can pickle it
class SharedMLP(nn.Module):
    def __init__(self, in_dim, targets, classes, h1, h2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            # Output size is (targets * classes) to predict everything at once
            nn.Linear(h2, targets * classes)
        )
        self.targets = targets
        self.classes = classes

    def forward(self, x):
        out = self.net(x)
        # Reshape to (batch_size, num_classes, num_targets) for PyTorch CrossEntropyLoss
        return out.view(-1, self.classes, self.targets)


# 2. The wrapper now calls the global SharedMLP class
class PyTorchMLPWrapper:
    """
    A PyTorch shared-representation Multi-Layer Perceptron that acts like a scikit-learn model.
    It predicts all basis statuses simultaneously to capture physical constraints.
    """
    def __init__(self, hidden_sizes=(100, 50), max_iter=500, batch_size=64, lr=0.001, random_state=None):
        self.hidden_sizes = hidden_sizes
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.encoders = []
        self.model = None
        self.num_classes = 0

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        # Encode targets just like the XGBWrapper
        self.encoders = [LabelEncoder() for _ in range(y.shape[1])]
        y_encoded = np.zeros_like(y)
        for i in range(y.shape[1]):
            y_encoded[:, i] = self.encoders[i].fit_transform(y[:, i])
            
        # Find the maximum number of unique classes across all target columns
        self.num_classes = int(np.max(y_encoded)) + 1
        num_features = X.shape[1]
        num_targets = y.shape[1]
        
        # Initialize the global model and move to CPU/GPU
        self.model = SharedMLP(num_features, num_targets, self.num_classes, 
                               self.hidden_sizes[0], self.hidden_sizes[1]).to(self.device)
        
        # Prepare PyTorch DataLoaders
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training Loop
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.max_iter):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Move model back to CPU at the end to ensure it saves safely via joblib
        self.model.to('cpu')
        return self
        
    def predict(self, X):
        self.model.to(self.device)
        self.model.eval()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # Get the class with the highest probability for each target
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
        # Decode back to original basis statuses (-1, 0, etc.)
        y_pred = np.zeros_like(predictions)
        for i in range(predictions.shape[1]):
            y_pred[:, i] = self.encoders[i].inverse_transform(predictions[:, i])
            
        self.model.to('cpu')
        return y_pred