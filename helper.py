import gurobipy as gp
from gurobipy import GRB
from matpowercaseframes import CaseFrames
import joblib
import numpy as np
from tqdm import tqdm
import pathlib, re

from sys import stderr
from numpy import zeros, arange, isscalar, dot, ix_, ones, r_, pi, flatnonzero as find
from scipy.sparse import csr_matrix as sparse
from pypower.idx_bus import BUS_TYPE, REF, BUS_I
from pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, SHIFT, BR_STATUS
from numpy.linalg import solve

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
    c2 = case.gencost.COST_2.values * case.baseMVA**2
    c1 = case.gencost.COST_1.values * case.baseMVA
    c0 = case.gencost.COST_0.values
    c = c2 + c1 + c0
    c = np.hstack([c,np.zeros(ngen*2+nbranch*2)])

    return c, d0, fmax, nbranch, ngen, nbus, pmin, pmax
