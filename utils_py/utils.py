# -*- coding: utf-8 -*-
#

"""
Spatially extended Epileptor model.

"""
import numpy
from tvb.simulator.models.base import ModelNumbaDfun
# from tvb.simulator.models import MontbrioPaxoRoxin
import numba
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final, HasTraits, Attr
from tvb.datatypes.equations import SpatialApplicableEquation, FiniteSupportEquation
from tvb.simulator.coupling import Coupling, SparseCoupling
from tvb.simulator.lab import *
import scipy, h5py
from scipy.optimize import fsolve
import nibabel as nib
import pyvista as pv
from tvb.simulator.history import SparseHistory,DenseHistory
from tvb.datatypes import connectivity
import scipy.sparse as scpsp
import numexpr
from tvb.simulator.descriptors import StaticAttr, Dim, NDArray




class csr_mat(Attr):
    def __init__(
            self, default=None, doc='', label='', required=False, final=False, choices=None,):
        super(csr_mat, self).__init__(
            field_type=scpsp.csr_matrix, default=None, doc='', label='', required=required, final=final, choices=choices
        )

class Connectivity_Sparse(connectivity.Connectivity):
    region_labels = NArray(required=False)
    centres = NArray(required=False)
    weights = csr_mat()
    tract_lengths = csr_mat()
    idelays = csr_mat(required=False)
    delays = csr_mat(required=False)

    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialization.
        """
        assert numpy.array_equal(self.tract_lengths.indices, self.weights.indices), "weights and tract lenghts don't have non-zero entries at same positions"
        assert numpy.array_equal(self.tract_lengths.indptr, self.weights.indptr), "weights and tract lenghts don't have non-zero entries at same positions"
        self.number_of_regions = int(self.weights.shape[0])
        self.number_of_connections = int(self.weights.nonzero()[0].shape[0])
        assert not self.speed is None, "Need to set conduction speed in connectivity to compute delays."
        self.delays = self.tract_lengths.copy()  #Change to restore: self.delays = scpsp.csr_matrix(self.tract_lengths / self.speed)
        self.delays.data /= self.speed.item()    #Remove to restore
        if (abs(self.weights-self.weights.T)>1e-10).nnz == 0:
            self.undirected = True
        self.validate()
    
    def set_idelays(self, dt):
        # Express delays in integration steps
        self.idelays = numpy.rint(self.delays / dt).astype(numpy.int32)
        self.has_delays = self.idelays.data.any()
        self._horizon = self.idelays.max() + 1
        # nn = self.idelays.shape[0]
        # self.inodes = numpy.tile(numpy.r_[:nn], (nn, 1))
        # self.delay_indices = self.idelays * nn + self.inodes


class SparseHistory_Sparse():
    # n_time, n_node, n_cvar, n_mode = Dim(), Dim(), Dim(), Dim()
    # delayed_state = NDArray(('n_node', 'n_cvar', 'n_node', 'n_mode'), 'f', read_only=False)
    # nnz_mask = NDArray(('n_node', 'n_node'), numpy.bool_)
    def __init__(self, sim, initial_conditions=None):
        self.n_time = sim.connectivity._horizon
        self.n_node = sim.connectivity.number_of_regions
        self.n_cvar = len(sim.model.cvar)
        self.n_mode = sim.model.number_of_modes
        self.cvars = sim.model.cvar
        if initial_conditions is None:
            self.buffer = numpy.zeros((self.n_time, self.n_cvar, self.n_node, self.n_mode))
        else : 
            assert initial_conditions.shape == (self.n_time, sim.model._nvar, self.n_node, self.n_mode), "Initial conditions are not in the correct shape, should be (max(time delay steps), n state var, n regions, n modes)"
            self.buffer = initial_conditions[:, self.cvars]
        
        
        self.idelays = sim.connectivity.idelays
        if hasattr(sim.model, 'nest_node'):
            self.subset_mask = numpy.isin(self.idelays.indices, sim.model.nest_node)
            self.nest_nodes=sim.model.nest_node

        else:
            self.subset_mask = numpy.zeros_like(self.idelays.indices, dtype=bool)  # oppure altro fallback
        # self.subset_mask = numpy.isin(self.idelays.indices, sim.model.nest_node)

        self.delays = sim.connectivity.delays
        self.weights_data = sim.connectivity.weights.data
        self.weights_indptr = sim.connectivity.weights.indptr
        self.coupling_array = numpy.zeros((self.n_cvar, self.n_node, self.n_mode))
        sim.current_state = initial_conditions[-1].copy()
        sim.current_step += initial_conditions.shape[0] - 1
        # self.delayed_state[:] = 0.0

    def update(self, step, new_state):
        self.buffer[step % self.n_time] = new_state[self.cvars]

    def query_sparse(self, step):
        # print('nest nodes of shape: ',self.nest_nodes.shape)
        # Only select the subset of neurons
        # print('idelays indices of shape ',self.idelays.indices.shape,' with vals between: ',numpy.min(self.idelays.indices),numpy.max(self.idelays.indices))
        # print('subset_mask with ',numpy.count_nonzero(subset_mask),' non null ids')

    
        time_indices = ((step - 1 - self.idelays.data + self.n_time) % self.n_time)
        delayed_state = self.buffer[time_indices,:,self.idelays.indices,:]
        # delayed_state_region = self.buffer[time_indices[self.subset_mask],:,self.idelays.indices[self.subset_mask],:]
        current_state = self.buffer[(step - 1) % self.n_time]
        return current_state, delayed_state.transpose((1,0,2))
        # return current_state, delayed_state.transpose((1,0,2)),delayed_state_region.transpose((1,0,2))
    
    def query_sparse_region(self, step):
        # print('nest nodes of shape: ',self.nest_nodes.shape)
        # Only select the subset of neurons
        # print('idelays indices of shape ',self.idelays.indices.shape,' with vals between: ',numpy.min(self.idelays.indices),numpy.max(self.idelays.indices))
        # print('subset_mask with ',numpy.count_nonzero(subset_mask),' non null ids')

    
        time_indices = ((step - 1 - self.idelays.data + self.n_time) % self.n_time)
        delayed_state_region = self.buffer[time_indices[self.subset_mask],:,self.idelays.indices[self.subset_mask],:]
        current_state = self.buffer[(step - 1) % self.n_time]
        return current_state,delayed_state_region.transpose((1,0,2))
    
    def query(self, step, out=None):
        current, delayed= self.query_sparse(step)
        # current, delayed,delayed_region = self.query_sparse(step)
        # self.delayed_state.transpose((1, 0, 2, 3))[:, self.nnz_mask] = delayed
        # print(type(delayed))
        # print('delayed shape: ',delayed.shape)
        # print(self.idelays.indices)
        return current, delayed
        # return current, delayed,delayed_region

    
        # return current, self.delayed_state
    @property
    def nbytes(self):
        # keep it simple, ignore the other parts of the history object
        # it seems this is just used to estimate and log the memory requirement
        return self.buffer.nbytes 
    
    def initialize(self, init):
        if init.shape[1] > len(self.cvars):
            init = init[:, self.cvars] # simulator still thinks history is (time, svar, ..)
        self.buffer = init

class Linear_Sparse(SparseCoupling):
    """
    Implement Linear step coupling function H(x).
    y=x
    """
    a = NArray(
        label=":math:`a`",
        default=numpy.array([1,]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="Rescales the connection strength while maintaining the ratio "
            "between different values.")
    
    b = NArray(
        label=":math:`b`",
        default=numpy.array([-1]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="Threshold of the heaviside step function.")
        
    def __call__(self, step, history):
        h = history # type: SparseHistory_Sparse
        x_i, x_j = h.query_sparse(step)
        # x_i, x_j,_ = h.query_sparse(step)
        pre = self.pre(x_i, x_j)
        self.csr_mult_sum(h.coupling_array, h.weights_data, pre, h.weights_indptr, h.n_cvar, h.n_mode)
        return self.post(h.coupling_array)

    def pre(self, x_i, x_j):
        return x_j

    def post(self, gx):
        return self.a * gx +self.b

    @staticmethod
    @numba.njit(parallel=True)
    def csr_mult_sum(out, weights_data, pre, indptr, n_cvar, n_mode):
        for i in numba.prange(n_cvar):
            for j in numba.prange(n_mode):
                for k in numba.prange(len(indptr)-1):
                    mult = weights_data[indptr[k]:indptr[k+1]] * pre[i,indptr[k]:indptr[k+1],j]
                    out[i,k,j] = numpy.sum(mult)

class Heaviside_Sparse(SparseCoupling):
    """
    Implement heaviside step coupling function H(x).
    0 if x < 0
    0.5 if x == 0
    1 if x > 0
    """
    a = NArray(
        label=":math:`a`",
        default=numpy.array([1,]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="Rescales the connection strength while maintaining the ratio "
            "between different values.")
    
    theta = NArray(
        label=":math:`\theta`",
        default=numpy.array([-1]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="Threshold of the heaviside step function.")
        
    def __call__(self, step, history):
        h = history # type: SparseHistory_Sparse
        x_i, x_j = h.query_sparse(step)
        # x_i, x_j,_ = h.query_sparse(step)
        pre = self.pre(x_i, x_j)
        self.csr_mult_sum(h.coupling_array, h.weights_data, pre, h.weights_indptr, h.n_cvar, h.n_mode)
        return self.post(h.coupling_array)

    def pre(self, x_i, x_j):
        return numpy.heaviside(x_j - self.theta, 0.5)

    def post(self, gx):
        return self.a * gx

    @staticmethod
    @numba.njit(parallel=True)
    def csr_mult_sum(out, weights_data, pre, indptr, n_cvar, n_mode):
        for i in numba.prange(n_cvar):
            for j in numba.prange(n_mode):
                for k in numba.prange(len(indptr)-1):
                    mult = weights_data[indptr[k]:indptr[k+1]] * pre[i,indptr[k]:indptr[k+1],j]
                    out[i,k,j] = numpy.sum(mult)


class Simulator_Sparse(simulator.Simulator):
    connectivity = Attr(
        field_type=Connectivity_Sparse,
        label="Long-range connectivity",
        default=None,
        required=True,
        doc="""A tvb.datatypes.Connectivity object which contains the
         structural long-range connectivity data (i.e., white-matter tracts). In
         combination with the ``Long-range coupling function`` it defines the inter-regional
         connections. These couplings undergo a time delay via signal propagation
         with a propagation speed of ``Conduction Speed``""")
    

    # def _configure_history(self):
    def _configure_history(self,initial_conditions=None):
        "Initialize history instance; cf. from_simulator for more information."
        self.history = SparseHistory_Sparse(self, self.initial_conditions)

    def _loop_compute_node_coupling(self, step): # overwrite to ignore surface
        """Compute delayed node coupling values."""
        return self.coupling(step, self.history)
    

    # def _loop_update_history(self, step,n_reg, state): # overwrite to ignore surface
    def _loop_update_history(self, step,state): # overwrite to ignore surface
        """Update history."""
        self.history.update(step, state)


def normalize_range(arr):
    """Normalize range of values in arr to [0,1]"""
    arr -= arr.min()
    return arr/arr.max()


def get_equilibrium(model, init):
    nvars = len(model.state_variables)
    cvars = len(model.cvar)
    def func(x):
        fx = model.dfun(x.reshape((nvars, 1, 1)),
                        numpy.zeros((cvars, 1, 1)))
        return fx.flatten()
    x = fsolve(func, init)
    return x

def zero_rows(M, rows):
    diag = scipy.sparse.eye(M.shape[0]).tolil()
    for r in rows:
        diag[r, r] = 0
    return diag.dot(M)

def zero_columns(M, columns):
    diag = scipy.sparse.eye(M.shape[1]).tolil()
    for c in columns:
        diag[c, c] = 0
    return M.dot(diag)

class LaplaceKernel(SpatialApplicableEquation, FiniteSupportEquation):
    """
    A Laplace kernel equation.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.
    """

    equation = Final(
        label="Laplace kernel",
        default="amp * (1./(2.*b)) * (exp(-abs(var)/b)) + offset",
        )

    parameters = Attr(
        field_type=dict,
        label="Laplace parameters",
        default=lambda: {"amp": 1.0, "b": 1.0, "offset": 0.0})
    def evaluate(self, data):
        # Make sure to pass parameters as a dictionary to numexpr
        params = self.parameters.copy()
        params["var"] = data  # Add data to the params dict for numexpr
        return numexpr.evaluate(self.equation, local_dict=params)

# overwrite function of local connectivity to take into account the vertex area
class LocalConnectivity_new(local_connectivity.LocalConnectivity):
    def compute(self, vertex_areas=None) :
        self.log.info("Mapping geodesic distance through the LocalConnectivity.")

        # Start with data being geodesic_distance_matrix, then map it through equation
        # Then replace original data with result...
        self.matrix_gdist.data = self.equation.evaluate(self.matrix_gdist.data)

        # scale by vertex areas and skip homogenization part
        if vertex_areas is None:
            area_mtx = scipy.sparse.diags(self.surface.vertex_areas)
        else: 
            area_mtx = scipy.sparse.diags(vertex_areas)
        self.matrix_gdist = self.matrix_gdist * area_mtx
        self.matrix = self.matrix_gdist.tocsr()

# add functionality to compute vertex area to surface
class CorticalSurface_new(surfaces.CorticalSurface) :
    _vertex_areas = None
    _triangle_areas = None
    @property
    def vertex_areas(self):
        """An array specifying the area belonging to the vertices of a surface."""
        if self._vertex_areas is None:
            self._vertex_areas  = self._find_vertex_areas()
        return self._vertex_areas

    def _find_vertex_areas(self):
        """Calculates the area belonging to the vertices of a surface."""
        vertex_areas = numpy.zeros(self.number_of_vertices)
        for i, triangle in enumerate(self.triangles):
            nverts = len(triangle) # This should always be 3 - it is a triangle afterall.
            for j in triangle:
                vertex_areas[j] += self.triangle_areas[i]/nverts
        return vertex_areas


class SpatEpi(ModelNumbaDfun):
    _ui_name = "SpatEpi"
    ui_configurable_parameters = []

    y0 = NArray(
        label="y0",
        default=numpy.array([1]),
        doc="Additive coefficient for the second state variable")

    tau0 = NArray(
        label="tau0",
        default=numpy.array([2857.0]),
        doc="Temporal scaling in the third state variable")

    tau2 = NArray(
        label="tau2",
        default=numpy.array([10.0]),
        doc="Temporal scaling in the fifth state variable")

    x0 = NArray(
        label="x0",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter")

    Iext = NArray(
        label="Iext",
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External inumpyut current to the first population")

    Iext2 = NArray(
        label="Iext2",
        domain=Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="External inumpyut current to the second population")

    gamma = NArray(
        label="gamma",
        default=numpy.array([0.01]),
        doc="Temporal integration scaling"
    )

    gamma11 = NArray(
        label="gamma11",
        default=numpy.array([1.0]),
        doc="Scaling of local connections 1-1"
    )

    gamma22 = NArray(
        label="gamma22",
        default=numpy.array([1.0]),
        doc="Scaling of local connections 2-2"
    )

    gamma12 = NArray(
        label="gamma12",
        default=numpy.array([1.0]),
        doc="Scaling of local connections 1-2"
    )

    gamma_glob = NArray(
        label="gamma_glob",
        default=numpy.array([1.0]),
        doc="Scaling of the global connections"
    )

    theta11 = NArray(
        label="theta11",
        default=numpy.array([-1.1]),
        doc="Firing threshold 1-1"
    )

    theta22 = NArray(
        label="theta22",
        default=numpy.array([-0.5]),
        doc="Firing threshold 2-2"
    )

    theta12 = NArray(
        label="theta12",
        default=numpy.array([-1.1]),
        doc="Firing threshold 1-2"
    )

    tt = NArray(
        label="tt",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system")

    state_variable_range = Final(
        label="State variable ranges [lo, hi]",
        default={"u1": numpy.array([-2., 1.]),
                 "u2": numpy.array([-20., 2.]),
                 "s": numpy.array([2.0, 5.0]),
                 "q1": numpy.array([-2., 0.]),
                 "q2": numpy.array([0., 2.]),
                 "g": numpy.array([-1., 1.])},
        doc="Typical bounds on state variables in the Epileptor model."
        )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=['u1', 'u2', 's', 'q1', 'q2', 'g', 'q1 - u1'],
        default=['q1 - u1', 's'],
        doc="Quantities of the Epileptor available to monitor.",
    )

    state_variables = ['u1', 'u2', 's', 'q1', 'q2', 'g']

    _nvar = 6
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, x, c, local_coupling=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T

        if type(local_coupling) == float:
            loc11 = self.gamma11 * local_coupling * (0.5 * (numpy.sign(x[0, :, 0] - self.theta11) + 1.0))
            loc22 = self.gamma22 * local_coupling * (0.5 * (numpy.sign(x[3, :, 0] - self.theta22) + 1.0))
            loc12 = self.gamma12 * local_coupling * (0.5 * (numpy.sign(x[0, :, 0] - self.theta12) + 1.0))
        else:
            loc11 = self.gamma11 * local_coupling.dot(0.5 * (numpy.sign(x[0, :, 0] - self.theta11) + 1.0))
            loc22 = self.gamma22 * local_coupling.dot(0.5 * (numpy.sign(x[3, :, 0] - self.theta22) + 1.0))
            loc12 = self.gamma12 * local_coupling.dot(0.5 * (numpy.sign(x[0, :, 0] - self.theta12) + 1.0))

        deriv = _numba_dfun(x_, self.gamma_glob * c_,
                            self.x0, self.Iext, self.Iext2,
                            loc11, loc22, loc12,
                            self.tt, self.y0,
                            self.tau0, self.tau2, self.gamma)
        return deriv.T[..., numpy.newaxis]


@guvectorize([(float64[:],) * 14], '(n),(m)' + ',()'*11 + '->(n)', nopython=True, target='cpu')
def _numba_dfun(y, c_pop, x0, Iext, Iext2, loc11, loc22, loc12, tt, y0, tau0, tau2, gamma, ydot):
    "Gufunc for Epileptor model equations."

    # population 1
    if y[0] < 0.0:
        ydot[0] = y[0]**3 - 3 * y[0]**2
    else:
        ydot[0] = (y[3] - 0.6 * (y[2] - 4.0) ** 2) * y[0]

    ydot[0] = tt[0] * (y[1] - ydot[0] - y[2] + Iext[0] + loc11[0] + c_pop[0])
    ydot[1] = tt[0] * (y0[0] - 5*y[0]**2 - y[1])

    # energy
    if y[2] < 0.0:
        ydot[2] = - 0.1 * y[2] ** 7
    else:
        ydot[2] = 0.0
    ydot[2] = tt[0] * (1.0/tau0[0] * (4.0 * (y[0] - x0[0]) - y[2] + ydot[2]))

    # population 2
    ydot[3] = tt[0] * (-y[4] + y[3] - y[3] ** 3 + Iext2[0] + 2 * y[5] - 0.3 * (y[2] - 3.5) + loc22[0])
    if y[3] < -0.25:
        ydot[4] = 0.0
    else:
        ydot[4] = 6.0 * (y[3] + 0.25)
    ydot[4] = tt[0] * ((-y[4] + ydot[4]) / tau2[0])

    # filter
    ydot[5] = tt[0] * (-0.01 * y[5] + 0.003 * y[0] + 0.01 * loc12[0])



class Spat2DEpi(ModelNumbaDfun):
    _ui_name = "Spat2DEpi"
    ui_configurable_parameters = []

    x0 = NArray(
        label="x0",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.3]),
        doc="Epileptogenicity parameter")

    I = NArray(
        label="Iext",
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([1]),
        doc="External inumpyut current to the first population")

    gamma_glob = NArray(
        label="gamma_glob",
        default=numpy.array([1.0]),
        doc="Scaling of the global connections"
    )

    gamma_loc = NArray(
        label="gamma_loc",
        default=numpy.array([1.0]),
        doc="Scaling of the local connections"
    )

    eps = NArray(
        label="eps",
        default=numpy.array([0.01]),
        doc="Firing threshold 1-1"
    )

    theta = NArray(
        label="theta",
        default=numpy.array([0.0]),
        doc="Firing threshold "
    )

    tt = NArray(
        label="tt",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system")

    state_variable_range = Final(
        label="State variable ranges [lo, hi]",
        default={"x": numpy.array([-4., 4.]),
                 "z": numpy.array([-2., 2.]),},
        doc="Typical bounds on state variables in the Epileptor model."
        )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=['x', 'z'],
        default=['x'],
        doc="Quantities of the Epileptor available to monitor.",
    )

    state_variables = ['x', 'z']

    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, x, c, local_coupling=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T

        if type(local_coupling) == float:
            loc = self.gamma_loc * local_coupling * (x[0, :, 0] > self.theta)
        else:
            loc = self.gamma_loc * local_coupling.dot(x[0, :, 0] > self.theta)
        

        deriv = _numba_spat_2depi_dfun(x_, self.gamma_glob * c_,
                            self.x0, self.I, loc, self.tt, self.eps)
        return deriv.T[..., numpy.newaxis]


@guvectorize([(float64[:],) * 8], '(n),(m)' + ',()'*5 + '->(n)', nopython=True, target='cpu')
def _numba_spat_2depi_dfun(y, c_pop, x0, I, loc, tt, eps, ydot):
    "Gufunc for 2D Epileptor model equations."
    ydot[0] = tt[0] * (-y[0]**3 - 2 * y[0]**2 - y[1] + I[0] + loc[0] + c_pop[0])
    ydot[1] = tt[0] * (eps[0] * ( 4 * ( y[0] - x0[0] ) - y[1] ))
    


def remove_isolated_vertices( vertices, triangles, list_of_metrics):
    """
    Remove vertices which are not used in any triangle.
    """
    n_vert = vertices.shape[0]
    used_vert = numpy.zeros((n_vert), dtype=numpy.bool_)
    used_vert[numpy.unique(triangles)] = 1
    keep_vert = numpy.where(used_vert)[0]

    # triangles need to be renumbered
    new_vert_idx = numpy.zeros((n_vert), dtype=numpy.int64) -1
    new_vert_idx[keep_vert] = numpy.arange(len(keep_vert))
    triangles = new_vert_idx[triangles.reshape(-1)].reshape(-1,3)
    tri_mask = numpy.all(triangles!=-1,axis=1)
    list_of_metrics = [metric[keep_vert] for metric in list_of_metrics]
    return vertices[keep_vert], triangles[tri_mask], list_of_metrics

def delete_vertices_from_triangular_mesh(vertices, triangles, mask, list_of_metrics, remove_new_isolated_vertices=True):
    """
    mask is a boolean array with as many entries as number of vertices.
    vertices labeled with a True value in mask will be deleted from the mesh.
    Corresponding triangles will be deleted too and vertices renumbered.
    list_of_metrics contains label or metric arrays with as many entries as number of vertices.
    The entries for deleted vertices will be removed from metrics too. 
    """
    n_vert = vertices.shape[0]
    assert n_vert == len(mask), "Number of vertices mask equal the length of mask."

    # delete all triangles which contain atleast one vertex that needs to be deleted
    keep_vert = numpy.where(numpy.invert(mask))[0]
    new_vert_idx = numpy.zeros((n_vert), dtype=numpy.int64) -1
    new_vert_idx[keep_vert] = numpy.arange(len(keep_vert), dtype=numpy.int64)
    triangles = new_vert_idx[triangles.reshape(-1)].reshape(-1,3)
    tri_mask = numpy.all(triangles!=-1,axis=1)
    triangles = triangles[tri_mask]
    vertices = vertices[keep_vert]
    list_of_metrics = [metric[keep_vert] for metric in list_of_metrics]

    # triangle deletion can cause some vertices which are not selected for deletion to be isolated
    if remove_new_isolated_vertices:
        vertices, triangles, list_of_metrics = remove_isolated_vertices(vertices, triangles, list_of_metrics)

    return vertices, triangles, list_of_metrics


def read_surf_from_gii(surf_gii_path):
    tri = nib.load(surf_gii_path).get_arrays_from_intent(1009)[0].data.astype(numpy.int32)
    vert = nib.load(surf_gii_path).get_arrays_from_intent(1008)[0].data
    return vert, tri

def convert_to_pyvista_mesh(vertices, triangles):
    """
    Take triangles and vertices as [nx3] and [mx3] arrays.
    To triangles add one row with number 3, to indicate for pyvista that face is a triangle
    Return a pyvista mesh
    """
    faces_ = numpy.ones((triangles.shape[0],triangles.shape[1]+1),dtype=int)
    faces_[:,:1] *= 3
    faces_[:,1:]  = triangles
    return pv.PolyData(vertices,faces_.flatten())

@numba.njit
def dilate_boolean_label_on_surface(vertices, triangles, label, n_iter=1):
    """
    label is a boolean array with as many entries as number of vertices.
    Run n_iter dilation operation, in which each vertex is relabeled True if it
    is connected to another vertex labeled True through an edge of a triangle.  
    """
    assert vertices.shape[0] == len(label), "Number of vertices mask equal the length of label."
    new_label = label.copy()
    for _ in range(n_iter):
        for tri in triangles:
            # only relabel vertices in triangles with 1 or 2 True labels
            # cause they indicate a connection between a True and a False labeled vertex
            if numpy.sum(label[tri]) in [1,2]:
                for v in tri:
                    new_label[v] = True
        label = new_label.copy()

    return new_label


def create_animation_with_data_idex(py_vista_mesh, data, data_index, save_fname, cpos=None, time=numpy.array([0])):
    """
    Create movie of surface simulation. 
    Slight addition to create_animation() for the case that there are subcortical areas to be displayed. 
    A subcortical vertices might be displayed as a sphere, with multiple vertices.
    data_index should take care of this. It points for each vertex (cortical surface vertex or subcortical spherical mesh vertices),
    to the place in the data array that should be taken. 


    vertices, triangles - numpy.array
    data - numpy.array [n_vertices, n_timepoints]
    data_index - numpy.array [n_vertices of mesh]
    save_fname - file name to save the movie
    cpos - pyvista camera position
    """

    p = pv.Plotter()#window_size=[1024, 768])
    p.set_background("white")
    p.open_movie(save_fname, framerate=24)
    if not cpos is None:
        p.camera_position = cpos
    kwargs = {"scalars" : 'values', 
            "cmap" : "viridis", 
            "clim" : [data.min(), data.max()],
            "smooth_shading" : True, 
            "interpolate_before_map" : False, 
            "opacity" : 1, 
            "show_scalar_bar" : False,
            "below_color" : "grey"
            }
    
    py_vista_mesh["values"] = data[data_index,0]
    p.add_mesh(py_vista_mesh, **kwargs)
    text_actor = p.add_text(f"{time[0]:.1f} ms", color="black", position="upper_edge")
    p.show( interactive=False, auto_close=False)
    p.write_frame()

    for i in range(1,data.shape[1]):
        py_vista_mesh["values"] = data[data_index,i]
        text_actor.SetText(7,f"{time[i]:.1f} ms")
        p.write_frame()
    p.close()


def create_animation_with_data_idex_subcort(py_vista_mesh,py_vista_mesh_sub, data,data_sub, data_index, save_fname, cpos=None, time_sim=numpy.array([0])):
    """
    Create movie of surface simulation. 
    Slight addition to create_animation() for the case that there are subcortical areas to be displayed. 
    A subcortical vertices might be displayed as a sphere, with multiple vertices.
    data_index should take care of this. It points for each vertex (cortical surface vertex or subcortical spherical mesh vertices),
    to the place in the data array that should be taken. 


    vertices, triangles - np.array
    data - np.array [n_vertices, n_timepoints]
    data_index - np.array [n_vertices of mesh]
    save_fname - file name to save the movie
    cpos - pyvista camera position
    """

    p = pv.Plotter()#window_size=[1024, 768])
    p.set_background("white")
    p.open_movie(save_fname, framerate=24)
    if not cpos is None:
        p.camera_position = cpos
    kwargs = {"scalars" : 'values', 
            "cmap" : "viridis", 
            "clim" : [data.min(), data.max()],
            "smooth_shading" : True, 
            "interpolate_before_map" : False, 
            "opacity" : 1, 
            "show_scalar_bar" : False,
            "below_color" : "grey"
            }
    
    py_vista_mesh["values"] = data[:,0]
    py_vista_mesh_sub["values"] = data_sub[:,0]
    p.add_mesh(py_vista_mesh, **kwargs)
    p.add_mesh(py_vista_mesh_sub, **kwargs)
    text_actor = p.add_text(f"{time_sim[0]:.1f} ms", color="black", position="upper_edge")
    p.show( interactive=False, auto_close=False)
    p.write_frame()

    for i in range(1,data.shape[1]):
        py_vista_mesh["values"] = data[:,i]
        py_vista_mesh_sub["values"] = data_sub[:,i]
        text_actor.SetText(7,f"{time_sim[i]:.1f} ms")
        p.write_frame()
    p.close()



class Heaviside(Coupling):
    """
    Implement heaviside step coupling function H(x).
    0 if x < 0
    0.5 if x == 0
    1 if x > 0
    """
    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.1,]),
        domain=Range(lo=0.0, hi=10., step=0.1),
        doc="Rescales the connection strength.",)

    theta = NArray(
            label=":math:`theta`",
            default=numpy.array([-1]),
            domain=Range(lo=-5., hi=5., step=0.1),
            doc="Threshold of the heaviside step function.",)

    def pre(self, x_i, x_j):
        return numpy.heaviside(x_j - self.theta, 0.5)

    def post(self, gx):
        return self.a * gx


class MontbrioPazoRoxinSparse(models.MontbrioPazoRoxin ):
    nest_node = NArray(
        dtype=int,
        label="nest_node",
        default=numpy.array([1], dtype=int),
        doc="Node simulated by NEST"
    )

    nest_region = NArray(
        dtype=int,
        label="nest_region",
        default=numpy.array([0], dtype=int),
        doc="Region simulated by NEST"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)