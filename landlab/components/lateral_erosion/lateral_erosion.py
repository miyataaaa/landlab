# -*- coding: utf-8 -*-
"""Grid-based simulation of lateral erosion by channels in a drainage network.

ALangston
"""

import numpy as np
import sys

from landlab import Component, RasterModelGrid
from landlab.components.flow_accum import FlowAccumulator

from .node_finder import node_finder
from .cfuncs import node_finder_use_fivebyfive_window_ver2, node_finder_use_fivebyfive_window_only_hill
from .cfuncs import _run_one_step_fivebyfive_window_ver2, _run_one_step_fivebyfive_window_only_hill

# Hard coded constants
cfl_cond = 0.3  # CFL timestep condition
wid_coeff = 0.4  # coefficient for calculating channel width
wid_exp = 0.35  # exponent for calculating channel width
# dp_coeff = 1.2 # 
# dp_exp = 0

class LateralEroder(Component):
    """Laterally erode neighbor node through fluvial erosion.

    Landlab component that finds a neighbor node to laterally erode and
    calculates lateral erosion.
    See the publication:

    Langston, A.L., Tucker, G.T.: Developing and exploring a theory for the
    lateral erosion of bedrock channels for use in landscape evolution models.
    Earth Surface Dynamics, 6, 1-27,
    `https://doi.org/10.5194/esurf-6-1-2018 <https://www.earth-surf-dynam.net/6/1/2018/>`_

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components import FlowAccumulator, LateralEroder
    >>> np.random.seed(2010)

    Define grid and initial topography

    * 5x4 grid with baselevel in the lower left corner
    * All other boundary nodes closed
    * Initial topography is plane tilted up to the upper right with noise

    >>> mg = RasterModelGrid((5, 4), xy_spacing=10.0)
    >>> mg.set_status_at_node_on_edges(
    ...     right=mg.BC_NODE_IS_CLOSED,
    ...     top=mg.BC_NODE_IS_CLOSED,
    ...     left=mg.BC_NODE_IS_CLOSED,
    ...     bottom=mg.BC_NODE_IS_CLOSED,
    ... )
    >>> mg.status_at_node[1] = mg.BC_NODE_IS_FIXED_VALUE
    >>> mg.add_zeros("topographic__elevation", at="node")
    array([ 0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.])
    >>> rand_noise=np.array(
    ...     [
    ...         0.00436992,  0.03225985,  0.03107455,  0.00461312,
    ...         0.03771756,  0.02491226,  0.09613959,  0.07792969,
    ...         0.08707156,  0.03080568,  0.01242658,  0.08827382,
    ...         0.04475065,  0.07391732,  0.08221057,  0.02909259,
    ...         0.03499337,  0.09423741,  0.01883171,  0.09967794,
    ...     ]
    ... )
    >>> mg.at_node["topographic__elevation"] += (
    ...     mg.node_y / 10. + mg.node_x / 10. + rand_noise
    ... )
    >>> U = 0.001
    >>> dt = 100

    Instantiate flow accumulation and lateral eroder and run each for one step

    >>> fa = FlowAccumulator(
    ...     mg,
    ...     surface="topographic__elevation",
    ...     flow_director="FlowDirectorD8",
    ...     runoff_rate=None,
    ...     depression_finder=None,
    ... )
    >>> latero = LateralEroder(mg, latero_mech="UC", Kv=0.001, Kl_ratio=1.5)

    Run one step of flow accumulation and lateral erosion to get the dzlat array
    needed for the next part of the test.

    >>> fa.run_one_step()
    >>> mg, dzlat = latero.run_one_step(dt)

    Evolve the landscape until the first occurence of lateral erosion. Save arrays
    volume of lateral erosion and topographic elevation before and after the first
    occurence of lateral erosion

    >>> while min(dzlat) == 0.0:
    ...     oldlatvol = mg.at_node["volume__lateral_erosion"].copy()
    ...     oldelev = mg.at_node["topographic__elevation"].copy()
    ...     fa.run_one_step()
    ...     mg, dzlat = latero.run_one_step(dt)
    ...     newlatvol = mg.at_node["volume__lateral_erosion"]
    ...     newelev = mg.at_node["topographic__elevation"]
    ...     mg.at_node["topographic__elevation"][mg.core_nodes] += U * dt

    Before lateral erosion occurs, *volume__lateral_erosion* has values at
    nodes 6 and 10.

    >>> np.around(oldlatvol, decimals=0)
    array([  0.,   0.,   0.,   0.,
             0.,   0.,  79.,   0.,
             0.,   0.,  24.,   0.,
             0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.])


    After lateral erosion occurs at node 6, *volume__lateral_erosion* is reset to 0

    >>> np.around(newlatvol, decimals=0)
    array([  0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,
             0.,   0.,  24.,   0.,
             0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.])


    After lateral erosion at node 6, elevation at node 6 is reduced by -1.41
    (the elevation change stored in dzlat[6]). It is also provided as the
    at-node grid field *lateral_erosion__depth_increment*.

    >>> np.around(oldelev, decimals=2)
    array([ 0.  ,  1.03,  2.03,  3.  ,
            1.04,  1.77,  2.45,  4.08,
            2.09,  2.65,  3.18,  5.09,
            3.04,  3.65,  4.07,  6.03,
            4.03,  5.09,  6.02,  7.1 ])

    >>> np.around(newelev, decimals=2)
    array([ 0.  ,  1.03,  2.03,  3.  ,
            1.04,  1.77,  1.03,  4.08,
            2.09,  2.65,  3.18,  5.09,
            3.04,  3.65,  4.07,  6.03,
            4.03,  5.09,  6.02,  7.1 ])

    >>> np.around(dzlat, decimals=2)
    array([ 0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  , -1.41,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0. ])

    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    Langston, A., Tucker, G. (2018). Developing and exploring a theory for the
    lateral erosion of bedrock channels for use in landscape evolution models.
    Earth Surface Dynamics  6(1), 1--27.
    https://dx.doi.org/10.5194/esurf-6-1-2018

    **Additional References**

    None Listed

    """

    _name = "LateralEroder"

    _unit_agnostic = False

    _cite_as = """
    @article{langston2018developing,
      author = {Langston, A. L. and Tucker, G. E.},
      title = {{Developing and exploring a theory for the lateral erosion of
      bedrock channels for use in landscape evolution models}},
      doi = {10.5194/esurf-6-1-2018},
      pages = {1---27},
      number = {1},
      volume = {6},
      journal = {Earth Surface Dynamics},
      year = {2018}
    }
    """
    _info = {
        "drainage_area": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m**2",
            "mapping": "node",
            "doc": "Upstream accumulated surface area contributing to the node's discharge",
        },
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "flow__upstream_node_order": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array containing downstream-to-upstream ordered list of node IDs",
        },
        "lateral_erosion__depth_increment": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Change in elevation at each node from lateral erosion during time step",
        },
        "sediment__influx": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/y",
            "mapping": "node",
            "doc": "Sediment flux (volume per unit time of sediment entering each node)",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
        "volume__lateral_erosion": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3",
            "mapping": "node",
            "doc": "Array tracking volume eroded at each node from lateral erosion",
        },
        "deposition__rate": { # 2022/09/03 added
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/y",
            "mapping": "node",
            "doc": "deposition rate at node",
        },
    }

    def __init__(
        self,
        grid,
        latero_mech="UC",
        alph=0.8,
        Kv=0.001,
        Kl_ratio=1.0,
        solver="langston",
        node_finder="langston",
        use_Q = False,
        inlet_on=False,
        inlet_node=None,
        inlet_area=None,
        qsinlet=0.0,
        flow_accumulator=None,
        dp_coef = 0.4, # 
        dp_exp = 0.35,
        wid_coef = 10,
        F = 0.02,
        thresh_da = 0,
        phase_delay_node_num = 0,
        fai_alpha = 3.3,
        fai_beta = -0.25,
        fai_gamma = -0.85,
        fai_C = -64,
        add_min_Q_or_da = 0.0,
        critical_erosion_volume_ratio = 1.0,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A Landlab square cell raster grid object
        latero_mech : string, optional (defaults to UC)
            Lateral erosion algorithm, choices are "UC" for undercutting-slump
            model and "TB" for total block erosion
        alph : float, optional (defaults to 0.8)
            Parameter describing potential for deposition, dimensionless
        Kv : float, node array, or field name
            Bedrock erodibility in vertical direction, 1/years
        Kl_ratio : float, optional (defaults to 1.0)
            Ratio of lateral to vertical bedrock erodibility, dimensionless
        solver : string
            Solver options:
                (1) 'langston' (default): explicit forward-time extrapolation.
                    Simple but will become unstable if time step is too large or
                    if bedrock erodibility is vry high.
                (2) 'ULE': Lateral erosion calculations using regression equations obtained from ULEmodel.
        node_finder: string
            node_finder options:
                (1) 'langston' (default): 
                    it is use in Langston & Tucker (2018) and defined in node_finder.py.
                (2) 'ffwindow': 
                    Determine the lateral erosion cell after defining the virtual flow path with a 5×5 local window.
                (3) 'ffwindow_only_hill':
                    Determine the lateral erosion cell after defining the virtual flow path with a 5×5 local window and choose only hillslope nodes.
        use_Q : bool, optional (defaults to False)
            If True, use the discharge to calculate erosion rate. If False, use the drainage area.
        inlet_node : integer, optional
            Node location of inlet (source of water and sediment)
        inlet_area : float, optional
            Drainage area at inlet node, must be specified if inlet node is "on", m^2
        qsinlet : float, optional
            Sediment flux supplied at inlet, optional. m3/year
        flow_accumulator : Instantiated Landlab FlowAccumulator, optional
            When solver is set to "adaptive", then a valid Landlab FlowAccumulator
            must be passed. It will be run within sub-timesteps in order to update
            the flow directions and drainage area.
        dp_coef : float, optional (defaults to 0.4)
            Coefficient for calculating water depth, dimensionless
        dp_exp : float, optional (defaults to 0.35)
            Exponent for calculating water depth, dimensionless
        wid_coef : float, optional (defaults to 10)
            Coefficient for calculating channel width, dimensionless
        F : float, optional (defaults to 0.02)
            Coefficient for calculating runoff, dimensionless
        thresh_da : float, optional (defaults to 0)
            Threshold of drainage area, m^2 if use_Q is False, m^3/s if use_Q is True
        phase_delay_node_num : int, optional (defaults to 0)
            Number of nodes to consider for phase delay curvature. this parameter is used only when solver is "delay_rs".
        fai_alpha : float, optional (defaults to 3.3)
            Parameter for calculating lateral/versonal erosion rate, dimensionless. 
            this parameter is used only when solver is "delay_rs". 
        fai_beta : float, optional (defaults to -0.25)
            Parameter for calculating lateral/versonal erosion rate, dimensionless. 
            this parameter is used only when solver is "delay_rs".
        fai_gamma : float, optional (defaults to -0.85)
            Parameter for calculating lateral/versonal erosion rate, dimensionless. 
            this parameter is used only when solver is "delay_rs".
        fai_C : float, optional (defaults to -64)
            Parameter for calculating lateral/versonal erosion rate, dimensionless. 
            this parameter is used only when solver is "delay_rs".
        add_min_Q_or_da : float, optional (defaults to 0)
            Add this value to drainage area or discharge. this parameter is used only when solver is "delay_rs".
        critical_erosion_volume_ratio : float, optional (defaults to 1.0)   
            If the volume of lateral erosion is larger than the critical_erosion_volume_ratio * volume of lateral node,
            lateral erosion occurs. if critical_erosion_volume_ratio is 1.0, lateral erosion occurs when the volume of 
            lateral erosion is larger than the volume of lateral node. it is equivalent to the original algorithm (UC&TB).
            if critical_erosion_volume_ratio is less than 1.0, lateral erosion ocuur earlier than the original algorithm.
        """
        super().__init__(grid)

        assert isinstance(
            grid, RasterModelGrid
        ), "LateralEroder requires a sqare raster grid."

        if "flow__receiver_node" in grid.at_node:
            if grid.at_node["flow__receiver_node"].size != grid.size("node"):
                msg = (
                    "A route-to-multiple flow director has been "
                    "run on this grid. The LateralEroder is not currently "
                    "compatible with route-to-multiple methods. Use a route-to-"
                    "one flow director."
                )
                raise NotImplementedError(msg)

        solver_list = ("langston", 
                       "ULE")
        
        if solver not in solver_list:
            raise ValueError(
                "value for solver not understood ({val} not one of {valid})".format(
                    val=solver, valid=", ".join(solver_list)
                )
            )
        
        node_finder_list = ("langston", 
                            "ffwindow",
                            "ffwindow_only_hill")
        
        if node_finder not in node_finder_list:
            raise ValueError(
                "value for node_finder not understood ({val} not one of {valid})".format(
                    val=node_finder, valid=", ".join(node_finder_list)
                )
            )

        if latero_mech not in ("UC", "TB"):
            raise ValueError(
                "value for latero_mech not understood ({val} not one of {valid})".format(
                    val=latero_mech, valid=", ".join(("UC", "TB"))
                )
            )

        if inlet_on and (inlet_node is None or inlet_area is None):
            raise ValueError(
                "inlet_on is True, but no inlet_node or inlet_area is provided."
            )

        if Kv is None:
            raise ValueError(
                "Kv must be set as a float, node array, or field name. It was None."
            )

        # Create fields needed for this component if not already existing
        if "volume__lateral_erosion" in grid.at_node:
            self._vol_lat = grid.at_node["volume__lateral_erosion"]
        else:
            self._vol_lat = grid.add_zeros("volume__lateral_erosion", at="node")

        if "sediment__influx" in grid.at_node:
            self._qs_in = grid.at_node["sediment__influx"]
        else:
            self._qs_in = grid.add_zeros("sediment__influx", at="node")

        if "lateral_erosion__depth_increment" in grid.at_node:
            self._dzlat = grid.at_node["lateral_erosion__depth_increment"]
        else:
            self._dzlat = grid.add_zeros("lateral_erosion__depth_increment", at="node")

        # 2022/09/30 add new 
        if "deposition__rate" in grid.at_node:
            self._deprate = grid.at_node["deposition__rate"]
        else:
            self._deprate = grid.add_zeros("deposition__rate", at="node")

        # for backward compatibility (remove in version 3.0.0+)
        grid.at_node["sediment__flux"] = grid.at_node["sediment__influx"]


        # 曲率を保存するためのフィールドを追加(2023/07/14)
        if "curvature" in grid.at_node:
            self._curvature = grid.at_node["curvature"]
        else:
            self._curvature = grid.add_zeros("curvature", at="node")

        # 位相遅れ曲率(phase_delay_curvature)を保存するためのフィールドを追加(2023/07/14)
        # phd_curvature = phase_delay_curvature
        if "phd_curvature" in grid.at_node:
            self._phd_curvature = grid.at_node["phd_curvature"]
        else:
            self._phd_curvature = grid.add_zeros("phd_curvature", at="node")

        # add "flow depth" field
        if "flow_depth" in grid.at_node:
            self._flow_depth = grid.at_node["flow_depth"]
        else:   
            self._flow_depth = grid.add_zeros("flow_depth", at="node")

        # 何回側方侵食を起こしたかを記録する配列を追加(2023/11/17)
        if "latero_nums" in grid.at_node:
            self._latero_nums = grid.at_node["latero_nums"].astype(np.int32)
        else:
            self._latero_nums = grid.add_zeros("latero_nums", at="node", dtype=np.int32)


        # you can specify the type of lateral erosion model you want to use.
        # But if you don't the default is the undercutting-slump model
        if latero_mech == "TB":
            self._TB = True
            self._UC = False
        else:
            self._UC = True
            self._TB = False
        # option use adaptive time stepping. Default is fixed dt supplied by user
        if solver == "langston" and node_finder == "langston":
            self.run_one_step = self.run_one_step_langston_langston
        elif solver == "langston" and node_finder == "ffwindow":
            self.run_one_step = self.run_one_step_langston_ffwindow
        elif solver == "langston" and node_finder == "ffwindow_only_hill":
            self.run_one_step = self.run_one_step_langston_ffwindow_only_hill
        elif solver == "ULE" and node_finder == "langston":
            self.run_one_step = self.run_one_step_ULE_langston
        elif solver == "ULE" and node_finder == "ffwindow":
            self.run_one_step = self.run_one_step_ULE_ffwindow
        elif solver == "ULE" and node_finder == "ffwindow_only_hill":
            self.run_one_step = self.run_one_step_ULE_ffwindow_only_hill

        self._alph = alph
        self._Kv = Kv  # can be overwritten with spatially variable
        self._Klr = float(Kl_ratio)  # default ratio of Kv/Kl is 1. Can be overwritten

        self._dzdt = grid.add_zeros(
            "dzdt", at="node", clobber=True
        )  # elevation change rate (M/Y)
        # optional inputs
        self._inlet_on = inlet_on
        if inlet_on:
            self._inlet_node = inlet_node
            self._inlet_area = inlet_area
            # commetn out, 2022/10/19
            # # runoff is an array with values of the area of each node (dx**2)
            # runoffinlet = np.full(grid.number_of_nodes, grid.dx**2, dtype=float)
            # # Change the runoff at the inlet node to node area + inlet node
            # runoffinlet[inlet_node] += inlet_area
            # grid.add_field("water__unit_flux_in", runoffinlet, at="node", clobber=True)
            # # set qsinlet at inlet node. This doesn't have to be provided, defaults
            # # to 0.
            self._qsinlet = qsinlet
            self._qs_in[self._inlet_node] = self._qsinlet

        # handling Kv for floats (inwhich case it populates an array N_nodes long) or
        # for arrays of Kv. Checks that length of Kv array is good.
        self._Kv = np.ones(self._grid.number_of_nodes, dtype=float) * Kv

        # add water depth coef and power exp（2022/07/06)
        self._dp_coef = dp_coef
        self._dp_exp = dp_exp
        self._wid_coef = wid_coef
        self._F = F

        # add threshold of drainage area
        self._thresh_da = thresh_da

        # add phase delay node number
        # 位相遅れを考慮するノード数
        if phase_delay_node_num > 1:
            # とりあえず実験的に１つ上流側まで位相をずらせるようにする
            # セルサイズに依存する仕組みなので、今後改良する必要あり
            # langston et al 2018での曲率計算をそのまま使って、位相遅れの影響を考慮する。
            raise ValueError("phase_delay_node_num must be 1 or 0.")
        self._phase_delay_node_num = phase_delay_node_num

        # add use_Q
        self._use_Q = use_Q # Stream Power Modelの計算において、流量を使うかどうか。使わない場合は流域面積が使われる。

        # add fai_alpha, fai_beta, fai_gamma, fai_C
        self._fai_alpha = fai_alpha
        self._fai_beta = fai_beta
        self._fai_gamma = fai_gamma
        self._fai_C = fai_C

        # add min_Q_or_da
        self._add_min_Q_or_da = add_min_Q_or_da

        # add critical_erosion_volume_ratio
        self._critical_erosion_volume_ratio = critical_erosion_volume_ratio

    def run_one_step_langston_langston(self, dt=1.0):
        """Calculate vertical and lateral erosion for a time period 'dt'.

        Parameters
        ----------
        dt : float
            Model timestep [T]

        Note
        ----------
        側方侵食アルゴリズム
            Langston & Tucker (2018)
        ノード選定アルゴリズム
            Langston & Tucker (2018)
        """
        Klr = self._Klr
        grid = self._grid
        UC = self._UC
        TB = self._TB
        inlet_on = self._inlet_on  # this is a true/false flag
        Kv = self._Kv
        qs_in = self._qs_in
        dzdt = self._dzdt
        alph = self._alph
        vol_lat = grid.at_node["volume__lateral_erosion"]
        
        # kw = 10.0
        # F = 0.02
        dp_coef = self._dp_coef
        dp_exp = self._dp_exp
        kw = self._wid_coef
        F = self._F
        thresh_da = self._thresh_da

        # May 2, runoff calculated below (in m/s) is important for calculating
        # 2022/07/07 unit of runoff is not m/s, maybe m/yr
        # discharge and water depth correctly. renamed runoffms to prevent
        # confusion with other uses of runoff
        runoffms = (Klr * F / kw) ** 2
        # Kl is calculated from ratio of lateral to vertical K parameters
        Kl = Kv * Klr
        z = grid.at_node["topographic__elevation"]
        # clear qsin for next loop
        qs_in = grid.add_zeros("sediment__influx", at="node", clobber=True)
        qs = grid.add_zeros("qs", at="node", clobber=True)
        lat_nodes = np.zeros(grid.number_of_nodes, dtype=int)
        dzver = np.zeros(grid.number_of_nodes)
        El = grid.add_zeros("latero__rate", at="node", clobber=True) 
        El = grid.at_node["latero__rate"] 
        fai = grid.add_zeros("fai", at="node", clobber=True)
        fai = grid.at_node["fai"]
        vol_lat_dt = np.zeros(grid.number_of_nodes)

        latero_nums = self._latero_nums
        latero_nums[:] = 0 # 何回側方侵食を起こしたかを記録する配列を初期化(2023/11/17)

        # dz_lat needs to be reset. Otherwise, once a lateral node erode's once, it will continue eroding
        # at every subsequent time setp. If you want to track all lateral erosion, create another attribute,
        # or add self.dzlat to itself after each time step.
        self._dzlat.fill(0.0)

        # critical_erosion_volume_ratio 
        critical_erosion_volume_ratio = self._critical_erosion_volume_ratio

        if inlet_on is True:
            inlet_node = self._inlet_node
            qsinlet = self._qsinlet
            qs_in[inlet_node] = qsinlet
            q = grid.at_node["surface_water__discharge"]
            # da = q / grid.dx**2
            da = q / runoffms # change, 2022/10/19
        # if inlet flag is not on, proceed as normal.
        else:
            if self._use_Q:
                # water discharge is calculated by flow router
                da = grid.at_node["surface_water__discharge"]
            else:
                # drainage area is calculated by flow router
                da = grid.at_node["drainage_area"]
        
        # add min_Q_or_da
        da += self._add_min_Q_or_da

        # water depth in meters, needed for lateral erosion calc
        dp = grid.at_node["flow_depth"]
        dp[:] = dp_coef * (da ** dp_exp)

        # flow__upstream_node_order is node array contianing downstream to
        # upstream order list of node ids
        # s contein ids
        s = grid.at_node["flow__upstream_node_order"]
        max_slopes = grid.at_node["topographic__steepest_slope"]
        flowdirs = grid.at_node["flow__receiver_node"]

        # make a list l, where node status is interior (signified by label 0) in s
        # make threshold mask, because apply equation only river. (2022/10/26)
        interior_mask = np.where(np.logical_and(grid.status_at_node == 0, da >= thresh_da))[0]
        interior_s = np.intersect1d(s, interior_mask)
        dwnst_nodes = interior_s.copy()
        # reverse list so we go from upstream to down stream
        dwnst_nodes = dwnst_nodes[::-1]
        max_slopes[:] = max_slopes.clip(0)
        for i in dwnst_nodes:
            # calc deposition and erosion
            dep = alph * qs_in[i] / da[i]
            ero = -Kv[i] * da[i] ** (0.5) * max_slopes[i]
            dzver[i] = dep + ero
            # potential lateral erosion initially set to 0
            petlat = 0.0

            # Choose lateral node for node i. If node i flows downstream, continue.
            # if node i is the first cell at the top of the drainage network, don't go
            # into this loop because in this case, node i won't have a "donor" node
            if i in flowdirs:
                # node_finder picks the lateral node to erode based on angle
                # between segments between three nodes
                [lat_node, inv_rad_curv] = node_finder(grid, i, flowdirs, da)
                # node_finder returns the lateral node ID and the radius of curvature
                lat_nodes[i] = lat_node
                
                # # latero_nums[i] = -1 # 河川ノードは侵食が起きない前提で-1を代入(2023/11/17)

                # if the lateral node is not 0 or -1 continue. lateral node may be
                # 0 or -1 if a boundary node was chosen as a lateral node. then
                # radius of curavature is also 0 so there is no lateral erosion
                if lat_node > 0:
                    # if the elevation of the lateral node is higher than primary node,
                    # calculate a new potential lateral erosion (L/T), which is negative
                    if z[lat_node] > z[i]:
                        petlat = -Kl[i] * da[i] * max_slopes[i] * inv_rad_curv
                        El[i] = petlat
                        fai[i] = petlat/ero
                        # the calculated potential lateral erosion is mutiplied by the length of the node
                        # and the bank height, then added to an array, vol_lat_dt, for volume eroded
                        # laterally  *per timestep* at each node. This vol_lat_dt is reset to zero for
                        # each timestep loop. vol_lat_dt is added to itself in case more than one primary
                        # nodes are laterally eroding this lat_node
                        # volume of lateral erosion per timestep
                        vol_lat_dt[lat_node] += abs(petlat) * grid.dx * dp[i]
                        latero_nums[lat_node] += 1 # 何回側方侵食を起こしたかを記録する配列を更新(2023/11/17)
                        # wd? may be H is true. how calc H ? 

            # send sediment downstream. sediment eroded from vertical incision
            # and lateral erosion is sent downstream
            #            print("debug before 406")
            qs_in[flowdirs[i]] += (
                qs_in[i] - (dzver[i] * grid.dx**2) - (petlat * grid.dx * dp[i])
            )  # qsin to next node
            # print(f"qs_in[i] = {qs_in[i]:.2f}, qs_in[flowdirs[i]] = {qs_in[flowdirs[i]]:.2f}, i = {i}, flowdirs[i] = {flowdirs[i]}")
        qs[:] = qs_in - (dzver * grid.dx**2)
        dzdt[:] = dzver * dt
        vol_lat[:] += vol_lat_dt * dt
        # this loop determines if enough lateral erosion has happened to change
        # the height of the neighbor node.
        for i in dwnst_nodes:
            lat_node = lat_nodes[i]
    
            if lat_node > 0:  # greater than zero now bc inactive neighbors are value -1
                if z[lat_node] > z[i]:
                    # vol_diff is the volume that must be eroded from lat_node so that its
                    # elevation is the same as node downstream of primary node
                    # UC model: this would represent undercutting (the water height at
                    # node i), slumping, and instant removal.
                    if UC == 1:
                        voldiff = critical_erosion_volume_ratio * (z[i] + dp[i] - z[flowdirs[i]]) * grid.dx**2 
                    # TB model: entire lat node must be eroded before lateral erosion
                    # occurs
                    if TB == 1:
                        voldiff = critical_erosion_volume_ratio * (z[lat_node] - z[flowdirs[i]]) * grid.dx**2
                    # if the total volume eroded from lat_node is greater than the volume
                    # needed to be removed to make node equal elevation,
                    # then instantaneously remove this height from lat node. already has
                    # timestep in it
                    if vol_lat[lat_node] >= voldiff:
                        self._dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                        # after the lateral node is eroded, reset its volume eroded to
                        # zero
                        vol_lat[lat_node] = 0.0
        # combine vertical and lateral erosion.
        dz = dzdt + self._dzlat
        # change height of landscape
        z[:] += dz
        return grid, self._dzlat

    def run_one_step_langston_ffwindow(self, dt=1.0):
        """Calculate vertical and lateral erosion for a time period 'dt'.

        Parameters
        ----------
        dt : float
            Model timestep [T]

        Note
        ----------
        側方侵食アルゴリズム
            Langston & Tucker (2018)
        ノード選定アルゴリズム
            Langston & Tucker (2018)
        """
        Klr = self._Klr
        grid = self._grid
        UC = self._UC
        TB = self._TB
        inlet_on = self._inlet_on  # this is a true/false flag
        Kv = self._Kv
        qs_in = self._qs_in
        dzdt = self._dzdt
        alph = self._alph
        vol_lat = grid.at_node["volume__lateral_erosion"]
        
        # kw = 10.0
        # F = 0.02
        dp_coef = self._dp_coef
        dp_exp = self._dp_exp
        kw = self._wid_coef
        F = self._F
        thresh_da = self._thresh_da

        # May 2, runoff calculated below (in m/s) is important for calculating
        # 2022/07/07 unit of runoff is not m/s, maybe m/yr
        # discharge and water depth correctly. renamed runoffms to prevent
        # confusion with other uses of runoff
        runoffms = (Klr * F / kw) ** 2
        # Kl is calculated from ratio of lateral to vertical K parameters
        Kl = Kv * Klr
        z = grid.at_node["topographic__elevation"]
        # clear qsin for next loop
        qs_in = grid.add_zeros("sediment__influx", at="node", clobber=True)
        qs = grid.add_zeros("qs", at="node", clobber=True)
        dzver = np.zeros(grid.number_of_nodes)
        El = grid.add_zeros("latero__rate", at="node", clobber=True) 
        El = grid.at_node["latero__rate"] 
        fai = grid.add_zeros("fai", at="node", clobber=True)
        fai = grid.at_node["fai"]
        vol_lat_dt = np.zeros(grid.number_of_nodes)
        cur = grid.at_node["curvature"]
        phd_cur = grid.at_node["phd_curvature"]

        latero_nums = self._latero_nums
        latero_nums[:] = 0 # 何回側方侵食を起こしたかを記録する配列を初期化(2023/11/17)

        # dz_lat needs to be reset. Otherwise, once a lateral node erode's once, it will continue eroding
        # at every subsequent time setp. If you want to track all lateral erosion, create another attribute,
        # or add self.dzlat to itself after each time step.
        self._dzlat.fill(0.0)

        # critical_erosion_volume_ratio 
        critical_erosion_volume_ratio = self._critical_erosion_volume_ratio

        if inlet_on is True:
            inlet_node = self._inlet_node
            qsinlet = self._qsinlet
            qs_in[inlet_node] = qsinlet
            q = grid.at_node["surface_water__discharge"]
            # da = q / grid.dx**2
            da = q / runoffms # change, 2022/10/19
        # if inlet flag is not on, proceed as normal.
        else:
            if self._use_Q:
                # water discharge is calculated by flow router
                da = grid.at_node["surface_water__discharge"]
            else:
                # drainage area is calculated by flow router
                da = grid.at_node["drainage_area"]
        
        # add min_Q_or_da
        da += self._add_min_Q_or_da

        # water depth in meters, needed for lateral erosion calc
        dp = grid.at_node["flow_depth"]
        dp[:] = dp_coef * (da ** dp_exp)

        # flow__upstream_node_order is node array contianing downstream to
        # upstream order list of node ids
        # s contein ids
        s = grid.at_node["flow__upstream_node_order"]
        max_slopes = grid.at_node["topographic__steepest_slope"]
        flowdirs = grid.at_node["flow__receiver_node"]

        # make a list l, where node status is interior (signified by label 0) in s
        # make threshold mask, because apply equation only river. (2022/10/26)
        interior_mask = np.where(np.logical_and(grid.status_at_node == 0, da >= thresh_da))[0]
        interior_s = np.intersect1d(s, interior_mask)
        dwnst_nodes = interior_s.copy()
        # reverse list so we go from upstream to down stream
        dwnst_nodes = dwnst_nodes[::-1]
        max_slopes[:] = max_slopes.clip(0)
        iterNum = len(dwnst_nodes)
        dummy_value = -99

        nodeNum = grid.shape[0]*grid.shape[1]
        lat_nodes = np.full(shape=(nodeNum, 4), fill_value=dummy_value) # node iの上下４つのノードが入るサイズ
        i = 0

        for j in range(iterNum):
            i = dwnst_nodes[j]
            # calc erosion
            #S = np.clip(float(max_slopes[i].real), 1e-8, None)
            S = np.clip(max_slopes[i], 1e-8, None).astype(np.float64)
            # ero = -Kv[i] * (da[i] ** (0.5)) * S
            ero = -Kv[i] * np.power(da[i], 0.5) * S
            dzver[i] = ero
            petlat = 0.0
            
            # Choose lateral node for node i. If node i flows downstream, continue.
            # if node i is the first cell at the top of the drainage network, don't go
            # into this loop because in this case, node i won't have a "donor" node
            if i in flowdirs:
                # node_finder picks the lateral node to erode based on angle
                # between segments between five nodes
                # node_finder returns the lateral node ID and the curvature and phase delay curvature
                lat_nodes_at_i, inv_rad_curv, phd_inv_rad_curv = node_finder_use_fivebyfive_window_ver2(grid, 
                                                                                                        i, 
                                                                                                        flowdirs, 
                                                                                                        da, 
                                                                                                        is_get_phd_cur=True,
                                                                                                        dummy_value=dummy_value)

                lat_nodes[i] = lat_nodes_at_i
                cur[i] = inv_rad_curv
                phd_cur[i] = phd_inv_rad_curv
                petlat = -Kl[i] * da[i] * max_slopes[i] * inv_rad_curv # 側方侵食速度
                El[i] = petlat
                fai[i] = petlat/ero #側方/下方侵食速度比率
                node_num_at_i = len(np.where(lat_nodes_at_i != dummy_value)[0])

                # latero_nums[i] = -1 # 河川ノードは侵食が起きない前提で-1を代入(2023/11/17)

                for k in range(node_num_at_i):
                    lat_node = lat_nodes_at_i[k]
                    if lat_node > 0:
                        # if the elevation of the lateral node is higher than primary node,
                        # calculate a new potential lateral erosion (L/T), which is negative
                        if z[lat_node] > z[i]:
                            # the calculated potential lateral erosion is mutiplied by the length of the node
                            # and the bank height, then added to an array, vol_lat_dt, for volume eroded
                            # laterally  *per timestep* at each node. This vol_lat_dt is reset to zero for
                            # each timestep loop. vol_lat_dt is added to itself in case more than one primary
                            # nodes are laterally eroding this lat_node
                            # volume of lateral erosion per timestep
                            vol_lat_dt[lat_node] += abs(petlat) * grid.dx * dp[i]
                            latero_nums[lat_node] += 1 # 何回側方侵食を起こしたかを記録する配列を更新(2023/11/17)
                            # wd? may be H is true. how calc H ? 

        dzdt[:] = dzver * dt
        vol_lat[:] += vol_lat_dt * dt
        # this loop determines if enough lateral erosion has happened to change
        # the height of the neighbor node.
        # print(f"len(lat_nodes): {len(lat_nodes)}, len(dwns_nodes): {len(dwnst_nodes)}")
        for j in range(iterNum):
            i = dwnst_nodes[j]
            lat_nodes_at_i = lat_nodes[i]

            #if lat_nodes_at_i[0] != dummy:
            node_num_at_i = len(np.where(lat_nodes_at_i != dummy_value)[0])

            for k in range(node_num_at_i):
                lat_node = lat_nodes_at_i[k]
                if lat_node > 0:  # greater than zero now bc inactive neighbors are value -1
                    if z[lat_node] > z[i]:
                        # vol_diff is the volume that must be eroded from lat_node so that its
                        # elevation is the same as node downstream of primary node
                        # UC model: this would represent undercutting (the water height at
                        # node i), slumping, and instant removal.
                        if UC:
                            voldiff = critical_erosion_volume_ratio * (z[i] + dp[i] - z[flowdirs[i]]) * grid.dx**2 
                        # TB model: entire lat node must be eroded before lateral erosion
                        # occurs
                        if TB:
                            voldiff = critical_erosion_volume_ratio * (z[lat_node] - z[flowdirs[i]]) * grid.dx**2
                        # if the total volume eroded from lat_node is greater than the volume
                        # needed to be removed to make node equal elevation,
                        # then instantaneously remove this height from lat node. already has
                        # timestep in it
                        if vol_lat[lat_node] >= voldiff:
                            self._dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                            # after the lateral node is eroded, reset its volume eroded to
                            # zero
                            vol_lat[lat_node] = 0.0
        # combine vertical and lateral erosion.
        dz = dzdt + self._dzlat
        # change height of landscape
        z[:] += dz

        return grid, self._dzlat
    
    def run_one_step_langston_ffwindow_only_hill(self, dt=1.0):
        """Calculate vertical and lateral erosion for a time period 'dt'.

        Parameters
        ----------
        dt : float
            Model timestep [T]

        Note
        ----------
        側方侵食アルゴリズム
            Langston & Tucker (2018)
        ノード選定アルゴリズム
            Langston & Tucker (2018)
        """
        Klr = self._Klr
        grid = self._grid
        UC = self._UC
        TB = self._TB
        inlet_on = self._inlet_on  # this is a true/false flag
        Kv = self._Kv
        qs_in = self._qs_in
        dzdt = self._dzdt
        alph = self._alph
        vol_lat = grid.at_node["volume__lateral_erosion"]
        
        # kw = 10.0
        # F = 0.02
        dp_coef = self._dp_coef
        dp_exp = self._dp_exp
        kw = self._wid_coef
        F = self._F
        thresh_da = self._thresh_da

        # May 2, runoff calculated below (in m/s) is important for calculating
        # 2022/07/07 unit of runoff is not m/s, maybe m/yr
        # discharge and water depth correctly. renamed runoffms to prevent
        # confusion with other uses of runoff
        runoffms = (Klr * F / kw) ** 2
        # Kl is calculated from ratio of lateral to vertical K parameters
        Kl = Kv * Klr
        z = grid.at_node["topographic__elevation"]
        # clear qsin for next loop
        qs_in = grid.add_zeros("sediment__influx", at="node", clobber=True)
        qs = grid.add_zeros("qs", at="node", clobber=True)
        dzver = np.zeros(grid.number_of_nodes)
        El = grid.add_zeros("latero__rate", at="node", clobber=True) 
        El = grid.at_node["latero__rate"] 
        fai = grid.add_zeros("fai", at="node", clobber=True)
        fai = grid.at_node["fai"]
        vol_lat_dt = np.zeros(grid.number_of_nodes)
        cur = grid.at_node["curvature"]
        phd_cur = grid.at_node["phd_curvature"]

        latero_nums = self._latero_nums
        latero_nums[:] = 0 # 何回側方侵食を起こしたかを記録する配列を初期化(2023/11/17)

        # dz_lat needs to be reset. Otherwise, once a lateral node erode's once, it will continue eroding
        # at every subsequent time setp. If you want to track all lateral erosion, create another attribute,
        # or add self.dzlat to itself after each time step.
        self._dzlat.fill(0.0)

        # critical_erosion_volume_ratio 
        critical_erosion_volume_ratio = self._critical_erosion_volume_ratio

        if inlet_on is True:
            inlet_node = self._inlet_node
            qsinlet = self._qsinlet
            qs_in[inlet_node] = qsinlet
            q = grid.at_node["surface_water__discharge"]
            # da = q / grid.dx**2
            da = q / runoffms # change, 2022/10/19
        # if inlet flag is not on, proceed as normal.
        else:
            if self._use_Q:
                # water discharge is calculated by flow router
                da = grid.at_node["surface_water__discharge"]
            else:
                # drainage area is calculated by flow router
                da = grid.at_node["drainage_area"]
        
        # add min_Q_or_da
        da += self._add_min_Q_or_da

        # water depth in meters, needed for lateral erosion calc
        dp = grid.at_node["flow_depth"]
        dp[:] = dp_coef * (da ** dp_exp)

        # flow__upstream_node_order is node array contianing downstream to
        # upstream order list of node ids
        # s contein ids
        s = grid.at_node["flow__upstream_node_order"]
        max_slopes = grid.at_node["topographic__steepest_slope"]
        flowdirs = grid.at_node["flow__receiver_node"]

        # make a list l, where node status is interior (signified by label 0) in s
        # make threshold mask, because apply equation only river. (2022/10/26)
        interior_mask = np.where(np.logical_and(grid.status_at_node == 0, da >= thresh_da))[0]
        interior_s = np.intersect1d(s, interior_mask)
        dwnst_nodes = interior_s.copy()
        # reverse list so we go from upstream to down stream
        dwnst_nodes = dwnst_nodes[::-1].astype(np.int32)
        max_slopes[:] = max_slopes.clip(0)
        iterNum = len(dwnst_nodes)
        dummy_value = -99

        nodeNum = grid.shape[0]*grid.shape[1]
        lat_nodes = np.full(shape=(nodeNum, 4), fill_value=dummy_value) # node iの上下４つのノードが入るサイズ
        i = 0

        for j in range(iterNum):
            i = dwnst_nodes[j]
            # calc erosion
            #S = np.clip(float(max_slopes[i].real), 1e-8, None)
            S = np.clip(max_slopes[i], 1e-8, None).astype(np.float64)
            # ero = -Kv[i] * (da[i] ** (0.5)) * S
            ero = -Kv[i] * np.power(da[i], 0.5) * S
            dzver[i] = ero
            petlat = 0.0
            
            # Choose lateral node for node i. If node i flows downstream, continue.
            # if node i is the first cell at the top of the drainage network, don't go
            # into this loop because in this case, node i won't have a "donor" node
            if i in flowdirs:
                # node_finder picks the lateral node to erode based on angle
                # between segments between five nodes
                # node_finder returns the lateral node ID and the curvature and phase delay curvature
                lat_nodes_at_i, inv_rad_curv, phd_inv_rad_curv = node_finder_use_fivebyfive_window_only_hill(
                                                                                                            grid, 
                                                                                                            i, 
                                                                                                            flowdirs, 
                                                                                                            da, 
                                                                                                            dwnst_nodes,
                                                                                                            is_get_phd_cur=True,
                                                                                                            dummy_value=dummy_value,
                                                                                                             )

                lat_nodes[i] = lat_nodes_at_i
                cur[i] = inv_rad_curv
                phd_cur[i] = phd_inv_rad_curv
                petlat = -Kl[i] * da[i] * max_slopes[i] * inv_rad_curv # 側方侵食速度
                El[i] = petlat
                fai[i] = petlat/ero #側方/下方侵食速度比率
                node_num_at_i = len(np.where(lat_nodes_at_i != dummy_value)[0])

                # latero_nums[i] = -1 # 河川ノードは侵食が起きない前提で-1を代入(2023/11/17)

                for k in range(node_num_at_i):
                    lat_node = lat_nodes_at_i[k]
                    latero_nums[lat_node] += 1 # 何回側方侵食を起こしたかを記録する配列を更新(2023/11/17)
                    if lat_node > 0:
                        # if the elevation of the lateral node is higher than primary node,
                        # calculate a new potential lateral erosion (L/T), which is negative
                        if z[lat_node] > z[i]:
                            # the calculated potential lateral erosion is mutiplied by the length of the node
                            # and the bank height, then added to an array, vol_lat_dt, for volume eroded
                            # laterally  *per timestep* at each node. This vol_lat_dt is reset to zero for
                            # each timestep loop. vol_lat_dt is added to itself in case more than one primary
                            # nodes are laterally eroding this lat_node
                            # volume of lateral erosion per timestep
                            vol_lat_dt[lat_node] += abs(petlat) * grid.dx * dp[i]
                            # wd? may be H is true. how calc H ? 

        dzdt[:] = dzver * dt
        vol_lat[:] += vol_lat_dt * dt
        # this loop determines if enough lateral erosion has happened to change
        # the height of the neighbor node.
        # print(f"len(lat_nodes): {len(lat_nodes)}, len(dwns_nodes): {len(dwnst_nodes)}")
        for j in range(iterNum):
            i = dwnst_nodes[j]
            lat_nodes_at_i = lat_nodes[i]

            #if lat_nodes_at_i[0] != dummy:
            node_num_at_i = len(np.where(lat_nodes_at_i != dummy_value)[0])

            for k in range(node_num_at_i):
                lat_node = lat_nodes_at_i[k]
                if lat_node > 0:  # greater than zero now bc inactive neighbors are value -1
                    if z[lat_node] > z[i]:
                        # vol_diff is the volume that must be eroded from lat_node so that its
                        # elevation is the same as node downstream of primary node
                        # UC model: this would represent undercutting (the water height at
                        # node i), slumping, and instant removal.
                        if UC:
                            voldiff = critical_erosion_volume_ratio * (z[i] + dp[i] - z[flowdirs[i]]) * grid.dx**2 
                        # TB model: entire lat node must be eroded before lateral erosion
                        # occurs
                        if TB:
                            voldiff = critical_erosion_volume_ratio * (z[lat_node] - z[flowdirs[i]]) * grid.dx**2
                        # if the total volume eroded from lat_node is greater than the volume
                        # needed to be removed to make node equal elevation,
                        # then instantaneously remove this height from lat node. already has
                        # timestep in it
                        if vol_lat[lat_node] >= voldiff:
                            self._dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                            # after the lateral node is eroded, reset its volume eroded to
                            # zero
                            vol_lat[lat_node] = 0.0
        # combine vertical and lateral erosion.
        dz = dzdt + self._dzlat
        # change height of landscape
        z[:] += dz

        return grid, self._dzlat

    def run_one_step_ULE_langston(self, dt=1.0):
        """Calculate vertical and lateral erosion for a time period 'dt'.
        lateral/vertical erosion rate is calculated by using phase delay curvature and exponential function.

        lateral/vertical erosion rate 'fai' = exp(fai_C) * Q**fai_alpah * S**fai_beta * R**fai_gamma
        Q: discharge, S: slope, R: radius of curvature

        Parameters
        ----------
        dt : float
            Model timestep [T]

        Note
        ----------
        側方侵食アルゴリズム
            ULEmodel regression
        ノード選定アルゴリズム
            Langston & Tucker (2018)
        """
        Klr = self._Klr
        grid = self._grid
        UC = self._UC
        TB = self._TB
        inlet_on = self._inlet_on  # this is a true/false flag
        Kv = self._Kv
        qs_in = self._qs_in
        dzdt = self._dzdt
        alph = self._alph
        vol_lat = grid.at_node["volume__lateral_erosion"]
        
        # kw = 10.0
        # F = 0.02
        dp_coef = self._dp_coef
        dp_exp = self._dp_exp
        kw = self._wid_coef
        F = self._F
        thresh_da = self._thresh_da

        cur = grid.at_node["curvature"]

        fai_alpha = self._fai_alpha
        fai_beta = self._fai_beta
        fai_gamma = self._fai_gamma
        fai_C = self._fai_C

        # May 2, runoff calculated below (in m/s) is important for calculating
        # 2022/07/07 unit of runoff is not m/s, maybe m/yr
        # discharge and water depth correctly. renamed runoffms to prevent
        # confusion with other uses of runoff
        runoffms = (Klr * F / kw) ** 2
        # Kl is calculated from ratio of lateral to vertical K parameters
        Kl = Kv * Klr
        z = grid.at_node["topographic__elevation"]
        # clear qsin for next loop
        qs_in = grid.add_zeros("sediment__influx", at="node", clobber=True)
        qs = grid.add_zeros("qs", at="node", clobber=True)
        lat_nodes = np.zeros(grid.number_of_nodes, dtype=int)
        dzver = np.zeros(grid.number_of_nodes)
        El = grid.add_zeros("latero__rate", at="node", clobber=True) 
        El = grid.at_node["latero__rate"] 
        fai = grid.add_zeros("fai", at="node", clobber=True)
        fai = grid.at_node["fai"]
        vol_lat_dt = np.zeros(grid.number_of_nodes)

        latero_nums = self._latero_nums
        latero_nums[:] = 0 # 何回側方侵食を起こしたかを記録する配列を初期化(2023/11/17)

        # dz_lat needs to be reset. Otherwise, once a lateral node erode's once, it will continue eroding
        # at every subsequent time setp. If you want to track all lateral erosion, create another attribute,
        # or add self.dzlat to itself after each time step.
        self._dzlat.fill(0.0)

        # critical_erosion_volume_ratio 
        critical_erosion_volume_ratio = self._critical_erosion_volume_ratio

        if inlet_on is True:
            inlet_node = self._inlet_node
            qsinlet = self._qsinlet
            qs_in[inlet_node] = qsinlet
            q = grid.at_node["surface_water__discharge"]
            da = q / grid.dx**2
        # if inlet flag is not on, proceed as normal.
        else:
            if self._use_Q:
                # water discharge is calculated by flow router
                da = grid.at_node["surface_water__discharge"]
            else:
                # drainage area is calculated by flow router
                da = grid.at_node["drainage_area"]

        # add min_Q_or_da
        da += self._add_min_Q_or_da

        # water depth in meters, needed for lateral erosion calc
        dp = grid.at_node["flow_depth"]
        dp[:] = dp_coef * (da ** dp_exp)

        # add min_Q_or_da
        da += self._add_min_Q_or_da
        # flow__upstream_node_order is node array contianing downstream to
        # upstream order list of node ids
        s = grid.at_node["flow__upstream_node_order"]
        max_slopes = grid.at_node["topographic__steepest_slope"]
        flowdirs = grid.at_node["flow__receiver_node"]

        # make a list l, where node status is interior (signified by label 0) in s
        # make threshold mask, because apply equation only river. (2022/10/26)
        interior_mask = np.where(np.logical_and(grid.status_at_node == 0, da >= thresh_da))[0]
        interior_s = np.intersect1d(s, interior_mask)
        dwnst_nodes = interior_s.copy()
        # reverse list so we go from upstream to down stream
        dwnst_nodes = dwnst_nodes[::-1]
        max_slopes[:] = max_slopes.clip(0)
        
        epsilon = 1e-10

        for i in dwnst_nodes:
            # calc deposition and erosion
            dep = alph * qs_in[i] / da[i]
            ero = -Kv[i] * da[i] ** (0.5) * max_slopes[i]
            dzver[i] = dep + ero

            # potential lateral erosion initially set to 0
            petlat = 0.0
            
            # Choose lateral node for node i. If node i flows downstream, continue.
            # if node i is the first cell at the top of the drainage network, don't go
            # into this loop because in this case, node i won't have a "donor" node
            if i in flowdirs:
                # node_finder picks the lateral node to erode based on angle
                # between segments between three nodes
                [lat_node, inv_rad_curv] = node_finder(grid, i, flowdirs, da)
                # node_finder returns the lateral node ID and the radius of curvature
                lat_nodes[i] = lat_node
                cur[i] = inv_rad_curv
                # latero_nums[i] = -1 # 河川ノードは侵食が起きない前提で-1を代入(2023/11/17)
                # if the lateral node is not 0 or -1 continue. lateral node may be
                # 0 or -1 if a boundary node was chosen as a lateral node. then
                # radius of curavature is also 0 so there is no lateral erosion
                if lat_node > 0:
                    # if the elevation of the lateral node is higher than primary node,
                    # calculate a new potential lateral erosion (L/T), which is negative
                    if z[lat_node] > z[i]:
                        R = 1/(inv_rad_curv+epsilon)
                        S = np.clip(max_slopes[i], 1e-8, None)
                        fai[i] = np.exp(fai_C) * (da[i]**fai_alpha) * (S**fai_beta) * (R**fai_gamma)
                        petlat = fai[i] * ero
                        El[i] = petlat
                        # the calculated potential lateral erosion is mutiplied by the length of the node
                        # and the bank height, then added to an array, vol_lat_dt, for volume eroded
                        # laterally  *per timestep* at each node. This vol_lat_dt is reset to zero for
                        # each timestep loop. vol_lat_dt is added to itself in case more than one primary
                        # nodes are laterally eroding this lat_node
                        # volume of lateral erosion per timestep
                        vol_lat_dt[lat_node] += abs(petlat) * grid.dx * dp[i]
                        latero_nums[lat_node] += 1 # 何回側方侵食を起こしたかを記録する配列を更新(2023/11/17)
                        # wd? may be H is true. how calc H ? 

            # send sediment downstream. sediment eroded from vertical incision
            # and lateral erosion is sent downstream
            #            print("debug before 406")
            qs_in[flowdirs[i]] += (
                qs_in[i] - (dzver[i] * grid.dx**2) - (petlat * grid.dx * dp[i])
            )  # qsin to next node
            # print(f"qs_in[i] = {qs_in[i]:.2f}, qs_in[flowdirs[i]] = {qs_in[flowdirs[i]]:.2f}, i = {i}, flowdirs[i] = {flowdirs[i]}")
        qs[:] = qs_in - (dzver * grid.dx**2)
        dzdt[:] = dzver * dt
        vol_lat[:] += vol_lat_dt * dt
        # this loop determines if enough lateral erosion has happened to change
        # the height of the neighbor node.
        for i in dwnst_nodes:
            lat_node = lat_nodes[i]
            if lat_node > 0:  # greater than zero now bc inactive neighbors are value -1
                if z[lat_node] > z[i]:
                    # vol_diff is the volume that must be eroded from lat_node so that its
                    # elevation is the same as node downstream of primary node
                    # UC model: this would represent undercutting (the water height at
                    # node i), slumping, and instant removal.
                    if UC == 1:
                        voldiff = critical_erosion_volume_ratio * (z[i] + dp[i] - z[flowdirs[i]]) * grid.dx**2 
                    # TB model: entire lat node must be eroded before lateral erosion
                    # occurs
                    if TB == 1:
                        voldiff = critical_erosion_volume_ratio * (z[lat_node] - z[flowdirs[i]]) * grid.dx**2
                    # if the total volume eroded from lat_node is greater than the volume
                    # needed to be removed to make node equal elevation,
                    # then instantaneously remove this height from lat node. already has
                    # timestep in it
                    if vol_lat[lat_node] >= voldiff:
                        self._dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                        # after the lateral node is eroded, reset its volume eroded to
                        # zero
                        vol_lat[lat_node] = 0.0
        # combine vertical and lateral erosion.
        dz = dzdt + self._dzlat
        # change height of landscape
        z[:] += dz
        return grid, self._dzlat
    
    def run_one_step_ULE_ffwindow(self, dt=1.0):
        """Calculate vertical and lateral erosion for a time period 'dt'.

        Parameters
        ----------
        dt : float
            Model timestep [T]

        Note
        ----------
        側方侵食アルゴリズム
            ULEmodel regression
        ノード選定アルゴリズム
            ffwindow

        lat_nodesの実装がnp.ndarray。
        アルゴリズムの本質はrun_one_step_fivebyfive_window_Cと同じだが高速
        """
        Klr = self._Klr
        grid = self._grid
        UC = self._UC
        TB = self._TB
        inlet_on = self._inlet_on  # this is a true/false flag
        Kv = self._Kv
        qs_in = self._qs_in
        dzdt = self._dzdt
        alph = self._alph
        vol_lat = grid.at_node["volume__lateral_erosion"]
        
        # kw = 10.0
        # F = 0.02
        dp_coef = self._dp_coef
        dp_exp = self._dp_exp
        kw = self._wid_coef
        F = self._F
        thresh_da = self._thresh_da

        # phd_node_num = 1
        cur = grid.at_node["curvature"]
        phd_cur = grid.at_node["phd_curvature"]

        fai_alpha = self._fai_alpha
        fai_beta = self._fai_beta
        fai_gamma = self._fai_gamma
        fai_C = self._fai_C

        z = grid.at_node["topographic__elevation"]
        # clear qsin for next loop
        qs_in = grid.add_zeros("sediment__influx", at="node", clobber=True)
        qs = grid.add_zeros("qs", at="node", clobber=True)
        dzver = np.zeros(grid.number_of_nodes)
        El = grid.add_zeros("latero__rate", at="node", clobber=True) 
        El = grid.at_node["latero__rate"] 
        fai = grid.add_zeros("fai", at="node", clobber=True)
        fai = grid.at_node["fai"]
        vol_lat_dt = np.zeros(grid.number_of_nodes)
        latero_nums = self._latero_nums
        latero_nums[:] = 0

        # dz_lat needs to be reset. Otherwise, once a lateral node erode's once, it will continue eroding
        # at every subsequent time setp. If you want to track all lateral erosion, create another attribute,
        # or add self.dzlat to itself after each time step.
        self._dzlat.fill(0.0)

        # critical_erosion_volume_ratio 
        critical_erosion_volume_ratio = self._critical_erosion_volume_ratio

        if inlet_on is True:
            inlet_node = self._inlet_node
            qsinlet = self._qsinlet
            qs_in[inlet_node] = qsinlet
            q = grid.at_node["surface_water__discharge"]
            da = q / grid.dx**2
        # if inlet flag is not on, proceed as normal.
        else:
            if self._use_Q:
                # water discharge is calculated by flow router
                da = grid.at_node["surface_water__discharge"]
            else:
                # drainage area is calculated by flow router
                da = grid.at_node["drainage_area"]

        # add min_Q_or_da
        da += self._add_min_Q_or_da

        # water depth in meters, needed for lateral erosion calc
        dp = grid.at_node["flow_depth"]
        dp[:] = dp_coef * (da ** dp_exp)

        # add min_Q_or_da
        da += self._add_min_Q_or_da
        # flow__upstream_node_order is node array contianing downstream to
        # upstream order list of node ids
        s = grid.at_node["flow__upstream_node_order"]
        max_slopes = grid.at_node["topographic__steepest_slope"]
        flowdirs = grid.at_node["flow__receiver_node"]

        # make a list l, where node status is interior (signified by label 0) in s
        # make threshold mask, because apply equation only river. (2022/10/26)
        interior_mask = np.where(np.logical_and(grid.status_at_node == 0, da >= thresh_da))[0]
        interior_s = np.intersect1d(s, interior_mask)
        dwnst_nodes = interior_s.copy()
        # reverse list so we go from upstream to down stream
        dwnst_nodes = dwnst_nodes[::-1].astype(np.int32)
        # lat_nodes = [[-99] for i in range(grid.shape[0]*grid.shape[1])]
        max_slopes[:] = max_slopes.clip(0)

        _run_one_step_fivebyfive_window_ver2(
            grid,
            dwnst_nodes,
            flowdirs,
            latero_nums,
            z,
            da,
            dp,
            Kv,
            max_slopes,
            dzver,
            cur,
            phd_cur,
            El,
            fai,
            vol_lat,
            self._dzlat,
            dzdt,
            grid.dx,
            fai_alpha,
            fai_beta,
            fai_gamma,
            fai_C,
            dt,
            critical_erosion_volume_ratio,
            UC,
            TB,
        )

        return grid, self._dzlat   
    

    def run_one_step_ULE_ffwindow_only_hill(self, dt=1.0):
        """Calculate vertical and lateral erosion for a time period 'dt'.

        Parameters
        ----------
        dt : float
            Model timestep [T]

        Note
        ----------
        側方侵食アルゴリズム
            ULEmodel regression
        ノード選定アルゴリズム
            ffwindow

        lat_nodesの実装がnp.ndarray。
        アルゴリズムの本質はrun_one_step_fivebyfive_window_Cと同じだが高速
        """
        Klr = self._Klr
        grid = self._grid
        UC = self._UC
        TB = self._TB
        inlet_on = self._inlet_on  # this is a true/false flag
        Kv = self._Kv
        qs_in = self._qs_in
        dzdt = self._dzdt
        alph = self._alph
        vol_lat = grid.at_node["volume__lateral_erosion"]
        
        # kw = 10.0
        # F = 0.02
        dp_coef = self._dp_coef
        dp_exp = self._dp_exp
        kw = self._wid_coef
        F = self._F
        thresh_da = self._thresh_da

        # phd_node_num = 1
        cur = grid.at_node["curvature"]
        phd_cur = grid.at_node["phd_curvature"]

        fai_alpha = self._fai_alpha
        fai_beta = self._fai_beta
        fai_gamma = self._fai_gamma
        fai_C = self._fai_C

        z = grid.at_node["topographic__elevation"]
        # clear qsin for next loop
        qs_in = grid.add_zeros("sediment__influx", at="node", clobber=True)
        qs = grid.add_zeros("qs", at="node", clobber=True)
        dzver = np.zeros(grid.number_of_nodes)
        El = grid.add_zeros("latero__rate", at="node", clobber=True) 
        El = grid.at_node["latero__rate"] 
        fai = grid.add_zeros("fai", at="node", clobber=True)
        fai = grid.at_node["fai"]
        vol_lat_dt = np.zeros(grid.number_of_nodes)
        latero_nums = self._latero_nums
        latero_nums[:] = 0

        # dz_lat needs to be reset. Otherwise, once a lateral node erode's once, it will continue eroding
        # at every subsequent time setp. If you want to track all lateral erosion, create another attribute,
        # or add self.dzlat to itself after each time step.
        self._dzlat.fill(0.0)

        # critical_erosion_volume_ratio 
        critical_erosion_volume_ratio = self._critical_erosion_volume_ratio

        if inlet_on is True:
            inlet_node = self._inlet_node
            qsinlet = self._qsinlet
            qs_in[inlet_node] = qsinlet
            q = grid.at_node["surface_water__discharge"]
            da = q / grid.dx**2
        # if inlet flag is not on, proceed as normal.
        else:
            if self._use_Q:
                # water discharge is calculated by flow router
                da = grid.at_node["surface_water__discharge"]
            else:
                # drainage area is calculated by flow router
                da = grid.at_node["drainage_area"]

        # add min_Q_or_da
        da += self._add_min_Q_or_da

        # water depth in meters, needed for lateral erosion calc
        dp = grid.at_node["flow_depth"]
        dp[:] = dp_coef * (da ** dp_exp)

        # add min_Q_or_da
        da += self._add_min_Q_or_da
        # flow__upstream_node_order is node array contianing downstream to
        # upstream order list of node ids
        s = grid.at_node["flow__upstream_node_order"]
        max_slopes = grid.at_node["topographic__steepest_slope"]
        flowdirs = grid.at_node["flow__receiver_node"]

        # make a list l, where node status is interior (signified by label 0) in s
        # make threshold mask, because apply equation only river. (2022/10/26)
        interior_mask = np.where(np.logical_and(grid.status_at_node == 0, da >= thresh_da))[0]
        interior_s = np.intersect1d(s, interior_mask)
        dwnst_nodes = interior_s.copy()
        # reverse list so we go from upstream to down stream
        dwnst_nodes = dwnst_nodes[::-1].astype(np.int32)
        # lat_nodes = [[-99] for i in range(grid.shape[0]*grid.shape[1])]
        max_slopes[:] = max_slopes.clip(0)

        _run_one_step_fivebyfive_window_only_hill(
            grid,
            dwnst_nodes,
            flowdirs,
            latero_nums,
            z,
            da,
            dp,
            Kv,
            max_slopes,
            dzver,
            cur,
            phd_cur,
            El,
            fai,
            vol_lat,
            self._dzlat,
            dzdt,
            grid.dx,
            fai_alpha,
            fai_beta,
            fai_gamma,
            fai_C,
            dt,
            critical_erosion_volume_ratio,
            UC,
            TB,
        )

        return grid, self._dzlat  
    