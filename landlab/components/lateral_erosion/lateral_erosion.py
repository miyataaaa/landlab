# -*- coding: utf-8 -*-
"""Grid-based simulation of lateral erosion by channels in a drainage network.

ALangston
"""

import numpy as np
import sys

from landlab import Component, RasterModelGrid
from landlab.components.flow_accum import FlowAccumulator

from .node_finder import node_finder, node_and_phdcur_finder

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
        solver="basic",
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
                (1) 'basic' (default): explicit forward-time extrapolation.
                    Simple but will become unstable if time step is too large or
                    if bedrock erodibility is vry high.
                (2) 'adaptive': subdivides global time step as needed to
                    prevent slopes from reversing.
                (3) 'delay_rs': use 'phase delay curvature' to calculate the lateral erosion rate.
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

        if solver not in ("basic", "adaptive", "delay_rs"):
            raise ValueError(
                "value for solver not understood ({val} not one of {valid})".format(
                    val=solver, valid=", ".join(("basic", "adaptive", "delay_rs"))
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

        if solver == "adaptive":
            if not isinstance(flow_accumulator, FlowAccumulator):
                raise ValueError(
                    (
                        "When the adaptive solver is used, a valid "
                        "FlowAccumulator must be passed on "
                        "instantiation."
                    )
                )
            self._flow_accumulator = flow_accumulator

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


        # you can specify the type of lateral erosion model you want to use.
        # But if you don't the default is the undercutting-slump model
        if latero_mech == "TB":
            self._TB = True
            self._UC = False
        else:
            self._UC = True
            self._TB = False
        # option use adaptive time stepping. Default is fixed dt supplied by user
        if solver == "basic":
            self.run_one_step = self.run_one_step_basic
        elif solver == "adaptive":
            self.run_one_step = self.run_one_step_adaptive
        elif solver == "delay_rs":
            self.run_one_step = self.run_one_step_basic_delay_rs

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

    def run_one_step_basic(self, dt=1.0):
        """Calculate vertical and lateral erosion for a time period 'dt'.

        Parameters
        ----------
        dt : float
            Model timestep [T]
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

        # dz_lat needs to be reset. Otherwise, once a lateral node erode's once, it will continue eroding
        # at every subsequent time setp. If you want to track all lateral erosion, create another attribute,
        # or add self.dzlat to itself after each time step.
        self._dzlat.fill(0.0)

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
        dp = dp_coef * (da ** dp_exp)

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
                        voldiff = (z[i] + dp[i] - z[flowdirs[i]]) * grid.dx**2
                    # TB model: entire lat node must be eroded before lateral erosion
                    # occurs
                    if TB == 1:
                        voldiff = (z[lat_node] - z[flowdirs[i]]) * grid.dx**2
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

    def run_one_step_adaptive(self, dt=1.0):
        """Run time step with adaptive time stepping to prevent slope
        flattening."""
        Klr = self._Klr
        grid = self._grid
        UC = self._UC
        TB = self._TB
        inlet_on = self._inlet_on  # this is a true/false flag
        Kv = self._Kv
        qs_in = self._qs_in
        dzdt = self._dzdt
        alph = self._alph
        vol_lat = self._grid.at_node["volume__lateral_erosion"]
        kw = 10.0
        F = 0.02
        runoffms = (Klr * F / kw) ** 2
        Kl = Kv * Klr
        z = grid.at_node["topographic__elevation"]
        # clear qsin for next loop
        qs_in = grid.add_zeros("sediment__influx", at="node", clobber=True)
        qs = grid.add_zeros("qs", at="node", clobber=True)
        lat_nodes = np.zeros(grid.number_of_nodes, dtype=int)
        dzver = np.zeros(grid.number_of_nodes)
        vol_lat_dt = np.zeros(grid.number_of_nodes)

        # dz_lat needs to be reset. Otherwise, once a lateral node erode's once, it will continue eroding
        # at every subsequent time setp. If you want to track all lateral erosion, create another attribute,
        # or add self.dzlat to itself after each time step.
        self._dzlat.fill(0.0)

        if inlet_on is True:
            # define inlet_node
            inlet_node = self._inlet_node
            qsinlet = self._qsinlet
            qs_in[inlet_node] = qsinlet
            q = grid.at_node["surface_water__discharge"]
            da = q / grid.dx**2
        # if inlet flag is not on, proceed as normal.
        else:
            # renamed this drainage area set by flow router
            da = grid.at_node["drainage_area"] if self._use_Q else grid.at_node["surface_water__discharge"] 
        s = grid.at_node["flow__upstream_node_order"]
        max_slopes = grid.at_node["topographic__steepest_slope"]
        flowdirs = grid.at_node["flow__receiver_node"]
        interior_s = s[np.where((grid.status_at_node[s] == 0))[0]]
        dwnst_nodes = interior_s.copy()
        # reverse list so we go from upstream to down stream
        dwnst_nodes = dwnst_nodes[::-1]
        # local time
        time = 0.0
        globdt = dt

        while time < globdt:
            max_slopes[:] = max_slopes.clip(0)
            # here calculate dzdt for each node, with initial time step
            for i in dwnst_nodes:
                dep = alph * qs_in[i] / da[i]
                ero = -Kv[i] * da[i] ** (0.5) * max_slopes[i]
                dzver[i] = dep + ero
                petlat = 0.0
                # water depth in meters, needed for lateral erosion calc
                wd = wid_coeff * (da[i] * runoffms) ** wid_exp

                if i in flowdirs:
                    # node_finder picks the lateral node to erode based on angle
                    # between segments between three nodes
                    [lat_node, inv_rad_curv] = node_finder(grid, i, flowdirs, da)
                    # node_finder returns the lateral node ID and the radius of
                    # curvature
                    lat_nodes[i] = lat_node
                    # if the lateral node is not 0 or -1 continue.
                    if lat_node > 0:
                        # if the elevation of the lateral node is higher than primary node,
                        # calculate a new potential lateral erosion (L/T), which is
                        # negative
                        if z[lat_node] > z[i]:
                            petlat = -Kl[i] * da[i] * max_slopes[i] * inv_rad_curv
                            # the calculated potential lateral erosion is mutiplied by the length of the node
                            # and the bank height, then added to an array, vol_lat_dt, for volume eroded
                            # laterally  *per timestep* at each node. This vol_lat_dt is reset to zero for
                            # each timestep loop. vol_lat_dt is added to itself more than one primary nodes are
                            # laterally eroding this lat_node
                            # volume of lateral erosion per timestep
                            vol_lat_dt[lat_node] += abs(petlat) * grid.dx * wd
                # send sediment downstream. sediment eroded from vertical incision
                # and lateral erosion is sent downstream
                qs_in[flowdirs[i]] += (
                    qs_in[i] - (dzver[i] * grid.dx**2) - (petlat * grid.dx * wd)
                )  # qsin to next node
            # summing qs for this entire timestep
            qs[:] += qs_in - (dzver * grid.dx**2)
            dzdt[:] = dzver
            # Do a time-step check
            # If the downstream node is eroding at a slower rate than the
            # upstream node, there is a possibility of flow direction reversal,
            # or at least a flattening of the landscape.
            # Limit dt so that this flattening or reversal doesn't happen.
            # How close you allow these two points to get to eachother is
            # determined by the cfl timestep condition, hard coded to equal 0.3
            # dtn is an arbitrarily large number to begin with, but will be adapted as we step through
            # the nodes
            dtn = dt * 50  # starting minimum timestep for this round
            for i in dwnst_nodes:
                # are points converging? ie, downstream eroding slower than upstream
                dzdtdif = dzdt[flowdirs[i]] - dzdt[i]
                # if points converging, find time to zero slope
                if dzdtdif > 1.0e-5 and max_slopes[i] > 1e-5:
                    # time to flat between points
                    dtflat = (z[i] - z[flowdirs[i]]) / dzdtdif
                    # if time to flat is smaller than dt, take the lower value
                    if dtflat < dtn:
                        dtn = dtflat
                    #                        assert dtn>0, "dtn <0 at dtflat"
                    # if dzdtdif*dtflat will make upstream lower than downstream, find
                    # time to flat
                    if dzdtdif * dtflat > (z[i] - z[flowdirs[i]]):
                        dtn = (z[i] - z[flowdirs[i]]) / dzdtdif
            dtn *= cfl_cond
            # new minimum timestep for this round of nodes
            dt = min(abs(dtn), dt)
            assert dt > 0.0, "timesteps less than 0."

            # vol_lat is the total volume eroded from the lateral nodes through
            # the entire model run. So vol_lat is itself plus vol_lat_dt (for current loop)
            # times stable timestep size
            vol_lat[:] += vol_lat_dt * dt
            # this loop determines if enough lateral erosion has happened to change
            # the height of the neighbor node.
            for i in dwnst_nodes:
                lat_node = lat_nodes[i]
                wd = wid_coeff * (da[i] * runoffms) ** wid_exp
                if (
                    lat_node > 0
                ):  # greater than zero now bc inactive neighbors are value -1
                    if z[lat_node] > z[i]:
                        # vol_diff is the volume that must be eroded from lat_node so that its
                        # elevation is the same as node downstream of primary node
                        # UC model: this would represent undercutting (the water height
                        # at node i), slumping, and instant removal.
                        if UC == 1:
                            voldiff = (z[i] + wd - z[flowdirs[i]]) * grid.dx**2
                        # TB model: entire lat node must be eroded before lateral
                        # erosion occurs
                        if TB == 1:
                            voldiff = (z[lat_node] - z[flowdirs[i]]) * grid.dx**2
                        # if the total volume eroded from lat_node is greater than the volume
                        # needed to be removed to make node equal elevation,
                        # then instantaneously remove this height from lat node. already
                        # has timestep in it
                        if vol_lat[lat_node] >= voldiff:
                            self._dzlat[lat_node] = (
                                z[flowdirs[i]] - z[lat_node]
                            )  # -0.001
                            # after the lateral node is eroded, reset its volume eroded
                            # to zero
                            vol_lat[lat_node] = 0.0

            # multiply dzdt by timestep size and combine with lateral erosion
            # self._dzlat, which is already a length for the calculated time step
            dz = dzdt * dt + self._dzlat
            # change height of landscape
            z[:] += dz
            # update elapsed time
            time = dt + time
            # check to see that you are within 0.01% of the global timestep, if so
            # done, if not continue

            if time > 0.9999 * globdt:
                time = globdt

            else:
                dt = globdt - time
                qs_in = grid.zeros(centering="node")

                # recalculate flow directions
                (da, q) = self._flow_accumulator.accumulate_flow()

                if inlet_on:
                    # if inlet on, reset drainage area and qsin to reflect inlet conditions
                    # this is the drainage area needed for code below with an inlet
                    # set by spatially varible runoff.
                    da = q / grid.dx**2
                    qs_in[inlet_node] = qsinlet
                else:
                    # otherwise, drainage area is just drainage area.
                    da = grid.at_node["drainage_area"]
                s = grid.at_node["flow__upstream_node_order"]
                max_slopes = grid.at_node["topographic__steepest_slope"]
                q = grid.at_node["surface_water__discharge"]
                flowdirs = grid.at_node["flow__receiver_node"]
                interior_s = s[np.where((grid.status_at_node[s] == 0))[0]]
                dwnst_nodes = interior_s.copy()
                dwnst_nodes = dwnst_nodes[::-1]

                lat_nodes = np.zeros(grid.number_of_nodes, dtype=int)
                self._dzlat = np.zeros(grid.number_of_nodes)
                vol_lat_dt = np.zeros(grid.number_of_nodes)
                dzver = np.zeros(grid.number_of_nodes)

        return grid, self._dzlat

    def run_one_step_basic_delay_rs(self, dt=1.0):
        """Calculate vertical and lateral erosion for a time period 'dt'.
        lateral/vertical erosion rate is calculated by using phase delay curvature and exponential function.

        lateral/vertical erosion rate 'fai' = exp(fai_C) * Q**fai_alpah * S**fai_beta * R**fai_gamma
        Q: discharge, S: slope, R: radius of curvature

        Parameters
        ----------
        dt : float
            Model timestep [T]
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

        # dz_lat needs to be reset. Otherwise, once a lateral node erode's once, it will continue eroding
        # at every subsequent time setp. If you want to track all lateral erosion, create another attribute,
        # or add self.dzlat to itself after each time step.
        self._dzlat.fill(0.0)

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
        dp = dp_coef * (da ** dp_exp)

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
                [lat_node, inv_rad_curv, phd_inv_rad_curv] = node_and_phdcur_finder(grid, i, flowdirs, da)
                # node_finder returns the lateral node ID and the radius of curvature
                lat_nodes[i] = lat_node
                cur[i] = inv_rad_curv
                phd_cur[i] = phd_inv_rad_curv
                # if the lateral node is not 0 or -1 continue. lateral node may be
                # 0 or -1 if a boundary node was chosen as a lateral node. then
                # radius of curavature is also 0 so there is no lateral erosion
                if lat_node > 0:
                    # if the elevation of the lateral node is higher than primary node,
                    # calculate a new potential lateral erosion (L/T), which is negative
                    if z[lat_node] > z[i]:
                        R = 1/(phd_inv_rad_curv+epsilon)
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
                        voldiff = (z[i] + dp[i] - z[flowdirs[i]]) * grid.dx**2
                    # TB model: entire lat node must be eroded before lateral erosion
                    # occurs
                    if TB == 1:
                        voldiff = (z[lat_node] - z[flowdirs[i]]) * grid.dx**2
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
