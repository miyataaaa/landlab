import numpy as np
from landlab.grid import RasterModelGrid

cimport cython
cimport numpy as np

DTYPE_INT = int
ctypedef np.int32_t DTYPE_INT_t

DTYPE_LONG = long
ctypedef np.longlong_t DTYPE_LONG_t

DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t

cpdef DTYPE_INT_t test(DTYPE_INT_t i, 
                np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area):
    return 1

cdef extern from "math.h":
    DTYPE_FLOAT_t acos(DTYPE_FLOAT_t)

@cython.boundscheck(False)
cpdef inline DTYPE_FLOAT_t calculate_angle(DTYPE_FLOAT_t xd, DTYPE_FLOAT_t yd, 
                                           DTYPE_FLOAT_t xi, DTYPE_FLOAT_t yi, 
                                           DTYPE_FLOAT_t xr, DTYPE_FLOAT_t yr):
    cdef:
         DTYPE_FLOAT_t di_x, di_y, ir_x, ir_y, dot_product, di_length, ir_length, angle_cosine, angle_radian
    
    # ベクトルdiの成分を計算
    di_x = xi - xd
    di_y = yi - yd

    # ベクトルirの成分を計算
    ir_x = xr - xi
    ir_y = yr - yi

    # ベクトルdiとベクトルirの内積を計算
    dot_product = di_x * ir_x + di_y * ir_y

    # ベクトルdiとベクトルirのノルム(長さ)を計算
    di_length = (di_x * di_x + di_y * di_y)**0.5
    ir_length = (ir_x * ir_x + ir_y * ir_y)**0.5

    # ベクトルdiとベクトルirの成す角のcosineを計算
    angle_cosine = dot_product / (di_length * ir_length)

    # cosineから成す角のradianを計算
    angle_radian = acos(angle_cosine)

    return angle_radian

@cython.boundscheck(False)
cpdef inline DTYPE_INT_t find_upstream_nodes(DTYPE_INT_t i, 
                                    np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                    np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area) except -1:

    """
    ノードiの上流側のノードのindexを返す。
    上流側ノードが複数ある場合は、その中で最も流域面積が大きいノードのindexを返す。
    flowdirsの要素はint型で、ノードiの流向方向のノードのindexを表す。
    drain_areaの要素はfloat型で、ノードiの流域面積を表す。
    """
    cdef:
        DTYPE_INT_t donor, ran_num, maxinfln
        DTYPE_FLOAT_t drmax
        np.ndarray[DTYPE_INT_t, ndim=1] ups_nodes 
        np.ndarray[DTYPE_FLOAT_t, ndim=1] drin
        np.ndarray[DTYPE_INT_t, ndim=1] maxinfl

    ups_nodes = np.where(flowdirs == i)[0].astype(np.int32) # 型に注意。ファイル先頭で定義したDTYPE_INT_tを使う。

    # if there are more than 1 donors, find the one with largest drainage area
    if len(ups_nodes) > 1:
        
        drin = drain_area[ups_nodes]
        drmax = max(drin)
        maxinfl = ups_nodes[np.where(drin == drmax)]

        # if donor nodes have same drainage area, choose one randomly
        if len(maxinfl) > 1:
            ran_num = np.random.randint(0, len(maxinfl))
            maxinfln = maxinfl[ran_num]
            donor = maxinfln
        else:
            donor = maxinfl

        # if inflow is empty, no donor
    elif len(ups_nodes) == 0:
        donor = i
    # else donor is the only one
    else:
        donor = ups_nodes[0]

    return donor  # Return the first element (int) as the result

@cython.boundscheck(False)
cpdef inline DTYPE_INT_t find_n_upstream_nodes(DTYPE_INT_t i, 
                                      np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                      np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area, 
                                      DTYPE_INT_t  n) except -1:
    cdef DTYPE_INT_t  j = 1
    cdef DTYPE_INT_t target_node = i
    cdef DTYPE_INT_t doner

    while j <= n:
        doner = find_upstream_nodes(target_node, flowdirs, drain_area)

        if doner == target_node:
            break
        else:
            target_node = doner
        j += 1

    return doner
    
@cython.boundscheck(False)
cpdef inline DTYPE_INT_t find_n_downstream_nodes(DTYPE_INT_t i, 
                                                 np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                                 DTYPE_INT_t n) except -1:
    cdef DTYPE_INT_t  j = 1
    cdef DTYPE_INT_t target_node = i
    cdef DTYPE_INT_t receiver

    while j <= n:
        receiver = flowdirs[target_node]

        if receiver == target_node:
            break
        else:
            target_node = receiver
        j += 1

    return receiver

@cython.boundscheck(False)
cpdef inline DTYPE_INT_t is_inside_triangle_and_is_triangle(DTYPE_FLOAT_t xp, DTYPE_FLOAT_t yp, 
                                                    DTYPE_FLOAT_t xa, DTYPE_FLOAT_t ya, 
                                                    DTYPE_FLOAT_t xb, DTYPE_FLOAT_t yb, 
                                                    DTYPE_FLOAT_t xc, DTYPE_FLOAT_t yc):
    # ベクトルAB, BC, CAを計算
    cdef DTYPE_FLOAT_t vector_ab_x = xb - xa
    cdef DTYPE_FLOAT_t vector_ab_y = yb - ya
    cdef DTYPE_FLOAT_t vector_bc_x = xc - xb
    cdef DTYPE_FLOAT_t vector_bc_y = yc - yb
    cdef DTYPE_FLOAT_t vector_ca_x = xa - xc
    cdef DTYPE_FLOAT_t vector_ca_y = ya - yc

    # ベクトルAP, BP, CPを計算
    cdef DTYPE_FLOAT_t vector_ap_x = xp - xa
    cdef DTYPE_FLOAT_t vector_ap_y = yp - ya
    cdef DTYPE_FLOAT_t vector_bp_x = xp - xb
    cdef DTYPE_FLOAT_t vector_bp_y = yp - yb
    cdef DTYPE_FLOAT_t vector_cp_x = xp - xc
    cdef DTYPE_FLOAT_t vector_cp_y = yp - yc

    # 外積を計算
    cdef DTYPE_FLOAT_t cross_product_ab_ap = vector_ab_x * vector_ap_y - vector_ab_y * vector_ap_x
    cdef DTYPE_FLOAT_t cross_product_bc_bp = vector_bc_x * vector_bp_y - vector_bc_y * vector_bp_x
    cdef DTYPE_FLOAT_t cross_product_ca_cp = vector_ca_x * vector_cp_y - vector_ca_y * vector_cp_x

    # 外積の結果が0の場合、三角形ABCは直線上にある
    if cross_product_ab_ap == 0 and cross_product_bc_bp == 0 and cross_product_ca_cp == 0:
        return -1

    # 外積の符号を確認して点Pが三角形ABCの内部にあるか外部にあるかを判定
    if (cross_product_ab_ap >= 0 and cross_product_bc_bp >= 0 and cross_product_ca_cp >= 0) or (cross_product_ab_ap <= 0 and cross_product_bc_bp <= 0 and cross_product_ca_cp <= 0):
        return 1  # 点Pは三角形ABCの内部にある
    else:
        return 0  # 点Pは三角形ABCの外部にある

@cython.boundscheck(False)
cpdef inline DTYPE_INT_t point_position_relative_to_line(DTYPE_FLOAT_t xp, DTYPE_FLOAT_t yp, 
                                                 DTYPE_FLOAT_t xa, DTYPE_FLOAT_t ya, 
                                                 DTYPE_FLOAT_t xb, DTYPE_FLOAT_t yb):
    cdef DTYPE_FLOAT_t vector_ap_x = xp - xa
    cdef DTYPE_FLOAT_t vector_ap_y = yp - ya
    cdef DTYPE_FLOAT_t vector_bp_x = xp - xb
    cdef DTYPE_FLOAT_t vector_bp_y = yp - yb
    cdef DTYPE_FLOAT_t vector_ab_x = xb - xa
    cdef DTYPE_FLOAT_t vector_ab_y = yb - ya

    cdef DTYPE_FLOAT_t cross_product_ab_ap = vector_ab_x * vector_ap_y - vector_ab_y * vector_ap_x
    cdef DTYPE_FLOAT_t cross_product_ab_bp = vector_ab_x * vector_bp_y - vector_ab_y * vector_bp_x

    if cross_product_ab_ap > 0 and cross_product_ab_bp > 0:
        return -1  # 点Pは直線ABの左側にある
    elif cross_product_ab_ap < 0 and cross_product_ab_bp < 0:
        return 1  # 点Pは直線ABの右側にある
    else:
        return 0  # 点Pは直線AB上にある

@cython.boundscheck(False)
cpdef tuple node_finder_use_fivebyfive_window(grid, 
                                              DTYPE_INT_t i, 
                                              np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                              np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area, 
                                              bint is_get_phd_cur=False):
    
    cdef list lat_nodes = []
    cdef DTYPE_FLOAT_t radcurv = 0.0
    cdef DTYPE_FLOAT_t phd_radcurv = 0.0
    
    cdef DTYPE_INT_t  n = 2
    cdef DTYPE_INT_t  donor = find_n_upstream_nodes(i, flowdirs, drain_area, n)
    cdef DTYPE_INT_t  receiver = find_n_downstream_nodes(i, flowdirs, n)

    if donor == i or receiver == i:
        return lat_nodes, radcurv, phd_radcurv

    cdef DTYPE_INT_t  side_flag = np.random.choice([-1, 1])
    cdef DTYPE_FLOAT_t x_don = grid.x_of_node[donor]
    cdef DTYPE_FLOAT_t y_don = grid.y_of_node[donor]
    cdef DTYPE_FLOAT_t x_i = grid.x_of_node[i]
    cdef DTYPE_FLOAT_t y_i = grid.y_of_node[i]
    cdef DTYPE_FLOAT_t x_rec = grid.x_of_node[receiver]
    cdef DTYPE_FLOAT_t y_rec = grid.y_of_node[receiver]

    cdef np.ndarray[DTYPE_INT_t, ndim=1] neighbors = grid.active_adjacent_nodes_at_node[i]

    cdef list temp_lat_nodes = []

    cdef j = 0
    cdef neig = 0
    cdef:
        DTYPE_FLOAT_t x_neig, y_neig
        DTYPE_INT_t  is_triangle_and_in_triangle, side_of_line
    for j in range(len(neighbors)):
        neig = neighbors[j]
        x_neig = grid.x_of_node[neig]
        y_neig = grid.y_of_node[neig]
        is_triangle_and_in_triangle = is_inside_triangle_and_is_triangle(x_neig, y_neig, x_don, y_don, x_i, y_i, x_rec, y_rec)
        side_of_line = -1

        if is_triangle_and_in_triangle == 0:
            side_of_line = point_position_relative_to_line(x_neig, y_neig, x_don, y_don, x_rec, y_rec)
            if side_of_line == side_flag:
                temp_lat_nodes.append(neig)
        elif is_triangle_and_in_triangle == -1:
            temp_lat_nodes.append(neig)

    lat_nodes = temp_lat_nodes

    cdef DTYPE_FLOAT_t angle = calculate_angle(x_don, y_don, x_i, y_i, x_rec, y_rec)
    
    if np.isclose(angle, 0.0) or np.isclose(angle, np.pi):
        angle = np.deg2rad(16.8)
    else:
        angle = np.pi - angle
    
    cdef DTYPE_FLOAT_t ds = np.hypot(x_don - x_i, y_don - y_i) + np.hypot(x_i - x_rec, y_i - y_rec)
    cdef DTYPE_FLOAT_t d_donor_receiver = np.hypot(x_don - x_rec, y_don - y_rec)
    ds = (ds + d_donor_receiver) * 0.5
    radcurv = angle / ds
    
    if is_get_phd_cur:
        donor = find_upstream_nodes(i, flowdirs, drain_area)
        cur = grid.at_node["curvature"]
        phd_radcurv = cur[donor]

    cdef tuple result = (lat_nodes, radcurv, phd_radcurv)

    return result

# ここから再開
cpdef void _run_one_step_fivebyfive_window(_self, DTYPE_FLOAT_t dt):
    
    grid = _self.grid
    UC = _self._UC
    TB = _self._TB
    inlet_on = _self._inlet_on  # this is a true/false flag
    Kv = _self._Kv
    qs_in = _self._qs_in
    dzdt = _self._dzdt
    alph = _self._alph
    vol_lat = grid.at_node["volume__lateral_erosion"]

    dp_coef = _self._dp_coef
    dp_exp = _self._dp_exp
    kw = _self._wid_coef
    F = _self._F
    thresh_da = _self._thresh_da

    # phd_node_num = 1
    cur = grid.at_node["curvature"]
    phd_cur = grid.at_node["phd_curvature"]

    fai_alpha = _self._fai_alpha
    fai_beta = _self._fai_beta
    fai_gamma = _self._fai_gamma
    fai_C = _self._fai_C

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

    # dz_lat needs to be reset. Otherwise, once a lateral node erode's once, it will continue eroding
    # at every subsequent time setp. If you want to track all lateral erosion, create another attribute,
    # or add _self.dzlat to itself after each time step.
    _self._dzlat.fill(0.0)

    # critical_erosion_volume_ratio 
    critical_erosion_volume_ratio = _self._critical_erosion_volume_ratio

    if inlet_on is True:
        inlet_node = _self._inlet_node
        qsinlet = _self._qsinlet
        qs_in[inlet_node] = qsinlet
        q = grid.at_node["surface_water__discharge"]
        da = q / grid.dx**2
    # if inlet flag is not on, proceed as normal.
    else:
        if _self._use_Q:
            # water discharge is calculated by flow router
            da = grid.at_node["surface_water__discharge"]
        else:
            # drainage area is calculated by flow router
            da = grid.at_node["drainage_area"]

    # add min_Q_or_da
    da += _self._add_min_Q_or_da

    # water depth in meters, needed for lateral erosion calc
    dp = dp_coef * (da ** dp_exp)

    # add min_Q_or_da
    da += _self._add_min_Q_or_da
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
    lat_nodes = [i for i in range(grid.shape[0]*grid.shape[1])]
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
            lat_nodes_at_i, inv_rad_curv, phd_inv_rad_curv = node_finder_use_C(grid, i, flowdirs, da, is_get_phd_cur=True)
            # lat_nodes_at_i, inv_rad_curv, phd_inv_rad_curv = node_finder_use_fivebyfive_window(grid, i, flowdirs, da, is_get_phd_cur=True)
            # print(f"lat_nodes_at_i: {lat_nodes_at_i}")
            # node_finder returns the lateral node ID and the radius of curvature
            if len(lat_nodes_at_i) > 0:
                lat_nodes[i] = lat_nodes_at_i
            cur[i] = inv_rad_curv
            phd_cur[i] = phd_inv_rad_curv
            # if the lateral node is not 0 or -1 continue. lateral node may be
            # 0 or -1 if a boundary node was chosen as a lateral node. then
            # radius of curavature is also 0 so there is no lateral erosion
            R = 1/(phd_inv_rad_curv+epsilon)
            S = np.clip(max_slopes[i], 1e-8, None)
            fai[i] = np.exp(fai_C) * (da[i]**fai_alpha) * (S**fai_beta) * (R**fai_gamma)
            petlat = fai[i] * ero
            El[i] = petlat
            for lat_node in lat_nodes_at_i:
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
        # else:
        #     lat_nodes[i] = [0]
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
    # print(f"len(lat_nodes): {len(lat_nodes)}, len(dwns_nodes): {len(dwnst_nodes)}")
    for i in dwnst_nodes:
        lat_nodes_at_i = lat_nodes[i]
        if isinstance(lat_nodes_at_i, list):
            for lat_node in lat_nodes_at_i:
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
                            _self._dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                            # after the lateral node is eroded, reset its volume eroded to
                            # zero
                            vol_lat[lat_node] = 0.0
    # combine vertical and lateral erosion.
    dz = dzdt + _self._dzlat
    # change height of landscape
    z[:] += dz
    return grid, _self._dzlat
