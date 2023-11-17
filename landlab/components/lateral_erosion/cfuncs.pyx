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

DTYPE_COMPLEX = np.cdouble
#ctypedef np.double_complex_t DTYPE_COMPLEX_t

cpdef DTYPE_INT_t test(DTYPE_INT_t i, 
                np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area):
    return 1

cdef extern from "math.h":
    DTYPE_FLOAT_t acos(DTYPE_FLOAT_t)

@cython.boundscheck(False)
@cython.wraparound(False)
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
@cython.wraparound(False)
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
@cython.wraparound(False)
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
@cython.wraparound(False)
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
@cython.wraparound(False)
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
    cdef DTYPE_FLOAT_t vector_ac_x = xc - xa
    cdef DTYPE_FLOAT_t vector_ac_y = yc - ya

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
    cdef DTYPE_FLOAT_t cross_product_ab_ac = vector_ab_x * vector_ac_y - vector_ab_y * vector_ac_x

    # 外積の結果が0の場合、三角形ABCは直線上にある
    if np.isclose(cross_product_ab_ac, 0.0):
        return -1

    # 外積の符号を確認して点Pが三角形ABCの内部にあるか外部にあるかを判定
    if (cross_product_ab_ap >= 0 and cross_product_bc_bp >= 0 and cross_product_ca_cp >= 0) or (cross_product_ab_ap <= 0 and cross_product_bc_bp <= 0 and cross_product_ca_cp <= 0):
        return 1  # 点Pは三角形ABCの内部にある
    else:
        return 0  # 点Pは三角形ABCの外部にある

@cython.boundscheck(False)
@cython.wraparound(False)
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
    #cdef DTYPE_FLOAT_t cross_product_ab_bp = vector_ab_x * vector_bp_y - vector_ab_y * vector_bp_x

    if cross_product_ab_ap < 0 :#and cross_product_ab_bp > 0:
        return -1  # 点Pは直線ABの左側にある
    elif cross_product_ab_ap > 0:# and cross_product_ab_bp < 0:
        return 1  # 点Pは直線ABの右側にある
    else:
        return 0  # 点Pは直線AB上にある

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline tuple node_finder_use_fivebyfive_window(grid, 
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

    cdef DTYPE_INT_t j = 0
    cdef DTYPE_INT_t neig = 0
    cdef:
        DTYPE_FLOAT_t x_neig, y_neig
        DTYPE_INT_t  is_triangle_and_in_triangle, side_of_line_i, side_of_line_neig
    for j in range(len(neighbors)):
        neig = neighbors[j]
        x_neig = grid.x_of_node[neig]
        y_neig = grid.y_of_node[neig]
        is_triangle_and_in_triangle = is_inside_triangle_and_is_triangle(x_neig, y_neig, x_don, y_don, x_i, y_i, x_rec, y_rec)
        side_of_line_i = point_position_relative_to_line(x_i, y_i, x_don, y_don, x_rec, y_rec)
        side_of_line_neig = point_position_relative_to_line(x_neig, y_neig, x_don, y_don, x_rec, y_rec)
        
        # -1: donor, i, receiverが１直線上にある場合(三角形を構成しない)
        # 1: donor, i, receiverが三角形を構成しており、neigが三角形の内部にある場合
        # 0: donor, i, receiverが三角形を構成しており、neigが三角形の外部にある場合
        if is_triangle_and_in_triangle == -1:
            if side_of_line_neig == side_flag:
                temp_lat_nodes.append(neig)
        elif (is_triangle_and_in_triangle == 0) and (side_of_line_i == side_of_line_neig):
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline void _run_one_step_fivebyfive_window(
                                           grid,
                                           np.ndarray[DTYPE_INT_t, ndim=1] dwnst_nodes,
                                           np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] z, 
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] da,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dp,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] Kv,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] max_slopes,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzver,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] cur,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] phd_cur,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] El,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] fai,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] vol_lat,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzlat,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzdt,
                                           DTYPE_FLOAT_t dx,
                                           double fai_alpha,
                                           double fai_beta,
                                           double fai_gamma,
                                           double fai_C,
                                           double dt,
                                           double critical_erosion_volume_ratio,
                                           bint UC,
                                           bint TB,
                                           ):

    cdef:
         DTYPE_INT_t i, j, k, node_num_at_i, lat_node
         DTYPE_INT_t iterNum = len(dwnst_nodes)
         DTYPE_INT_t nodeNum = grid.shape[0]*grid.shape[1]
         DTYPE_FLOAT_t inv_rad_curv, phd_inv_rad_curv, R, S, ero
         DTYPE_FLOAT_t petlat = 0. # potential lateral erosion initially set to 0
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] vol_lat_dt = np.zeros(grid.number_of_nodes)
    cdef double epsilon = 1e-10
    cdef int dummy = -99
    cdef list lat_nodes_at_i = []
    cdef list lat_nodes = [[dummy] for i in range(nodeNum)]

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
            lat_nodes_at_i, inv_rad_curv, phd_inv_rad_curv = node_finder_use_fivebyfive_window(grid, i, flowdirs, da, is_get_phd_cur=True)

            if len(lat_nodes_at_i) > 0:
                lat_nodes[i] = lat_nodes_at_i
            cur[i] = inv_rad_curv
            phd_cur[i] = phd_inv_rad_curv
            # if the lateral node is not 0 or -1 continue. lateral node may be
            # 0 or -1 if a boundary node was chosen as a lateral node. then
            # radius of curavature is also 0 so there is no lateral erosion
            R = 1/(phd_inv_rad_curv+epsilon)
            fai[i] = np.exp(fai_C) * (da[i]**fai_alpha) * (S**fai_beta) * (R**fai_gamma)
            petlat = fai[i] * ero
            El[i] = petlat
            node_num_at_i = len(lat_nodes_at_i)

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
                        vol_lat_dt[lat_node] += abs(petlat) * dx * dp[i]
                        # wd? may be H is true. how calc H ? 

    dzdt[:] = dzver * dt
    vol_lat[:] += vol_lat_dt * dt
    # this loop determines if enough lateral erosion has happened to change
    # the height of the neighbor node.
    # print(f"len(lat_nodes): {len(lat_nodes)}, len(dwns_nodes): {len(dwnst_nodes)}")
    for j in range(iterNum):
        i = dwnst_nodes[j]
        lat_nodes_at_i = lat_nodes[i]

        if lat_nodes_at_i[0] != dummy:
            node_num_at_i = len(lat_nodes_at_i)

            for k in range(node_num_at_i):
                lat_node = lat_nodes_at_i[k]
                if lat_node > 0:  # greater than zero now bc inactive neighbors are value -1
                    if z[lat_node] > z[i]:
                        # vol_diff is the volume that must be eroded from lat_node so that its
                        # elevation is the same as node downstream of primary node
                        # UC model: this would represent undercutting (the water height at
                        # node i), slumping, and instant removal.
                        if UC:
                            voldiff = critical_erosion_volume_ratio * (z[i] + dp[i] - z[flowdirs[i]]) * dx**2 
                        # TB model: entire lat node must be eroded before lateral erosion
                        # occurs
                        if TB:
                            voldiff = critical_erosion_volume_ratio * (z[lat_node] - z[flowdirs[i]]) * dx**2
                        # if the total volume eroded from lat_node is greater than the volume
                        # needed to be removed to make node equal elevation,
                        # then instantaneously remove this height from lat node. already has
                        # timestep in it
                        if vol_lat[lat_node] >= voldiff:
                            dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                            # after the lateral node is eroded, reset its volume eroded to
                            # zero
                            vol_lat[lat_node] = 0.0
    # combine vertical and lateral erosion.
    dz = dzdt + dzlat
    # change height of landscape
    z[:] += dz


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline tuple node_finder_use_fivebyfive_window_ver2(grid, 
                                                          DTYPE_INT_t i, 
                                                          np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                                          np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area, 
                                                          bint is_get_phd_cur=False,
                                                          DTYPE_INT_t dummy_value=-99):

    # Ver1とVer2でセルの選定規則は変わらないが、
    # Ver2では、lat_nodeをnp.ndarrayを使って実装している。
    # Ver1ではリストを使って実装しているので、Ver2の方が実行速度が速い  
    
    #cdef list lat_nodes = []
    cdef np.ndarray[DTYPE_INT_t, ndim=1] lat_nodes = np.full(shape=(4), fill_value=dummy_value)

    lat_nodes = lat_nodes.astype(dtype=np.int32)

    cdef DTYPE_FLOAT_t radcurv = 0.0
    cdef DTYPE_FLOAT_t phd_radcurv = 0.0
    
    cdef DTYPE_INT_t  n = 2
    cdef DTYPE_INT_t  donor = find_n_upstream_nodes(i, flowdirs, drain_area, n)
    cdef DTYPE_INT_t  receiver = find_n_downstream_nodes(i, flowdirs, n)

    #cdef np.ndarray[DTYPE_INT_t, ndim=1] donors = np.where(flowdirs == i)[0].astype(np.int32)
    
    #donors = donors.astype(dtype=np.int32) # nodes that flow into node i
    cdef DTYPE_INT_t  donor_one_ups = find_n_upstream_nodes(i, flowdirs, drain_area, 1)

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

    #cdef list temp_lat_nodes = []
    #cdef np.ndarray[DTYPE_INT_t, ndim=2] temp_lat_nodes = np.full(shape=(4), fill_value=dummy_value)

    cdef DTYPE_INT_t j = 0
    cdef DTYPE_INT_t k = 0
    cdef DTYPE_INT_t neig = 0
    cdef:
        DTYPE_FLOAT_t x_neig, y_neig
        DTYPE_INT_t  is_triangle_and_in_triangle, side_of_line_i, side_of_line_neig
    for j in range(len(neighbors)):
        neig = neighbors[j]

        if (neig == donor_one_ups) or (neig == -1) or (neig == i) or (neig == donor) or (neig == receiver):
            continue
        else:
            x_neig = grid.x_of_node[neig]
            y_neig = grid.y_of_node[neig]
            is_triangle_and_in_triangle = is_inside_triangle_and_is_triangle(x_neig, y_neig, x_don, y_don, x_i, y_i, x_rec, y_rec)
            side_of_line_i = point_position_relative_to_line(x_i, y_i, x_don, y_don, x_rec, y_rec)
            side_of_line_neig = point_position_relative_to_line(x_neig, y_neig, x_don, y_don, x_rec, y_rec)
            
            # -1: donor, i, receiverが１直線上にある場合(三角形を構成しない)
            # 1: donor, i, receiverが三角形を構成しており、neigが三角形の内部にある場合
            # 0: donor, i, receiverが三角形を構成しており、neigが三角形の外部にある場合
            if is_triangle_and_in_triangle == -1:
                if side_of_line_neig == side_flag:
                    #temp_lat_nodes.append(neig)
                    lat_nodes[k] = neig
                    k += 1
            elif (is_triangle_and_in_triangle == 0) and (side_of_line_i == side_of_line_neig):
                #temp_lat_nodes.append(neig)
                lat_nodes[k] = neig
                k += 1

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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline void _run_one_step_fivebyfive_window_ver2(
                                           grid,
                                           np.ndarray[DTYPE_INT_t, ndim=1] dwnst_nodes,
                                           np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] z, 
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] da,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dp,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] Kv,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] max_slopes,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzver,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] cur,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] phd_cur,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] El,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] fai,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] vol_lat,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzlat,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzdt,
                                           DTYPE_FLOAT_t dx,
                                           double fai_alpha,
                                           double fai_beta,
                                           double fai_gamma,
                                           double fai_C,
                                           double dt,
                                           double critical_erosion_volume_ratio,
                                           bint UC,
                                           bint TB,
                                           ):

    cdef:
         DTYPE_INT_t i, j, k, node_num_at_i, lat_node
         DTYPE_INT_t iterNum = len(dwnst_nodes)
         DTYPE_INT_t nodeNum = grid.shape[0]*grid.shape[1]
         DTYPE_FLOAT_t inv_rad_curv, phd_inv_rad_curv, R, S, ero
         DTYPE_FLOAT_t petlat = 0. # potential lateral erosion initially set to 0
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] vol_lat_dt = np.zeros(grid.number_of_nodes)
    cdef double epsilon = 1e-10
    cdef int dummy = -99
    #cdef list lat_nodes_at_i = []
    #cdef list lat_nodes = [[dummy] for i in range(nodeNum)]

    cdef DTYPE_INT_t dummy_value = -99
    cdef np.ndarray[DTYPE_INT_t, ndim=1] lat_nodes_at_i = np.full(shape=(4), fill_value=dummy_value)
    lat_nodes_at_i = lat_nodes_at_i.astype(dtype=np.int32)
    cdef np.ndarray[DTYPE_INT_t, ndim=2] lat_nodes = np.full(shape=(nodeNum, 4), fill_value=dummy_value)
    lat_nodes = lat_nodes.astype(dtype=np.int32)
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

            #if len(lat_nodes_at_i) > 0:
                #lat_nodes[i] = lat_nodes_at_i
            lat_nodes[i] = lat_nodes_at_i
            cur[i] = inv_rad_curv
            phd_cur[i] = phd_inv_rad_curv
            # if the lateral node is not 0 or -1 continue. lateral node may be
            # 0 or -1 if a boundary node was chosen as a lateral node. then
            # radius of curavature is also 0 so there is no lateral erosion
            R = 1/(phd_inv_rad_curv+epsilon)
            fai[i] = np.exp(fai_C) * (da[i]**fai_alpha) * (S**fai_beta) * (R**fai_gamma)
            petlat = fai[i] * ero
            El[i] = petlat
            node_num_at_i = len(np.where(lat_nodes_at_i != dummy_value)[0])

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
                        vol_lat_dt[lat_node] += abs(petlat) * dx * dp[i]
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
                        voldiff = critical_erosion_volume_ratio * (z[i] + dp[i] - z[flowdirs[i]]) * dx**2 
                    # TB model: entire lat node must be eroded before lateral erosion
                    # occurs
                    if TB:
                        voldiff = critical_erosion_volume_ratio * (z[lat_node] - z[flowdirs[i]]) * dx**2
                    # if the total volume eroded from lat_node is greater than the volume
                    # needed to be removed to make node equal elevation,
                    # then instantaneously remove this height from lat node. already has
                    # timestep in it
                    if vol_lat[lat_node] >= voldiff:
                        dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                        # after the lateral node is eroded, reset its volume eroded to
                        # zero
                        vol_lat[lat_node] = 0.0
    # combine vertical and lateral erosion.
    dz = dzdt + dzlat
    # change height of landscape
    z[:] += dz

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline tuple node_finder_use_fivebyfive_window_diag(grid, 
                                                          DTYPE_INT_t i, 
                                                          np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                                          np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area, 
                                                          bint is_get_phd_cur=False,
                                                          DTYPE_INT_t dummy_value=-99):
    
    #cdef list lat_nodes = []
    cdef np.ndarray[DTYPE_INT_t, ndim=1] lat_nodes = np.full(shape=(8), fill_value=dummy_value)

    lat_nodes = lat_nodes.astype(dtype=np.int32)

    cdef DTYPE_FLOAT_t radcurv = 0.0
    cdef DTYPE_FLOAT_t phd_radcurv = 0.0
    
    cdef DTYPE_INT_t  n = 2
    cdef DTYPE_INT_t  donor = find_n_upstream_nodes(i, flowdirs, drain_area, n)
    cdef DTYPE_INT_t  receiver = find_n_downstream_nodes(i, flowdirs, n)

    cdef np.ndarray[DTYPE_INT_t, ndim=1] donors = np.where(flowdirs == i)[0].astype(np.int32)
    
    donors = donors.astype(dtype=np.int32) # nodes that flow into node i

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
    cdef np.ndarray[DTYPE_INT_t, ndim=1] neighbors_diag = grid.diagonal_adjacent_nodes_at_node[i]

    cdef total_neighbors = np.concatenate((neighbors, neighbors_diag))

    #cdef list temp_lat_nodes = []
    #cdef np.ndarray[DTYPE_INT_t, ndim=2] temp_lat_nodes = np.full(shape=(4), fill_value=dummy_value)

    cdef DTYPE_INT_t j = 0
    cdef DTYPE_INT_t k = 0
    cdef DTYPE_INT_t neig = 0
    cdef:
        DTYPE_FLOAT_t x_neig, y_neig
        DTYPE_INT_t  is_triangle_and_in_triangle, side_of_line_i, side_of_line_neig
    for j in range(len(total_neighbors)):
        neig = total_neighbors[j]

        if neig in donors:
            continue
        else:
            x_neig = grid.x_of_node[neig]
            y_neig = grid.y_of_node[neig]
            is_triangle_and_in_triangle = is_inside_triangle_and_is_triangle(x_neig, y_neig, x_don, y_don, x_i, y_i, x_rec, y_rec)
            side_of_line_i = point_position_relative_to_line(x_i, y_i, x_don, y_don, x_rec, y_rec)
            side_of_line_neig = point_position_relative_to_line(x_neig, y_neig, x_don, y_don, x_rec, y_rec)
            
            # -1: donor, i, receiverが１直線上にある場合(三角形を構成しない)
            # 1: donor, i, receiverが三角形を構成しており、neigが三角形の内部にある場合
            # 0: donor, i, receiverが三角形を構成しており、neigが三角形の外部にある場合
            if is_triangle_and_in_triangle == -1:
                if side_of_line_neig == side_flag:
                    #temp_lat_nodes.append(neig)
                    lat_nodes[k] = neig
                    k += 1
            elif (is_triangle_and_in_triangle == 0) and (side_of_line_i == side_of_line_neig):
                #temp_lat_nodes.append(neig)
                lat_nodes[k] = neig
                k += 1

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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline void _run_one_step_fivebyfive_window_ver3(
                                           grid,
                                           np.ndarray[DTYPE_INT_t, ndim=1] dwnst_nodes,
                                           np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] z, 
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] da,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dp,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] Kv,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] max_slopes,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzver,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] cur,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] phd_cur,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] El,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] fai,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] vol_lat,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzlat,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzdt,
                                           DTYPE_FLOAT_t dx,
                                           double fai_alpha,
                                           double fai_beta,
                                           double fai_gamma,
                                           double fai_C,
                                           double dt,
                                           double critical_erosion_volume_ratio,
                                           bint UC,
                                           bint TB,
                                           ):

    # Ver3では、側方侵食セルの選定対象に対角線上のセルも含めている

    cdef:
         DTYPE_INT_t i, j, k, node_num_at_i, lat_node
         DTYPE_INT_t iterNum = len(dwnst_nodes)
         DTYPE_INT_t nodeNum = grid.shape[0]*grid.shape[1]
         DTYPE_FLOAT_t inv_rad_curv, phd_inv_rad_curv, R, S, ero
         DTYPE_FLOAT_t petlat = 0. # potential lateral erosion initially set to 0
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] vol_lat_dt = np.zeros(grid.number_of_nodes)
    cdef double epsilon = 1e-10
    cdef int dummy = -99
    #cdef list lat_nodes_at_i = []
    #cdef list lat_nodes = [[dummy] for i in range(nodeNum)]

    cdef DTYPE_INT_t dummy_value = -99
    cdef np.ndarray[DTYPE_INT_t, ndim=1] lat_nodes_at_i = np.full(shape=(8), fill_value=dummy_value).astype(dtype=np.int32)
    cdef np.ndarray[DTYPE_INT_t, ndim=2] lat_nodes = np.full(shape=(nodeNum, 8), fill_value=dummy_value).astype(dtype=np.int32)

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
            lat_nodes_at_i, inv_rad_curv, phd_inv_rad_curv = node_finder_use_fivebyfive_window_diag(grid, 
                                                                                                    i, 
                                                                                                    flowdirs, 
                                                                                                    da, 
                                                                                                    is_get_phd_cur=True,
                                                                                                    dummy_value=dummy_value)

            #if len(lat_nodes_at_i) > 0:
                #lat_nodes[i] = lat_nodes_at_i
            lat_nodes[i] = lat_nodes_at_i
            cur[i] = inv_rad_curv
            phd_cur[i] = phd_inv_rad_curv
            # if the lateral node is not 0 or -1 continue. lateral node may be
            # 0 or -1 if a boundary node was chosen as a lateral node. then
            # radius of curavature is also 0 so there is no lateral erosion
            R = 1/(phd_inv_rad_curv+epsilon)
            fai[i] = np.exp(fai_C) * (da[i]**fai_alpha) * (S**fai_beta) * (R**fai_gamma)
            petlat = fai[i] * ero
            El[i] = petlat
            node_num_at_i = len(np.where(lat_nodes_at_i != dummy_value)[0])

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
                        vol_lat_dt[lat_node] += abs(petlat) * dx * dp[i]
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
                        voldiff = critical_erosion_volume_ratio * (z[i] + dp[i] - z[flowdirs[i]]) * dx**2 
                    # TB model: entire lat node must be eroded before lateral erosion
                    # occurs
                    if TB:
                        voldiff = critical_erosion_volume_ratio * (z[lat_node] - z[flowdirs[i]]) * dx**2
                    # if the total volume eroded from lat_node is greater than the volume
                    # needed to be removed to make node equal elevation,
                    # then instantaneously remove this height from lat node. already has
                    # timestep in it
                    if vol_lat[lat_node] >= voldiff:
                        dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                        # after the lateral node is eroded, reset its volume eroded to
                        # zero
                        vol_lat[lat_node] = 0.0
    # combine vertical and lateral erosion.
    dz = dzdt + dzlat
    # change height of landscape
    z[:] += dz

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline tuple node_finder_use_fivebyfive_window_only_hill(grid, 
                                                               DTYPE_INT_t i, 
                                                               np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                                               np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area, 
                                                               np.ndarray[DTYPE_INT_t, ndim=1] dwnst_nodes,
                                                               bint is_get_phd_cur=False,
                                                               DTYPE_INT_t dummy_value=-99,
                                                               ):

    # node_finder_use_fivebyfive_window_only_hillはver2の派生形で、
    # 側方侵食対象ノードに河川ノードを含めず、ヒルスロープ領域のノードだけを含める
    # dwnst_nodesは河川ノードが格納された配列
    
    cdef np.ndarray[DTYPE_INT_t, ndim=1] lat_nodes = np.full(shape=(4), fill_value=dummy_value)

    lat_nodes = lat_nodes.astype(dtype=np.int32)

    cdef DTYPE_FLOAT_t radcurv = 0.0
    cdef DTYPE_FLOAT_t phd_radcurv = 0.0
    
    cdef DTYPE_INT_t  n = 2
    cdef DTYPE_INT_t  donor = find_n_upstream_nodes(i, flowdirs, drain_area, n)
    cdef DTYPE_INT_t  receiver = find_n_downstream_nodes(i, flowdirs, n)

    #cdef np.ndarray[DTYPE_INT_t, ndim=1] donors = np.where(flowdirs == i)[0].astype(np.int32)
    
    #donors = donors.astype(dtype=np.int32) # nodes that flow into node i
    cdef DTYPE_INT_t  donor_one_ups = find_n_upstream_nodes(i, flowdirs, drain_area, 1)

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

    #cdef list temp_lat_nodes = []
    #cdef np.ndarray[DTYPE_INT_t, ndim=2] temp_lat_nodes = np.full(shape=(4), fill_value=dummy_value)

    cdef DTYPE_INT_t j = 0
    cdef DTYPE_INT_t k = 0
    cdef DTYPE_INT_t neig = 0
    cdef:
        DTYPE_FLOAT_t x_neig, y_neig
        DTYPE_INT_t  is_triangle_and_in_triangle, side_of_line_i, side_of_line_neig
    for j in range(len(neighbors)):
        neig = neighbors[j]

        # ここを変更
        if neig in dwnst_nodes:
            continue
        else:
            x_neig = grid.x_of_node[neig]
            y_neig = grid.y_of_node[neig]
            is_triangle_and_in_triangle = is_inside_triangle_and_is_triangle(x_neig, y_neig, x_don, y_don, x_i, y_i, x_rec, y_rec)
            side_of_line_i = point_position_relative_to_line(x_i, y_i, x_don, y_don, x_rec, y_rec)
            side_of_line_neig = point_position_relative_to_line(x_neig, y_neig, x_don, y_don, x_rec, y_rec)
            
            # -1: donor, i, receiverが１直線上にある場合(三角形を構成しない)
            # 1: donor, i, receiverが三角形を構成しており、neigが三角形の内部にある場合
            # 0: donor, i, receiverが三角形を構成しており、neigが三角形の外部にある場合
            if is_triangle_and_in_triangle == -1:
                if side_of_line_neig == side_flag:
                    #temp_lat_nodes.append(neig)
                    lat_nodes[k] = neig
                    k += 1
            elif (is_triangle_and_in_triangle == 0) and (side_of_line_i == side_of_line_neig):
                #temp_lat_nodes.append(neig)
                lat_nodes[k] = neig
                k += 1

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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline void _run_one_step_fivebyfive_window_only_hill(
                                           grid,
                                           np.ndarray[DTYPE_INT_t, ndim=1] dwnst_nodes,
                                           np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] z, 
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] da,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dp,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] Kv,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] max_slopes,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzver,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] cur,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] phd_cur,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] El,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] fai,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] vol_lat,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzlat,
                                           np.ndarray[DTYPE_FLOAT_t, ndim=1] dzdt,
                                           DTYPE_FLOAT_t dx,
                                           double fai_alpha,
                                           double fai_beta,
                                           double fai_gamma,
                                           double fai_C,
                                           double dt,
                                           double critical_erosion_volume_ratio,
                                           bint UC,
                                           bint TB,
                                           ):

    # _run_one_step_fivebyfive_window_only_hillはver2の派生形で、
    # 側方侵食対象ノードに河川ノードを含めず、ヒルスロープ領域のノードだけを含める
    # 側方侵食対象ノードの選定アルゴリズムにはnode_finder_use_fivebyfive_window_only_hillを用いる

    cdef:
         DTYPE_INT_t i, j, k, node_num_at_i, lat_node
         DTYPE_INT_t iterNum = len(dwnst_nodes)
         DTYPE_INT_t nodeNum = grid.shape[0]*grid.shape[1]
         DTYPE_FLOAT_t inv_rad_curv, phd_inv_rad_curv, R, S, ero
         DTYPE_FLOAT_t petlat = 0. # potential lateral erosion initially set to 0
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] vol_lat_dt = np.zeros(grid.number_of_nodes)
    cdef double epsilon = 1e-10
    cdef int dummy = -99
    #cdef list lat_nodes_at_i = []
    #cdef list lat_nodes = [[dummy] for i in range(nodeNum)]

    cdef DTYPE_INT_t dummy_value = -99
    cdef np.ndarray[DTYPE_INT_t, ndim=1] lat_nodes_at_i = np.full(shape=(4), fill_value=dummy_value)
    lat_nodes_at_i = lat_nodes_at_i.astype(dtype=np.int32)
    cdef np.ndarray[DTYPE_INT_t, ndim=2] lat_nodes = np.full(shape=(nodeNum, 4), fill_value=dummy_value)
    lat_nodes = lat_nodes.astype(dtype=np.int32)
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
            lat_nodes_at_i, inv_rad_curv, phd_inv_rad_curv = node_finder_use_fivebyfive_window_only_hill(grid, 
                                                                                                        i, 
                                                                                                        flowdirs, 
                                                                                                        da, 
                                                                                                        dwnst_nodes,
                                                                                                        is_get_phd_cur=True,
                                                                                                        dummy_value=dummy_value,
                                                                                                        )

            #if len(lat_nodes_at_i) > 0:
                #lat_nodes[i] = lat_nodes_at_i
            lat_nodes[i] = lat_nodes_at_i
            cur[i] = inv_rad_curv
            phd_cur[i] = phd_inv_rad_curv
            # if the lateral node is not 0 or -1 continue. lateral node may be
            # 0 or -1 if a boundary node was chosen as a lateral node. then
            # radius of curavature is also 0 so there is no lateral erosion
            R = 1/(phd_inv_rad_curv+epsilon)
            fai[i] = np.exp(fai_C) * (da[i]**fai_alpha) * (S**fai_beta) * (R**fai_gamma)
            petlat = fai[i] * ero
            El[i] = petlat
            node_num_at_i = len(np.where(lat_nodes_at_i != dummy_value)[0])

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
                        vol_lat_dt[lat_node] += abs(petlat) * dx * dp[i]
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
                        voldiff = critical_erosion_volume_ratio * (z[i] + dp[i] - z[flowdirs[i]]) * dx**2 
                    # TB model: entire lat node must be eroded before lateral erosion
                    # occurs
                    if TB:
                        voldiff = critical_erosion_volume_ratio * (z[lat_node] - z[flowdirs[i]]) * dx**2
                    # if the total volume eroded from lat_node is greater than the volume
                    # needed to be removed to make node equal elevation,
                    # then instantaneously remove this height from lat node. already has
                    # timestep in it
                    if vol_lat[lat_node] >= voldiff:
                        dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                        # after the lateral node is eroded, reset its volume eroded to
                        # zero
                        vol_lat[lat_node] = 0.0
    # combine vertical and lateral erosion.
    dz = dzdt + dzlat
    # change height of landscape
    z[:] += dz