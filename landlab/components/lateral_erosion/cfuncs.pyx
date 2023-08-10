import numpy as np
from landlab.grid import RasterModelGrid

cimport cython
cimport numpy as np

DTYPE_INT = int
ctypedef np.int_t DTYPE_INT_t

DTYPE_LONG = long
ctypedef np.longlong_t DTYPE_LONG_t

DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t


cdef extern from "math.h":
    double acos(double)

cpdef inline double calculate_angle(double xd, double yd, double xi, double yi, double xr, double yr):
    cdef:
         double di_x, di_y, ir_x, ir_y, dot_product, di_length, ir_length, angle_cosine, angle_radian
    
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

cpdef int test(int i, 
                np.ndarray[DTYPE_LONG_t, ndim=1] flowdirs, 
                np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area):
    return 1

cpdef inline int find_upstream_nodes(int i, 
                                    np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                    np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area) except -1:

    """
    ノードiの上流側のノードのindexを返す。
    上流側ノードが複数ある場合は、その中で最も流域面積が大きいノードのindexを返す。
    flowdirsの要素はint型で、ノードiの流向方向のノードのindexを表す。
    drain_areaの要素はfloat型で、ノードiの流域面積を表す。
    """
    cdef:
        int donor, ran_num, maxinfln
        float drmax
        np.ndarray[DTYPE_INT_t, ndim=1] ups_nodes 
        np.ndarray[DTYPE_FLOAT_t, ndim=1] drin
        np.ndarray[DTYPE_INT_t, ndim=1] maxinfl

    ups_nodes = np.where(flowdirs == i)[0]

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

# find_n_upstream_nodes関数を定義
@cython.boundscheck(False)
cpdef inline int find_n_upstream_nodes(int i, 
                                      np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                      np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area, int n) except -1:
    cdef int j = 1
    cdef int target_node = i
    cdef int doner

    while j <= n:
        doner = find_upstream_nodes(target_node, flowdirs, drain_area)

        if doner == target_node:
            break
        else:
            target_node = doner
        j += 1

    return doner
    
@cython.boundscheck(False)
cpdef inline int find_n_downstream_nodes(int i, np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, int n) except -1:
    cdef int j = 1
    cdef int target_node = i
    cdef int receiver

    while j <= n:
        receiver = flowdirs[target_node]

        if receiver == target_node:
            break
        else:
            target_node = receiver
        j += 1

    return receiver

cpdef inline int is_inside_triangle_and_is_triangle(double xp, double yp, double xa, double ya, double xb, double yb, double xc, double yc):
    # ベクトルAB, BC, CAを計算
    cdef double vector_ab_x = xb - xa
    cdef double vector_ab_y = yb - ya
    cdef double vector_bc_x = xc - xb
    cdef double vector_bc_y = yc - yb
    cdef double vector_ca_x = xa - xc
    cdef double vector_ca_y = ya - yc

    # ベクトルAP, BP, CPを計算
    cdef double vector_ap_x = xp - xa
    cdef double vector_ap_y = yp - ya
    cdef double vector_bp_x = xp - xb
    cdef double vector_bp_y = yp - yb
    cdef double vector_cp_x = xp - xc
    cdef double vector_cp_y = yp - yc

    # 外積を計算
    cdef double cross_product_ab_ap = vector_ab_x * vector_ap_y - vector_ab_y * vector_ap_x
    cdef double cross_product_bc_bp = vector_bc_x * vector_bp_y - vector_bc_y * vector_bp_x
    cdef double cross_product_ca_cp = vector_ca_x * vector_cp_y - vector_ca_y * vector_cp_x

    # 外積の結果が0の場合、三角形ABCは直線上にある
    if cross_product_ab_ap == 0 and cross_product_bc_bp == 0 and cross_product_ca_cp == 0:
        return -1

    # 外積の符号を確認して点Pが三角形ABCの内部にあるか外部にあるかを判定
    if (cross_product_ab_ap >= 0 and cross_product_bc_bp >= 0 and cross_product_ca_cp >= 0) or (cross_product_ab_ap <= 0 and cross_product_bc_bp <= 0 and cross_product_ca_cp <= 0):
        return 1  # 点Pは三角形ABCの内部にある
    else:
        return 0  # 点Pは三角形ABCの外部にある

cpdef inline int point_position_relative_to_line(double xp, double yp, double xa, double ya, double xb, double yb):
    cdef double vector_ap_x = xp - xa
    cdef double vector_ap_y = yp - ya
    cdef double vector_bp_x = xp - xb
    cdef double vector_bp_y = yp - yb
    cdef double vector_ab_x = xb - xa
    cdef double vector_ab_y = yb - ya

    cdef double cross_product_ab_ap = vector_ab_x * vector_ap_y - vector_ab_y * vector_ap_x
    cdef double cross_product_ab_bp = vector_ab_x * vector_bp_y - vector_ab_y * vector_bp_x

    if cross_product_ab_ap > 0 and cross_product_ab_bp > 0:
        return -1  # 点Pは直線ABの左側にある
    elif cross_product_ab_ap < 0 and cross_product_ab_bp < 0:
        return 1  # 点Pは直線ABの右側にある
    else:
        return 0  # 点Pは直線AB上にある

@cython.boundscheck(False)
cpdef tuple node_finder_use_fivebyfive_window(grid, 
                                              int i, 
                                              np.ndarray[DTYPE_INT_t, ndim=1] flowdirs, 
                                              np.ndarray[DTYPE_FLOAT_t, ndim=1] drain_area, 
                                              bint is_get_phd_cur=False):
    
    cdef list lat_nodes = []
    cdef double radcurv = 0.0
    cdef double phd_radcurv = 0.0
    
    cdef int n = 2
    cdef int donor = find_n_upstream_nodes(i, flowdirs, drain_area, n)
    cdef int receiver = find_n_downstream_nodes(i, flowdirs, n)

    if donor == i or receiver == i:
        return lat_nodes, radcurv, phd_radcurv

    cdef int side_flag = np.random.choice([-1, 1])
    cdef double x_don = grid.x_of_node[donor]
    cdef double y_don = grid.y_of_node[donor]
    cdef double x_i = grid.x_of_node[i]
    cdef double y_i = grid.y_of_node[i]
    cdef double x_rec = grid.x_of_node[receiver]
    cdef double y_rec = grid.y_of_node[receiver]

    cdef list neighbors = grid.active_adjacent_nodes_at_node[i]

    cdef list temp_lat_nodes = []

    cdef j = 0
    cdef neig = 0
    cdef:
        double x_neig, y_neig
        int is_triangle_and_in_triangle, side_of_line
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

    cdef double angle = calculate_angle(x_don, y_don, x_i, y_i, x_rec, y_rec)
    
    if np.isclose(angle, 0.0) or np.isclose(angle, np.pi):
        angle = np.deg2rad(16.8)
    else:
        angle = np.pi - angle
    
    cdef double ds = np.hypot(x_don - x_i, y_don - y_i) + np.hypot(x_i - x_rec, y_i - y_rec)
    cdef double d_donor_receiver = np.hypot(x_don - x_rec, y_don - y_rec)
    ds = (ds + d_donor_receiver) * 0.5
    radcurv = angle / ds
    
    if is_get_phd_cur:
        donor = find_upstream_nodes(i, flowdirs, drain_area)
        cur = grid.at_node["curvature"]
        phd_radcurv = cur[donor]

    cdef tuple result = (lat_nodes, radcurv, phd_radcurv)

    return result