import numpy as np
from typing import Tuple, Union
from landlab.grid import RadialModelGrid, RasterModelGrid

def angle_finder(grid, dn, cn, rn):
    """Find the interior angle between two vectors on a grid.

    Parameters
    ----------
    grid : ModelGrid
        A landlab grid.
    dn : int or array of int
        Node or nodes at the end of the first vector.
    cn : int or array of int
        Node or nodes at the vertex between vectors.
    rn : int or array of int
        Node or nodes at the end of the second vector.

    Returns
    -------
    float or array of float
        Angle between vectors (in radians).

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components.lateral_erosion.node_finder import angle_finder

    >>> grid = RasterModelGrid((3, 4))
    >>> np.rad2deg(angle_finder(grid, 8, 5, 0))
    90.0
    >>> np.rad2deg(angle_finder(grid, (8, 9, 10, 6), 5, 6))
    array([ 135.,   90.,   45.,    0.])
    """
    vertex = np.take(grid.x_of_node, cn), np.take(grid.y_of_node, cn)
    vec_1 = [
        np.take(grid.x_of_node, dn) - vertex[0],
        np.take(grid.y_of_node, dn) - vertex[1],
    ]
    vec_2 = [
        np.take(grid.x_of_node, rn) - vertex[0],
        np.take(grid.y_of_node, rn) - vertex[1],
    ]

    return np.arccos(
        (vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1])
        / np.sqrt((vec_1[0] ** 2 + vec_1[1] ** 2) * (vec_2[0] ** 2 + vec_2[1] ** 2))
    )


def forty_five_node(donor, i, receiver, neighbors, diag_neigh):
    radcurv_angle = 0.67

    lat_node = 0
    # In Landlab 2019: diagonal list goes [NE, NW, SW, SE]. Node list are ordered as [E,N,W,S]
    # if water flows SE-N OR if flow NE-S or E-NW or E-SW, erode west node
    if (
        donor == diag_neigh[0]
        and receiver == neighbors[3]
        or donor == diag_neigh[3]
        and receiver == neighbors[1]
        or donor == neighbors[0]
        and receiver == diag_neigh[2]
        or donor == neighbors[0]
        and receiver == diag_neigh[1]
    ):
        lat_node = neighbors[2]
    # if flow is from SW-N or NW-S or W-NE or W-SE, erode east node
    elif (
        donor == diag_neigh[1]
        and receiver == neighbors[3]
        or donor == diag_neigh[2]
        and receiver == neighbors[1]
        or donor == neighbors[2]
        and receiver == diag_neigh[3]
        or donor == neighbors[2]
        and receiver == diag_neigh[0]
    ):
        lat_node = neighbors[0]
    # if flow is from SE-W or SW-E or S-NE or S-NW, erode north node
    elif (
        donor == diag_neigh[3]
        and receiver == neighbors[2]
        or donor == diag_neigh[2]
        and receiver == neighbors[0]
        or donor == neighbors[3]
        and receiver == diag_neigh[0]
        or donor == neighbors[3]
        and receiver == diag_neigh[1]
    ):
        lat_node = neighbors[1]
    # if flow is from NE-W OR NW-E or N-SE or N-SW, erode south node
    elif (
        donor == diag_neigh[0]
        and receiver == neighbors[2]
        or donor == diag_neigh[1]
        and receiver == neighbors[0]
        or donor == neighbors[1]
        and receiver == diag_neigh[3]
        or donor == neighbors[1]
        and receiver == diag_neigh[2]
    ):
        lat_node = neighbors[3]
    return lat_node, radcurv_angle


def ninety_node(donor, i, receiver, link_list, neighbors, diag_neigh):
    # if flow is 90 degrees
    if donor in diag_neigh and receiver in diag_neigh:
        radcurv_angle = 1.37
        # if flow is NE-SE or NW-SW, erode south node
        if (
            donor == diag_neigh[0]
            and receiver == diag_neigh[3]
            or donor == diag_neigh[1]
            and receiver == diag_neigh[2]
        ):
            lat_node = neighbors[3]
        # if flow is SW-NW or SE-NE, erode north node
        elif (
            donor == diag_neigh[2]
            and receiver == diag_neigh[1]
            or donor == diag_neigh[3]
            and receiver == diag_neigh[0]
        ):
            lat_node = neighbors[1]
        # if flow is SW-SE or NW-NE, erode east node
        elif (
            donor == diag_neigh[2]
            and receiver == diag_neigh[3]
            or donor == diag_neigh[1]
            and receiver == diag_neigh[0]
        ):
            lat_node = neighbors[0]
        # if flow is SE-SW or NE-NW, erode west node
        elif (
            donor == diag_neigh[3]
            and receiver == diag_neigh[2]
            or donor == diag_neigh[0]
            and receiver == diag_neigh[1]
        ):
            lat_node = neighbors[2]
    elif donor not in diag_neigh and receiver not in diag_neigh:
        radcurv_angle = 1.37
        # if flow is from east, erode west node
        if donor == neighbors[0]:
            lat_node = neighbors[2]
        # if flow is from north, erode south node
        elif donor == neighbors[1]:
            lat_node = neighbors[3]
        # if flow is from west, erode east node
        elif donor == neighbors[2]:
            lat_node = neighbors[0]
        # if flow is from south, erode north node
        elif donor == neighbors[3]:
            lat_node = neighbors[1]
    return lat_node, radcurv_angle


def straight_node(donor, i, receiver, neighbors, diag_neigh):
    # ***FLOW LINK IS STRAIGHT, NORTH TO SOUTH***#
    if donor == neighbors[1] or donor == neighbors[3]:
        # print "flow is stright, N-S from ", donor, " to ", flowdirs[i]
        radcurv_angle = 0.23
        # neighbors are ordered E,N,W, S
        # if the west cell is boundary (neighbors=-1), erode from east node
        if neighbors[2] == -1:
            lat_node = neighbors[0]
        elif neighbors[0] == -1:
            lat_node = neighbors[2]
        else:
            # if could go either way, choose randomly. 0 goes East, 1 goes west
            ran_num = np.random.randint(0, 2)
            if ran_num == 0:
                lat_node = neighbors[0]
            if ran_num == 1:
                lat_node = neighbors[2]
    # ***FLOW LINK IS STRAIGHT, EAST-WEST**#
    elif donor == neighbors[0] or donor == neighbors[2]:
        radcurv_angle = 0.23
        #  Node list are ordered as [E,N,W,S]
        # if the north cell is boundary (neighbors=-1), erode from south node
        if neighbors[1] == -1:
            lat_node = neighbors[3]
        elif neighbors[3] == -1:
            lat_node = neighbors[1]
        else:
            # if could go either way, choose randomly. 0 goes south, 1 goes north
            ran_num = np.random.randint(0, 2)
            if ran_num == 0:
                lat_node = neighbors[1]
            if ran_num == 1:
                lat_node = neighbors[3]
    # if flow is straight across diagonal, choose node to erode at random
    elif donor in diag_neigh and receiver in diag_neigh:
        radcurv_angle = 0.23
        if receiver == diag_neigh[0]:
            poss_diag_nodes = neighbors[0 : 1 + 1]
        elif receiver == diag_neigh[1]:
            poss_diag_nodes = neighbors[1 : 2 + 1]
        elif receiver == diag_neigh[2]:
            poss_diag_nodes = neighbors[2 : 3 + 1]
        elif receiver == diag_neigh[3]:
            poss_diag_nodes = [neighbors[3], neighbors[0]]
        ran_num = np.random.randint(0, 2)
        if ran_num == 0:
            lat_node = poss_diag_nodes[0]
        if ran_num == 1:
            lat_node = poss_diag_nodes[1]
    return lat_node, radcurv_angle


def node_finder(grid, i, flowdirs, drain_area):
    """Find lateral neighbor node of the primary node for straight, 45 degree,
    and 90 degree channel segments.

    Parameters
    ----------
    grid : ModelGrid
        A Landlab grid object
    i : int
        node ID of primary node
    flowdirs : array
        Flow direction array
    drain_area : array
        drainage area array

    Returns
    -------
    lat_node : int
        node ID of lateral node
    radcurv_angle : float
        inverse radius of curvature of channel at lateral node
    """
    # receiver node of flow is flowdirs[i]
    receiver = flowdirs[i]

    # find indicies of where flowdirs=i to find donor nodes.
    # will donor nodes always equal the index of flowdir list?
    inflow = np.where(flowdirs == i)

    # if there are more than 1 donors, find the one with largest drainage area

    if len(inflow[0]) > 1:
        drin = drain_area[inflow]
        drmax = max(drin)
        maxinfl = inflow[0][np.where(drin == drmax)]
        # if donor nodes have same drainage area, choose one randomly
        if len(maxinfl) > 1:
            ran_num = np.random.randint(0, len(maxinfl))
            maxinfln = maxinfl[ran_num]
            donor = [maxinfln]
        else:
            donor = maxinfl
        # if inflow is empty, no donor
    elif len(inflow[0]) == 0:
        donor = i
    # else donor is the only inflow
    else:
        donor = inflow[0]
    # now we have chosen donor cell, next figure out if inflow/outflow lines are
    # straight, 45, or 90 degree angle. and figure out which node to erode
    link_list = grid.links_at_node[i]
    # this gives list of active neighbors for specified node
    # the order of this list is: [E,N,W,S]
    neighbors = grid.active_adjacent_nodes_at_node[i]
    # this gives list of all diagonal neighbors for specified node
    # the order of this list is: [NE,NW,SW,SE]
    diag_neigh = grid.diagonal_adjacent_nodes_at_node[i]
    angle_diff = np.rad2deg(angle_finder(grid, donor, i, receiver))

    if donor == flowdirs[i]:
        # this is a sink. no lateral ero
        radcurv_angle = 0.0
        lat_node = 0
    elif donor == i:
        # this is a sink. no lateral ero
        radcurv_angle = 0.0
        lat_node = 0
    elif np.isclose(angle_diff, 0.0) or np.isclose(angle_diff, 180.0):
        [lat_node, radcurv_angle] = straight_node(
            donor, i, receiver, neighbors, diag_neigh
        )
    elif np.isclose(angle_diff, 45.0) or np.isclose(angle_diff, 135.0):
        [lat_node, radcurv_angle] = forty_five_node(
            donor, i, receiver, neighbors, diag_neigh
        )
    elif np.isclose(angle_diff, 90.0):
        [lat_node, radcurv_angle] = ninety_node(
            donor, i, receiver, link_list, neighbors, diag_neigh
        )
    else:
        lat_node = 0
        radcurv_angle = 0.0

    dx = grid.dx
    # INVERSE radius of curvature.
    radcurv_angle = radcurv_angle / dx
    return int(lat_node), radcurv_angle

def node_and_phdcur_finder(grid, i, flowdirs, drain_area):
    """Find lateral neighbor node of the primary node for straight, 45 degree,
    and 90 degree channel segments. さらに、そのノードの位相遅れ曲率（１つ上流側ノードでの曲率）
    を求める。上流から下流に向かって曲率を計算していることを想定している。

    Parameters
    ----------
    grid : ModelGrid
        A Landlab grid object
    i : int
        node ID of primary node
    flowdirs : array
        Flow direction array
    drain_area : array
        drainage area array

    Returns
    -------
    lat_node : int
        node ID of lateral node
    radcurv_angle : float
        inverse radius of curvature of channel at lateral node
    phd_radcurv_angle : float
        phase delay curvature of channel at lateral node
    """
    # receiver node of flow is flowdirs[i]
    receiver = flowdirs[i]

    # find indicies of where flowdirs=i to find donor nodes.
    # will donor nodes always equal the index of flowdir list?
    inflow = np.where(flowdirs == i)

    # if there are more than 1 donors, find the one with largest drainage area

    if len(inflow[0]) > 1:
        drin = drain_area[inflow]
        drmax = max(drin)
        maxinfl = inflow[0][np.where(drin == drmax)]
        # if donor nodes have same drainage area, choose one randomly
        if len(maxinfl) > 1:
            ran_num = np.random.randint(0, len(maxinfl))
            maxinfln = maxinfl[ran_num]
            donor = [maxinfln]
        else:
            donor = maxinfl
        # if inflow is empty, no donor
    elif len(inflow[0]) == 0:
        donor = i
    # else donor is the only inflow
    else:
        donor = inflow[0]
    # now we have chosen donor cell, next figure out if inflow/outflow lines are
    # straight, 45, or 90 degree angle. and figure out which node to erode
    link_list = grid.links_at_node[i]
    # this gives list of active neighbors for specified node
    # the order of this list is: [E,N,W,S]
    neighbors = grid.active_adjacent_nodes_at_node[i]
    # this gives list of all diagonal neighbors for specified node
    # the order of this list is: [NE,NW,SW,SE]
    diag_neigh = grid.diagonal_adjacent_nodes_at_node[i]
    angle_diff = np.rad2deg(angle_finder(grid, donor, i, receiver))

    if donor == flowdirs[i]:
        # this is a sink. no lateral ero
        radcurv_angle = 0.0
        lat_node = 0
    elif donor == i:
        # this is a sink. no lateral ero
        radcurv_angle = 0.0
        lat_node = 0
    elif np.isclose(angle_diff, 0.0) or np.isclose(angle_diff, 180.0):
        [lat_node, radcurv_angle] = straight_node(
            donor, i, receiver, neighbors, diag_neigh
        )
    elif np.isclose(angle_diff, 45.0) or np.isclose(angle_diff, 135.0):
        [lat_node, radcurv_angle] = forty_five_node(
            donor, i, receiver, neighbors, diag_neigh
        )
    elif np.isclose(angle_diff, 90.0):
        [lat_node, radcurv_angle] = ninety_node(
            donor, i, receiver, link_list, neighbors, diag_neigh
        )
    else:
        lat_node = 0
        radcurv_angle = 0.0

    dx = grid.dx
    # INVERSE radius of curvature.
    radcurv_angle = radcurv_angle / dx

    # 上流側の曲率は既に計算してある前提で位相遅れ曲率を取得する。
    cur = grid.at_node["curvature"]
    phd_radcurv_angle = cur[donor]

    return int(lat_node), radcurv_angle, phd_radcurv_angle

def find_upstream_nodes(i: int, flowdirs: np.ndarray, drain_area: np.ndarray) -> int:

    """
    ノードiの上流側のノードのindexを返す。
    上流側ノードが複数ある場合は、その中で最も流域面積が大きいノードのindexを返す。
    """    

    # 流向方向がi自身(つまり、iに流れ込んでくる)ノードが上流側の―ド
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
            donor = [maxinfln]
        else:
            donor = maxinfl
        # if inflow is empty, no donor
    elif len(ups_nodes) == 0:
        donor = i
    # else donor is the only one
    else:
        donor = ups_nodes

    return donor

def find_n_upstream_nodes(i: int, flowdirs: np.ndarray, drain_area: np.ndarray, n: int) -> int:

    """
    ノードiのn個上流側のノードのindexを返す。

    Returns
    -------
    donor : int
        n個上流側のノードのindex
    """    

    j = 1
    target_node = i

    while j <= n:

        doner = find_upstream_nodes(target_node, flowdirs, drain_area)

        if doner == target_node:
            break
        else:
            target_node = doner
        j += 1

    return doner

def find_n_downstream_nodes(i: int, flowdirs: np.ndarray, n: int) -> int:

    """
    ノードiのn個下流側のノードのindexを返す。

    Returns
    -------
    receiver : int
        n個下流側のノードのindex
    """    

    j = 1
    target_node = i

    while j <= n:

        receiver = flowdirs[target_node]

        if receiver == target_node:
            break
        else:
            target_node = receiver
        j += 1

    return receiver

def is_inside_triangle_and_is_triangle(xp: float, yp: float, xa: float, 
                                       ya: float, xb: float, yb: float, 
                                       xc: float, yc: float) -> int:

    """
    点Pが三角形ABCの内部にあるかどうか、点ABCが三角形を構成するかどうかを判定する。

    Returns
    -------
    flag_int: int
        点ABCが直線上に存在する場合は-1 \n
        点ABCが三角形を構成しており、点pが三角形内部にある場合に1\n
        点ABCが三角形を構成するが、点pが三角形外部にある場合に0\n
        を返す。
    """    

    # ベクトルAB, BC, CAを計算
    vector_ab = (xb - xa, yb - ya) # float
    vector_bc = (xc - xb, yc - yb) # float
    vector_ca = (xa - xc, ya - yc) # float
    vector_ac = (xc - xa, yc - ya) # float

    # ベクトルAP, BP, CPを計算
    vector_ap = (xp - xa, yp - ya) # float 
    vector_bp = (xp - xb, yp - yb) # float
    vector_cp = (xp - xc, yp - yc) # float

    # 外積を計算
    cross_product_ab_ap = vector_ab[0] * vector_ap[1] - vector_ab[1] * vector_ap[0] # float
    cross_product_bc_bp = vector_bc[0] * vector_bp[1] - vector_bc[1] * vector_bp[0] # float
    cross_product_ca_cp = vector_ca[0] * vector_cp[1] - vector_ca[1] * vector_cp[0] # float
    cross_product_ab_ac = vector_ab[0] * vector_ac[1] - vector_ab[1] * vector_ac[0] # float

    # 外積の結果が0の場合、三角形ABCは直線上にある
    if np.isclose(cross_product_ab_ac, 0):
        return -1

    # 外積の符号を確認して点Pが三角形ABCの内部にあるか外部にあるかを判定
    if (cross_product_ab_ap >= 0 and cross_product_bc_bp >= 0 and cross_product_ca_cp >= 0) or \
       (cross_product_ab_ap <= 0 and cross_product_bc_bp <= 0 and cross_product_ca_cp <= 0):
        return 1  # 点Pは三角形ABCの内部にある
    else:
        return 0  # 点Pは三角形ABCの外部にある

def point_position_relative_to_line(xp: float, yp: float, xa: float, ya: float, xb: float, yb: float) -> int:
    """
    点Pが直線ABの右側か左側か直線上にあるかを判定します。

    Parameters
    ----------
    xp: float
        点Pのx座標
    yp: float
        点Pのy座標
    xa: float
        直線ABの始点のx座標
    ya: float
        直線ABの始点のy座標
    xb: float
        直線ABの終点のx座標
    yb: float   
        直線ABの終点のy座標

    Returns
    -------
    flag_: int
        1 if 点Pが直線ABの右側にある場合, \n
        -1 if 点Pが直線ABの左側にある場合, \n
        0 if 点Pが直線AB上にある場合\n
    """
    def cross_product(x1, y1, x2, y2):
        return x1 * y2 - x2 * y1

    # ベクトルAP, BPを計算
    vector_ap_x = xp - xa
    vector_ap_y = yp - ya
    # vector_bp_x = xp - xb
    # vector_bp_y = yp - yb

    # ベクトルABを計算
    vector_ab_x = xb - xa
    vector_ab_y = yb - ya

    # 外積を計算
    cross_product_ab_ap = cross_product(vector_ab_x, vector_ab_y, vector_ap_x, vector_ap_y)
    # cross_product_ab_bp = cross_product(vector_ab_x, vector_ab_y, vector_bp_x, vector_bp_y)

    # 外積の結果をもとに点Pが直線ABの右側か左側か直線上にあるかを判定
    if cross_product_ab_ap < 0: #and cross_product_ab_bp > 0:
        return -1  # 点Pは直線ABの左側にある
    elif cross_product_ab_ap > 0:# and cross_product_ab_bp < 0:
        return 1  # 点Pは直線ABの右側にある
    else:
        return 0  # 点Pは直線AB上にある

def node_finder_use_fivebyfive_window(grid: RadialModelGrid, i: int, 
                                     flowdirs: np.ndarray, drain_area: np.ndarray,
                                     is_get_phd_cur: bool=False, dummy_value: np.int64=-99) -> Tuple[list, float, float]: 
    
    """
    5x5のウィンドウ範囲の情報を使って、ノード探索を行う。ノードi、ノードiの2つ上流側のノード(upstream node: node A)と
    2つ下流側のノード(downstream node: node B)の計3つのノードを結ぶ線分を大局的流線と定義する。
    このように定義すると、表現できる角度が[33.7, 45, 56.4, 90, 123.7, 135, 146.4, 180]度の8種類になる。
    このとき、側方侵食を受けるノードは以下の3つの条件を満たすノードである。

    1. node iの上下左右の4つのノードに含まれる
    2. 条件1を満たすノードのうち、三角形AiBの外側にある
    3. 条件2を満たすノードのうち、流線ノードではないもの(ノードiに流れ込んでくるノードではないもの)

    また、曲率は以下の近似式で求める。このとき、角度AiBをθとすると、曲率(1/R)は以下のようになる。

    1/R = θ / (Ai + Bi)

    ここで、Aiはnode iとnode Aの距離、Biはnode iとnode Bの距離である。

    このとき、直線の場合はθ=180となるので曲率は0となり側方侵食は発生しないが、
    直線流路の場合、河床租度の影響によりランダムに側方侵食が発生することを仮定し直線AiBの左右にあるノードのいずれかをランダムに選択する。
    このとき、直線AiBの右側にあるときはθ=163.2, 左側にあるときはθ=-163.2として計算する。

    Parameters
    ----------
    grid : ModelGrid
        A Landlab grid object
    i : int
        node ID of primary node
    flowdirs : array
        Flow direction array
    drain_area : array
        drainage area array
    is_get_phd_cur : bool, optional
        Trueの場合は位相遅れ曲率も計算する, by default False

    Returns
    -------
    lat_nodes : list
        node ID of lateral node
    radcurv : float
        curvature of channel at node i
    phd_radcurv : float
        phase delay curvature at node i, if is_get_phd_cur is True. default is 0.
    """
    
    # 側方侵食ノードを格納するリストと曲率, 位相ずれ曲率を初期化
    lat_nodes = []
    radcurv = 0.
    phd_radcurv = 0.

    # 2つ上流側、2つ下流側のノードを取得する
    n = 2
    donor = find_n_upstream_nodes(i, flowdirs, drain_area, n)
    receiver = find_n_downstream_nodes(i, flowdirs, n)

    if donor == i or receiver == i:
        # 上流側ノード、下流側ノードがない場合は側方侵食は発生しないので、空のリストと0を返す
        return lat_nodes, radcurv, phd_radcurv
    
    # this gives list of active neighbors for specified node
    # the order of this list is: [E,N,W,S]
    neighbors = grid.active_adjacent_nodes_at_node[i]

    side_flag = np.random.choice([-1, 1]) # donerとreceiverを結ぶ直線の右側か左側かをランダムに決定する
    # temp_lat_nodes = []
    temp_lat_nodes = np.full(shape=(4), fill_value=dummy_value, dtype=np.int64)
    x_don = grid.x_of_node[donor]
    y_don = grid.y_of_node[donor]
    x_i = grid.x_of_node[i]
    y_i = grid.y_of_node[i]
    x_rec = grid.x_of_node[receiver]
    y_rec = grid.y_of_node[receiver]

    k = 0
    for neig in neighbors:
        x_neig = grid.x_of_node[neig]
        y_neig = grid.y_of_node[neig]
        
        # iの上下左右の4つのノードがdonor, i, receiverの三角形の内部にあるかどうかを判定する
        # また、donor, i, receiverの三角形を構成するかどうかも判定する
        # -1: donor, i, receiverが１直線上にある場合(三角形を構成しない)
        # 1: donor, i, receiverが三角形を構成しており、neigが三角形の内部にある場合
        # 0: donor, i, receiverが三角形を構成しており、neigが三角形の外部にある場合
        is_triangle_and_in_triangle = is_inside_triangle_and_is_triangle(x_neig, y_neig, x_don, y_don, x_i, y_i, x_rec, y_rec)
        side_of_line_i = point_position_relative_to_line(x_i, y_i, x_don, y_don, x_rec, y_rec)
        side_of_line_neig = point_position_relative_to_line(x_neig, y_neig, x_don, y_don, x_rec, y_rec)
        if is_triangle_and_in_triangle == -1:
            # donor, i, receiverが１直線上にある場合は、直線の右側か左側かを判定する 
            if side_of_line_neig == side_flag:
                # 直線の右側にある場合は側方侵食ノードとする
                # temp_lat_nodes.append(neig)
                temp_lat_nodes[k] = neig
                k += 1
        elif (is_triangle_and_in_triangle == 0) and (side_of_line_i == side_of_line_neig):
            # donor, i, receiverが三角形を構成しており、neigが三角形の内部にあるかつ、
            # 直線donor-receiverに対してノードiとノードneigが同じ側にある場合は側方侵食ノードとする
            # temp_lat_nodes.append(neig)
            temp_lat_nodes[k] = neig
            k += 1

        # sent = f""" i: {i}, p: {neig}, donor: {donor}, receiver: {receiver}, is_triangle_and_in_triangle: {is_triangle_and_in_triangle}, 
        # side_of_line: {side_of_line}, side_flag: {side_flag}, temp_lat_nodes: {temp_lat_nodes}"""
        # print(sent)
                
    
    # ノードiに流れ込んでくるノードは側方侵食ノードに含めない
    # donors = np.where(flowdirs == i)[0]
    # lat_nodes = [node for node in temp_lat_nodes if node not in donors]
    lat_nodes = temp_lat_nodes
    # 曲率を計算する
    angle = angle_finder(grid, donor, i, receiver) # rad

    if np.isclose(angle, 0.) or np.isclose(angle, np.pi):
        angle = np.deg2rad(16.8) # 180-163.2=16.8度,
    else:
        angle = np.pi - angle

    ds = np.hypot(x_don - x_i, y_don - y_i) + np.hypot(x_i - x_rec, y_i - y_rec)
    d_donor_receiver = np.hypot(x_don - x_rec, y_don - y_rec)
    ds = (ds + d_donor_receiver) * 0.5 # 平均を使用 

    radcurv = angle / ds
    
    # 位相遅れを仮定する場合
    # 上流側の曲率は既に計算してある前提で位相遅れ曲率を取得する。
    if is_get_phd_cur:
        donor = find_upstream_nodes(i, flowdirs, drain_area)
        cur = grid.at_node["curvature"]
        phd_radcurv = cur[donor]

    return lat_nodes, radcurv, phd_radcurv

def node_finder_use_fivebyfive_window_diag(grid: RadialModelGrid, i: int, 
                                     flowdirs: np.ndarray, drain_area: np.ndarray,
                                     is_get_phd_cur: bool=False, dummy_value: np.int64=-99) -> Tuple[list, float, float]: 
    
    """
    5x5のウィンドウ範囲の情報を使って、ノード探索を行う。ノードi、ノードiの2つ上流側のノード(upstream node: node A)と
    2つ下流側のノード(downstream node: node B)の計3つのノードを結ぶ線分を大局的流線と定義する。
    このように定義すると、表現できる角度が[33.7, 45, 56.4, 90, 123.7, 135, 146.4, 180]度の8種類になる。
    このとき、側方侵食を受けるノードは以下の3つの条件を満たすノードである。

    1. node iの上下左右の4つのノードに含まれる
    2. 条件1を満たすノードのうち、三角形AiBの外側にある
    3. 条件2を満たすノードのうち、流線ノードではないもの(ノードiに流れ込んでくるノードではないもの)

    また、曲率は以下の近似式で求める。このとき、角度AiBをθとすると、曲率(1/R)は以下のようになる。

    1/R = θ / (Ai + Bi)

    ここで、Aiはnode iとnode Aの距離、Biはnode iとnode Bの距離である。

    このとき、直線の場合はθ=180となるので曲率は0となり側方侵食は発生しないが、
    直線流路の場合、河床租度の影響によりランダムに側方侵食が発生することを仮定し直線AiBの左右にあるノードのいずれかをランダムに選択する。
    このとき、直線AiBの右側にあるときはθ=163.2, 左側にあるときはθ=-163.2として計算する。

    Parameters
    ----------
    grid : ModelGrid
        A Landlab grid object
    i : int
        node ID of primary node
    flowdirs : array
        Flow direction array
    drain_area : array
        drainage area array
    is_get_phd_cur : bool, optional
        Trueの場合は位相遅れ曲率も計算する, by default False

    Returns
    -------
    lat_nodes : list
        node ID of lateral node
    radcurv : float
        curvature of channel at node i
    phd_radcurv : float
        phase delay curvature at node i, if is_get_phd_cur is True. default is 0.
    """
    
    # 側方侵食ノードを格納するリストと曲率, 位相ずれ曲率を初期化
    lat_nodes = []
    radcurv = 0.
    phd_radcurv = 0.

    # 2つ上流側、2つ下流側のノードを取得する
    n = 2
    donor = find_n_upstream_nodes(i, flowdirs, drain_area, n)
    receiver = find_n_downstream_nodes(i, flowdirs, n)

    if donor == i or receiver == i:
        # 上流側ノード、下流側ノードがない場合は側方侵食は発生しないので、空のリストと0を返す
        return lat_nodes, radcurv, phd_radcurv
    
    # this gives list of active neighbors for specified node
    # the order of this list is: [E,N,W,S]
    neighbors = grid.active_adjacent_nodes_at_node[i]
    diag_neighs = grid.diagonal_adjacent_nodes_at_node[i]

    neighbors = np.concatenate([neighbors, diag_neighs])

    side_flag = np.random.choice([-1, 1]) # donerとreceiverを結ぶ直線の右側か左側かをランダムに決定する
    # temp_lat_nodes = []
    temp_lat_nodes = np.full(shape=(8), fill_value=dummy_value, dtype=np.int64)
    x_don = grid.x_of_node[donor]
    y_don = grid.y_of_node[donor]
    x_i = grid.x_of_node[i]
    y_i = grid.y_of_node[i]
    x_rec = grid.x_of_node[receiver]
    y_rec = grid.y_of_node[receiver]

    donors = np.where(flowdirs == i)[0]
    k = 0
    for neig in neighbors:

        if neig not in donors:

            x_neig = grid.x_of_node[neig]
            y_neig = grid.y_of_node[neig]
            
            # iの上下左右の4つのノードがdonor, i, receiverの三角形の内部にあるかどうかを判定する
            # また、donor, i, receiverの三角形を構成するかどうかも判定する
            # -1: donor, i, receiverが１直線上にある場合(三角形を構成しない)
            # 1: donor, i, receiverが三角形を構成しており、neigが三角形の内部にある場合
            # 0: donor, i, receiverが三角形を構成しており、neigが三角形の外部にある場合
            is_triangle_and_in_triangle = is_inside_triangle_and_is_triangle(x_neig, y_neig, x_don, y_don, x_i, y_i, x_rec, y_rec)
            side_of_line_i = point_position_relative_to_line(x_i, y_i, x_don, y_don, x_rec, y_rec)
            side_of_line_neig = point_position_relative_to_line(x_neig, y_neig, x_don, y_don, x_rec, y_rec)
            if is_triangle_and_in_triangle == -1:
                # donor, i, receiverが１直線上にある場合は、直線の右側か左側かを判定する 
                if side_of_line_neig == side_flag:
                    # 直線の右側にある場合は側方侵食ノードとする
                    # temp_lat_nodes.append(neig)
                    temp_lat_nodes[k] = neig
                    k += 1
            elif (is_triangle_and_in_triangle == 0) and (side_of_line_i == side_of_line_neig):
                # donor, i, receiverが三角形を構成しており、neigが三角形の内部にあるかつ、
                # 直線donor-receiverに対してノードiとノードneigが同じ側にある場合は側方侵食ノードとする
                # temp_lat_nodes.append(neig)
                temp_lat_nodes[k] = neig
                k += 1

        # sent = f""" i: {i}, p: {neig}, donor: {donor}, receiver: {receiver}, is_triangle_and_in_triangle: {is_triangle_and_in_triangle}, 
        # side_of_line: {side_of_line}, side_flag: {side_flag}, temp_lat_nodes: {temp_lat_nodes}"""
        # print(sent)
                
    
    # ノードiに流れ込んでくるノードは側方侵食ノードに含めない
    # donors = np.where(flowdirs == i)[0]
    # lat_nodes = [node for node in temp_lat_nodes if node not in donors]
    lat_nodes = temp_lat_nodes
    # 曲率を計算する
    angle = angle_finder(grid, donor, i, receiver) # rad

    if np.isclose(angle, 0.) or np.isclose(angle, np.pi):
        angle = np.deg2rad(16.8) # 180-163.2=16.8度,
    else:
        angle = np.pi - angle

    ds = np.hypot(x_don - x_i, y_don - y_i) + np.hypot(x_i - x_rec, y_i - y_rec)
    d_donor_receiver = np.hypot(x_don - x_rec, y_don - y_rec)
    ds = (ds + d_donor_receiver) * 0.5 # 平均を使用 

    radcurv = angle / ds
    
    # 位相遅れを仮定する場合
    # 上流側の曲率は既に計算してある前提で位相遅れ曲率を取得する。
    if is_get_phd_cur:
        donor = find_upstream_nodes(i, flowdirs, drain_area)
        cur = grid.at_node["curvature"]
        phd_radcurv = cur[donor]

    return lat_nodes, radcurv, phd_radcurv


