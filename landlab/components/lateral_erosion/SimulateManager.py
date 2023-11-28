"""
各クラスの略記号
----------------
HDFfileController: HDFファイルを読み書きするためのクラス
    略記号: HDF
SimulateHDFhandler: 計算実行時にHDFファイルに書き込むことを目的としたクラスで、HDFhandlerクラスを継承している。
    略記号: SimHDF 
LataralSimilateManager: 側方侵食モデルの計算を統括するためのクラス。
    略記号: SimManager
InitTpgraphyMaker: 初期地形を作成するためのクラス
    略記号: InitTp
LateroEroder: 側方侵食モデルを実行するためのクラス
    略記号: Latero
FlowAccumulator: 流量分配を実行するためのクラス
    略記号: FlowAcc
"""

import datetime
import os
import h5py
import multiprocessing
from copy import deepcopy, copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import pandas as pd
from tqdm import tqdm
from landlab import RasterModelGrid
from landlab.components.flow_accum import FlowAccumulator
from landlab.components.lateral_erosion import LateralEroder
from landlab.components.profiler import ChannelProfiler
from landlab.plot import imshow_grid
from typing import Union, Tuple, List, Dict, Any, Optional


class HDFhandler:

    """
    HDFhandler
    ----------
    HDFファイルを読み書きするためのクラス

    HDFファイルの構成は以下の通り。
    Gはグループ、Dsetはデータセットを示す。また、カッコ内はプログラム内での参照名を示す。

    G1計算実験(simulateGroup)
    G1.1 計算パラメータ(Calculation_parameters)
        G1.1.1. シミュレーション管理条件(simManage_conditions) -> 計算時間間隔dtや計算年数など
            Dset(param_df)
        G1.1.2. 初期地形条件(InitTP_conditions) -> InitTpクラスで使用するパラメータ群
            Dset(param_df)
        G1.1.3. 流量条件(Flow_conditions) -> FlowAccクラスで使用するパラメータ群
            Dset(param_df)
        G1.1.4. 河床侵食条件(Eroder_conditions) -> Lateroクラスで使用するパラメータ群
            Dset(param_df)
    G1.2 境界条件(Boundary_conditions)
        Dset (status_of_nodeをそのまま利用→numpyの1次元配列)
    G1.3. フィールドデータ(Field_datas)
        G1.3.0. グループ名一覧 (Group_names)
            Dset (Names) -> フィールドデータの名前を格納したデータセット
                          例: z, latero, deprate
        G1.3.1. 標高値 (z)
            Dset (0) 0~999yr
            Dset (1) 1000~1999yr
        G1.3.2. 側方侵食量 (latero)
            Dset (0) 0~999yr
            Dset (1) 1000~1999yr
    G2解析結果

    Gropu_names
    ----------
    フィールドデータのグループ名として以下を慣習的に使用する

    z: 標高\n
    latero: 側方侵食量\n
    deprate: 堆積速度\n
    curvature: 曲率\n
    phd_curvature: 位相ずれ曲率\n
    """    


    def __init__(self, dirpath: str, fname: str):

        """
        Parameters
        ----------
        dirpath : str
            HDFファイルが格納されているディレクトリの絶対パス
        fname : str
            HDFファイル名
        """        

        self.dirpath = dirpath # 保存先ディレクトリ
        self.fname = fname # 拡張子込みのファイル名
        self.fpath = os.path.join(dirpath, fname) # 絶対パス

    def _check_hierarchy(self):

        """
        HDFファイル内のデータ構造を確認し、存在しない場合は作成するメソッド
        """        
        
        with h5py.File(self.fpath, "a") as f:
            if ("simulateGroup" in f):
                sGroup = f["simulateGroup"]
            else:
                sGroup = f.create_group("/simulateGroup")
            if not ("Field_datas" in f["simulateGroup"]):
                sGroup.create_group("Field_datas")

    def PrintAll(self):

        """
        HDFファイル内のデータ構造を出力するメソッド
        """        
        
        def _PrintAllObjects(name):
            print(name)

        # h5py.File.visitは引数の関数の返り値がNoneになるまで再帰関数として動く
        with h5py.File(self.fpath) as f:
            f.visit(_PrintAllObjects)

    def _converted_value(self, value_and_dtype_df: pd.DataFrame) -> pd.DataFrame:
        
        """
        pd.DataFrameのvalue行の値をdtype行のdtypeに変換するメソッド

        Parameters
        ----------
        value_and_dtype_df : pd.DataFrame
            value行とdtype行を持つpd.DataFrame

        Returns
        -------
        pd.DataFrame
            dtype行のdtypeに変換したvalue行を持つpd.DataFrame
        """
        # display(value_and_dtype_df)
        converter = {"int": int, "float": float, "bool": bool, "str": str, "dict": dict, "list": list,}
        value_series = value_and_dtype_df.iloc[0]
        dtype_series = value_and_dtype_df.iloc[1]
        for key in value_series.keys():
            data_type = dtype_series[key]
            if data_type == "dict" or data_type == "list":
                value_series[key] = eval(value_series[key])
            elif data_type == "NoneType":
                value_series[key] = None
            else:
                value_series[key] = converter[data_type](value_series[key])

        value_df = pd.DataFrame(value_series).T
        return value_df

    def read_calc_condition_param(self, condition_keys: list=[]) -> list:
        """
        計算条件のパラメータを読み込むメソッド

        Parameters
        ----------
        condition_keys : list, optional
            読み込む計算条件のパラメータのキーを格納したリスト, by default []
            例：["simManage_conditions", "InitTP_conditions", "FlowAcc_conditions", "Latero_conditions"]

        Returns
        -------
        calc_conditions :  list
            condition_keysで指定した数の計算条件のpd.DataFrameを格納したlist

        Raises
        ------
        ValueError
            condition_keysで指定したキーが存在しない場合に発生
        
        """     
        calc_conditions = []   
        with h5py.File(self.fpath, "r") as f:
            assert "simulateGroup" in f, "there is no group which name is = simulateGroup"
            assert "Calculation_parameters" in f["/simulateGroup"], "there is no group which name is = Calculation_parametersin simulateGroup"
            
            for key in condition_keys:
                if not (key in f["/simulateGroup/Calculation_parameters"]):
                    raise ValueError(f"there is no key which name is = {key} in Calculation_parameters")
                df = pd.read_hdf(self.fpath, key=f"/simulateGroup/Calculation_parameters/{key}/param_df")
                calc_conditions.append(self._converted_value(df))

        return calc_conditions

    def read_boundary_condition(self) -> Tuple[np.ndarray, str]:

        """
        境界条件を読み込むメソッド

        Returns
        -------
        value, comment : Tuple[np.ndarray, str]

        value : np.ndarray
            ノード種類を整数値で表現した物。Landlab.RasterModelGridクラスを参照
        comment : str
            境界条件に関する説明文
        """        

        # retrun: [np.ndarray, str] -> [ノード情報、境界条件の説明文]
        with h5py.File(self.fpath, "r") as f:
            assert "simulateGroup" in f, "there is no group which name is = simulateGroup"
            assert "Boundary_conditions" in f["/simulateGroup"], "there is no group which name is = Boundary_conditionsin simulateGroup"
            _comment = f["/simulateGroup/Boundary_conditions/boundary_condition"].attrs["type of boundary"]
            _value = np.array(f["/simulateGroup/Boundary_conditions/boundary_condition"])
        return _value, _comment
    
    def read_fildvalue_names(self) -> list:
                        
        """
        Returns
        -------
        value_names : list
            フィールドデータの名前一覧を格納したリスト
        """        

        with h5py.File(self.fpath, "r") as f:
            fGroup = f["/simulateGroup/Field_datas"]
            assert "Group_names" in fGroup, "there is no group which name is = Group_names in simulateGroup/Field_datas"
            assert "Names" in fGroup["Group_names"], f"there is no group which name is = Names in simulateGroup/Field_datas/Group_names"
            ds = fGroup["Group_names"]["Names"]
            value_names = [ds[i].decode("utf-8") for i in range(ds.shape[0])] # decode("utf-8")でバイト列を文字列に変換。ds.shape[0]はdsの要素数
            
        return value_names

    def read_fildvalue_Matrix(self, Gname: str, Dname: str) -> Tuple[np.ndarray, str]:
         
        """
        フィールドデータを読み込むメソッド

        Parameters
        ----------
        Gname : str
            対象のフィールドデータが格納されたグループ名
        Dname: str
            対象のGname内のデータセット名

        Returns
        -------
        value, period : Tuple[np.ndarray, str]

        value : np.ndarray
            フィールドデータ
        period : str
            データが示す期間
        """        

        with h5py.File(self.fpath, "r") as f:
            assert "simulateGroup" in f, "there is no group which name is = simulateGroup"
            assert "Field_datas" in f["/simulateGroup"], "there is no group which name is = Field_datas in simulateGroup"
            assert Gname in f["/simulateGroup/Field_datas"], f"there is no group which name is = {Gname} in simulateGroup/Field_datas"
            targetGroup = f[f"/simulateGroup/Field_datas/{Gname}"]
            _period = targetGroup[Dname].attrs["period"]
            _value = np.array(targetGroup[Dname])
        
        return _value, _period

    def write_calc_condition_parameter(self, calc_condition_params: dict):

        """
        計算条件(ex. 初期地形条件, 流量条件)のパラメータをHDFファイルに書き込む関数\n
        (使用例)\n
            simManage_conditions_dict = {"start_time": 0, "end_time": 100, "dt": 1, "dataset_size": 100}\n
            calc_condition_param = {"simManage_conditions": simManage_conditions_dict}\n
            self.wirte_calc_condition_parameter(calc_condition_params=calc_condition_param)\n

            LateralErosion_LEMでは、計算条件のパラメータは以下の4つである。\n
            ["simManage_conditions", "InitTP_conditions", "FlowAcc_conditions", "Latero_conditions"]

            書き込まれるデータはpd.DataFrameとして処理され、以下のような形式である。\n
            df.columns = calc_condition_params.keys()\n
            df.index = ["value", "dtype"]\n

        Parameters
        ----------
        calc_condition_param : dict
            計算条件のパラメータの名前をkeyにもち、それに該当する計算条件値のdictオブジェクトをvalueに持つdictオブジェクト。上記の使用例を参照。
        """        

        with h5py.File(self.fpath, "a") as f:
            # Calculation_parametersグループの存在確認
            # もし、存在する場合は一度削除したうえで上書きする。
            if ("Calculation_parameters" in f["/simulateGroup"]):
                del f["/simulateGroup/Calculation_parameters"]
            
            for key, param_dict in calc_condition_params.items():
                # pandasのDataFrameを利用して一気に書き込む
                # データ型を統一するためにstrに変換
                # また、データ型を別の列に格納する
                value_list = [[str(value), str(type(value))[len("<class '"):-2]] for value in param_dict.values()] 
                df = pd.DataFrame(value_list, index=param_dict.keys(), columns=["value", "dtype"])
                df = df.T
                df.to_hdf(self.fpath, key=f"/simulateGroup/Calculation_parameters/{key}/param_df")

    def write_boundary_condition(self, status_of_node: np.ndarray, comment: str):

        """
        境界条件を書き込むメソッド。

        Parameters
        ----------
        status_of_node : np.ndarray
            全ノードのノード種類を整数値で表現した物。Landlab.RasterModelGridクラスを参照
        comment : str
            境界条件に関する説明文
        """        

        with h5py.File(self.fpath, "a") as f:
            # Boundary_conditionsグループの存在確認
            # もし、存在する場合は一度削除したうえで上書きする。
            if ("Boundary_conditions" in f["/simulateGroup"]):
                del f["/simulateGroup/Boundary_conditions"]

            bGroup = f["/simulateGroup"].create_group("Boundary_conditions")
            bGroup.create_dataset("boundary_condition", data=status_of_node)
            # 境界条件に関する説明を"type of boundary"という属性として設定
            bGroup["boundary_condition"].attrs["type of boundary"] = comment 

    def write_fildvalue_names(self, value_names: list):

        """
        フィールドデータの名前一覧を書き込むメソッド。
        f["/simulateGroup/Field_datas/Group_names/Names"]に書き込まれる。
        既に存在する場合は上書きされる。
        """        

        with h5py.File(self.fpath, "a") as f:
            fGroup = f["/simulateGroup/Field_datas"]
            # print(fGroup.keys())
            if ("Group_names" in fGroup):
                targetGroup = fGroup["Group_names"]
                if "Names" in targetGroup:
                    del f[f"/simulateGroup/Field_datas/Group_names/Names"]
            else:
                targetGroup = fGroup.create_group("Group_names")
            
            ds = targetGroup.create_dataset("Names", shape=(len(value_names),), dtype=h5py.special_dtype(vlen=str))
            ds[...] = value_names
            
    def write_fildvalue_Matrix(self, value: np.ndarray, Gname: str, Dname: str, period: str=None): 

        """
        フィールドデータを書き込むメソッド。

        Parameters
        ----------
        value : np.ndarray
            保存するフィールドデータ。時間×空間の二次元配で時間はdataset_sizeで規定される大きさ。
            空間が2次元の場合は1次元に変換して、時間×空間の二次元配列になるようにしてから保存する。
        Gname : str
            保存するフィールドデータの名前。フィールドグループ直下に作成されるグループ名に使用される。
        Dname : str
            保存するフィールドデータの名前。フィールドグループ直下に作成されるデータセット名に使用される。

        """        
        with h5py.File(self.fpath, "a") as f:
            fGroup = f["/simulateGroup/Field_datas"]
            # print(fGroup.keys())
            if (Gname in fGroup):
                targetGroup = fGroup[Gname]
                if Dname in targetGroup:
                    ds = targetGroup[Dname]
                else:
                    ds = targetGroup.create_dataset(Dname, data=value)
            else:
                targetGroup = fGroup.create_group(Gname)
                ds = targetGroup.create_dataset(Dname, data=value)
            ds.attrs["period"] = period if period != None else "None"

        
class LataralSimilateManager(HDFhandler):

    # 追加計算やデータ解析をする際に、過去のデータから復元計算ができ必要なパラメータのリスト
    _required_additional_calculation_params = ['topographic__elevation', 'curvature', 'phd_curvature'] 

    def __init__(self, dirpath: str, HDF_fname: str, start_time: int=None, end_time: int=None, time_unit: str=None, 
                 space_unit: str=None, wight_unit: str=None, dt: int=None, t_dataset_size: int=None, 
                 InitTP_param: dict=None, FlowAcc_param: dict=None, Latero_param: dict=None, 
                 additional_calculation_flag: bool=False, additional_calculation_yr: int=None,
                 progress_to_txt: bool=False, recorded_variables: list=[]) -> None:
        
        """
        Parameters
        ----------
        """        
        
        super().__init__(dirpath=dirpath, fname=HDF_fname)
        self.additional_calculation_flag = additional_calculation_flag

        # 追加計算の場合は、HDFファイルに追加計算のためのパラメータを記録する
        if self.additional_calculation_flag:
            if additional_calculation_yr is None:
                raise ValueError("additional_calculation_yr must be specified.")
            
            calc_condition_params = self.read_calc_condition_param(condition_keys=["simManage_conditions", 
                                                                                   "InitTP_conditions", 
                                                                                   "FlowAcc_conditions", 
                                                                                   "Latero_conditions"])
            self.simManage_dict = calc_condition_params[0].iloc[0].to_dict()
            self.InitTP_dict = calc_condition_params[1].iloc[0].to_dict()
            self.FlowAcc_dict = calc_condition_params[2].iloc[0].to_dict()
            self.Latero_dict = calc_condition_params[3].iloc[0].to_dict()
            self.time_unit = self.simManage_dict["time_unit"]
            self.space_unit = self.simManage_dict["space_unit"]
            self.wight_unit = self.simManage_dict["wight_unit"]
            self.dt = self.simManage_dict["dt"]
            self.t_dataset_size = self.simManage_dict["t_dataset_size"]
            self.start_time = self.simManage_dict["end_time"] + 1
            self.end_time = self.start_time + additional_calculation_yr -1
            self.recorded_variables = self.read_fildvalue_names()
            self.simManage_dict["end_time"] = self.end_time # 追加計算のために計算終了時刻を更新する
      
        else:
        # 追加計算でない場合は、インスタンス時に与えられたパラメータを記録する
            if os.path.exists(self.fpath):
                os.remove(self.fpath)
            self._check_hierarchy()  
            self.start_time = start_time
            self.end_time = end_time
            self.time_unit = time_unit
            self.space_unit = space_unit
            self.wight_unit = wight_unit
            self.dt = dt
            self.t_dataset_size = t_dataset_size
            self.InitTP_dict = InitTP_param
            self.FlowAcc_dict = FlowAcc_param
            self.Latero_dict = Latero_param
            self.recorded_variables = recorded_variables

            self.simManage_dict = {"start_time": self.start_time,
                                    "end_time": self.end_time,
                                    "time_unit": self.time_unit,
                                    "space_unit": self.space_unit,
                                    "wight_unit": self.wight_unit,
                                    "dt": self.dt,
                                    "t_dataset_size": self.t_dataset_size,
                                    }
        
        # 共通の処理
        self._check_required_calculation_params() # 追加計算や解析をするために保存されていなけらばならない変数群が保存変数に指定されているか確認する
        self.total_yr = self.end_time + 1 # 総計算時間数。0時刻を含めるために+1する 
        self.Current_time = deepcopy(self.start_time) # 現在の年代
        self.Dataset_Pointer, self.time_index = self.get_DatasetPointer_and_TimeIndex(Yr=self.Current_time) # 計算開始時刻のデータセットのインデックスと、そのデータセット内での開始時刻のインデックスを取得する
        self.Dataset_Pointer_at_start_time, self.start_time_index = deepcopy(self.Dataset_Pointer), deepcopy(self.time_index) # 計算開始時刻のデータセットのインデックスと、そのデータセット内での開始時刻のインデックスを取得する
        self.Dataset_Pointer_at_end_time, self.end_time_index = self.get_DatasetPointer_and_TimeIndex(Yr=self.end_time) # 計算終了時刻のデータセットのインデックスと、そのデータセット内での終了時刻のインデックスを取得する                      
        self.recorded_value_index_dict = self._get_index_dict(valiable_names=self.recorded_variables) # フィールドデータの名前をキーとして、インデックスのリストを値とする辞書を作成する
        self.total_loop_num = self.Dataset_Pointer_at_end_time - self.Dataset_Pointer_at_start_time + 1 # 総計算ループ数
        self._is_first_loop = True # main計算の初回ループかどうかを判定するフラグ
        self.progress_to_txt = progress_to_txt # 計算進捗をtxtファイルに出力するかどうか
        if progress_to_txt: # 計算進捗をtxtファイルに出力するかどうか
            fname_key = "progress"
            self.delete_files(fname_key=fname_key, type_of_file=".txt") # 既存の計算進捗txtファイルを削除する
            self.progress_txtfile_path = os.path.join(self.dirpath, f'{fname_key}_{datetime.datetime.now().strftime(f"%Y-%m-%d %H-%M.txt")}')

    def _get_index_dict(self, valiable_names: list) -> dict:

        """
        引数に指定した変数名のリストをキーとして、インデックスのリストを値とする辞書を返す関数。

        (例):\n
        valiable_names = ['x', 'y', 'z']\n
        index_dict = self._get_index_dict(valiable_names)\n
        print(index_dict) # {'x': 0, 'y': 1, 'z': 2}\n

        Parameters
        ----------
        valiable_names : list
            インデックスを取得したい変数名のリスト

        Returns
        -------
        index_dict : dict
            引数に指定した変数名のリストをキーとして、インデックスのリストを値とする辞書
        """        

        return dict(zip(valiable_names, np.arange(len(valiable_names))))

    def delete_files(self, fname_key: str, type_of_file: str=".txt"):

        """
        指定されたディレクトリ内のファイル(type_of_file 拡張子）を削除する関数

        Parameters
        ----------
        fname_key : str
            削除したいファイルのファイル名の一部
        type_of_file : str, optional
            削除したいファイルの拡張子, by default ".txt"
        """

        for filename in os.listdir(self.dirpath):

            if filename.startswith(fname_key) and filename.endswith(type_of_file):
                filepath = os.path.join(self.dirpath, filename)
                os.remove(filepath)
    
    def _check_required_calculation_params(self):

        """
        追加計算や解析をするために保存されていなけらばならない変数群が保存変数に指定されているか確認するメソッド
        """        

        for param in self._required_additional_calculation_params:
            if not (param in self.recorded_variables):
                raise ValueError(f"{param} is not in recorded_variables.")
            
    def get_DatasetPointer_and_TimeIndex(self, Yr: int) -> Tuple[int, int]:

        """
        指定した年が格納されているデータセットのインデックスと、そのデータセット内での該当年のインデックスを取得するメソッド

        Returns
        -------
        Dataset_Pointer, time_index : Tuple[int, int]
        
        Dataset_Pointer : int
            指定した年が格納されているデータセットのインデックス
        time_index : int
            指定した年が格納されているデータセット内での該当年のインデックス
        """        

        Dataset_Pointer, time_index = divmod(Yr, self.t_dataset_size*self.dt)
        time_index = int(time_index/self.dt)

        return Dataset_Pointer, time_index
    
    def read_oneYr_filedvalue(self, Yr: int, name: str) -> np.ndarray:

        """
        あるフィールドデータを１タイムステップ分だけ読み込むメソッド

        Parameters
        ----------
        Yr : int
            取り出したい年
        name : str
            取り出したいデータが格納されたグループ名

        Returns
        -------
        value : np.ndarray
            取り出したい年のフィールドデータ
        """        
        if Yr > self.end_time:
            raise ValueError(f"Yr must be less than or equal to {self.end_time}.")

        if name not in self.recorded_variables:
            raise ValueError(f"{name} is not recorded.")

        _dataset_pointer, _row_index = self.get_DatasetPointer_and_TimeIndex(Yr=Yr)
        valueMatrix = self.read_fildvalue_Matrix(Gname=name, Dname=str(_dataset_pointer))[0]
        return valueMatrix[_row_index]
    
    def create_mg(self, Yr: int) -> RasterModelGrid:

        """
        指定した年の地形を作成するメソッド

        Parameters
        ----------
        Yr : int
            地形を作成したい年

        Returns
        -------
        mg : RasterModelGrid
            指定した年の地形を格納したLandlab.RasterModelGridクラスのインスタンス.
            recorded_variablesに指定された変数の値も格納されている。
        """        

        z = self.read_oneYr_filedvalue(Yr=Yr, name="topographic__elevation")
        ncols = self.InitTP_dict['ncols']
        nrows = self.InitTP_dict['nrows']
        dx = self.InitTP_dict['dx']

        initTP = InitialTopographyMaker(**self.InitTP_dict)
        mg = initTP.arbitrary_topography(z=z, ncols=ncols, nrows=nrows, dx=dx)

        for name in self.recorded_variables:
            if name == "topographic__elevation":
                continue
            value = self.read_oneYr_filedvalue(Yr=Yr, name=name)

            if name in ['flow__receiver_node', 'flow__upstream_node_order']:
                mg.add_zeros(name, at="node", dtype=int)
                value = value.astype(int)
            else:
                mg.add_zeros(name, at="node", dtype=float)
            mg.at_node[name] = value

        return mg
    
    def create_fa(self, mg: RasterModelGrid, flowdir_key: str=None) -> FlowAccumulator:

        """
        設定した流量条件でのFlowAccumulatorクラスをインスタンス化する

        Parameters
        ----------
        mg : RasterModelGrid
            標高情報等を格納したグリッドオブジェクト
        flowdir_key : str, optional
            流量条件のキー, by default None

        Returns
        -------
        _type_
            _description_
        """        

        # 流量条件の設定INF
        if flowdir_key == "FlowDirectorD8":
            self.FlowAcc_dict['flow_director'] = flowdir_key
            mg.at_node['flow__receiver_node'] = mg.add_zeros("flow__receiver_node", at="node", dtype=int, clobber=True)
            mg.at_node['flow__upstream_node_order'] = mg.add_zeros("flow__upstream_node_order", at="node", dtype=int, clobber=True)
            mg.at_node["topographic__steepest_slope"] = np.zeros(mg.number_of_nodes)
            mg.at_node["flow__link_to_receiver_node"] = np.zeros(mg.number_of_nodes, dtype=int)

        fa = FlowAccumulator(grid=mg, **self.FlowAcc_dict)

        return fa
    
    def init_model(self) -> Tuple[RasterModelGrid, FlowAccumulator, LateralEroder]:

        """
        初期地形、流量条件、側方侵食条件の設定を行うメソッド
        additional_calculation_flagがTrueの場合は、初期地形条件を前回の計算での最終ステップでの地形に設定する

        Returns
        -------
        mg, fa, latero : Tuple[RasterModelGrid, FlowAccumulator, LateralEroder]

        mg : RasterModelGrid
            初期地形条件を格納したLandlab.RasterModelGridクラスのインスタンス
        fa : FlowAccumulator
            流量条件を格納したLandlab.FlowAccumulatorクラスのインスタンス
        latero : LateralEroder
            側方侵食条件を格納したLandlab.LateralEroderクラスのインスタンス
        """        

        initTP_dict = deepcopy(self.InitTP_dict)
        
        # 初期地形条件と境界条件の設定
        # 境界条件はInitTpクラス内で設定する
        if self.additional_calculation_flag:
            last_time = self.start_time - self.dt
            initTP_dict['initz_func'] = 'arbitrary_topography'
            initTP_dict['create_tp_hyper_param'] =  {
                                                     "z": self.read_oneYr_filedvalue(Yr=last_time, name="topographic__elevation"),
                                                     "ncols": initTP_dict['ncols'],
                                                     "nrows": initTP_dict['nrows'],
                                                     "dx": initTP_dict['dx'],
                                                    }
        initTP = InitialTopographyMaker(**initTP_dict)
        mg = initTP.create_initial_topography()

        # 流量条件の設定
        fa = FlowAccumulator(grid=mg, **self.FlowAcc_dict)

        # 側方侵食条件の設定
        Latero_dict = deepcopy(self.Latero_dict)
        if 'U' in Latero_dict:
            del Latero_dict['U']
        latero = LateralEroder(grid=mg, **Latero_dict)

        return mg, fa, latero
    
    def init_dataset(self) -> np.ndarray:

        """
        初期データセットを作成するメソッド。
        初回ループかつ追加計算かつスタートでのタイムインデックスが0じゃないの場合は、前回の計算での最終ステップでのデータセットを読み込む。
        それ以外の場合は、ゼロ行列を作成する。

        Returns
        -------
        init_data_matrix : np.ndarray
            初期データセット。shape=(self.t_dataset_size, len(self.recorded_variables), self.InitTP_dict['ncols'] * self.InitTP_dict['nrows'])
            つまり、時間データサイズ×変数数×空間の三次元配列。
        """        

        # 初回ループかつ追加計算かつスタートでのタイムインデックスが0じゃないの場合は、前回の計算での最終ステップでのデータセットを読み込む
        flag = self._is_first_loop and self.additional_calculation_flag and (self.start_time_index!=0) 
        axis_0_shape = self.t_dataset_size
        axis_1_shape = len(self.recorded_variables)
        axis_2_shape = self.InitTP_dict['ncols'] * self.InitTP_dict['nrows']
        init_data_matrix = np.zeros(shape=(axis_0_shape, axis_1_shape, axis_2_shape))

        if flag:
            for name in self.recorded_variables:
                i = self.recorded_value_index_dict[name]
                init_data_matrix[:, i, :] = self.read_fildvalue_Matrix(Gname=name, Dname=str(self.Dataset_Pointer_at_start_time))[0]

        return init_data_matrix

    def do_computing(self, mg: RasterModelGrid, fa: FlowAccumulator, latero: LateralEroder, dataset: np.ndarray):

        """
        1つのデータセットのサイズ分だけ計算を実行するメソッド

        Parameters
        ----------
        mg : RasterModelGrid
            初期地形条件を格納したLandlab.RasterModelGridクラスのインスタンス
        fa : FlowAccumulator
            流量条件を格納したLandlab.FlowAccumulatorクラスのインスタンス
        latero : LateralEroder
            側方侵食条件を格納したLandlab.LateralEroderクラスのインスタンス
        dataset : np.ndarray
            記録用のデータセット。shape=(self.t_dataset_size, len(self.recorded_variables), self.InitTP_dict['ncols'] * self.InitTP_dict['nrows'])

        Returns
        -------
        mg, dataset : Tuple[RasterModelGrid, np.ndarray]
            計算後のmgとdataset
        """        
        value_id_dict = self.recorded_value_index_dict
        
        # 初期標高を格納
        initial_flag = self._is_first_loop and (self.additional_calculation_flag==False)
        _start = self.time_index

        if initial_flag:
            # 追加計算ではなく、初回ループの場合は、初期標高を格納する
            dataset[0, value_id_dict["topographic__elevation"], :] = mg.at_node["topographic__elevation"]
            _start += 1

        _end = self.t_dataset_size
        dt = self.simManage_dict['dt']
        U = self.Latero_dict['U']

        try:
        
            for i in range(_start, _end, 1):

                if self.Current_time > self.end_time:
                    break
            
                # 流れ方向の更新
                fa.run_one_step()

                # 侵食量、隆起量の計算
                mg, _ = latero.run_one_step(dt) # 侵食項・堆積項
                mg.at_node["topographic__elevation"][mg.core_nodes] += U * dt # 隆起項
                
                # データセットに格納
                for var_name in self.recorded_variables:
                    j = value_id_dict[var_name]
                    dataset[i, j, :] = mg.at_node[var_name]

                self.Current_time += dt

            self.time_index = 0
        
        except Exception as e:

            # 例外が発生する前までのデータセットをインスタンス変数として保存
            self.mg = mg
            self.dataset = dataset

            raise Exception(f"Error occurred in {self.__class__.__name__} at time = {self.Current_time}. Error message: {str(e)}") from e

        return mg, dataset

    def main(self):

        """
        LateralErosion_LEMのメイン計算を実行するメソッド
        """        

        # 記録変数のインデックス辞書
        value_id_dict = self.recorded_value_index_dict

        # モデルの初期化
        mg, fa, latero = self.init_model()

        # 計算実行前に、計算パラメータと境界条件を保存する。
        calc_condition_params = {
                                 "simManage_conditions": self.simManage_dict, 
                                 "InitTP_conditions": self.InitTP_dict, 
                                 "FlowAcc_conditions": self.FlowAcc_dict,
                                 "Latero_conditions": self.Latero_dict,
                                 }
        if not self.additional_calculation_flag:
            self.write_fildvalue_names(value_names=self.recorded_variables)
        self.write_calc_condition_parameter(calc_condition_params=calc_condition_params)
        self.write_boundary_condition(np.array(mg.status_at_node), comment=f"{self.InitTP_dict['boundary_condition_status']}")
        
        # 計算開始
        progress_bar = np.arange(self.total_loop_num)
        print("Let's start", end="\n")
        for i in tqdm(progress_bar, desc="Processing..."):

            try:

                if self.progress_to_txt:
                    with open(self.progress_txtfile_path, "a") as f:
                        print(f"{i+1}/{self.total_loop_num}, {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}", end="\n", file=f)

                # 計算開始時間
                start_time = copy(self.Current_time)

                # 記録用のデータセットの作成
                dataset = self.init_dataset()

                # 計算実行
                mg, dataset = self.do_computing(mg=mg, fa=fa, latero=latero, dataset=dataset)
                
                # 記録用のデータセットをHDFファイルに書き込む
                for name in self.recorded_variables:
                    j = value_id_dict[name]
                    self.write_fildvalue_Matrix(value=dataset[:, j, :], Gname=name, Dname=str(self.Dataset_Pointer), 
                                                period=f"{start_time}-{self.Current_time}")

                self.Dataset_Pointer += 1

                if self._is_first_loop:
                    self._is_first_loop = False

            except Exception as e:

                # do_computing()内でエラーが発生した場合、計算を中断する
                # ただし、計算途中のデータは保存する
                if self.progress_to_txt:
                    with open(self.progress_txtfile_path, mode="a") as f:
                        now = datetime.datetime.now() # 現在の時刻を取得
                        current_time = now.strftime("%Y-%m-%d %H:%M:%S") # 時刻を文字列に変換
                        f.write(f"now ={current_time}")
                        f.write(f"Error occured in the main loop. Error message: {str(e)}")
                print("Error occured in the main loop")

                # 記録用のデータセットをHDFファイルに書き込む
                dataset = self.dataset
                for name in self.recorded_variables:
                    j = value_id_dict[name]
                    self.write_fildvalue_Matrix(value=dataset[:, j, :], Gname=name, Dname=str(self.Dataset_Pointer), 
                                                period=f"{start_time}-{self.Current_time}")
                
                raise e

        self.mg = mg
        self.fa = fa
        self.latero = latero

        if self.progress_to_txt:
            with open(self.progress_txtfile_path, "a") as f:
                print(f"finish. {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}", end="\n", file=f)


        print("\nFinish!")

class InitialTopographyMaker:

    """
    InitialTopographyMaker
    ----------------------
    初期地形を作成するためのクラス。初期地形の形成と同時に境界条件も設定する。
    """    

    def __init__(self, 
                 dx: int, 
                 nrows: int,
                 ncols: int, 
                 boundary_condition_status: int=0,
                 initz_func: str="create_flat_elevation", 
                 initial_elevation: float=1000.0,
                 create_tp_hyper_param: dict=None,
                 initz_fpath: str=None,
                 random_seed: int=1,):
        """
        Parameters
        ----------
        dx : int
            グリッドサイズ[L]
        nrows : int
            領域の行数
        ncols : int
            領域の列数
        boundary_condition_status : int, optional
            境界条件の種類を指定する整数値（値の詳細はboundary_conditionsを参照）, by default 0
        initz_func : str, optional
            初期地形の作成方法を指定する文字列（値の詳細はcreate_initial_topographyメソッドを参照）, by default "create_flat_elevation"
            {"create_flat_elevation", "arbitrary_topography_define_at_excelfile", "arbitrary_topography"} のみ対応
        initial_elevation : float, optional
            initz_func=="create_flat_elevation"の場合に使用する。境界以外を平行に上昇させる高さ[m], by default 1000.0
        create_tp_hyper_param: dict, optional
            self.create_initial_topographyメソッドに渡す引数を辞書型で指定する。, by default None
        initz_fpath : str, optional
            _description_, by default None
        """        
        
        self.dx = dx
        self.nrows = nrows
        self.ncols = ncols
        self.boundary_condition_status = boundary_condition_status
        self.initz_func = initz_func
        self.initial_elevation = initial_elevation
        self.initz_fpath = initz_fpath
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)
        self.outlet_node = None
        self.mg = self._create_zero_elevation()
        self.mg, self.bd_comment = self.boundary_condition(self.mg)
        self.create_tp_hyper_param = create_tp_hyper_param
        

    def create_initial_topography(self) -> RasterModelGrid:

        """
        初期地形を作成するメソッド。初期地形の作成方法は、initz_funcで指定する。

        Parameters
        ----------
        kwargs : dict, optional
            initz_funcによっては、引数を渡す必要がある。, by default None

        Returns
        -------
        mg : RasterModelGrid
            初期地形を格納したRasterModelGridクラスのインスタンス
        """        
        kwargs = self.create_tp_hyper_param

        if self.initz_func == "create_flat_elevation":
            return self.create_flat_elevation()
        elif self.initz_func == "arbitrary_topography_define_at_excelfile":
            return self.arbitrary_topography_define_at_excelfile()
        elif self.initz_func == "arbitrary_topography":
            if kwargs is None:
                raise ValueError("kwargs must be specified.")
            return self.arbitrary_topography(**kwargs)
        elif self.initz_func == "sine_wave_channel":
            if kwargs is None:
                raise ValueError("kwargs must be specified.")
            return self.sine_wave_channel(**kwargs)
        elif self.initz_func == "Vvalley_topography":
            if kwargs is None:
                raise ValueError("kwargs must be specified.")
            return self.Vvalley_topography(**kwargs)
        else:
            raise ValueError(f"initz_func={self.initz_func} is not supported")

    def create_flat_elevation(self) -> RasterModelGrid:

        """
        境界以外を平行にinitial_elevation[m]だけ上昇させた初期地形を作成するメソッド

        Returns
        -------
        mg : RasterModelGrid
            平行に上昇させた初期地形を格納したRasterModelGridクラスのインスタンス
        """        

        mg = self.mg
        initial_roughness = self.random_state.rand(self.nrows*self.ncols)
        mg.at_node["topographic__elevation"] += initial_roughness
        mg.at_node["topographic__elevation"][mg.core_nodes] += self.initial_elevation # コアノード（境界ではないノード）の標高値を設定
        mg.at_node["topographic__elevation"][mg.boundary_nodes] = 0.0 # 境界ノードは標高０に設定

        return mg
    
    def arbitrary_topography_define_at_excelfile(self) -> RasterModelGrid:

        """
        初期地形を別ファイル（xlsx形式）で定義しておいて、それを読み込むメソッド。
        excelファイルはself.initz_fpathで指定する。また、そのファイルは'z'と'param'の２つのシートを持つ必要がある。

        Returns
        -------
        mg : RasterModelGrid
            初期地形を格納したRasterModelGridクラスのインスタンス
        """        

        if self.initz_fpath != None:
            input_path = self.initz_fpath
        else: 
            raise ValueError("instance param 'initz_fpath' is None")
        
        if not os.path.isfile(input_path):
            raise ValueError(f"{self.initz_fpath} is not exist")
        
        initz_df = pd.read_excel(input_path, sheet_name="z", index_col=0)
        param_df = pd.read_excel(input_path, sheet_name="param", index_col=0)
        assert int(param_df['nrows']) == self.nrows, "nrows does not match"
        assert int(param_df['ncols']) == self.ncols, "ncols does not match"
        assert int(param_df['dx']) == self.dx, "dx does not match"
        
        mg = self.mg
        mg.at_node["topographic__elevation"] += initz_df.iloc[:, 0].to_numpy()

        return mg
    
    def arbitrary_topography(self, z: np.ndarray, ncols: int, nrows: int, dx: int) -> RasterModelGrid:

        """
        引数に与えられた標高値を初期地形として設定するメソッド

        Parameters
        ----------
        z : np.ndarray
            初期地形の標高値を格納した一次元配列。二次元であれば、一次元に変換してから渡す。
        ncols : int
            領域の列数。インスタンス生成時に与えた値と一致している必要がある。
        nrows : int
            領域の行数。インスタンス生成時に与えた値と一致している必要がある。

        Returns
        -------
        mg : RasterModelGrid
            初期地形を格納したRasterModelGridクラスのインスタンス
        """        
        
        if ncols != self.ncols:
            raise ValueError("ncols does not match")
        if nrows != self.nrows:
            raise ValueError("nrows does not match")
        if dx != self.dx:
            raise ValueError("dx does not match")
        
        mg = self.mg
        mg.at_node["topographic__elevation"] = z

        return mg
    
    def Vvalley_topography(self, y_intercept: int=500, ganmma: float=1.5, y_gradient: float=0.01) -> RasterModelGrid:

        """
        V字谷の形状の初期地形を作成するメソッド

        Parameters
        ----------
        y_intercept : int, optional
            y切片, by default 500
        ganmma : float, optional
            y軸方向の勾配はx軸方向の勾配のganmma倍, by default 1.5
        y_gradient : float, optional
            y軸方向の勾配, by default 0.01

        Returns
        -------
        mg : RasterModelGrid
            V字谷の形状の初期地形を格納したRasterModelGridクラスのインスタンス
        """        

        ncols = self.ncols
        nrows = self.nrows
        dx = self.dx
        midx_id = int(np.trunc(ncols/2)) if self.outlet_node is None else self.outlet_node
        
        x = np.arange(0, (ncols)*dx, dx)
        y = np.arange(0, (nrows)*dx, dx)
        xx, yy = np.meshgrid(x, y)

        midx = x[midx_id] # x軸方向の中点
        x_gradient = ganmma * y_gradient # x軸方向の勾配はy軸方向の勾配のganmma倍
        zz = np.where(xx<=midx, -x_gradient*(xx-midx), 0) # 東側の斜面を計算。xxを負の領域にシフトして負の勾配を計算することで、東側の標高は正になる。
        zz = np.where(xx<=midx, zz, x_gradient*(x-midx)) # 西側の斜面を計算。中点を原点にして、東側の斜面と同じ勾配で計算する。
        zz[1:, :] += yy[1:, :]*y_gradient + y_intercept # y軸方向の勾配による標高の上昇と、y切片による標高の上昇を計算する。
        
        mg = self.mg
        mg.at_node["topographic__elevation"] = zz

        return mg
    
    def sine_wave_channel(self, A: float, L: float, gradient_y: float, 
                          channel_depth: float=3.0, intercept_y: float=1000) -> RasterModelGrid:

        """
        x = A * sin(2πy/L) の形のsin波を作成するメソッド。

        Parameters
        ----------
        A : float
            振幅
        L : float
            波長
        gradient_y : float
            y軸方向の標高勾配。
        channel_depth : float, optional
            河道の深さでsinカーブの地点が周囲よりこの数値だけ低くなる, by default 3.0
        intercept_y : float, optional
            下流端での標高, by default 1000

        Returns
        -------
        mg : RasterModelGrid
            sin波を格納したRasterModelGridクラスのインスタンス
        """        
        x_max = self.ncols * self.dx
        y_max = self.nrows * self.dx
        mid_x = np.trunc(self.ncols/2)*self.dx
        xx, yy = np.meshgrid(np.linspace(0, x_max, self.ncols), np.linspace(0, y_max, self.nrows))
        zz = gradient_y * yy + intercept_y # 標高値
        y_at_channel = np.linspace(0, y_max, self.nrows)

        omega = 2 * np.pi / L # 各速度は1周期で2π進むので、1周期での角速度を求める。
        t = y_at_channel / L # 一波長Lで１周期なので1周期でy軸の方向の距離を規格化する。

        x_at_channel = A * np.sin(omega * t) + mid_x # x軸方向の位置を求める。この際、x方向の中点を中心とする。
        x_at_channel = np.ceil(x_at_channel / self.dx) * self.dx # x軸方向の位置をグリッドサイズで切り上げる。

        # 河道の座標をマスクする
        mask = np.full(shape=(self.nrows, self.ncols), fill_value=False)
        for i in range(self.ncols):
            x_c = x_at_channel[i]
            for j in range(self.nrows):
                y_c = y_at_channel[j]
                if np.logical_and(xx[i, j]==x_c, yy[i, j]==y_c):
                    mask[i, j] = True
        print("mask", np.where(mask))
        
        # 河道の座標をchannel_depthだけ下げる
        zz[mask] -= channel_depth

        # zz = np.flip(zz, axis=0) # y軸方向を反転する。北側(上側)が上流になるようにする。

        mg = self.mg
        mg.at_node["topographic__elevation"] = zz

        return mg

    def _create_zero_elevation(self) -> RasterModelGrid:

        """
        標高が全てゼロの地形を作成するメソッド

        Returns
        -------
        mg : RasterModelGrid
            標高が全てゼロの地形を格納したRasterModelGridクラスのインスタンス
        """        

        mg = RasterModelGrid((self.nrows, self.ncols), xy_spacing=self.dx)
        mg.add_zeros("topographic__elevation", at="node")

        return mg
    
    def boundary_condition(self, mg: RasterModelGrid) -> Tuple[RasterModelGrid, str]:

        """
        境界条件を設定するメソッド。境界条件の種類の設定は、boundary_condition_statusで指定する。

        Returns
        -------
        Tupple[mg, comment]

        mg: RaseterModelGrid
            境界条件を設定したRasterModelGridクラスのインスタンス
        comment: str
            境界条件の説明文
        """        

        bd_type_status = self.boundary_condition_status

        if bd_type_status == 0:
            return self.boundary_condition_0(mg)
        elif bd_type_status == 1:
            return self.boundary_condition_1(mg)
        else:
            raise ValueError(f"Not set bd_type_status={bd_type_status}")

    def boundary_condition_0(self, mg: RasterModelGrid) -> Tuple[RasterModelGrid, str]:
        
        """
        領域の下側(南側)のみ開放した境界条件を設定するメソッド

        Returns
        -------
        Tupple[mg, comment]

        mg: RaseterModelGrid
            境界条件を設定したRasterModelGridクラスのインスタンス
        comment: str
            境界条件の説明文
        """        
        _comment = "領域の下側のみ開放した境界"
        is_closed_boundary = np.logical_and(
            mg.status_at_node != mg.BC_NODE_IS_CORE, # コアノードではない　かつ
            mg.y_of_node > np.amin(mg.y_of_node)
        )
        mg.status_at_node[is_closed_boundary] = mg.BC_NODE_IS_CLOSED

        return mg, _comment

    def boundary_condition_1(self, mg: RasterModelGrid) -> Tuple[RasterModelGrid, str]:
        """
        下側(南側)の中央セル（河口）のみを解放した境界

        Returns
        -------
        Tupple[mg, comment]

        mg: RaseterModelGrid
            境界条件を設定したRasterModelGridクラスのインスタンス
        comment: str
            境界条件の説明文
        """
        _comment = "下境界（河口）の中央セルのみを解放した境界"
        mg.set_status_at_node_on_edges(
            right=mg.BC_NODE_IS_CLOSED,
            top=mg.BC_NODE_IS_CLOSED,
            left=mg.BC_NODE_IS_CLOSED,
            bottom=mg.BC_NODE_IS_CLOSED,
        )
        mid_x = np.trunc(self.ncols/2)*self.dx
        outlet_flag = np.logical_and(mg.x_of_node==mid_x, mg.y_of_node==np.amin(mg.y_of_node))
        inlet_flag = np.logical_and(mg.x_of_node==mid_x, mg.y_of_node==np.amax(mg.y_of_node))
        # print(np.where(inlet_flag), np.where(outlet_flag))
        self.inlet_node = np.where(inlet_flag)[0][0]

        if self.outlet_node != None: # 初期地形を先に設定し、outlet_nodeが先に決まっていた場合
            assert self.outlet_node == np.where(outlet_flag)[0][0], "outlet_node does not match"
        else: # 初期地形よりも境界条件の設定を先にした場合
            self.outlet_node = np.where(outlet_flag)[0][0]
        flag = outlet_flag#+inlet_flag
        mg.status_at_node[flag] = mg.BC_NODE_IS_FIXED_VALUE
        # print(mid_x, flag)
        return mg, _comment


class AnimationMaker(LataralSimilateManager):

    def __init__(self, dirpath: str, HDF_fname: str, draw_start_time: int=0, draw_end_time: int=None, 
                 draw_dt: int=1, outfname_key: str=None, draw_val_name: str='topographic__elevation'):
                 
        super().__init__(dirpath=dirpath, HDF_fname=HDF_fname, additional_calculation_flag=True, additional_calculation_yr=0)
        self.draw_start_time = draw_start_time
        self.draw_end_time = draw_end_time
        self.draw_dt = draw_dt
        self._check_draw_time()
        self.outfname_key = outfname_key
        self.draw_val_name = draw_val_name

        outfname_templete = f"{draw_val_name}_s{self.draw_start_time}_e{self.draw_end_time}_dt{self.draw_dt}"
        self.outfname = f"{outfname_templete}.mp4" if outfname_key is None else f"{outfname_key}_{outfname_templete}.mp4"
        self.outfpath = os.path.join(self.dirpath, self.outfname)
        
        self.draw_times = np.arange(self.draw_start_time, self.draw_end_time, self.draw_dt)

    def _check_draw_time(self):

        SimManage_param = self.read_calc_condition_param(condition_keys=["simManage_conditions"])[0]
        dt = SimManage_param['dt'].value
        start_time = SimManage_param['start_time'].value
        end_time = SimManage_param['end_time'].value

        if self.draw_end_time is None:
            self.draw_end_time = end_time

        if self.draw_start_time < start_time:
            raise ValueError(f"draw_start_time={self.draw_start_time} is smaller than start_time={start_time}")
        
        if self.draw_end_time > end_time:
            raise ValueError(f"draw_end_time={self.draw_end_time} is larger than end_time={end_time}")
        
        if self.draw_dt % dt != 0:
            raise ValueError(f"draw_dt={self.draw_dt} must be a multiple of dt={dt}")
        
        print(f"draw_start_time={self.draw_start_time}, draw_end_time={self.draw_end_time}, draw_dt={self.draw_dt}")

    def _label_name(self, value: str) -> str:
        if value == "topographic__elevation":
            return "elevation [m]"
        elif value == 'phd_curvature':
            return "phase delay curvature [1/m]"
        elif value == "latero__rate":
            return "lateral erosion rate [m/yr]"
        elif value == "drainage_area":
            return "drainage_area [$m^2$]"
        elif value == "fai":
            return "lateral/vertical ratio [-]"
        elif value == "surface_water__discharge":
            return "surface water discharge [$m^3/s$]"
        elif value == "flow_depth":
            return "flow depth [m]"
        else:
            raise ValueError("wrong filed name!, check HDFfile dataset name")
        
    def get_minmax(self, Yr: int, value: str) -> Tuple[float, float]:

        """
        指定した年の指定した変数の最小値と最大値を返すメソッド

        Returns
        -------
        Tuple[min_val, max_val]
            指定した年の指定した変数の最小値と最大値
        """        

        val = self.read_oneYr_filedvalue(Yr=Yr, name=value)
        min_val = np.min(val)
        max_val = np.max(val)

        return min_val, max_val
    
    def make_anim(self, color_limit: bool=True):
        plt.rcParams['figure.figsize'] = (7, 5)

        tp_param = self.read_calc_condition_param(condition_keys=["InitTP_conditions"])[0]
        ncols = tp_param['ncols'].value
        nrows = tp_param['nrows'].value
        dx = tp_param['dx'].value

        cmap = copy(mpl.cm.get_cmap("gist_earth"))
        mg_video = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
        mg_video.status_at_node, _comment = self.read_boundary_condition()
        fig, ax = plt.subplots(1, 1)
        writer = animation.FFMpegWriter(fps=10)
        writer.setup(fig, self.outfpath, 200)
        # print(_comment)
        colorbar_lavel = self._label_name(value=self.draw_val_name)

        limits = None
        if color_limit:
            min_val, max_val = self.get_minmax(Yr=self.draw_times[-1], value=self.draw_val_name)
            limits = (min_val, max_val)

        for _t in self.draw_times:
            
            val = self.read_oneYr_filedvalue(Yr=_t, name=self.draw_val_name)
            imshow_grid(mg_video, val, grid_units=["m", "m"], 
                        colorbar_label=colorbar_lavel, cmap=cmap, 
                        plot_name=f"{_t} years",
                        limits=limits,)
            # plt.title(f"{self.base_yr+_yr} years") limits=limits ,

            # Capture the state of `fig`.
            writer.grab_frame()

            # Remove the colorbar and clear the axis to reset the
            # figure for the next animation timestep.
            plt.gci().colorbar.remove()
            ax.cla()

        writer.saving(fig, self.outfpath, 200)
        plt.close()
        writer.finish()

    def make_anim_longitudinal_profile(self, outlet_node: int=None):

        plt.rcParams['figure.figsize'] = (7, 5)

        tp_param = self.read_calc_condition_param(condition_keys=["InitTP_conditions"])[0]
        ncols = tp_param['ncols'].value
        nrows = tp_param['nrows'].value
        dx = tp_param['dx'].value

        mg_video = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
        mg_video.add_zeros("topographic__elevation", at="node")
        mg_video.status_at_node, _comment = self.read_boundary_condition()

        mg_video.at_node["topographic__elevation"] = self.read_oneYr_filedvalue(Yr=self.draw_start_time, name="topographic__elevation")

        fig, ax = plt.subplots(1, 1)
        writer = animation.FFMpegWriter(fps=10)
        outpath = copy(self.outfpath[:-4]) + "_longitudinal_profile.mp4"
        writer.setup(fig, outpath, 200)
        # print(_comment)
        y_label = self._label_name(value=self.draw_val_name)

        fa_video = FlowAccumulator(
            mg_video, 
            surface="topographic__elevation", # FlowAccumlatorクラスの初期化に使用する地形データ
            flow_director="FlowDirectorD8", # D8アルゴリズムの使用
            runoff_rate=None,
            depression_finder="DepressionFinderAndRouter", #"DepressionFinderAndRouter"
        )
        fa_video.run_one_step()

        if outlet_node != None:
            prf = ChannelProfiler(mg_video, number_of_watersheds=1, main_channel_only=True, outlet_nodes=[outlet_node])
        else:
            prf = ChannelProfiler(mg_video, number_of_watersheds=1, main_channel_only=True)

        for _t in self.draw_times:
            
            if _t: # 初期地形の場合は上記で設定済み
                mg_video.at_node[self.draw_val_name] = self.read_oneYr_filedvalue(Yr=_t, name=self.draw_val_name) # 一年分のデータを読み込む
                fa_video.run_one_step() # 流域の流出量, 流向を計算する
            prf.run_one_step() # 流路のプロファイルを計算する
            prf.plot_profiles(color="blue", ylabel=y_label, title=f"{_t} years")
            
            # Capture the state of `fig`.
            writer.grab_frame()

            # Remove the colorbar and clear the axis to reset the
            # figure for the next animation timestep.

            ax.cla()

        writer.saving(fig, self.outfpath, 200)
        plt.close()
        writer.finish()

    def make_anim_with_channelLine(self, 
                                   fps: int=5,
                                   color_limit: bool=True,
                                   minimum_outlet_threshold: float=0.,
                                   minimum_channel_threshold: float=0.,
                                  ):
        
        """
        標高と共に河道の流線を描画するメソッド

        Parameters
        ----------
        minimum_outlet_threshold : float, optional
            河口が持つ流域面積の閾値（河口と認識される最小値）, by default 0.
        minimum_channel_threshold : float, optional
            谷頭が持つ流域面積の閾値（これ以上で河川と認識）, by default 0.
        """        

        self.draw_val_name = "topographic__elevation"
        plt.rcParams['figure.figsize'] = (7, 5)

        tp_param = self.read_calc_condition_param(condition_keys=["InitTP_conditions"])[0]
        ncols = tp_param['ncols'].value
        nrows = tp_param['nrows'].value
        dx = tp_param['dx'].value

        mg_video = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
        mg_video.add_zeros("topographic__elevation", at="node")
        mg_video.status_at_node, _comment = self.read_boundary_condition()

        mg_video.at_node["topographic__elevation"] = self.read_oneYr_filedvalue(Yr=self.draw_start_time, name="topographic__elevation")

        fig, ax = plt.subplots(1, 1)
        writer = animation.FFMpegWriter(fps=fps)
        outpath = copy(self.outfpath[:-4]) + "_with_channelLine.mp4"
        writer.setup(fig, outpath, 200)
        # print(_comment)
        
        fa_video = FlowAccumulator(
            mg_video, 
            **self.FlowAcc_dict,
        )
        fa_video.run_one_step()

        prf = ChannelProfiler(
                mg_video, 
                number_of_watersheds=1, #出力する流域の数
                main_channel_only=False, #本流だけ描画
                minimum_outlet_threshold=minimum_outlet_threshold, #河口が持つ流域面積の閾値（河口と認識される最小値）
                minimum_channel_threshold=minimum_channel_threshold, #谷頭が持つ流域面積の閾値（これ以上で河川と認識）
            )
        
        limits = None
        if color_limit:
            min_val, max_val = self.get_minmax(Yr=self.draw_times[-1], value=self.draw_val_name)
            limits = (min_val, max_val)
        
        kwds = {
                "grid_units":["m", "m"],
                "colorbar_label":"elevation (m)",
                "cmap": copy(mpl.cm.get_cmap("gist_earth")),
                "limits": limits,
            }
        
        _first_loop = True
        
        with writer.saving(fig, outpath, dpi=200):

            for _t in self.draw_times:
                
                if not _first_loop: # 初期地形の場合は上記で設定済み
                    mg_video.at_node[self.draw_val_name] = self.read_oneYr_filedvalue(Yr=_t, name=self.draw_val_name) # 一年分のデータを読み込む
                    fa_video.run_one_step() # 流域の流出量, 流向を計算する
                else:
                    _first_loop = False
                prf.run_one_step() # 流路のプロファイルを計算する
                prf.plot_profiles_in_map_view(color="black", plot_name=f"{_t} years",**kwds)
                
                # Capture the state of `fig`.
                writer.grab_frame()

                # Remove the colorbar and clear the axis to reset the
                # figure for the next animation timestep.
                plt.gci().colorbar.remove()
                ax.cla()
        plt.close()
        # writer.saving(fig, self.outfpath, 200)
        
        # writer.finish()
        
    def make_anim_with_channelLine_useMask(self, 
                                   fps: int=10,
                                   minimum_channel_threshold: float=0.,
                                  ):
        
        """
        標高と共に河道の流線を描画するメソッド

        Parameters
        ----------
        fps: int, optional
            1秒あたりのフレーム数, by default 5
        color_limit: bool, optional
            標高のカラーバーの最大値と最小値を時間変化を通じて揃えるかどうか, by default True
        minimum_channel_threshold : float, optional
            谷頭が持つ流域面積の閾値（これ以上で河川と認識）, by default 0.
        """        

        plt.rcParams['figure.figsize'] = (7, 5)

        self.draw_val_name = "topographic__elevation"

        tp_param = self.read_calc_condition_param(condition_keys=["InitTP_conditions"])[0]
        ncols = tp_param['ncols'].value
        nrows = tp_param['nrows'].value
        dx = tp_param['dx'].value

        cmap = copy(mpl.cm.get_cmap("gist_earth"))
        mg_video = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dx)
        mg_video.status_at_node, _comment = self.read_boundary_condition()
        fig, ax = plt.subplots(1, 1)
        writer = animation.FFMpegWriter(fps=fps)
        outpath = copy(self.outfpath[:-4]) + "_with_channelLine.mp4"
        writer.setup(fig, outpath, 200)
        # print(_comment)
        colorbar_lavel = self._label_name(value=self.draw_val_name)

        min_val, max_val = self.get_minmax(Yr=self.draw_times[-1], value=self.draw_val_name)
        limits = (min_val, max_val)

        _is_first_loop = True

        for _t in self.draw_times:
            
            # valは全てのノードの標高値を持つ一次元配列
            val = self.read_oneYr_filedvalue(Yr=_t, name=self.draw_val_name)

            if _is_first_loop:
                allow_colorbar = True
                _is_first_loop = False
            else:
                allow_colorbar = False

            imshow_grid(mg_video, val, grid_units=["m", "m"], 
                        colorbar_label=colorbar_lavel, cmap=cmap, 
                        plot_name=f"{_t} years",
                        limits=limits,
                        allow_colorbar=allow_colorbar,)
            
            da = self.read_oneYr_filedvalue(Yr=_t, name="drainage_area")
            masked_array = (da < minimum_channel_threshold) # 流域面積がminimum_channel_threshold以下のノードをTrueとするマスク配列
            imshow_grid(mg_video, masked_array, color_for_closed=None, allow_colorbar=False, show_elements=True) # Trueの部分が透過される

            # Capture the state of `fig`.
            writer.grab_frame()

            # Remove the colorbar and clear the axis to reset the
            # figure for the next animation timestep.
            # plt.gci().colorbar.remove()
            ax.cla()

        writer.saving(fig, self.outfpath, 200)
        plt.close()
        writer.finish()


class AnalysisManager(LataralSimilateManager):

    def __init__(self, dirpath: str, HDF_fname: str):
                 
        super().__init__(dirpath=dirpath, HDF_fname=HDF_fname, additional_calculation_flag=True, additional_calculation_yr=0)
        
    def test(self):
        print("test function")
        