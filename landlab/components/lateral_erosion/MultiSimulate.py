from landlab.components.lateral_erosion import LataralSimilateManager, AnimationMaker
from copy import deepcopy
import numpy as np
import multiprocessing
import os

class MultiSimilateManager:
    
    def __init__(self, ParamList: list):
        self.processingNum = len(ParamList)
        self.instanceList = [] # それぞれ異なる計算条件でのLataralSimilateManagerクラスのインスタンスを要素にもつ
        for pm in ParamList:
            self.instanceList.append(LataralSimilateManager(**pm))
                           
    def main(self):
        p = multiprocessing.Pool(self.processingNum)
        print("Now doing main calculation......................")
        p.map(self.does_computing, self.instanceList)
        print("--------Finish--------------------")
        
    def does_computing(self, SimManage: LataralSimilateManager):
        return SimManage.main()
    
def delete_files(dirpath: str, type_of_file: str=".txt"):
    """
    指定されたディレクトリ内のファイル(type_of_file 拡張子）を削除する関数

    Args:
        dirpath: ディレクトリの絶対パス
    """
    # print(f"Delete text files in {dirpath}")
    # print(type(dirpath))
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        if os.path.isfile(filepath) and filename.endswith(type_of_file):
            os.remove(filepath)
            # print(f"Deleted file: {filename}")

if __name__ == "__main__":
    
    """
    このセクションの構成
    1. 基本パラメータセットの作成
    2. 基本パラメータセットをもとに、複数のパラメータセットを作成
    3. 複数のパラメータセットをもとに、MultiSimilateManagerクラスのインスタンスを作成
    4. MultiSimilateManagerクラスのインスタンスをもとに、並列計算を実行
    5. 1つ1つの計算結果をもとに、アニメーションを作成
    """    
    
    # 1. 基本パラメータセットの作成
    #==========================================================================================================================================================================================
    initz_func = "Vvalley_topography" # 初期地形を作成する関数を指定

    # 1.1 初期地形、境界条件の設定
    if initz_func == "sine_wave_channel":
        # sine_wave_channel関数は離散化表現のためあまり良い初期地形にはならないので使用頻度は低い
        create_tp_hyper_param = {
            'A' : 30,
            'L' : 200,
            'gradient_y' : 0.001,
            'channel_depth' : 100,
            'intercept_y' : 1000,
        }
    elif initz_func == "Vvalley_topography":
        create_tp_hyper_param = {
            "y_intercept" : 10,  # y切片->下流端での標高値
            "ganmma" : 1.1, # V字谷の開き具合を決めるx軸方向の傾きに関するパラメータ。x_gradient = ganmma * y_gradientになる
            "y_gradient" : 0.01, # y軸方向の傾き。初期地形での河床勾配に相当する。
        }
    else:
        create_tp_hyper_param = None

    InitTP_param = {
        'dx' : 100, # グリッドサイズ[m]
        'nrows' : 200, # 領域の行数
        'ncols' : 50, # 領域の列数
        'boundary_condition_status' : 1, # 境界条件の種類を指定(0: 下流端を解放, 1: 河口部分の１セルのみ解放)
        'initial_elevation' : 1000.0, # 初期地形の高さ. initz_func = "sine_wave_channel" or "Vvalley_topography"の場合は無視される
        'initz_func' : initz_func, # 初期地形を作成する関数を指定。上で指定したものを選択
        'create_tp_hyper_param' : create_tp_hyper_param, # 初期地形を作成する関数に渡す追加パラメータを指定。initz_funcの種類によって内容が異なるので上で分岐
    }

    # 1.2 流出方向の計算に関するパラメータ
    FlowAcc_param = {
        'surface' : 'topographic__elevation', # FlowAccumlatorクラスの初期化に使用する地形データ
        'flow_director' : 'FlowDirectorD8', # D8アルゴリズムの使用
        'runoff_rate' : 15, # 降水量の設定
        'depression_finder' : 'DepressionFinderAndRouter', # 窪地に対する処理アルゴリズムの指定。詳しくはLandlabのドキュメントを参照
    }

    # 1.3 側方侵食に関するパラメータ
    Latero_param = {
        "U" : 0.001, #隆起速度 [m/yr] 
        "Kv" : 1e-4, #下刻侵食係数Kv[1/yr] ,
        "Kl_ratio" : 1.0, #下刻侵食係数Kvと側刻侵食係数Klの比率
        "dp_coef" : 0.0008, # 水深をべき乗近似した際の係数
        "dp_exp" : 0.4, # 水深をべき乗近似した際の指数
        "wid_coef" : 0.004, # 川幅をべき乗近似した際の係数
        "F" : 0.02, # 摩擦係数
        "latero_mech" : "TB", #トータルブロックモデルTB or アンダーカッティングスランプモデルを指定
        "alph": 0.0, # 堆積物項の影響度(0~1で指定する。 0: 影響なし, 1: 影響大)
        "inlet_area": 0, # 上流端流域面積[m^3]
        "qsinlet": 0, # 流入土砂量[m^3/yr]
        "solver": "delay_rs", # １ステップの計算に使用するLateralEroderクラスのメソッドタイプ{"basic", "adaptive", "delay_rs"}
        "use_Q": True, # StreamPowerModelの計算に流量を使用するかどうか。しない場合は流域面積を使用する。
        "thresh_da" : 8e7, # 流路と認識する閾値流域面積[m^2] use_Q=Trueの場合は単位が[m^3/s]
        "phase_delay_node_num": 1, # 位相遅れ曲率を計算する際に何個上流側のノードを参照するか
        "fai_alpha" : 3.3, # 側方/下方刻侵食係数の計算に使用するパラメータ。sol_type="delay_rs"の場合のみ使用
        "fai_beta" : -0.25, # 側方/下方刻侵食係数の計算に使用するパラメータ。sol_type="delay_rs"の場合のみ使用
        "fai_gamma" : -0.85, # 側方/下方刻侵食係数の計算に使用するパラメータ。sol_type="delay_rs"の場合のみ使用
        "fai_C": -64.0, # 側方/下方刻侵食係数の計算に使用するパラメータ。sol_type="delay_rs"の場合のみ使用
    }

    # 1.4 計算結果としてHDFファイルに保存する変数を指定
    recorded_variables = ['topographic__elevation', 'curvature', 'phd_curvature',
                        'latero__rate', 'fai', "drainage_area"] 

    # 1.5 計算マネジメントに関するパラメータ
    SimManage_param = {
        'dirpath' : r'Z:\miyata\landlab_exp\phase delay model\test', # 計算結果の保存先ディレクトリ
        'HDF_fname' : 'test_Vvalley_topography.h5', # 計算結果を保存するHDFファイル名
        'start_time' : 0, # 計算開始時刻[T]
        'end_time' : 40000, # 計算終了時刻[T]
        'time_unit' : 'yr', # 時間の単位
        'space_unit' : 'm', # 空間の単位
        'wight_unit' : 'kg', # 重さの単位
        'dt' : 1, # 時間刻み幅
        't_dataset_size' : 200, # HDFファイルに保存するデータセットのサイズ。全ての結果をN=総計算時間/t_dataset_size個のチャンクに分割して保存される
        'InitTP_param' : InitTP_param, # 初期地形に関するパラメータ
        'FlowAcc_param' : FlowAcc_param, # 流出方向の計算に関するパラメータ
        'Latero_param' : Latero_param, # 側方侵食に関するパラメータ
        'additional_calculation_flag' : False, # 計算モードの指定。False: 全て初期化して計算を実行, True: HDFファイルに保存されている計算結果を読み込んで追加計算を実行
        'additional_calculation_yr' : 1, # 追加計算を実行する場合の計算期間
        'progress_to_txt' : True, # 計算の進捗状況をtxtファイルに出力するかどうか
        'recorded_variables' : recorded_variables, # 計算結果としてHDFファイルに保存する変数を指定
    }

    # 2. 基本パラメータセットをもとに、複数のパラメータセットを作成
    #==========================================================================================================================================================================================
    
    solvers = ["basic", "delay_rs"] # 使用するLateralEroderクラスのメソッドタイプ
    dirpaths = [ 
                r'Z:\miyata\landlab_exp\phase delay model\test',
                r'Z:\miyata\landlab_exp\phase delay model\test_delay_rs',
               ]   
    HDF_fnames = [
                  'test_Vvalley_topography.h5',
                  'test_Vvalley_topography_delay_rs.h5',
                ]
    
    ParamList = []

    for solver, dirpath, HDF_fname in zip(solvers, dirpaths, HDF_fnames):

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        SimManage_param_copy = deepcopy(SimManage_param)
        SimManage_param_copy["Latero_param"]["solver"] = solver
        SimManage_param_copy["dirpath"] = dirpath
        SimManage_param_copy["HDF_fname"] = HDF_fname
        ParamList.append(SimManage_param_copy)

    # 3. 複数のパラメータセットをもとに、MultiSimilateManagerクラスのインスタンスを作成
    #==========================================================================================================================================================================================
    MultiSimManage = MultiSimilateManager(ParamList)

    # 4. MultiSimilateManagerクラスのインスタンスをもとに、並列計算を実行
    # ==========================================================================================================================================================================================
    try:
        MultiSimManage.main()
    except Exception as e:
        print(e)
        print("Error occured. Delete all text files in the directory.")
        pass

    # 5. 1つ1つの計算結果をもとに、アニメーションを作成
    #==========================================================================================================================================================================================
    draw_values = ["topographic__elevation", "latero__rate",] # 
    
    for draw_val_name in draw_values:
        for dirpath, HDF_fname in zip(dirpaths, HDF_fnames):
            anim_param = {
                'dirpath' : dirpath,
                'HDF_fname' : HDF_fname,
                'draw_start_time' : 0,
                'draw_end_time' : None,
                'draw_dt' : 200,
                'outfname_key' : None,
                'draw_val_name' : draw_val_name,
            }

            anim_Maker = AnimationMaker(**anim_param)
            anim_Maker.make_anim()

    
