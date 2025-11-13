"""
Summary:
    このモジュールは国土地理院で公開されている、セミダイナミック補正を行うための
    モジュールです。セミダイナミック補正は地殻変動を考慮した位置補正を行うための手法であり、
    国土地理院が提供するパラメーターファイルを使用して補正を実施します。
    詳しくは以下のURLを参照してください。
    https://www.gsi.go.jp/sokuchikijun/semidyna.html
"""

import datetime
import os
from typing import Optional

import pandas as pd

from .config import XY, XYZ, SemiDynaParam
from .utils import dimensional_count
from .web import fetch_corrected_semidynamic_from_web

global PARAM_FILE_DIR
PARAM_FILE_DIR = os.path.join(os.path.dirname(__file__), "data")

global DIGITS_SCALE
DIGITS_SCALE = 10**10


class SemiDynamic(object):
    def __init__(self, measurement_date: datetime.date):
        if datetime.datetime.now().date() < measurement_date:
            raise ValueError("Measurement date cannot be in the future.")

        if measurement_date.month < 4:
            # 3月以前の場合、前年のパラメーターファイルを使用する
            self.year = measurement_date.year - 1
        else:
            # 4月以降の場合、当年のパラメーターファイルを使用する
            self.year = measurement_date.year
        self.datetime = measurement_date
        # パラメーターファイルのデータフレームを初期化
        self.param_df = None

    def read_parameter(self) -> pd.DataFrame:
        """
        ## Summary:
            セミダイナミック補正用のパラメーターファイルを読み込み、データフレームとして返す。
            MeshCodeは基準地域メッシュのメッシュコードを示し、dB(sec), dL(sec), dH(m)は
            それぞれ経度、緯度、高度の補正値を示す。lon_min, lat_min, lon_max, lat_max
            はメッシュの範囲を示す。
        Returns:
            pd.DataFrame: セミダイナミック補正用のパラメーターファイルのデータフレーム
        ## DataFrame:
            MeshCode  dB(sec)  dL(sec)    dH(m)   lon_min    lat_min   lon_max    lat_max
            36230600 -0.05283  0.03851  0.07238  123.7500  24.000000  123.7625  24.008333
            36230605 -0.05226  0.03863  0.06994  123.8125  24.000000  123.8250  24.008333
            36230700 -0.05100  0.03841  0.06324  123.8750  24.000000  123.8875  24.008333
            36230555 -0.05345  0.03806  0.07369  123.6875  24.041667  123.7000  24.050000
            36230650 -0.05260  0.03858  0.07091  123.7500  24.041667  123.7625  24.050000
        """
        if self.param_df is not None:
            return self.param_df
        param_file_name = f"SemiDyna{self.year}.csv"
        param_file_path = os.path.join(PARAM_FILE_DIR, param_file_name)
        if not os.path.exists(param_file_path):
            raise FileNotFoundError(f"Parameter file for year {self.year} not found.")
        self.param_df = pd.read_csv(param_file_path)
        return self.param_df

    def query_parameters(self, lon: float, lat: float) -> dict[str, SemiDynaParam]:
        """
        ## Summary:
            指定された経度・緯度に基づいて、対応するセミダイナミック補正パラメーターを
            取得する。この戻り値のリストは、4つのパラメーターを返すはずです。4つでない
            場合は、なにかしらの問題が発生している可能性があります。
        Args:
            lon (float): 経度(10進数)
            lat (float): 緯度(10進数)
        Returns:
            dict[str, SemiDynaParam]:
            4つのセミダイナミック補正パラメーターを含む辞書。キーは"upper_left",
            "lower_left", "upper_right", "lower_right"であり、値はSemiDynaParam
            オブジェクトです。
        ## SemiDynaParam:
            code (str): メッシュコード
            lon (float): メッシュの最小経度
            lat (float): メッシュの最大緯度
            delta_x (float): 経度補正値(秒)
            delta_y (float): 緯度補正値(秒)
            delta_z (float): 高度補正値(メートル)
        """
        if self.param_df is None:
            self.read_parameter()

        # 指定座標を囲む4つのメッシュを正しく選択
        # 最も近い数個のメッシュから、実際に4象限に配置されているものを選択
        self.param_df["distance"] = (
            (self.param_df["lon_min"] + self.param_df["lon_max"]) / 2 - lon
        ) ** 2 + ((self.param_df["lat_min"] + self.param_df["lat_max"]) / 2 - lat) ** 2

        # より多くのメッシュを候補として取得
        candidate_df = self.param_df.nsmallest(20, "distance").copy()

        result = {}

        # 各象限から最も近いメッシュを選択
        for _, row in candidate_df.iterrows():
            mesh_center_lon = (row["lon_min"] + row["lon_max"]) / 2
            mesh_center_lat = (row["lat_min"] + row["lat_max"]) / 2

            param = SemiDynaParam(
                code=str(int(row["MeshCode"])),
                lon=float(row["lon_min"]),
                lat=float(row["lat_max"]),
                delta_x=float(row["dL(sec)"]),
                delta_y=float(row["dB(sec)"]),
                delta_z=float(row["dH(m)"]),
            )

            # 象限に基づいてメッシュを分類
            if mesh_center_lon <= lon and mesh_center_lat >= lat:
                if "upper_left" not in result:
                    result["upper_left"] = param
            elif mesh_center_lon > lon and mesh_center_lat >= lat:
                if "upper_right" not in result:
                    result["upper_right"] = param
            elif mesh_center_lon <= lon and mesh_center_lat < lat:
                if "lower_left" not in result:
                    result["lower_left"] = param
            elif mesh_center_lon > lon and mesh_center_lat < lat:
                if "lower_right" not in result:
                    result["lower_right"] = param

            # 4つすべてが見つかったら終了
            if len(result) == 4:
                break

        # 不足している象限を最も近いメッシュで補完
        if len(result) < 4:
            missing_positions = ["upper_left", "upper_right", "lower_left", "lower_right"]
            existing_positions = list(result.keys())
            for pos in existing_positions:
                missing_positions.remove(pos)

            for i, pos in enumerate(missing_positions):
                if i < len(candidate_df):
                    row = candidate_df.iloc[i + len(result)]
                    result[pos] = SemiDynaParam(
                        code=str(int(row["MeshCode"])),
                        lon=float(row["lon_min"]),
                        lat=float(row["lat_max"]),
                        delta_x=float(row["dL(sec)"]),
                        delta_y=float(row["dB(sec)"]),
                        delta_z=float(row["dH(m)"]),
                    )
        if len(result) != 4:
            raise ValueError(
                f"Expected 4 parameters, but got {len(result)}. "
                "Check the input coordinates."
            )
        return result

    def correction_delta(self, lon: float, lat: float) -> XYZ:
        """
        ## Summary:
            指定された経度・緯度・標高に基づいて、セミダイナミック補正値を計算する。
            セミダイナミック補正の計算は、指定した位置に最も近い4つのメッシュの補正値を
            使用して、Bilinear補完を行うことで実施されます。
        Args:
            lon (float): 経度(10進数)
            lat (float): 緯度(10進数)
        Returns:
            XYZ: XYZ形式のセミダイナミック補正値
        """
        lon = lon
        lat = lat
        # 補正値の取得
        params = self.query_parameters(lon=lon, lat=lat)
        upper_left = params["upper_left"]
        upper_right = params["upper_right"]
        lower_left = params["lower_left"]
        lower_right = params["lower_right"]
        # Bilinear補間のための座標系設定
        # 4つのメッシュの位置を確認
        x1, y2 = upper_left.lon, upper_left.lat  # 左上 (x1, y2)
        x2, y1 = lower_right.lon, lower_right.lat  # 右下 (x2, y1)

        # ターゲット座標
        x, y = lon, lat

        # ゼロ除算を避けるためのチェック
        if x2 == x1 or y1 == y2:
            # 距離がゼロの場合、最も近いメッシュの値を使用
            return XYZ(
                x=upper_left.delta_x / 3600,
                y=upper_left.delta_y / 3600,
                z=upper_left.delta_z,
            )

        data = {"x": x, "y": y, "x1": x1, "x2": x2, "y1": y1, "y2": y2}
        # 標準的なバイリニア補間の計算
        delta_x = self._bilinear_interpolation_standard(
            f_ll=lower_left.delta_x,  # f(x1, y1) 左下
            f_lr=lower_right.delta_x,  # f(x2, y1) 右下
            f_ul=upper_left.delta_x,  # f(x1, y2) 左上
            f_ur=upper_right.delta_x,  # f(x2, y2) 右上
            **data,
        )
        delta_y = self._bilinear_interpolation_standard(
            f_ll=lower_left.delta_y,  # f(x1, y1) 左下
            f_lr=lower_right.delta_y,  # f(x2, y1) 右下
            f_ul=upper_left.delta_y,  # f(x1, y2) 左上
            f_ur=upper_right.delta_y,  # f(x2, y2) 右上
            **data,
        )
        delta_z = self._bilinear_interpolation_standard(
            f_ll=lower_left.delta_z,  # f(x1, y1) 左下
            f_lr=lower_right.delta_z,  # f(x2, y1) 右下
            f_ul=upper_left.delta_z,  # f(x1, y2) 左上
            f_ur=upper_right.delta_z,  # f(x2, y2) 右上
            **data,
        )
        # 補正値を秒から度に変換（経度・緯度のみ、高度はメートル単位のまま）
        return XYZ(x=delta_x / 3600, y=delta_y / 3600, z=delta_z)

    def _bilinear_interpolation_standard(
        self,
        f_ll: float,  # f(x1, y1) 左下
        f_lr: float,  # f(x2, y1) 右下
        f_ul: float,  # f(x1, y2) 左上
        f_ur: float,  # f(x2, y2) 右上
        x: float,  # 補間したい点のx座標
        y: float,  # 補間したい点のy座標
        x1: float,  # 左側のx座標
        x2: float,  # 右側のx座標
        y1: float,  # 下側のy座標
        y2: float,  # 上側のy座標
    ) -> float:
        """
        ## Summary:
            標準的なバイリニア補間を使用して補間値を計算する。

            バイリニア補間の公式：
            f(x,y) ≈ f(x1,y1)(x2-x)(y2-y) + f(x2,y1)(x-x1)(y2-y) +
                     f(x1,y2)(x2-x)(y-y1) + f(x2,y2)(x-x1)(y-y1)
                     ────────────────────────────────────────────────
                                    (x2-x1)(y2-y1)

        Args:
            f_ll, f_lr, f_ul, f_ur: 4つの格子点での値
            x, y: 補間したい点の座標
            x1, x2, y1, y2: 格子点の座標
        Returns:
            float: 補間された値
        """
        # バイリニア補間の標準公式
        numerator = (
            f_ll * (x2 - x) * (y2 - y)  # 左下
            + f_lr * (x - x1) * (y2 - y)  # 右下
            + f_ul * (x2 - x) * (y - y1)  # 左上
            + f_ur * (x - x1) * (y - y1)  # 右上
        )
        denominator = (x2 - x1) * (y2 - y1)

        return numerator / denominator

    def correction(
        self,  #
        lon: float,
        lat: float,
        alt: Optional[float] = None,
        original: bool = False,
    ) -> XYZ:
        """
        ## Summary:
            指定された経度・緯度・標高に基づいて、セミダイナミック補正を適用した
            座標を計算する。このメソッドでは国土地理院からDLした補正パラメーターを使用して、
            補正を行います。WebAPIを使用した補正とは微妙に結果が異なる場合があります。
            テストでは1cm未満の差異が見られます。確実な結果を得るためには、
            correction_2d_with_web_apiメソッドを使用してください。
        Args:
            lon (float): 経度(10進数)
            lat (float): 緯度(10進数)
            alt (Optional[float]): 標高(メートル)。指定しない場合はNone。
            original (bool): Trueの場合、元期への補正を適用し、Falseの場合今期への補正
            を適用する。
        Returns:
            XYZ: 補正後の座標
        """
        # original=Trueの場合は、今期から元期への補正なので、補正値の符号を反転
        sign = -1.0 if original else 1.0
        # 補正値の取得
        delta = self.correction_delta(lon=lon, lat=lat)
        lon = lon + delta.x * sign
        lat = lat + delta.y * sign
        if alt is not None:
            alt = alt + delta.z * sign
        else:
            alt = 0.0
        return XYZ(x=lon, y=lat, z=alt)

    def correction_2d_with_web_api(
        self,  #
        lon: float | list[float],
        lat: float | list[float],
        original: bool = True,
    ):
        """
        ## Description:
            経緯度に対してセミダイナミック補正を行う。補正パラメーターの適用年はインスタンス化時に指定されたdatetime_に基づく。
            補正値をWeb APIから取得します。
        Args:
            lon (float | list[float]):
                経度(10進数)または経度のリスト
            lat (float | list[float]):
                緯度(10進数)または緯度のリスト
            alt (float | list[float], optional):
                標高(メートル)または標高のリスト。デフォルトはNone。
            original (bool, optional):
                Trueは今期から元期への補正を行う。Falseは元期から今期への補正を行う。
                デフォルトはTrue。
        Returns:
        """
        dimensional = dimensional_count(lon)
        iterable = 0 < dimensional
        lons = lon if iterable else [lon]
        lats = lat if iterable == 1 else [lat]
        resps = fetch_corrected_semidynamic_from_web(
            correction_datetime=self.datetime,
            lons=lons,
            lats=lats,
            dimension=2,
            return_to_original=original,
        )
        if iterable:
            return [XY(x=resp.x, y=resp.y) for resp in resps]
        return XY(x=resps[0].x, y=resps[0].y)

    def correction_3d_with_web_api(
        self,  #
        lon: float | list[float],
        lat: float | list[float],
        altitude: float | list[float],
        original: bool = True,
    ) -> XYZ | list[XYZ]:
        """
        ## Description:
            経緯度と標高に対してWebAPIでセミダイナミック補正を行う。
            補正パラメーターの適用年はインスタンス化時に指定されたdatetime_に基づく。
        Args:
            original (bool, optional):
                Trueは今期から元期への補正を行う。Falseは元期から今期への補正を行う。
                デフォルトはTrue。
        Returns:
            XYZ | list[XYZ]:
                補正後の経度、緯度、標高を含むXYZオブジェクトまたはXYZオブジェクトのリスト。
        """
        dimensional = dimensional_count(lon)
        iterable = 0 < dimensional
        lons = lon if iterable else [lon]
        lats = lat if iterable == 1 else [lat]
        alts = altitude if iterable else [altitude]
        resps = fetch_corrected_semidynamic_from_web(
            correction_datetime=self.datetime,
            lons=lons,
            lats=lats,
            altis=alts,
            dimension=3,
            return_to_original=original,
        )
        if iterable:
            return [XYZ(x=resp.x, y=resp.y, z=resp.z) for resp in resps]
        return XYZ(x=resps[0].x, y=resps[0].y, z=resps[0].z)
