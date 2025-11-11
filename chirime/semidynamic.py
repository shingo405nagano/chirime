import datetime
from decimal import Decimal
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from .config import XY, XYZ, Delta, MeshDesign, semidynamic_correction_file
from .formatter import (
    datetime_formatter,
    iterable_decimalize_formatter,
    type_checker_decimal,
)
from .mesh import MeshCode
from .utils import dimensional_count
from .web import fetch_corrected_semidynamic_from_web


class SemiDynamic(object):
    def __init__(
        self,
        lon: float | Iterable[float],
        lat: float | Iterable[float],
        datetime_: Union[datetime.datetime, Iterable[datetime.datetime]],
        altitude: Optional[float | Iterable[float]] = None,
        is_dms: bool = False,
    ):
        self.lon = lon
        self.lat = lat
        self.altitude = altitude
        self.datetime = datetime_formatter(datetime_)
        self._is_dms = is_dms
        self._is_iterable = False
        # Convert longitude and latitude to float or iterable of floats
        self._convert_lon_lat()
        self._param_df = self._read_parameters()

    def _convert_lon_lat(self):
        """
        ## Description:
            経緯度をDecimal型に変換する。オブジェクトが繰り返し可能な場合は、Decimalのリストに変換する。
        """
        count_lon = dimensional_count(self.lon)
        count_lat = dimensional_count(self.lat)
        assert count_lon == count_lat, (
            "longitude and latitude must have the same dimensionality."
        )
        if count_lon == 0:
            self.lon = Decimal(f"{float(self.lon)}")
            self.lat = Decimal(f"{float(self.lat)}")
            if self.altitude is not None:
                self.altitude = Decimal(f"{float(self.altitude)}")
        else:
            self.lon = iterable_decimalize_formatter(self.lon)
            self.lat = iterable_decimalize_formatter(self.lat)
            if self.altitude is not None:
                self.altitude = iterable_decimalize_formatter(self.altitude)
            self._is_iterable = True

    def _read_parameters(self) -> pd.DataFrame:
        """
        ## Description:
            セミダイナミック補正のパラメータを読み込む。
        ## Returns:
            pd.DataFrame: deltaは秒単位の補正値を表す。
                Index: Mesh code(int)
                Columns: delta_x(float), delta_y(float), delta_z(float)
        """
        return semidynamic_correction_file(self.datetime)

    @type_checker_decimal(arg_index=1, kward="lon")
    @type_checker_decimal(arg_index=2, kward="lat")
    def mesh_design(self, lon: float, lat: float) -> dict[str, MeshDesign]:
        """
        ## Description:
            指定された緯度経度のメッシュを設計する。
        ## Args:
            lon (Decimal):
                10進数経度
            lat (Decimal):
                10進数緯度
        ## Returns:
            dict[str, MeshDesign] | list[MeshDesign]:
                メッシュの四隅のデータ
                Mesh designs include:
                - lower_left: 左下の設計
                - lower_right: 右下の設計
                - upper_left: 左上の設計
                - upper_right: 右上の設計
            MeshDesign:
                - name: 識別名
                - lon: 経度（秒単位）
                - lat: 緯度（秒単位）
                - standard_mesh_code: 標準メッシュコード
        """
        lon_param = 225
        lat_param = 150
        lower_left_sec_lon = round(lon * 3600, 1)
        lower_left_sec_lat = round(lat * 3600, 1)
        m = int(lower_left_sec_lon / lon_param)
        n = int(lower_left_sec_lat / lat_param)
        lower_left_sec_lon = m * lon_param
        lower_left_sec_lat = n * lat_param
        lower_left_deg_lon = lower_left_sec_lon / 3600
        lower_left_deg_lat = lower_left_sec_lat / 3600
        # Create MeshCode and MeshDesign for lower left corner
        lower_left_mesh_code = MeshCode(lower_left_deg_lon, lower_left_deg_lat)
        lower_left_design = MeshDesign(
            "lower_left",
            lower_left_sec_lon,
            lower_left_sec_lat,
            lower_left_mesh_code.standard_mesh_code,
        )
        lower_right_design = self._adjust_mesh_code(
            lower_left_sec_lon, lower_left_sec_lat, lon_param, 0, "lower_right"
        )
        upper_left_design = self._adjust_mesh_code(
            lower_left_sec_lon, lower_left_sec_lat, 0, lat_param, "upper_left"
        )
        upper_right_design = self._adjust_mesh_code(
            lower_left_sec_lon,
            lower_left_sec_lat,
            lon_param,
            lat_param,
            "upper_right",
        )
        return {
            "lower_left": lower_left_design,
            "lower_right": lower_right_design,
            "upper_left": upper_left_design,
            "upper_right": upper_right_design,
        }

    def _adjust_mesh_code(
        self,
        lower_left_sec_lon: float,
        lower_left_sec_lat,
        lon_param: int = 225,
        lat_param: int = 150,
        name: str = "lower_right",
    ) -> MeshDesign:
        """
        ## Description:
            緯度経度からメッシュコードを秒単位で調整する。
        Args:
            lower_left_sec_lon (float): 左下の経度（秒）
            lower_left_sec_lat (float): 左下の緯度（秒）
            lon_param (int, optional): 経度のパラメータ。デフォルトは225。
            lat_param (int, optional): 緯度のパラメータ。デフォルトは150。
            name (str, optional): 識別名
        Returns:
            MeshDesign:
                - name: 識別名
                - lon: 秒単位経度
                - lat: 秒単位緯度
                - standard_mesh_code: 標準メッシュコード
        """
        sec_lon = lower_left_sec_lon + lon_param
        sec_lat = lower_left_sec_lat + lat_param
        mesh_code = MeshCode(sec_lon / 3600, sec_lat / 3600)
        return MeshDesign(
            name=name,
            lon=sec_lon,
            lat=sec_lat,
            standard_mesh_code=mesh_code.standard_mesh_code,
        )

    def _get_delta_sets(self, mesh_designs: dict[str, MeshDesign]) -> dict[str, Delta]:
        """
        ## Description:
            セミダイナミック補正のパラメータから4方向の補正値を求める。
        Args:
            mesh_designs (dict[str, MeshDesign]):
                メッシュ設計の辞書。キーは "lower_left", "lower_right", "upper_left", "upper_right"。
                各値はMeshDesignオブジェクト。
        Returns:
            DeltaSet:
                各方向の補正値を含む辞書。
                - lower_left: 左下の補正値
                - lower_right: 右下の補正値
                - upper_left: 左上の補正値
                - upper_right: 右上の補正値
        """
        lower_left_delta = self._get_delta(mesh_designs["lower_left"].standard_mesh_code)
        lower_right_delta = self._get_delta(
            mesh_designs["lower_right"].standard_mesh_code
        )
        upper_left_delta = self._get_delta(mesh_designs["upper_left"].standard_mesh_code)
        upper_right_delta = self._get_delta(
            mesh_designs["upper_right"].standard_mesh_code
        )
        return {
            "lower_left": lower_left_delta,
            "lower_right": lower_right_delta,
            "upper_left": upper_left_delta,
            "upper_right": upper_right_delta,
        }

    def _get_delta(self, mesh_code: str) -> Delta:
        """
        ## Description:
            セミダイナミック補正のパラメータから指定されたメッシュコードの補正値を取得する。
        Args:
            mesh_code (str):
                補正値を取得するメッシュコード。
        Returns:
            Delta:
                - delta_x(Decimal): 経度の補正値（秒単位）
                - delta_y(Decimal): 緯度の補正値（秒単位）
                - delta_z(Decimal): 標高の補正値（メートル単位）
        """
        try:
            row = self._param_df.loc[int(mesh_code)]
        except KeyError:
            print(f"Mesh code {mesh_code} not found in parameters.")
            row = {
                "delta_x": None,
                "delta_y": None,
                "delta_z": None,
            }
            return Delta(**row)
        return Delta(
            delta_x=Decimal(f"{row['delta_x']}"),
            delta_y=Decimal(f"{row['delta_y']}"),
            delta_z=Decimal(f"{row['delta_z']}"),
        )

    def _fill_delta_zero(self, delta_sets: dict[str, Delta]) -> dict[str, Delta]:
        """
        ## Description:
            補正値が取得出来なかった場合、他の補正値の平均で埋める。
        Args:
            delta_sets (dict[str, Delta]):
                補正値のセット。
        Returns:
            dict[str, Delta]:
                ゼロで埋められた補正値のセット。
        """
        # DataFrame化
        delta_df = pd.DataFrame(delta_sets)
        # 欠損値を平均値で埋める
        delta_df = delta_df.T.fillna(delta_df.T.mean()).T
        delta_sets["lower_left"] = Delta(*delta_df["lower_left"])
        delta_sets["lower_right"] = Delta(*delta_df["lower_right"])
        delta_sets["upper_left"] = Delta(*delta_df["upper_left"])
        delta_sets["upper_right"] = Delta(*delta_df["upper_right"])
        return delta_sets

    @type_checker_decimal(arg_index=1, kward="lon")
    @type_checker_decimal(arg_index=2, kward="lat")
    def _calc_correction_delta(
        self, lon: float, lat: float, return_to_original: bool = True
    ) -> Delta:
        """
        ## Description:
            経緯度（10進法）を受け取り、セミダイナミック補正を行う。
        Args:
            lon (float):
                ターゲットの経度（10進法）
            lat (float):
                ターゲットの緯度（10進法）
            return_to_original (bool, optional):
                Trueは今期から元期への補正を行う。Falseは元期から今期への補正を行う。
        Returns:
            dict[str, float]:
                補正後の経度と緯度を含む辞書。
                {'lon': 経度, 'lat': 緯度}
        """
        mesh_designs = self.mesh_design(lon, lat)
        if not mesh_designs:
            return {"lon": False, "lat": False}
        # 経度と緯度を秒単位に変換
        lon = lon * 3600
        lat = lat * 3600
        # MeshDesign(name, lon, lat, standard_mesh_code)
        lower_left_design = mesh_designs["lower_left"]
        lower_right_design = mesh_designs["lower_right"]
        upper_left_design = mesh_designs["upper_left"]
        # Delta(delta_x, delta_y, delta_z)
        delta_sets = self._get_delta_sets(mesh_designs)
        if None in np.array([delta for delta in delta_sets.values()]).flatten().tolist():
            # 欠損値がある場合は、他の補正値の平均で埋める
            delta_sets = self._fill_delta_zero(delta_sets)
        lower_left_delta = delta_sets["lower_left"]
        lower_right_delta = delta_sets["lower_right"]
        upper_left_delta = delta_sets["upper_left"]
        upper_right_delta = delta_sets["upper_right"]
        data = {
            "lon_sec": lon,
            "lat_sec": lat,
            "lower_left_design": lower_left_design,
            "lower_right_design": lower_right_design,
            "upper_left_design": upper_left_design,
        }
        # バイリニア補間により補正値を計算
        delta_lon = self._bilinear_interpolation_delta(
            lower_left_delta=lower_left_delta.delta_x,
            lower_right_delta=lower_right_delta.delta_x,
            upper_left_delta=upper_left_delta.delta_x,
            upper_right_delta=upper_right_delta.delta_x,
            **data,
        )
        delta_lat = self._bilinear_interpolation_delta(
            lower_left_delta=lower_left_delta.delta_y,
            lower_right_delta=lower_right_delta.delta_y,
            upper_left_delta=upper_left_delta.delta_y,
            upper_right_delta=upper_right_delta.delta_y,
            **data,
        )
        delta_alti = self._bilinear_interpolation_delta(
            lower_left_delta=lower_left_delta.delta_z,
            lower_right_delta=lower_right_delta.delta_z,
            upper_left_delta=upper_left_delta.delta_z,
            upper_right_delta=upper_right_delta.delta_z,
            **data,
        )

        # 元期から今期へのパラメーターなので、今期から元期へは -1 を掛ける
        if return_to_original:
            delta_lon *= -1
            delta_lat *= -1
            delta_alti *= -1
        return Delta(delta_x=delta_lon, delta_y=delta_lat, delta_z=delta_alti)

    @type_checker_decimal(arg_index=1, kward="lon_sec")
    @type_checker_decimal(arg_index=2, kward="lat_sec")
    @type_checker_decimal(arg_index=6, kward="lower_left_delta")
    @type_checker_decimal(arg_index=7, kward="lower_right_delta")
    @type_checker_decimal(arg_index=8, kward="upper_left_delta")
    @type_checker_decimal(arg_index=9, kward="upper_right_delta")
    def _bilinear_interpolation_delta(
        self,
        lon_sec: float | Decimal,
        lat_sec: float | Decimal,
        lower_left_design: MeshDesign,
        lower_right_design: MeshDesign,
        upper_left_design: MeshDesign,
        lower_left_delta: float | Decimal,
        lower_right_delta: float | Decimal,
        upper_left_delta: float | Decimal,
        upper_right_delta: float | Decimal,
    ) -> Decimal:
        """
        ## Description:
            バイリニア補間を使用して、指定された緯度経度の補正値を計算する。
        Args:
            lon_sec (float | Decimal):
                ターゲットの経度（秒単位）
            lat_sec (float | Decimal):
                ターゲットの緯度（秒単位）
            lower_left_design (MeshDesign):
                左下のメッシュ設計
            lower_right_design (MeshDesign):
                右下のメッシュ設計
            upper_left_design (MeshDesign):
                左上のメッシュ設計
            lower_left_delta (float | Decimal):
                左下の補正値（秒単位）
            lower_right_delta (float | Decimal):
                右下の補正値（秒単位）
            upper_left_delta (float | Decimal):
                左上の補正値（秒単位）
            upper_right_delta (float | Decimal):
                右上の補正値（秒単位）
        Returns:
            Decimal:
                補正値（秒単位）
        """
        vertical_distance = upper_left_design.lat - lower_left_design.lat
        horizontal_distance = lower_right_design.lon - lower_left_design.lon
        y_norm_lw = (lat_sec - lower_left_design.lat) / vertical_distance
        y_norm_up = (upper_left_design.lat - lat_sec) / vertical_distance
        x_norm_left = (lon_sec - lower_left_design.lon) / horizontal_distance
        x_norm_right = (lower_right_design.lon - lon_sec) / horizontal_distance
        # バイリニア補間の計算
        delta = (
            y_norm_lw * x_norm_right * upper_left_delta
            + y_norm_up * x_norm_right * upper_right_delta
            + y_norm_lw * x_norm_left * lower_left_delta
            + y_norm_up * x_norm_left * lower_right_delta
        )
        return delta

    def correction_2d(self, return_to_original: bool = True) -> XY | list[XY]:
        """
        ## Description:
            経緯度に対してセミダイナミック補正を行う。補正パラメーターの適用年はインス
            タンス化時に指定されたdatetime_に基づく。
        Args:
            return_to_original (bool, optional):
                Trueは今期から元期への補正を行う。Falseは元期から今期への補正を行う。
                デフォルトはTrue。
        ## Returns:
            XY | list[XY]:
                補正後の経度と緯度を含むXYオブジェクトまたはXYオブジェクトのリスト。
                - XY: 補正後の経度と緯度を含むオブジェクト
                - list[XY]: 複数のXYオブジェクトを含むリスト
        Returns:
        """
        if not self._is_iterable:
            delta = self._calc_correction_delta(self.lon, self.lat, return_to_original)
            corrected_lon = float(self.lon + (delta.delta_x / 3600))
            corrected_lat = float(self.lat + (delta.delta_y / 3600))
            return XY(x=corrected_lon, y=corrected_lat)
        # If the input is iterable, apply the correction to each element
        lst = []
        previous_mesh_code = None
        for lon, lat in zip(self.lon, self.lat, strict=True):
            mesh_code = MeshCode(float(lon), float(lat)).standard_mesh_code
            if mesh_code is None:
                delta = self._calc_correction_delta(lon, lat, return_to_original)
                previous_mesh_code = mesh_code
            elif mesh_code != previous_mesh_code:
                delta = self._calc_correction_delta(lon, lat, return_to_original)
                previous_mesh_code = mesh_code
            corrected_lon = lon + (delta.delta_x / 3600)
            corrected_lat = lat + (delta.delta_y / 3600)
            lst.append(XY(x=float(corrected_lon), y=float(corrected_lat)))
        return lst

    def correction_2d_with_web_api(self, return_to_original: bool = True):
        """
        ## Description:
            経緯度に対してセミダイナミック補正を行う。補正パラメーターの適用年はインスタンス化時に指定されたdatetime_に基づく。
            補正値をWeb APIから取得します。
        Args:
            return_to_original (bool, optional):
                Trueは今期から元期への補正を行う。Falseは元期から今期への補正を行う。
                デフォルトはTrue。
        Returns:
            XY | list[XY]:
                補正後の経度と緯度を含むXYオブジェクトまたはXYオブジェクトのリスト。
        """
        dimensional = dimensional_count(self.lon)
        iterable = 0 < dimensional
        lons = self.lon if iterable else [self.lon]
        lats = self.lat if iterable == 1 else [self.lat]
        resps = fetch_corrected_semidynamic_from_web(
            correction_datetime=self.datetime,
            lons=lons,
            lats=lats,
            dimension=2,
            return_to_original=return_to_original,
        )
        if iterable:
            return [XY(x=resp.x, y=resp.y) for resp in resps]
        return XY(x=resps[0].x, y=resps[0].y)

    def correction_3d_with_web_api(
        self, return_to_original: bool = True
    ) -> XYZ | list[XYZ]:
        """
        ## Description:
            経緯度と標高に対してセミダイナミック補正を行う。補正パラメーターの適用年はインスタンス化時に指定されたdatetime_に基づく。
            補正値をWeb APIから取得します。
        Args:
            return_to_original (bool, optional):
                Trueは今期から元期への補正を行う。Falseは元期から今期への補正を行う。
                デフォルトはTrue。
        Returns:
            XY | list[XY]:
                補正後の経度、緯度、標高を含むXYオブジェクトまたはXYオブジェクトのリスト。
        """
        dimensional = dimensional_count(self.lon)
        iterable = 0 < dimensional
        lons = self.lon if iterable else [self.lon]
        lats = self.lat if iterable == 1 else [self.lat]
        alts = self.altitude if iterable else [self.altitude]
        resps = fetch_corrected_semidynamic_from_web(
            correction_datetime=self.datetime,
            lons=lons,
            lats=lats,
            altis=alts,
            dimension=3,
            return_to_original=return_to_original,
        )
        if iterable:
            return [XYZ(x=resp.x, y=resp.y, z=resp.z) for resp in resps]
        return XYZ(x=resps[0].x, y=resps[0].y, z=resps[0].z)
