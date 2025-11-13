import datetime
import os
from dataclasses import dataclass
from decimal import Decimal
from glob import glob
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import pyproj

from .formatter import datetime_formatter

# 地磁気値（偏角）のデータを読み込み辞書型に変換。辞書のキーは整数型の第二次メッシュコード
_mag_df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data", "mag_2020.csv"),
    dtype={"mesh_code": int, "mag": float},
)
_mag_df["mesh_code"] = _mag_df["mesh_code"].astype(int).astype(str)
MAG_DATA: dict[int, float] = {
    mesh_code: mag_value
    for mesh_code, mag_value in zip(_mag_df["mesh_code"], _mag_df["mag"], strict=False)
}


FilePath = str


class XY(NamedTuple):
    """2次元座標を格納するクラス"""

    x: float | Decimal
    y: float | Decimal


class XYZ(NamedTuple):
    """3次元座標を格納するクラス"""

    x: float
    y: float
    z: float


class MeshDesign(NamedTuple):
    """
    name(str): 識別名
    lon(float): 経度
    lat(float): 緯度
    standard_mesh_code(str): 標準メッシュコード
    """

    name: str
    lon: float
    lat: float
    standard_mesh_code: str


class Delta(NamedTuple):
    """
    ## Summary:
        3次元座標の補正値を格納するクラス。
    """

    delta_x: float
    delta_y: float
    delta_z: float


class SemiDynaParam(NamedTuple):
    code: str
    lon: float
    lat: float
    delta_x: float
    delta_y: float
    delta_z: float


class RelativePosition(NamedTuple):
    xyz1: XY | XYZ
    xyz2: XY | XYZ
    azimuth: float
    level_distance: float
    angle: float = 0.0
    slope_distance: float = 0.0


class SemidynamicCorrectionFiles(object):
    def __init__(self):
        self.files = glob(os.path.join(os.path.dirname(__file__), "data", "*Semi*.par"))

    def _get_file_path(self, datetime_: datetime.datetime) -> FilePath:
        """
        ## Summary:
            地殻変動補正のパラメーターファイルを探すクラス
        ## Args:
            datetime_ (datetime.datetime): 補正値を取得したい日時
        ## Returns:

        """
        datetime_ = datetime_formatter(datetime_)
        year = datetime_.year
        if datetime_.month < 4:
            # 年度の開始月が4月なので、1月から3月は前年のデータを使用
            year -= 1
        file_path = [file for file in self.files if str(year) in file][0]
        return file_path

    def _clean_line(self, line: list[str]) -> list[list[str]]:
        """
        ## Summary:
            セミダイナミック補正のパラメータを読み込む際に、行ごとに読み込んでいるが
            その際に改行文字を削除し、数値に変換できるものは変換する為の関数。
        Args:
            line (list[str]):
                セミダイナミック補正のパラメータの行
        Returns:
            list[list[str]]:
                改行文字を削除し、数値に変換できるものは変換したリスト
        Example:
            >>> line = ['MeshCode', 'dB(sec)', '', 'dL(sec)', 'dH(m)\n']
            >>> xxx._clean_line(line)
            ['MeshCode', 'dB(sec)', 'dL(sec)', 'dH(m)']
            >>> line = ['36230600', '', '-0.05708', '', '',
                        '0.04167', '', '', '0.05603\n']
            >>> xxx._clean_line(line)
            ['36230600', -0.05708, 0.04167, 0.05603]
        """
        result = []
        for txt in line:
            # 改行文字を削除
            txt = txt.replace("\n", "") if "\n" in txt else txt
            try:
                if "." in txt:
                    # 小数点が含まれている場合はfloatに変換
                    result.append(float(txt))
                else:
                    # 小数点が含まれていない場合はintに変換
                    result.append(int(txt))
            except Exception:
                # 変換できない場合はそのまま文字列として追加
                if txt == "":
                    # 空文字は無視
                    pass
                else:
                    result.append(txt)
        return result

    def read_file(
        self, datetime_: datetime.datetime, encoding: str = "utf-8"
    ) -> pd.DataFrame:
        """
        ## Summary:
            地殻変動補正のパラメーターファイルを読み込む
        ## Args:
            datetime_ (datetime.datetime):
                補正値を取得したい日時
        ## Returns:
            pd.DataFrame:
                補正値のデータフレーム
        """
        file_path = self._get_file_path(datetime_)
        with open(file_path, mode="r", encoding=encoding) as f:
            # 16行目からパラメータが定義されている。
            lines = f.readlines()[15:]
            headers = self._clean_line(lines[0].split(" "))
            data = [self._clean_line(line.split(" ")) for line in lines[1:]]
            df = pd.DataFrame(data, columns=headers)
        # Breedte（緯度）Lengte（経度）
        df = df.rename(
            columns={"dB(sec)": "delta_y", "dL(sec)": "delta_x", "dH(m)": "delta_z"}
        )
        return df.set_index("MeshCode")


def semidynamic_correction_file(datetime_: datetime.datetime) -> Optional[pd.DataFrame]:
    """
    ## Summary:
        地殻変動補正のパラメーターファイルを読み込む
    ## Args:
        datetime_ (datetime.datetime):
            補正値を取得したい日時
    ## Returns:
        pd.DataFrame:
            補正値のデータフレーム。
            補正値は秒単位の値なので、10進法で表現する時は3600で割る必要がある。
            {index(int): MeshCode, columns: delta_x, delta_y, delta_z}
    ## Raises:
        RuntimeError:
            ファイルの読み込みに失敗した場合
        ValueError:
            すべてのエンコーディングでファイルの読み込みに失敗した場合
    """
    semi = SemidynamicCorrectionFiles()
    encodings = ["Shift-JIS", "utf-8", "cp932"]
    try:
        for encoding in encodings:
            try:
                return semi.read_file(datetime_, encoding=encoding)
            except UnicodeDecodeError:
                pass
    except Exception:
        raise ValueError(  # noqa: B904
            "Failed to read the file with all encodings.use encofings: "
            + ", ".join(encodings)
        )


class ChiriinWebApi(object):
    """
    ## Summary:
        国土地理院の測量計算サイトで利用可能なAPIのURLを提供するクラス。
        ただし同一IPアドレスからのリクエストは、10秒間で10回までに制限されているため、
        連続してリクエストを送信する場合は注意が必要。
        https://vldb.gsi.go.jp/sokuchi/surveycalc/api_help.html
    """

    @staticmethod
    def elevation_url() -> str:
        """
        ## Summary:
            地理院APIで標高値を取得するためのURL。
        ## Returns:
            str:
                地理院APIの標高値を取得するためのURL
        ## Example:
            >>> api = ChiriinWebApi()
            >>> url = api.elevation_url()
            >>> elev_url = url.format(lon=139.6917, lat=35.6895)
        """
        url = (
            "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php?"
            "outputtype=JSON&"
            "lon={lon}&"
            "lat={lat}"
        )
        return url

    @staticmethod
    def geoid_height_2011_url() -> str:
        """
        ## Summary:
            地理院APIで2011年の日本の測地系におけるジオイド高を取得するためのURL。
        ## Returns:
            str:
                地理院APIのジオイド高を取得するためのURL
        """
        url = (
            "http://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh2011/cgi/geoidcalc.pl?"
            "outputType=json&"
            "longitude={lon}&"
            "latitude={lat}"
        )
        return url

    @staticmethod
    def geoid_height_2024_url() -> str:
        """
        ## Summary:
            地理院APIで2024年の日本の測地系におけるジオイド高を取得するためのURL。
        ## Returns:
            str:
                地理院APIのジオイド高を取得するためのURL
        """
        url = (
            "http://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl?"
            "outputType=json&"
            "longitude={lon}&"
            "latitude={lat}"
        )
        return url

    @staticmethod
    def distance_and_azimuth_url() -> str:
        """
        ## Summary:
            地理院APIで2点間の距離と方位角を取得するためのURL。
        ## Returns:
            str:
                地理院APIの距離と方位角を取得するためのURL
                ellipsoidは、'GRS80'は世界測地系で計算する。'bessel'は日本測地系で計算する。
        """
        url = (
            "http://vldb.gsi.go.jp/sokuchi/surveycalc/surveycalc/bl2st_calc.pl?"
            "outputType=json&"
            "ellipsoid={ellipsoid}&"
            "longitude1={lon1}&"
            "latitude1={lat1}&"
            "longitude2={lon2}&"
            "latitude2={lat2}"
        )
        return url

    @staticmethod
    def semidynamic_correction_url() -> str:
        """
        ## Summary:
            地理院APIでセミダイナミック補正を行う為のURL。
        ## Returns:
            str:
                地理院APIのセミダイナミック補正を行う為のURL
        """
        url = (
            "http://vldb.gsi.go.jp/sokuchi/surveycalc/semidyna/web/semidyna_r.php?"
            "outputType=json&"
            "chiiki=SemiDyna{year}.par&"  # パラメーターファイル名（.par）
            "sokuchi={sokuchi}&"  # 0: 元期 -> 今期, 1: 今期 -> 元期
            "Place=0&"
            "Hosei_J={dimension}&"  # 2: 2次元補正, 3: 3次元補正
            "longitude={lon}&"
            "latitude={lat}&"
            "altitude1={alti}"
        )
        return url


class TileScope(NamedTuple):
    x_min: float = -20037508.342789244
    y_min: float = -20037508.342789244
    x_max: float = 20037508.342789244
    y_max: float = 20037508.342789244


class TileInfo(NamedTuple):
    """
    ## Summary:
        タイルの情報を格納するクラス。
    """

    zoom_level: int
    x_idx: int
    y_idx: int
    tile_scope: TileScope
    x_resolution: float
    y_resolution: float
    width: int = 256
    height: int = 256
    crs: pyproj.CRS = pyproj.CRS.from_epsg(3857)

    def __repr__(self):
        return f"""
TileInfo:
    - zoom_level  : {self.zoom_level}
    - x_idx       : {self.x_idx}
    - y_idx       : {self.y_idx}
    - tile_scope  : {self.tile_scope}
    - x_resolution: {self.x_resolution}
    - y_resolution: {self.y_resolution}
    - width       : {self.width}
    - height      : {self.height}
    - crs         : {self.crs.to_epsg()}
"""


@dataclass
class TileData:
    """
    ## Summary:
        タイルのデータを格納するクラス。
    """

    zoom_level: int
    x_idx: int
    y_idx: int
    tile_scope: TileScope
    x_resolution: float
    y_resolution: float
    crs: pyproj.CRS
    ary: np.ndarray
    width: int = 256
    height: int = 256

    def get_gdal_transform(self) -> tuple[float, float, float, float, float, float]:
        """
        ## Summary:
            GDALの変換用のタプルを返す。
        ## Returns:
            tuple[float]: GDALの変換用のタプル
        """
        return (
            self.tile_scope.x_min,
            self.x_resolution,
            0.0,
            self.tile_scope.y_max,
            0.0,
            self.y_resolution * -1,
        )


class TileUrls(object):
    def __init__(self):
        self._base_url = "https://cyberjapandata.gsi.go.jp/xyz/{t}/{z}/{x}/{y}.txt"
        self._dem_types = ["dem10b", "dem5a", "dem5b"]
        self._img_types = ["standard", "photo", "slope"]
        self._chiriin_source = "出典：国土地理院 地理院タイル"

    @property
    def dem_10b(self) -> str:
        """
        ## Summary:
            地理院タイルの標高タイル（DEM10b）のURLを生成する。
            ZoomLevelは1~14の範囲で指定する必要がある。
        Returns:
            str: 標高タイルのURL。ズームレベル、X座標、Y座標は後から指定する必要がある。
        """
        return self._base_url.replace("{t}", "dem")

    @property
    def dem_5a(self) -> str:
        """
        ## Summary:
            地理院タイルの標高タイル（DEM5a）のURLを生成する。
            ZoomLevelは1~15の範囲で指定する必要がある。
        Returns:
            str: 標高タイルのURL。ズームレベル、X座標、Y座標は後から指定する必要がある。
        """
        return self._base_url.replace("{t}", "dem5a")

    @property
    def dem_5b(self) -> str:
        """
        ## Summary:
            地理院タイルの標高タイル（DEM5b）のURLを生成する。
            ZoomLevelは1~15の範囲で指定する必要がある。
        Returns:
            str: 標高タイルのURL。ズームレベル、X座標、Y座標は後から指定する必要がある。
        """
        return self._base_url.replace("{t}", "dem5b")

    @property
    def standard_map(self) -> str:
        """
        ## Summary:
            地理院タイルの標準地図タイルのURLを生成する。
            ZoomLevelは5~18の範囲で指定する必要がある。
        Returns:
            str: 標準地図タイルのURL。ズームレベル、X座標、Y座標は後から指定する必要がある。
        """
        return self._base_url.replace("{t}", "std").replace(".txt", ".png")

    @property
    def pale_map(self) -> str:
        """
        ## Summary:
            地理院タイルの淡色地図タイルのURLを生成する。
            ZoomLevelは5~18の範囲で指定する必要がある。
        Returns:
            str: 淡色地図タイルのURL。ズームレベル、X座標、Y座標は後から指定する必要がある。
        """
        return self._base_url.replace("{t}", "pale").replace(".txt", ".png")

    @property
    def photo_map(self) -> str:
        """
        ## Summary:
            地理院タイルの空中写真タイルのURLを生成する。
            ZoomLevelは2~18の範囲で指定する必要がある。
        Returns:
            str: シームレス写真タイルのURL。ズームレベル、X座標、Y座標は後から指定する
            必要がある。
        """
        return self._base_url.replace("{t}", "seamlessphoto").replace(".txt", ".jpg")

    @property
    def slope_map(self) -> str:
        """
        ## Summary:
            地理院タイルの傾斜タイルのURLを生成する。
            ZoomLevelは3~15の範囲で指定する必要がある。
        Returns:
            str: 傾斜タイルのURL。ズームレベル、X座標、Y座標は後から指定する必要がある。
        """
        return self._base_url.replace("{t}", "slopemap").replace(".txt", ".png")

    @property
    def google_satellite(self) -> str:
        """
        ## Summary:
            Googleの衛星画像タイルのURLを生成する。
            ZoomLevelは5~18の範囲で指定する必要がある。
        Returns:
            str: Googleの衛星画像タイルのURL。ズームレベル、X座標、Y座標は後から指定する
            必要がある。
        """
        return "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga"

    @property
    def micro_topo_miyagi(self):
        """
        ## Summary:
            G空間センターで公開されている、宮城県の微地形図タイル。
            https://www.geospatial.jp/ckan/dataset/rinya-miyagi-maptiles
            出典：宮城県・微地形表現図マップタイル（林野庁加工）
        """
        return "https://forestgeo.info/opendata/4_miyagi/topography_2023/{z}/{x}/{y}.webp"


class FigureSize(NamedTuple):
    """
    ## Summary:
        matplotlibのFigureオブジェクトのサイズを表すクラス。
        幅と高さをインチ単位で指定する。
    """

    width: float
    height: float


class Scope(NamedTuple):
    """
    ## Summary:
        2次元座標の範囲を表すクラス。
        x_min, y_min, x_max, y_maxの4つの値を持つ。
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class PaperSize(object):
    """
    ## Summary:
        PDFとして印刷する為にmatplotlibのFigureオブジェクトとして設定するタプルの
        オブジェクトを計算するクラス。
    """

    def __init__(self):
        self.inches_per_cm = 1 / 2.54
        self.a4_cm_long = 29.7
        self.a4_cm_short = 21.0
        self.a3_cm_long = 42.0
        self.a3_cm_short = self.a4_cm_long

    def _calc_size(self, w: float, h: float):
        fig_width = w * self.inches_per_cm
        fig_height = h * self.inches_per_cm
        return FigureSize(fig_width, fig_height)

    def portrait_a4_size(self) -> FigureSize:
        """
        ## Summary:
            A4用紙の縦向きのサイズを計算するプロパティ。
        Args:
            cm_w (float): 幅（センチメートル単位）
            cm_h (float): 高さ（センチメートル単位）
        Returns:
            FigureSize:
                A4用紙の縦向きのサイズを計算したタプル。
                幅と高さをmatplotlibのFigureオブジェクトとして設定できるタプル。
                - width (float): 幅（インチ単位）
                - height (float): 高さ（インチ単位）
        """
        return self._calc_size(self.a4_cm_short, self.a4_cm_long)

    def landscape_a4_size(self) -> FigureSize:
        """
        ## Summary:
            A4用紙の横向きのサイズを計算するプロパティ。
        Args:
            cm_w (float): 幅（センチメートル単位）
            cm_h (float): 高さ（センチメートル単位）
        Returns:
            FigureSize:
                A4用紙の縦向きのサイズを計算したタプル。
                幅と高さをmatplotlibのFigureオブジェクトとして設定できるタプル。
                - width (float): 幅（インチ単位）
                - height (float): 高さ（インチ単位）
        """
        return self._calc_size(self.a4_cm_long, self.a4_cm_short)

    def portrait_a3_size(self) -> FigureSize:
        """
        ## Summary:
            A3用紙の縦向きのサイズを計算するプロパティ。
        Args:
            cm_w (float): 幅（センチメートル単位）
            cm_h (float): 高さ（センチメートル単位）
        Returns:
            FigureSize:
                A4用紙の縦向きのサイズを計算したタプル。
                幅と高さをmatplotlibのFigureオブジェクトとして設定できるタプル。
                - width (float): 幅（インチ単位）
                - height (float): 高さ（インチ単位）
        """
        return self._calc_size(self.a3_cm_short, self.a3_cm_long)

    def landscape_a3_size(self) -> FigureSize:
        """
        ## Summary:
            A3用紙の横向きのサイズを計算するプロパティ。
        Args:
            cm_w (float): 幅（センチメートル単位）
            cm_h (float): 高さ（センチメートル単位）
        Returns:
            FigureSize:
                A4用紙の縦向きのサイズを計算したタプル。
                幅と高さをmatplotlibのFigureオブジェクトとして設定できるタプル。
                - width (float): 幅（インチ単位）
                - height (float): 高さ（インチ単位）
        """
        return self._calc_size(self.a3_cm_long, self.a3_cm_short)


class Icons(object):
    """
    ## Summary:
        アイコンのパスを管理するクラス。
        アイコンのパスは、chiriinのアイコンフォルダ内にあるPNGファイルのパスを返す。
    """

    @staticmethod
    def get_icon_path(icon_name: str) -> str:
        """
        ## Summary:
            アイコンのパスを取得するメソッド。
        Args:
            icon_name (str): アイコンの名前（拡張子なし）
        Returns:
            str: アイコンのパス
        """
        return os.path.join(os.path.dirname(__file__), "data", "imgs", f"{icon_name}.png")

    @property
    def true_north(self) -> str:
        """
        ## Summary:
            位置情報アイコンのパスを取得するプロパティ。
        Returns:
            str: 位置情報アイコンのパス
        """
        return self.get_icon_path("true_north")

    @property
    def compass(self) -> str:
        """
        ## Summary:
            コンパスアイコンのパスを取得するプロパティ。
        Returns:
            str: コンパスアイコンのパス
        """
        return self.get_icon_path("compass")

    @property
    def simple_compass(self) -> str:
        """
        ## Summary:
            方位角アイコンのパスを取得するプロパティ。
        Returns:
            str: 方位角アイコンのパス
        """
        return self.get_icon_path("simple_compass")
