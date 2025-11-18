"""
参考情報：
[地域メッシュ統計の特質・沿革](https://www.stat.go.jp/data/mesh/pdf/gaiyo1.pdf)
[セミダイナミック補正マニュアル](https://www.gsi.go.jp/common/000258815.pdf)
[セミダイナミック補正のパラメーターファイル](https://www.gsi.go.jp/sokuchikijun/semidyna.html)
[高精度衛星測位システムの開発](https://kagoshima.daiichi-koudai.ac.jp/wp-content/uploads/2023/07/R5_report1_11.pdf)
[磁気図](https://www.gsi.go.jp/buturisokuchi/menu03_magnetic_chart.html)
[測量計算サイト](https://vldb.gsi.go.jp/sokuchi/surveycalc/api_help.html)
[標高タイルURL](https://maps.gsi.go.jp/development/ichiran.html#dem-1)
[標高タイルの詳細](https://maps.gsi.go.jp/development/demtile.html)
"""

import datetime
from typing import Iterable

import numpy as np
import pyproj
import shapely.geometry

from .config import XY, XYZ, RelativePosition, TileData, TileUrls
from .formatter import (
    dimensional_count,
    type_checker_crs,
    type_checker_elev_type,
    type_checker_img_type,
    type_checker_iterable,
)
from .geomesh import geomesh
from .geometries import transform_xy
from .mag import get_magnetic_declination
from .semidynamic import SemiDynamic
from .tile import search_tile_info_from_geometry, search_tile_info_from_xy
from .web import (
    fetch_distance_and_azimuth_from_web,
    fetch_elevation_from_web,
    fetch_elevation_tiles_from_web,
    fetch_geocentric_orthogonal_coordinates_from_web,
    fetch_geoid_height_from_web,
    fetch_img_map_tiles_from_web,
)


class _ChiriinDrawer(object):
    @staticmethod
    def magnetic_declination(lon: float, lat: float, is_dms: bool = False) -> float:
        """
        ## Summary:
            国土地理院の公開している地磁気値（2020.0年値）のパラメーターファイルを
            用いて、任意の地点の地磁気偏角を計算します。この地磁気値は通常5年ごとに
            更新されるので、最新の値を使用することをお勧めします。
            [磁器図](https://www.gsi.go.jp/buturisokuchi/menu03_magnetic_chart.html)
        Args:
            lon (float):
                経度（10進法） or 度分秒形式
            lat (float):
                緯度（10進法）or 度分秒形式
            is_dms (bool):
                経緯度が度分秒形式かどうか
        Returns:
            float:
                地磁気偏角（度）
        """
        return get_magnetic_declination(lon, lat, is_dms)

    @staticmethod
    def get_mesh_code(lon: float, lat: float, is_dms: bool = False) -> geomesh.MeshCodeJP:
        """
        ## Summary:
            任意の地点のメッシュコードを取得します。
            経緯度が度分秒形式の場合は、10進法に変換してから計算します。
            地域メッシュコードの計算方法は、総務省が公開しているPDFファイルを
            参照しています。
            [地域メッシュ統計の特質・沿革](https://www.stat.go.jp/data/mesh/pdf/gaiyo1.pdf)
        Args:
            lon (float):
                経度（10進法） or 度分秒形式
            lat (float):
                緯度（10進法）or 度分秒形式
            is_dms (bool):
                経緯度が度分秒形式かどうか
        Returns:
            geomesh.MeshCodeJP:
                メッシュコードオブジェクト
        """
        return geomesh.jpmesh.MeshCodeJP(lon, lat, is_dms)

    @staticmethod
    def semidynamic(
        lon: float | Iterable[float],
        lat: float | Iterable[float],
        measurement_date: datetime.datetime,
        alt: float | Iterable[float] = None,
        return_to_original: bool = True,
    ) -> XY | list[XY]:
        """
        ## Summary:
            セミダイナミック補正による2次元の地殻変動補正を行い、補正後の座標を
            返すメソッドです。補正には「今期から元期へ」と「元期から今期へ」の変
            換があり、それぞれパラメーターファイルを取得する為に座標の計測日が必
            要です。この補正は国土地理院で公開されているパラメーターファイルを使用
            して補正を行います。バイリニア補完などは独自で実装している為、数ミリ程度
            の誤差が生じる可能性があります。
        Args:
            lon (float | Iterable[float]):
                経度（10進法）の数値またはリストなどの反復可能なオブジェクト
            lat (float | Iterable[float]):
                緯度（10進法）の数値またはリストなどの反復可能なオブジェクト
            alt (float | Iterable[float]):
                標高（メートル単位）の数値またはリストなどの反復可能なオブジェクト
            datetime_ (datetime.datetime):
                座標の計測日時。この日時によって使用するパラメーターファイルが
                決まります。
            return_to_original (bool):
                True: 「今期から元期へ」の補正を行い、補正後の座標を返す
                False: 「元期から今期へ」の補正を行い、補正後の座標を返す
        """
        semidynamic = SemiDynamic(measurement_date=measurement_date)
        if dimensional_count(lon) == 0:
            return semidynamic.correction(lon, lat, alt, return_to_original)
        results = []
        for i, lon_, lat_ in enumerate(zip(lon, lat, strict=False)):
            alt_ = alt[i] if alt is not None else None
            results.append(semidynamic.correction(lon_, lat_, alt_, return_to_original))
        return results

    @staticmethod
    def fetch_semidynamic_2d(
        lon: float | Iterable[float],
        lat: float | Iterable[float],
        measurement_date: datetime.datetime,
        return_to_original: bool = True,
    ) -> XY | list[XY]:
        """
        ## Summary:
            セミダイナミック補正による2次元の地殻変動補正を行い、補正後の座標を
            返すメソッドです。補正には「今期から元期へ」と「元期から今期へ」の
            変換があり、それぞれパラメーターファイルを取得する為に座標の計測日が
            必要です。
            通常のセミダイナミック補正メソッド（`semidynamic_2d`）とは異なり、こ
            のメソッドは国土地理院の公開しているWeb APIを使用して、セミダイナミ
            ック補正を行います。その為、API利用制限があり10秒間に10回のリクエス
            トに制限しています。
        Args:
            lon (float | Iterable[float]):
                経度（10進法）の数値またはリストなどの反復可能なオブジェクト
            lat (float | Iterable[float]):
                緯度（10進法）の数値またはリストなどの反復可能なオブジェクト
            measurement_date (datetime.datetime):
                座標の計測日時。この日時によって使用するパラメーターファイルが
                決まります。
            return_to_original (bool):
                True: 「今期から元期へ」の補正を行い、補正後の座標を返す
                False: 「元期から今期へ」の補正を行い、補正後の座標を返す
        """
        semidynamic = SemiDynamic(measurement_date=measurement_date)
        return semidynamic.correction_2d_with_web_api(lon, lat, return_to_original)

    @staticmethod
    def fetch_semidynamic_3d(
        lon: float | Iterable[float],
        lat: float | Iterable[float],
        altitude: float | Iterable[float],
        measurement_date: datetime.datetime,
        return_to_original: bool = True,
    ) -> XYZ | list[XYZ]:
        """
        ## Summary:
            セミダイナミック補正による3次元の地殻変動補正を行い、補正後の座標を
            返すメソッドです。補正には「今期から元期へ」と「元期から今期へ」の
            変換があり、それぞれパラメーターファイルを取得する為に座標の計測日が
            必要です。
            通常のセミダイナミック補正メソッド（`semidynamic_3d`）とは異なり、この
            メソッドは国土地理院の公開しているWeb APIを使用して、セミダイナミッ
            ク補正を行います。その為、API利用制限があり10秒間に10回のリクエス
            トに制限しています。
        Args:
            lon (float | Iterable[float]):
                経度（10進法）の数値またはリストなどの反復可能なオブジェクト
            lat (float | Iterable[float]):
                緯度（10進法）の数値またはリストなどの反復可能なオブジェクト
            altitude (float | Iterable[float]):
                標高（メートル単位）の数値またはリストなどの反復可能なオブジェクト
            measurement_date (datetime.datetime):
                座標の計測日時。この日時によって使用するパラメーターファイルが
                決まります。
            return_to_original (bool):
                True: 「今期から元期へ」の補正を行い、補正後の座標を返す
                False: 「元期から今期へ」の補正を行い、補正後の座標を返す
        """
        semidynamic = SemiDynamic(measurement_date=measurement_date)
        return semidynamic.correction_3d_with_web_api(
            lon, lat, altitude, return_to_original
        )

    @staticmethod
    @type_checker_iterable(arg_index=0, kward="lon1")
    @type_checker_iterable(arg_index=1, kward="lat1")
    @type_checker_iterable(arg_index=2, kward="lon2")
    @type_checker_iterable(arg_index=3, kward="lat2")
    def fetch_distance_and_azimuth(
        lon1: float | Iterable[float],
        lat1: float | Iterable[float],
        lon2: float | Iterable[float],
        lat2: float | Iterable[float],
        ellipsoid: str = "bessel",
        slope_distance: bool = True,
    ) -> RelativePosition | list[RelativePosition]:
        """
        ## Summary:
            2点間の距離と方位角を計算するメソッドです。
            国土地理院の公開しているWeb APIを使用して、距離と方位角を計算します。
            このメソッドは、10秒間に10回のリクエスト制限があります。
        Args:
            lon1 (float | Iterable[float]):
                1点目の経度（10進法）の数値またはリストなどの反復可能なオブジェクト
            lat1 (float | Iterable[float]):
                1点目の緯度（10進法）の数値またはリストなどの反復可能なオブジェクト
            lon2 (float | Iterable[float]):
                2点目の経度（10進法）の数値またはリストなどの反復可能なオブジェクト
            lat2 (float | Iterable[float]):
                2点目の緯度（10進法）の数値またはリストなどの反復可能なオブジェクト
            ellipsoid (str):
                計算に使用する楕円体。'GRS80'は世界測地系で計算する。'bessel'は日本測地系で計算する。
        Returns:
            RelativePosition | list[RelativePosition]:
                距離と方位角を含むRelativePositionオブジェクトまたはそのリスト
                - xyz1: 1点目の座標（XYオブジェクト）
                - xyz2: 2点目の座標（XYオブジェクト）
                - azimuth: 方位角（度）
                - level_distance: 水平距離（メートル）
                - angle: 高度角（度）
                - slope_distance: 斜距離（メートル）
        """
        resps = fetch_distance_and_azimuth_from_web(
            lons1=lon1,
            lats1=lat1,
            lons2=lon2,
            lats2=lat2,
            ellipsoid=ellipsoid,
        )
        relative_objs = []
        for data, lon1_, lat1_, lon2_, lat2_ in zip(
            resps, lon1, lat1, lon2, lat2, strict=False
        ):
            xy1 = XY(lon1_, lat1_)
            xy2 = XY(lon2_, lat2_)
            level = data["distance"]
            if slope_distance:
                elev1 = chirime.fetch_elevation(lon1_, lat1_, "EPSG:4326")
                elev2 = chirime.fetch_elevation(lon2_, lat2_, "EPSG:4326")
                elev_diff = elev2 - elev1
                slope = np.sqrt(level**2 + elev_diff**2)
                vertical_angle_rad = np.atan2(elev_diff, level)
                vertical_angle_deg = np.degrees(vertical_angle_rad)
                relative_obj = RelativePosition(
                    xyz1=xy1,
                    xyz2=xy2,
                    azimuth=data["azimuth"],
                    level_distance=level,
                    angle=float(vertical_angle_deg),
                    slope_distance=float(slope),
                )
            else:
                relative_obj = RelativePosition(
                    xyz1=xy1,
                    xyz2=xy2,
                    azimuth=data["azimuth"],
                    level_distance=data["distance"],
                    angle=0.0,
                    slope_distance=0.0,
                )
            relative_objs.append(relative_obj)
        if len(relative_objs) == 1:
            return relative_objs[0]
        return relative_objs

    @type_checker_crs(arg_index=3, kward="in_crs")
    def fetch_elevation(
        self,  #
        x: float | Iterable[float],
        y: float | Iterable[float],
        in_crs: str | int | pyproj.CRS,
    ) -> float | list[float]:
        """
        ## Summary:
            国土地理院の公開しているWeb APIを使用して、標高を取得します。
            このメソッドは、10秒間に10回のリクエスト制限があります。
        Args:
            lon (float | Iterable[float]):
                経度（10進法）の数値またはリストなどの反復可能なオブジェクト
            lat (float | Iterable[float]):
                緯度（10進法）の数値またはリストなどの反復可能なオブジェクト
            in_crs (str | int | pyproj.CRS):
                入力座標系を指定するオプションの引数。
        Returns:
            float | list[float]:
                標高（メートル単位）の数値またはリスト
                - 単一の座標の場合はfloatを返す
                - 複数の座標の場合はlistを返す
        """
        if in_crs.to_epsg() != 4326:
            xy = transform_xy(x, y, in_crs, "EPSG:4326")
            if isinstance(xy, list):
                x = [xy_.x for xy_ in xy]
                y = [xy_.y for xy_ in xy]
            else:
                x, y = xy.x, xy.y
        return fetch_elevation_from_web(x, y)

    @type_checker_elev_type(arg_index=5, kward="elev_type")
    def fetch_elevation_tile_xy(
        self,
        x: float,
        y: float,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        elev_type: str,
        **kwargs,
    ) -> TileData:
        """
        ## Summary:
            指定した座標とズームレベルに対応する標高タイルの情報を取得します。
        Args:
            x (float):
                タイルのx座標
            y (float):
                タイルのy座標
            zoom_level (int):
                ズームレベルを指定する整数値。
                - dem10b: 1 ~ 14 の範囲にある整数。
                - dem5a, dem5b: 1 ~ 15 の範囲にある整数。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            elev_type (str):
                タイルの種類を指定する文字列。デフォルトは'dem10b'（10mメッシュ標高タイル）。
                他には 'dem5a'や'dem5b'（5mメッシュ標高タイル）などがあります。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            TileData:
                指定された座標とズームレベルに対応するタイルの情報を含むTileDataオブジェクト。
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 標高値の配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        # タイルのURLを取得
        tile_urls = TileUrls()
        url_template = {
            "dem10b": tile_urls.dem_10b,
            "dem5a": tile_urls.dem_5a,
            "dem5b": tile_urls.dem_5b,
        }.get(elev_type)
        self._check_elev_zl(elev_type, zoom_level)
        tile_info = search_tile_info_from_xy(x, y, zoom_level, in_crs, **kwargs)
        url = url_template.format(
            z=tile_info.zoom_level,
            x=tile_info.x_idx,
            y=tile_info.y_idx,
        )
        # URLから標高タイルを取得
        resps = fetch_elevation_tiles_from_web([url])
        ary = resps[url]
        return TileData(
            zoom_level=tile_info.zoom_level,
            x_idx=tile_info.x_idx,
            y_idx=tile_info.y_idx,
            tile_scope=tile_info.tile_scope,
            x_resolution=tile_info.x_resolution,
            y_resolution=tile_info.y_resolution,
            crs=tile_info.crs,
            ary=ary,
            width=tile_info.width,
            height=tile_info.height,
        )

    @type_checker_elev_type(arg_index=5, kward="elev_type")
    def fetch_elevation_tile_geometry(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        elev_type: str = "dem10b",
        **kwargs,
    ) -> list[TileData]:
        """
        ## Summary:
            指定したジオメトリとズームレベルに対応する標高タイルの情報を取得します。
        Args:
            geometry (shapely.geometry.base.BaseGeometry):
                タイルを検索するためのジオメトリ。
                例: shapely.geometry.Point, shapely.geometry.Polygonなど、
                `geometry.bounds`でgeometryの範囲を取得できるオブジェクト。
            zoom_level (int):
                ズームレベルを指定する整数値。
                - dem10b: 1 ~ 14 の範囲にある整数。
                - dem5a, dem5b: 1 ~ 15 の範囲にある整数。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            elev_type (str):
                タイルの種類を指定する文字列。デフォルトは'dem10b'（10mメッシュ標高タイル）。
                他には 'dem5a'や'dem5b'（5mメッシュ標高タイル）などがあります。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            `geometry`をカバーするタイルの情報を含むTileDataオブジェクトのリスト。
            各TileDataオブジェクトは以下の属性を持ちます:
            list[TileData]:
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 標高値の配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        # タイルのURLテンプレートを取得
        tile_urls = TileUrls()
        url_template = {
            "dem10b": tile_urls.dem_10b,
            "dem5a": tile_urls.dem_5a,
            "dem5b": tile_urls.dem_5b,
        }.get(elev_type)
        if elev_type == "dem10b":
            if (zoom_level < 1) or (14 < zoom_level):
                raise ValueError("dem10b is only available for zoom levels 1 to 14.")
        elif elev_type in ["dem5a", "dem5b"]:
            if (zoom_level < 1) or (15 < zoom_level):
                raise ValueError(
                    "dem5a and dem5b are only available for zoom levels 1 to 15."
                )
        # geometryをカバーするタイル情報を取得
        tile_infos = search_tile_info_from_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            **kwargs,
        )
        # URLを生成
        urls = []
        for tile_info in tile_infos:
            url = url_template.format(
                z=tile_info.zoom_level,
                x=tile_info.x_idx,
                y=tile_info.y_idx,
            )
            urls.append(url)
        # URLから標高タイルを取得し、TileDataオブジェクトを生成
        resps = fetch_elevation_tiles_from_web(urls)
        tile_datasets = []
        for tile_info, ary in zip(tile_infos, resps.values(), strict=False):
            tile_data = TileData(
                zoom_level=tile_info.zoom_level,
                x_idx=tile_info.x_idx,
                y_idx=tile_info.y_idx,
                tile_scope=tile_info.tile_scope,
                x_resolution=tile_info.x_resolution,
                y_resolution=tile_info.y_resolution,
                crs=tile_info.crs,
                ary=ary,
                width=tile_info.width,
                height=tile_info.height,
            )
            tile_datasets.append(tile_data)
        return tile_datasets

    def fetch_elevation_tile_mesh_with_dem10b(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        **kwargs,
    ) -> list[TileData]:
        """
        ## Summary:
            指定したジオメトリとズームレベルに対応する10mメッシュ標高タイルの
            情報を取得します。
        Args:
            geometry (shapely.geometry.base.BaseGeometry):
                タイルを検索するためのジオメトリ。
                例: shapely.geometry.Point, shapely.geometry.Polygonなど、
                `geometry.bounds`でgeometryの範囲を取得できるオブジェクト。
            zoom_level (int):
                ズームレベルを指定する整数値。1 ~ 14 の範囲にある整数。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            `geometry`をカバーする10mメッシュ標高タイルの情報を含むTileDataオブジェク
            トのリスト。各TileDataオブジェクトは以下の属性を持ちます:
            list[TileData]:
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 標高値の配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        return self.fetch_elevation_tile_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            elev_type="dem10b",
            **kwargs,
        )

    def fetch_elevation_tile_mesh_with_dem5a(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        **kwargs,
    ) -> list[TileData]:
        """
        ## Summary:
            指定したジオメトリとズームレベルに対応する5mメッシュ標高タイルの
            情報を取得します。dem5aのデータソースにはレーザー測量データが
            使用されており、より高精度な標高情報を提供します。
        Args:
            geometry (shapely.geometry.base.BaseGeometry):
                タイルを検索するためのジオメトリ。
                例: shapely.geometry.Point, shapely.geometry.Polygonなど、
                `geometry.bounds`でgeometryの範囲を取得できるオブジェクト。
            zoom_level (int):
                ズームレベルを指定する整数値。1 ~ 15 の範囲にある整数。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            `geometry`をカバーする5mメッシュ標高タイルの情報を含むTileDataオブジェクトの
            リスト。各TileDataオブジェクトは以下の属性を持ちます:
            list[TileData]:
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 標高値の配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        return self.fetch_elevation_tile_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            elev_type="dem5a",
            **kwargs,
        )

    def fetch_elevation_tile_mesh_with_dem5b(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        **kwargs,
    ) -> list[TileData]:
        """
        ## Summary:
            指定したジオメトリとズームレベルに対応する5mメッシュ標高タイルの
            情報を取得します。dem5bのデータソースには写真測量データが使用されており、
            レーザーよりは精度は劣ります。
        Args:
            geometry (shapely.geometry.base.BaseGeometry):
                タイルを検索するためのジオメトリ。
                例: shapely.geometry.Point, shapely.geometry.Polygonなど、
                `geometry.bounds`でgeometryの範囲を取得できるオブジェクト。
            zoom_level (int):
                ズームレベルを指定する整数値。1 ~ 15 の範囲にある整数。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            `geometry`をカバーする5mメッシュ標高タイルの情報を含むTileDataオブジェクトの
            リスト。各TileDataオブジェクトは以下の属性を持ちます:
            list[TileData]:
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 標高値の配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        return self.fetch_elevation_tile_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            elev_type="dem5b",
            **kwargs,
        )

    def _check_elev_zl(self, elev_type: str, zoom_level: int) -> bool:
        """
        ## Summary:
            指定された標高タイルの種類とズームレベルが有効かどうかをチェックします。
        Args:
            elev_type (str):
                標高タイルの種類を指定する文字列。'dem10b', 'dem5a', 'dem5b'のいずれか。
            zoom_level (int):
                ズームレベルを指定する整数値。
        Returns:
            bool:
                ズームレベルが有効な範囲内であればTrueを返します。
                無効な場合はValueErrorを発生させます。
        """
        if elev_type == "dem10b":
            if 1 <= zoom_level <= 14:
                return True
            else:
                raise ValueError(
                    "dem10b tiles are only available for zoom levels 1 to 14."
                )
        elif elev_type in ["dem5a", "dem5b"]:
            if 1 <= zoom_level <= 15:
                return True
            else:
                raise ValueError(
                    "dem5a and dem5b tiles are only available for zoom levels 1 to 15."
                )
        else:
            raise ValueError(
                f"Unknown elevation type: {elev_type}. "
                "Please use 'dem10b', 'dem5a', or 'dem5b'."
            )

    @type_checker_img_type(arg_index=4, kward="image_type")
    def fetch_img_tile_xy(
        self,
        x: float,
        y: float,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        image_type: str = "standard",
        **kwargs,
    ) -> TileData:
        """
        ## Summary:
            指定した座標とズームレベルに対応する画像タイルの情報を取得します。
        Args:
            x (float):
                タイルのx座標（経度）
            y (float):
                タイルのy座標（緯度）
            zoom_level (int):
                ズームレベルを指定する整数値。
            in_crs (str | int):
                入力座標の座標系を指定。
            image_type (str):
                タイルの種類を指定する文字列。追加する場合は'config.TileUrls'にURLを定義し、
                'formatter.py'の"IMG_TYPES"に追加してください。
                - "standard": 標準地図タイル
                - "photo": 空中写真タイル
                - "slope": 傾斜タイル
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            TileData:
                指定された座標とズームレベルに対応するタイルの情報を含むTileDataオブジェクト。
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 画像データの配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        # タイルのURLを取得
        tile_urls = TileUrls()
        url_template = {
            "standard": tile_urls.standard_map,
            "photo": tile_urls.photo_map,
            "slope": tile_urls.slope_map,
        }.get(image_type, tile_urls.standard_map)
        assert self._check_img_zl(image_type, zoom_level)
        tile_info = search_tile_info_from_xy(x, y, zoom_level, in_crs, **kwargs)
        url = url_template.format(
            z=tile_info.zoom_level,
            x=tile_info.x_idx,
            y=tile_info.y_idx,
        )
        # URLからタイルを取得
        resps = fetch_img_map_tiles_from_web([url])
        img = resps[url]
        return TileData(
            zoom_level=tile_info.zoom_level,
            x_idx=tile_info.x_idx,
            y_idx=tile_info.y_idx,
            tile_scope=tile_info.tile_scope,
            x_resolution=tile_info.x_resolution,
            y_resolution=tile_info.y_resolution,
            crs=tile_info.crs,
            ary=img,
            width=tile_info.width,
            height=tile_info.height,
        )

    def fetch_img_tile_geometry(
        self,
        geometry: Iterable[XY],
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        image_type: str = "standard",
        **kwargs,
    ) -> TileData:
        """
        ## Summary:
            指定したジオメトリとズームレベルに対応する画像タイルの情報を取得します。
        Args:
            geometry (shapely.geometry.base.BaseGeometry):
                タイルを検索するためのジオメトリ。
                例: shapely.geometry.Point, shapely.geometry.Polygonなど、
                `geometry.bounds`でgeometryの範囲を取得できるオブジェクト。
            zoom_level (int):
                ズームレベルを指定する整数値。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            image_type (str):
                タイルの種類を指定する文字列。デフォルトは'standard'（標準地図タイル）。
                他には 'photo'や'slope'（空中写真タイル、傾斜タイル）などがあります。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            `geometry`をカバーするタイルの情報を含むTileDataオブジェクトのリスト。
            各TileDataオブジェクトは以下の属性を持ちます:
            list[TileData]:
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 画像データの配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        # タイルのURLテンプレートを取得
        tile_urls = TileUrls()
        url_template = {
            "standard": tile_urls.standard_map,
            "pale": tile_urls.pale_map,
            "photo": tile_urls.photo_map,
            "slope": tile_urls.slope_map,
            "google_satellite": tile_urls.google_satellite,
            "micro_topo_miyagi": tile_urls.micro_topo_miyagi,
        }.get(image_type, tile_urls.standard_map)
        self._check_img_zl(image_type, zoom_level)
        # geometryをカバーするタイル情報を取得
        tile_infos = search_tile_info_from_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            **kwargs,
        )
        # URLを生成
        urls = []
        for tile_info in tile_infos:
            url = url_template.format(
                z=tile_info.zoom_level,
                x=tile_info.x_idx,
                y=tile_info.y_idx,
            )
            urls.append(url)
        # URLから画像タイルを取得し、TileDataオブジェクトを生成
        resps = fetch_img_map_tiles_from_web(urls)
        tile_datasets = []
        for tile_info, img in zip(tile_infos, resps.values(), strict=False):
            tile_data = TileData(
                zoom_level=tile_info.zoom_level,
                x_idx=tile_info.x_idx,
                y_idx=tile_info.y_idx,
                tile_scope=tile_info.tile_scope,
                x_resolution=tile_info.x_resolution,
                y_resolution=tile_info.y_resolution,
                crs=tile_info.crs,
                ary=img,
                width=tile_info.width,
                height=tile_info.height,
            )
            tile_datasets.append(tile_data)
        return tile_datasets

    def fetch_img_tile_geometry_with_standard_map(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        **kwargs,
    ) -> list[TileData]:
        """
        ## Summary:
            指定したジオメトリとズームレベルに対応する標準地図タイルの情報を取得します。
        Args:
            geometry (shapely.geometry.base.BaseGeometry):
                タイルを検索するためのジオメトリ。
                例: shapely.geometry.Point, shapely.geometry.Polygonなど、
                `geometry.bounds`でgeometryの範囲を取得できるオブジェクト。
            zoom_level (int):
                ズームレベルを指定する整数値。Zoomレベルは5から18の範囲で指定します。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            `geometry`をカバーする標準地図タイルの情報を含むTileDataオブジェクトのリスト。
            各TileDataオブジェクトは以下の属性を持ちます:
            list[TileData]:
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 画像データの配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        return self.fetch_img_tile_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            image_type="standard",
            **kwargs,
        )

    def fetch_img_tile_geometry_with_pale_map(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        **kwargs,
    ) -> list[TileData]:
        """
        ## Summary:
            指定したジオメトリとズームレベルに対応する淡色地図タイルの情報を取得します。
        Args:
            geometry (shapely.geometry.base.BaseGeometry):
                タイルを検索するためのジオメトリ。
                例: shapely.geometry.Point, shapely.geometry.Polygonなど、
                `geometry.bounds`でgeometryの範囲を取得できるオブジェクト。
            zoom_level (int):
                ズームレベルを指定する整数値。Zoomレベルは5から18の範囲で指定します。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            `geometry`をカバーするパレ地図タイルの情報を含むTileDataオブジェクトのリスト。
            各TileDataオブジェクトは以下の属性を持ちます:
            list[TileData]:
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 画像データの配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        return self.fetch_img_tile_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            image_type="pale",
            **kwargs,
        )

    def fetch_img_tile_geometry_with_photo_map(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        **kwargs,
    ) -> list[TileData]:
        """
        ## Summary:
            指定したジオメトリとズームレベルに対応する空中写真タイルの情報を取得します。
        Args:
            geometry (shapely.geometry.base.BaseGeometry):
                タイルを検索するためのジオメトリ。
                例: shapely.geometry.Point, shapely.geometry.Polygonなど、
                `geometry.bounds`でgeometryの範囲を取得できるオブジェクト。
            zoom_level (int):
                ズームレベルを指定する整数値。Zoomレベルは2から18の範囲で指定します。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            `geometry`をカバーする空中写真タイルの情報を含むTileDataオブジェクトのリスト。
            各TileDataオブジェクトは以下の属性を持ちます:
            list[TileData]:
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 画像データの配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        return self.fetch_img_tile_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            image_type="photo",
            **kwargs,
        )

    def fetch_img_tile_geometry_with_slope_map(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        **kwargs,
    ) -> list[TileData]:
        """
        ## Summary:
            指定したジオメトリとズームレベルに対応する傾斜タイルの情報を取得します。
        Args:
            geometry (shapely.geometry.base.BaseGeometry):
                タイルを検索するためのジオメトリ。
                例: shapely.geometry.Point, shapely.geometry.Polygonなど、
                `geometry.bounds`でgeometryの範囲を取得できるオブジェクト。
            zoom_level (int):
                ズームレベルを指定する整数値。Zoomレベルは3から15の範囲で指定します。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            `geometry`をカバーする傾斜タイルの情報を含むTileDataオブジェクトのリスト。
            各TileDataオブジェクトは以下の属性を持ちます:
            list[TileData]:
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 画像データの配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        return self.fetch_img_tile_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            image_type="slope",
            **kwargs,
        )

    def fetch_img_tile_geometry_with_google_satellite(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        **kwargs,
    ) -> list[TileData]:
        """
        ## Summary:
            指定したジオメトリとズームレベルに対応するGoogle衛星画像タイルの情報を取得します。
        Args:
            geometry (shapely.geometry.base.BaseGeometry):
                タイルを検索するためのジオメトリ。
                例: shapely.geometry.Point, shapely.geometry.Polygonなど、
                `geometry.bounds`でgeometryの範囲を取得できるオブジェクト。
            zoom_level (int):
                ズームレベルを指定する整数値。
            in_crs (str | int):
                入力座標系を指定するオプションの引数。
            **kwargs:
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        Returns:
            `geometry`をカバーするGoogle衛星画像タイルの情報を含むTileDataオブジェクトのリスト。
            各TileDataオブジェクトは以下の属性を持ちます:
            list[TileData]:
                - zoom_level(int): ズームレベル
                - x_idx(int): タイルのx座標
                - y_idx(int): タイルのy座標
                - tile_scope(TileScope): タイルの範囲を表す(x_min, y_min, x_max, y_max)
                - x_resolution(float): タイルのx方向の解像度
                - y_resolution(float): タイルのy方向の解像度
                - crs(pyproj.CRS): タイルの座標系を表すpyproj.CRSオブジェクト。
                - ary(numpy.ndarray): 画像データの配列。
                - width(int): タイルの幅（ピクセル単位）。デフォルトは256。
                - height(int): タイルの高さ（ピクセル単位）。デフォルトは256。
        """
        return self.fetch_img_tile_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            image_type="google_satellite",
            **kwargs,
        )  # type: ignore

    def fetch_img_tile_geometry_with_miyagi_micro_topo(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        zoom_level: int,
        in_crs: str | int | pyproj.CRS,
        **kwargs,
    ) -> list[TileData]:
        return self.fetch_img_tile_geometry(
            geometry=geometry,
            zoom_level=zoom_level,
            in_crs=in_crs,
            image_type="micro_topo_miyagi",
            **kwargs,
        )  # type: ignore

    def _check_img_zl(self, img_type: str, zoom_level: int) -> bool:
        """
        ## Summary:
            指定された画像タイプとズームレベルが有効かどうかをチェックします。
        Args:
            img_type (str):
                画像の種類を指定する文字列。'standard', 'photo', 'slope'のいずれか。
            zoom_level (int):
                ズームレベルを指定する整数値。
        Returns:
            bool:
                ズームレベルが有効な範囲内であればTrueを返します。
                無効な場合はValueErrorを発生させます。
        """
        if img_type in ["standard", "pale"]:
            if 5 <= zoom_level <= 18:
                return True
            else:
                raise ValueError(
                    "Standard map tiles are only available for zoom levels 5 to 18."
                )
        elif img_type in ["photo", "google_satellite"]:
            if 2 <= zoom_level <= 18:
                return True
            else:
                raise ValueError(
                    "Photo map tiles are only available for zoom levels 2 to 18."
                )
        elif img_type == "slope":
            if 3 <= zoom_level <= 15:
                return True
            else:
                raise ValueError(
                    "Slope map tiles are only available for zoom levels 5 to 18."
                )
        elif img_type in ["micro_topo_miyagi"]:
            if 2 <= zoom_level <= 18:
                return True
            else:
                raise ValueError(
                    "Micro topographic map tiles are only available for zoom levels 2 to 18."
                )
        else:
            raise ValueError(
                f"Unknown image type: {img_type}. "
                "Please use 'standard', 'photo', or 'slope'."
            )

    @type_checker_crs(arg_index=4, kward="in_crs")
    def fetch_geoid_height(
        self,
        x: float | list[float],
        y: float | list[float],
        in_crs: str | int | pyproj.CRS,
        year: int = 2011,
    ) -> float | list[float]:
        """
        ## Summary:
            指定した座標のジオイド高を取得します。
        Args:
            x (float | list[float]):
                ジオイド高を取得する経度（または経度のリスト）。
            y (float | list[float]):
                ジオイド高を取得する緯度（または緯度のリスト）。
            year (int):
                ジオイドモデルの年。デフォルトは2023年。
            in_crs (str | int | pyproj.CRS):
                入力座標系を指定するオプションの引数。
        Returns:
            float | list[float]:
                指定された座標のジオイド高（メートル単位）。
                単一の座標の場合はfloat、複数の座標の場合はlist[float]を返します。
        """
        if in_crs.to_epsg() != 4326:
            xy = transform_xy(x, y, in_crs, "EPSG:4326")
            if isinstance(xy, list):
                x = [xy_.x for xy_ in xy]
                y = [xy_.y for xy_ in xy]
            else:
                x, y = xy.x, xy.y
        return fetch_geoid_height_from_web(x, y, year)

    def fetch_geocentric_orthogonal_coordinates(
        self,
        params: list[dict[str, float]] = None,
        **kwargs,
    ) -> list[XYZ]:
        """
        ## Summary:
            経緯度と地心直交座標の相互変換を行います。
        Args:
            params (list[dict[str, float]], optional):
                経緯度から地心直交座標への変換を行うためのパラメータのリスト。
                - 'lat' (float): 緯度（度単位）
                - 'lon' (float): 経度（度単位）
                - 'alt' (float): 高度（メートル単位）
                - 'geoid_height' (float): ジオイド高（メートル単位）
                地心直交座標から経緯度への変換を行う場合は、'x', 'y', 'z'を使用します。
                - 'x' (float): 地心直交座標のX成分（メートル単位）
                - 'y' (float): 地心直交座標のY成分（メートル単位）
                - 'z' (float): 地心直交座標のZ成分（メートル単位）
            **kwargs:
                複数の座標を変換する場合は、paramsを使用してください。
                単一の座標を変換する場合は、以下のキーワード引数を使用できます。
                - 'lat' (float): 緯度（度単位）
                - 'lon' (float): 経度（度単位）
                - 'alt' (float): 高度（メートル単位）
                - 'geoid_height' (float): ジオイド高（メートル単位）
                地心直交座標から経緯度への変換を行う場合は、'x', 'y', 'z'を使用します。
                - 'x' (float): 地心直交座標のX成分（メートル単位）
                - 'y' (float): 地心直交座標のY成分（メートル単位）
                - 'z' (float): 地心直交座標のZ成分（メートル単位）
        Returns:
            list[XYZ]:
                変換後の座標のリスト。XYZオブジェクトのリストを返します。
                - LonLat to XYZ Example: "geocentricX", "geocentricY", "geocentricZ"
                - XYZ to LonLat Example: "longitude", "latitude", "ellipsoidHeight"
        """
        if params is not None:
            return fetch_geocentric_orthogonal_coordinates_from_web(params)
        return fetch_geocentric_orthogonal_coordinates_from_web([kwargs])


chirime = _ChiriinDrawer()


def calc_slope(dtm: np.ndarray, x_resol: float, y_resol: float) -> np.ndarray:
    """
    ---------------------------------------------------------------------------
    ## Summary:
        標高データから傾斜を計算する関数
    ---------------------------------------------------------------------------
    Args:
        dtm (np.ndarray):
            標高データの2次元配列
        x_resol (float):
            x方向の解像度（メートル単位）
        y_resol (float):
            y方向の解像度（メートル単位）
    ---------------------------------------------------------------------------
    Returns:
        np.ndarray:
            傾斜の2次元配列（度単位）
    """
    dy, dx = np.gradient(dtm, y_resol, x_resol)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg.astype(np.float16)


def calculate_mean_slope_in_polygon(
    poly: shapely.Polygon,
    in_crs: str | int | pyproj.CRS,
) -> float:
    """
    ---------------------------------------------------------------------------
    ## Summary:
        指定したポリゴン内の平均傾斜を計算する関数
    ---------------------------------------------------------------------------
    Args:
        poly (shapely.Polygon):
            ポリゴンジオメトリ
        in_crs (str | int | pyproj.CRS):
            入力データの投影法。この投影法はメルカトル図法である必要があります。
    ---------------------------------------------------------------------------
    Returns:
        float:
            平均傾斜（度単位）
    """
    resps = chiriin_drawer.fetch_elevation_tile_mesh_with_dem10b(poly, 14, in_crs)
    means = []
    for tile_data in resps:
        x_range = np.arange(
            tile_data.tile_scope.x_min,
            tile_data.tile_scope.x_max,
            tile_data.x_resolution,
        )
        if 256 < x_range.size:
            x_range = x_range[:256]
        y_range = np.arange(
            tile_data.tile_scope.y_min,
            tile_data.tile_scope.y_max,
            tile_data.y_resolution,
        )
        if 256 < y_range.size:
            y_range = y_range[:256]
        slope = calc_slope(tile_data.ary, tile_data.x_resolution, tile_data.y_resolution)
        means.append(np.mean(slope))
    return float(np.mean(means))
