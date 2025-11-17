"""
## Summary:
    このモジュールは国土地理院で公開されている、セミダイナミック補正を行うための
    パラメーターファイルをダウンロードした後に、後処理する為のスクリプトです。
    パラメーターファイルは各年ごとに公開されており、現在は2009年から2025年まで対応しています。
    新しいファイルが公開された場合は、対応するURLを追加することでダウンロード可能です。

    パラメーターファイル：https://www.gsi.go.jp/sokuchikijun/semidyna.html#file
## Method of Execution:
    このスクリプトは直接実行することができます。実行すると、指定された各年の
    セミダイナミック補正パラメーターファイルをダウンロードし、後処理を行い、
    指定されたディレクトリにCSV形式で保存します。
    directory: chirime/data/
"""

import os
import subprocess
import tempfile
import zipfile

import pandas as pd

from chirime import geomesh

params_urls = {
    "2009": "https://www.gsi.go.jp/common/000185050.zip",
    "2010": "https://www.gsi.go.jp/common/000185049.zip",
    "2011": "https://www.gsi.go.jp/common/000185048.zip",
    "2012": "https://www.gsi.go.jp/common/000185047.zip",
    "2013": "https://www.gsi.go.jp/common/000185046.zip",
    "2014": "https://www.gsi.go.jp/common/000185045.zip",
    "2015": "https://www.gsi.go.jp/common/000185044.zip",
    "2016": "https://www.gsi.go.jp/common/000185043.zip",
    "2017": "https://www.gsi.go.jp/common/000186321.zip",
    "2018": "https://www.gsi.go.jp/common/000198965.zip",
    "2019": "https://www.gsi.go.jp/common/000224044.zip",
    "2020": "https://www.gsi.go.jp/common/000221430.zip",
    "2021": "https://www.gsi.go.jp/common/000239301.zip",
    "2022": "https://www.gsi.go.jp/common/000248856.zip",
    "2023": "https://www.gsi.go.jp/common/000248852.zip",
    "2024": "https://www.gsi.go.jp/common/000256608.zip",
    "2025": "https://www.gsi.go.jp/common/000268763.zip",
}


def get_semidyna_param(url: str) -> pd.DataFrame:
    """
    国土地理院のセミダイナミック補正パラメーターファイルをダウンロードし、
    後処理を行ってDataFrameとして返す。
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        output_file = tmp_file.name
        subprocess.run(["wget", url, "-O", output_file])

    with zipfile.ZipFile(output_file, "r") as zip_ref:
        extracted_files = zip_ref.namelist()
        # パラメーターファイルを一時ファイルに抽出
        for file_name in extracted_files:
            if file_name.endswith(".par"):
                with zip_ref.open(file_name) as source_file:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".par"
                    ) as dest_file:
                        dest_file.write(source_file.read())

    # パラメーターファイルを読み込み、不要なヘッダー部分をスキップしてDataFrameに格納
    with open(dest_file.name, "r", encoding="Shift-JIS") as f:
        skip = True
        lines = []
        for line in f.readlines():
            if "MeshCode" in line:
                skip = False
            if not skip:
                lines.append(line.replace("\n", ""))

    os.remove(dest_file.name)
    os.remove(output_file)

    header = [v for v in lines[0].split(" ") if v != ""]
    data = [v.split("  ") for v in lines[1:]]
    df = pd.DataFrame(data, columns=header)
    df["MeshCode"] = df["MeshCode"].astype(str)
    # 各メッシュコードに対応する境界座標を計算してDataFrameに追加

    x_mins = []
    y_mins = []
    x_maxs = []
    y_maxs = []
    for mesh_code in df["MeshCode"]:
        bounds = geomesh.jpmesh.mesh_code_to_bounds(mesh_code)
        x_mins.append(bounds.x_min)
        y_mins.append(bounds.y_min)
        x_maxs.append(bounds.x_max)
        y_maxs.append(bounds.y_max)

    df["lon_min"] = x_mins
    df["lat_min"] = y_mins
    df["lon_max"] = x_maxs
    df["lat_max"] = y_maxs
    return df


if __name__ == "__main__":
    dirname = os.path.join(os.path.dirname(__file__), "chirime/data")
    for year, url in params_urls.items():
        output_path = os.path.join(dirname, f"SemiDyna{year}.csv")
        if os.path.exists(output_path):
            print(
                f"Semi-dynamic parameter file for {year} already exists. Skipping download."
            )
            continue
        df = get_semidyna_param(url)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved semi-dynamic parameter file for {year} to: {output_path}")
