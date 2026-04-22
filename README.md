# Vanning Layout Algorithm

40ft コンテナ向けのバンニングレイアウト生成用リポジトリです。  
本リポジトリでは、まず `small / medium / large` の荷物データを CSV で生成し、その CSV を入力として配置アルゴリズムを実行します。

## Overview

このリポジトリは、以下の 2 段構成で動作します。

1. `generate_items_csv.py`
   - 大・中・小の荷物数を指定して入力 CSV を生成します。
   - 各荷物には寸法、重量、目的地 ID を付与します。
2. `algorithm.py`
   - 生成された CSV を読み込み、40ft コンテナへの配置を行います。
   - 制約チェックと要件書形式の JSON 出力を行います。

## Files

- [generate_items_csv.py](./generate_items_csv.py)
  - 荷物データ生成スクリプト
- [algorithm.py](./algorithm.py)
  - バンニング配置アルゴリズム本体
- [generated_items.csv](./generated_items.csv)
  - 生成された入力 CSV の例
- [output_solution_spec.json](./output_solution_spec.json)
  - アルゴリズム出力例
- [algorithm.ipynb](./algorithm.ipynb)
  - Notebook ベースの検証用ファイル

## Container Specification

要件定義書に合わせて、以下の 40ft コンテナ条件を使用しています。

- Length: `5900 mm`
- Width: `2350 mm`
- Height: `2390 mm`
- Max payload: `24000 kg`

## Supported Item Types

荷物は以下の 3 種類を前提にしています。

- `small`: `760 x 1130 x 550`
- `medium`: `1490 x 2260 x 900`
- `large`: `2550 x 2280 x 2355`

回転は水平方向の 90 度回転のみ許可し、天地は固定です。

## Input CSV Format

`algorithm.py` は、以下の列を持つ CSV を入力とします。

- `item_id`
- `size_type`
- `width`
- `length`
- `height`
- `weight`
- `destination_id`

`generate_items_csv.py` はこの形式の CSV をそのまま出力します。

## Output JSON Format

出力は要件書に合わせて、以下の構造を持つ JSON です。

- `project_info`
- `containers`
  - `container_id`
  - `destination_id`
  - `total_weight`
  - `items`

各 `items` 要素には以下を出力します。

- `item_id`
- `size_type`
- `dimensions`
- `position`
- `weight`
- `is_rotated`

## Constraints Considered

`algorithm.py` では、少なくとも以下を見ています。

- 荷物同士の重複禁止
- コンテナ外へのはみ出し禁止
- 接地制約
- コンテナごとの重量上限
- 同一コンテナ内での `destination_id` 統一
- 長手方向 Y 軸の重心偏差制約
- 充填率の算出

## Algorithm Outline

配置アルゴリズムは、実行可能解を安定して返すことを優先したヒューリスティックです。

- First Fit Decreasing ベース
- 既存荷物の端点から候補座標を生成
- 回転 `0 / 90` を両方試行
- 既存コンテナに入るなら優先して配置
- 入らなければ新規コンテナを開く
- 候補評価では以下を優先
  - Y 軸重心偏差
  - デッドスペース
  - 配置高さ

## How To Run

### 1. Generate Input CSV

```powershell
cd "c:\Users\taiga\Downloads\バンニングレイアウト"
python generate_items_csv.py --small 8 --medium 12 --large 4 --destinations 2 --output generated_items.csv
```

### 2. Run Packing Algorithm

```powershell
cd "c:\Users\taiga\Downloads\バンニングレイアウト"
python algorithm.py --input generated_items.csv --output output_solution_spec.json --team-name "Team_Alpha"
```

## Example

生成と配置をまとめて実行する例です。

```powershell
cd "c:\Users\taiga\Downloads\バンニングレイアウト"
python generate_items_csv.py --small 8 --medium 12 --large 4 --destinations 2 --output generated_items.csv
python algorithm.py --input generated_items.csv --output output_solution_spec.json --team-name "Team_Alpha"
```

## Current Characteristics

現状の実装には以下の特徴があります。

- 目的地制約を満たすようにコンテナを分離
- 中央寄せ候補を追加して重心悪化を抑制
- 充填率や重心偏差を簡易評価
- 厳密最適化ではなく、まず feasible な解を返す構成

## Future Improvements

- 同一目的地内での荷物順序最適化
- コンテナ間 swap による重心改善
- 低充填率コンテナの再配置
- 候補点生成の高度化
- 焼きなまし、ビームサーチ、GA などへの拡張

