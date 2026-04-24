# Vanning Layout Algorithm

与えられた積荷データをもとに、コンテナへの最適な配置を計算するアルゴリズムを複数人で開発・比較するリポジトリです。

## フォルダ構成

| フォルダ | 担当者 |
|---|---|
| `shisa/` | https://github.com/rinseishisa |
| `taiga/` | https://github.com/hiramatsutaiga |
| `algo/kojima` | https://github.com/iput2023-kojima |

各メンバーが独自のアルゴリズムを実装し、同一の積荷データに対する結果を比較します。メンバーの作成が終わり次第、随時更新する予定

## コンテナ仕様

要件定義書に合わせて、40ft コンテナの次の定数を使用しています。

- Length: `12000 mm`
- Width: `2300 mm`
- Height: `2400 mm`
- Max payload: `24000 kg`

## 対応積荷タイプ

積荷は以下の 3 種類です。

- `small`: `760(W) x 1130(L) x 550(H)`
- `medium`: `1490(W) x 2260(L) x 900(H)`
- `large`: `2280(W) x 2550(L) x 2355(H)`

回転は水平面での `90°` 回転のみ許可し、天地は固定です。

## 入力データ仕様

ファイル名：`items_input.json`（shisa/items_input.json）

```json
{
  "dataset_info": {
    "dataset_name": "case_01",
    "seed": 42,
    "item_count": 100
  },
  "items": [
    {
      "item_id": "P001",
      "size_type": "large",
      "dimensions": { "w": 2280, "l": 2550, "h": 2355 },
      "weight": 1778.33,
      "destination_id": "DEST_C"
    }
  ]
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `dataset_name` | string | データセット識別名 |
| `seed` | int | 乱数シード |
| `item_count` | int | 積荷の総数 |
| `item_id` | string | 各積荷の識別ID（例：P001） |
| `size_type` | string | サイズ区分（`small` / `medium` / `large`） |
| `dimensions` | object | 寸法（単位：mm）。`w`=幅, `l`=長さ, `h`=高さ |
| `weight` | float | 重量（単位：kg） |
| `destination_id` | string | 荷降ろし先ID（例：DEST_A） |

## 出力形式

ファイル名：`layout_result.json`

アルゴリズムが計算したコンテナへの積荷配置結果を出力します。

```json
{
  "project_info": {
    "team_name": "Team_Alpha",
    "execution_time_ms": 2,
    "input_file": "items_input.json"
  },
  "containers": [
    {
      "container_id": 1,
      "destination_id": "DEST_A",
      "total_weight": 7891.72,
      "items": [
        {
          "item_id": "P001",
          "size_type": "large",
          "dimensions": { "w": 2280, "l": 2550, "h": 2355 },
          "position": { "x": 0, "y": 0, "z": 0 },
          "weight": 3883.75,
          "is_rotated": false,
          "destination_id": "DEST_A"
        }
      ]
    }
  ]
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `project_info.team_name` | string | チーム名 |
| `project_info.execution_time_ms` | int | アルゴリズムの実行時間（単位：ms） |
| `project_info.input_file` | string | 使用した入力ファイル名 |
| `containers` | array | 使用したコンテナの一覧 |
| `container_id` | int | コンテナの通し番号 |
| `destination_id` | string | このコンテナの配送先ID（例：DEST_A） |
| `total_weight` | float | コンテナ内の積荷合計重量（単位：kg） |
| `items` | array | このコンテナに積まれた積荷の一覧 |
| `item_id` | string | 積荷の識別ID（例：P001） |
| `size_type` | string | サイズ区分（`small` / `medium` / `large`） |
| `dimensions` | object | 配置時の寸法（単位：mm）。`w`=幅, `l`=長さ, `h`=高さ |
| `position` | object | コンテナ内の配置座標（単位：mm）。`x`=幅方向, `y`=奥行方向, `z`=高さ方向 |
| `weight` | float | 積荷の重量（単位：kg） |
| `is_rotated` | bool | XY平面での90度回転の有無 |
| `item.destination_id` | string | 積荷の配送先ID |

詳細な制約・評価ルールは[要件定義書](要件定義書v1.md)を参照。

## 今後追記予定

開発が進むごとに以下の内容を順次追記していきます。

- 各アルゴリズムの説明
- 実行方法・比較方法
- 評価指標・スコアリング結果
