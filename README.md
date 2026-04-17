# Vanning Layout Algorithm

与えられた積荷データをもとに、コンテナへの最適な配置を計算するアルゴリズムを複数人で開発・比較するリポジトリです。

## フォルダ構成

| フォルダ | 担当者 |
|---|---|
| `shisa/` | https://github.com/rinseishisa |

各メンバーが独自のアルゴリズムを実装し、同一の積荷データに対する結果を比較します。メンバーの作成が終わり次第、随時更新する予定

## 入力データ仕様

ファイル名：`items_input.json`（リポジトリルートに配置）

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

ファイル名：`config.json`（各メンバーのフォルダ内に配置）

アルゴリズムが使用するコンテナ仕様・制約・評価基準を定義します。

```json
{
  "container": {
    "type": "40ft",
    "dimensions": { "l": 5900, "w": 2350, "h": 2390 },
    "max_weight": 24000
  },
  "rotation_rule": {
    "z_axis_rotation_forbidden": true,
    "xy_90deg_rotation_only": true
  },
  "constraints": {
    "no_overlap": true,
    "ground_contact_required": true,
    "same_destination_only": true,
    "min_fill_rate": 0.5
  },
  "evaluation": {
    "disqualify_on": ["overlap", "out_of_bounds", "weight_over"],
    "fill_rate_penalty_threshold": 0.5,
    "efficiency": {
      "fewer_containers_is_better": true,
      "higher_average_fill_rate_is_better": true
    }
  }
}
```

| フィールド | 説明 |
|---|---|
| `container` | コンテナの種類・寸法（mm）・最大積載重量（kg） |
| `rotation_rule` | 回転制約（Z軸回転禁止、XY平面90度回転のみ許可） |
| `constraints` | 配置制約（重複禁止、接地必須、同一配送先のみ、最低充填率など） |
| `evaluation` | 評価基準（失格条件・充填率ペナルティ・重心位置・効率指標） |

詳細な制約・評価ルールは[要件定義書](要件定義書v1.md)を参照。

## 今後追記予定

開発が進むごとに以下の内容を順次追記していきます。

- 各アルゴリズムの説明
- 実行方法・比較方法
- 評価指標・スコアリング結果
