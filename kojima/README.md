# kojima - バンニングレイアウトアルゴリズム

kojimaが作成したコンテナへの積荷配置アルゴリズムです。配送先ごとにグループ化し、重い・大きい積荷から順にコンテナへ詰めていく方式を採用しています。

## ファイル構成

| ファイル | 説明 |
|---|---|
| `algorithm.py` | `items_input.json` を読み込み、コンテナへの配置を計算して `layout_result.json` を出力する |
| `items_input.json` | 入力データ（積荷情報） |
| `layout_result.json` | 実行結果（配置結果） |

## 実行方法

### 前提

- Python 3 がインストールされていること
- このディレクトリ（`kojima/`）に `items_input.json` が配置されていること

### 手順

```bash
cd kojima
python algorithm.py
```

実行が完了すると、同じディレクトリに `layout_result.json` が出力され、コンソールに以下のメッセージが表示されます。

```
layout_result.json を出力しました
```

## アルゴリズムの概要

### コンテナ仕様（algorithm.py 内で定義）

| 項目 | 値 |
|---|---|
| 幅（CONTAINER_WIDTH） | 2300 mm |
| 長さ（CONTAINER_LENGTH） | 12000 mm |
| 高さ（CONTAINER_HEIGHT） | 2400 mm |
| 最大積載重量（MAX_WEIGHT） | 24000 kg |

### 処理フロー

1. **入力読み込み**：`items_input.json` から積荷リストを読み込む
2. **ソート**：以下の優先順位で並び替える
   - 配送先ID（`destination_id`）昇順
   - 重量（`weight`）降順 ← 重いものから先に積む
   - 体積（`w × l × h`）降順 ← 大きいものから先に積む
3. **配送先ごとにグループ化**：同じ配送先の積荷を1つのコンテナにまとめる
4. **詰め込み（行ベースのパッキング）**：
   - 幅方向（x軸）に並べる
   - 幅を超えたら、奥行方向（y軸）に次の行へ進む
   - 奥行も超えたら、高さ方向（z軸）に積み上げる
   - 高さも超えたら、次のコンテナへ移る
5. **回転**：幅に収まらない積荷は90度回転を試みる（`is_rotated: true`）
6. **重量制限**：コンテナの合計重量が 24000 kg を超える場合は、その積荷を次のコンテナへ送る
7. **結果出力**：`layout_result.json` に書き出す

### 出力例

```json
{
  "project_info": {
    "team_name": "kojima",
    "execution_time_ms": 2
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
          "is_rotated": false
        }
      ]
    }
  ]
}
```

## 注意事項

- `items_input.json` のパスは `algorithm.py` 内で相対パス（`"items_input.json"`）として指定されています。実行時のカレントディレクトリに注意してください。
- 入力データの仕様（フィールドの意味など）はリポジトリルートの [README.md](../../README.md) を参照してください。
