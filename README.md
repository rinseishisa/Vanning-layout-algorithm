# Vanning Layout Algorithm

40ft コンテナ向けのバンニングレイアウト設計実装です。  
`items_input.json` を入力として読み込み、要件定義書に沿った `layout_result.json` を出力します。  
出力結果は `vanning-eval` の評価器にそのまま渡せます。

## Overview

このリポジトリは次の 2 つの役割を持ちます。

1. `taiga/generate_items.py`
   - `small / medium / large` の件数から `items_input.json` を生成
   - 要件定義書の JSON 形式に合わせて、`dataset_info` と `items` を出力
2. `taiga/algorithm.py`
   - `items_input.json` を読み込み、40ft コンテナへの配置を決定
   - 評価器互換の `layout_result.json` を出力

## Files

- [taiga/generate_items.py](./taiga/generate_items.py)
  - 要件定義書どおりの入力データ生成スクリプト
- [taiga/generate_items_json.py](./taiga/generate_items_json.py)
  - `generate_items.py` と同内容の JSON 生成スクリプト
- [taiga/algorithm.py](./taiga/algorithm.py)
  - バンニング配置アルゴリズム本体
- [README.md](./README.md)
  - 実行方法と仕様の概要

## Container Specification

要件定義書に合わせて、40ft コンテナの次の定数を使用しています。

- Length: `12000 mm`
- Width: `2300 mm`
- Height: `2400 mm`
- Max payload: `24000 kg`

## Supported Item Types

積荷は以下の 3 種類です。

- `small`: `760(W) x 1130(L) x 550(H)`
- `medium`: `1490(W) x 2260(L) x 900(H)`
- `large`: `2280(W) x 2550(L) x 2355(H)`

回転は水平面での `90°` 回転のみ許可し、天地は固定です。

## Input Format

入力は `items_input.json` です。

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
      "size_type": "medium",
      "dimensions": {"w": 1490, "l": 2260, "h": 900},
      "weight": 1200.5,
      "destination_id": "DEST_A"
    }
  ]
}
```

## Output Format

出力は `layout_result.json` です。評価器が要求する形式に合わせています。

```json
{
  "project_info": {
    "team_name": "Team_Alpha",
    "execution_time_ms": 1250
  },
  "containers": [
    {
      "container_id": 1,
      "destination_id": "DEST_A",
      "total_weight": 18500,
      "items": [
        {
          "item_id": "P001",
          "size_type": "medium",
          "dimensions": {"w": 1490, "l": 2260, "h": 900},
          "position": {"x": 0, "y": 0, "z": 0},
          "weight": 1200,
          "is_rotated": true,
          "destination_id": "DEST_A"
        }
      ]
    }
  ]
}
```

## Constraints Considered

`taiga/algorithm.py` では次の制約を見ています。

- 積荷同士の重複禁止
- コンテナ外へのはみ出し禁止
- 接地制約
- 同一コンテナ内での `destination_id` 統一
- コンテナ総重量 `24000 kg` 以下
- Y 軸重心偏差の制約
- 容積利用率の確認

## Algorithm Outline

配置アルゴリズムは、実行可能解を安定して返すことを優先したヒューリスティックです。

- First Fit Decreasing ベース
- 目的地ごとに積荷を分割
- 候補点を既存積荷の端点から生成
- 回転 `0 / 90` を試行
- Y 軸重心偏差とデッドスペースを使って候補を評価
- 入らなければ新しいコンテナを追加

## Weight Generation

要件定義書に合わせて、生成される積荷重量は各サイズ共通で次の範囲です。

- `1000 kg` から `15000 kg`

## How To Run

### 1. 入力データ生成

```bash
cd "/c/Users/taiga/Downloads/バンニングレイアウト"
python taiga/generate_items.py --small 8 --medium 12 --large 4 --destinations 2 --output items_input.json
```

### 2. レイアウト設計

```bash
cd "/c/Users/taiga/Downloads/バンニングレイアウト"
python taiga/algorithm.py --input items_input.json --output layout_result.json --team-name "Team_Alpha"
```

### 3. 評価器へ直接出力

```bash
cd "/c/Users/taiga/Downloads/バンニングレイアウト"
python taiga/algorithm.py --input items_input.json --submission-name taiga --eval-root "/c/Users/taiga/Downloads/Vanning-layout-algorithm/vanning-eval/vanning_eval_rui" --team-name "Team_Alpha"
```

### 4. Batch 評価

```bash
cd "/c/Users/taiga/Downloads/Vanning-layout-algorithm/vanning-eval/vanning_eval_rui"
python main.py --batch
```

### 5. WebUI 起動

```bash
cd "/c/Users/taiga/Downloads/Vanning-layout-algorithm/vanning-eval/vanning_eval_rui"
python main.py
```

## Current Status

現状のコードは以下を満たしています。

- `items_input.json` 生成に対応
- `layout_result.json` 出力に対応
- evaluator 互換のスキーマに対応
- `taiga` 提出で batch 評価の合格実績あり

## Future Improvements

- コンテナ本数削減のための再配置
- 重心改善の局所探索
- 充填率向上のための候補点再評価
- 重量分布ルールの詳細化
- ビームサーチや焼きなましへの拡張
