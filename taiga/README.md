# taiga Implementation

`taiga/` は個人実装ディレクトリです。  
共通仕様、入力 JSON 形式、出力 JSON 形式、評価ルールは要件定義書を参照してください。

## Files

- [generate_items.py](./generate_items.py)
  - `items_input.json` を生成する入口スクリプト
- [generate_items_json.py](./generate_items_json.py)
  - JSON 生成本体
- [algorithm.py](./algorithm.py)
  - taiga 実装の配置アルゴリズム本体

## Implementation Notes

この実装は、まず違反を出さずに実行可能解を返すことを優先しています。

- 目的地ごとに積荷を分割してから配置
- First Fit Decreasing ベースで既存コンテナへ優先配置
- 候補点は既存積荷の端点と中央寄せ候補から生成
- 回転 `0 / 90` を試行
- 候補評価では Y 軸重心偏差とデッドスペースを優先
- 入らない場合のみ新しいコンテナを追加

## Current Characteristics

- 目的地混載を避ける構成
- Y 軸重心偏差を抑える方向に寄せたヒューリスティック
- 厳密最適化ではなく、安定して feasible な解を返す設計
- evaluator 互換の `layout_result.json` を出力

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

## Result Snapshot

- `items_input.json` 生成に対応
- `layout_result.json` 出力に対応
- evaluator 互換のスキーマに対応
- `taiga` 提出で batch 評価の合格実績あり

## Future Improvements

- コンテナ本数削減のための再配置
- 重心改善の局所探索
- 充填率向上のための候補点再評価
- ビームサーチや焼きなましへの拡張
