# shisa - バンニングレイアウトアルゴリズム

shisaが作成したコンテナへの積荷配置アルゴリズムです。

## ファイル構成

| ファイル | 説明 |
|---|---|
| `generate_items.py` | 積荷データをランダム生成し `items_input.json` を出力する |
| `layout_desinger.py` | `items_input.json` を読み込み、コンテナへの配置を計算して `layout_result.json` を出力する |

## 実行方法

### Step 1: 積荷データを生成

```bash
python generate_items.py
```

`items_input.json` が生成されます。

> **注意**: 毎回 `generate_items.py` を実行すると積荷データが変わり、結果の比較が難しくなります。一度生成した `items_input.json` をそのまま使い回してStep 2を実行することを推奨します。

### Step 2: レイアウトを計算

```bash
python layout_desinger.py
```

`layout_result.json` が生成されます。

## 出力ファイル

| ファイル | 説明 |
|---|---|
| `items_input.json` | 生成された積荷データ |
| `layout_result.json` | バンニング配置結果 |

## アルゴリズム概要

- **Best Fit Decreasing**: 体積の大きい荷物から順にコンテナへ配置する
- **目的地制約**: 同一目的地ID（DEST_A / DEST_B / DEST_C）の荷物のみ同一コンテナに積む
- **回転ルール**: 天地固定。水平方向（X-Y平面）での90度回転のみ許可

## コンテナ仕様

| 項目 | 値 |
|---|---|
| サイズ | 5900(L) × 2350(W) × 2390(H) mm |
| 最大積載重量 | 24,000 kg |
