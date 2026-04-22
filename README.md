# Vanning Layout Algorithm

実装本体は [taiga/README.md](./taiga/README.md) にまとめています。

含まれるもの:
- `taiga/algorithm.py`
- `taiga/generate_items.py`
- `taiga/generate_items_json.py`

基本実行:

```bash
cd "/c/Users/taiga/Downloads/バンニングレイアウト"
python taiga/generate_items.py --small 8 --medium 12 --large 4 --destinations 2 --output items_input.json
python taiga/algorithm.py --input items_input.json --output layout_result.json --team-name "Team_Alpha"
```
