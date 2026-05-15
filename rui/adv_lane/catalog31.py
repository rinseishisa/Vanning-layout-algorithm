"""実 31 種ケースカタログ（純データ・ロジック無し）。

出典: 開示資料 (抜粋)ケースリスト.xlsx（いすゞロジスティクス, 2026-05-15 入手）。
木箱10種(id 1-10) + スチール21種(id 11-31)。寸法 mm。
h = シート H 列（天地固定の鉛直）。w,l = シート L,W 列（X-Y 90 度回転等価）。
原典 名称列は日本語 xlsx encoding 罠で openpyxl が非可逆破損 → 非本質ゆえ
ラベルは material+id の ASCII 合成（寸法/材料/id は検証済で正）。
重量列は原典に無し → weight は密度モデル（rui/要件定義書 §2）維持。
SSOT: rui/adv_lane/design_catalog31.md §2。"""
from __future__ import annotations

# (id, material, label, w, l, h)   w,l = sheet L,W ; h = sheet H
CATALOG_31 = [
    (1, 'wood', 'wood_01', 1490, 2260, 1050),
    (2, 'wood', 'wood_02', 1932, 2233, 990),
    (3, 'wood', 'wood_03', 1990, 2260, 900),
    (4, 'wood', 'wood_04', 1990, 2260, 2070),
    (5, 'wood', 'wood_05', 2550, 2280, 2355),
    (6, 'wood', 'wood_06', 1490, 2260, 900),
    (7, 'wood', 'wood_07', 1490, 2260, 980),
    (8, 'wood', 'wood_08', 1452, 1116, 1640),
    (9, 'wood', 'wood_09', 1452, 1116, 1270),
    (10, 'wood', 'wood_10', 1130, 1490, 1460),
    (11, 'steel', 'steel_11', 1190, 2260, 550),
    (12, 'steel', 'steel_12', 1190, 2260, 730),
    (13, 'steel', 'steel_13', 1190, 2260, 1100),
    (14, 'steel', 'steel_14', 1490, 2260, 550),
    (15, 'steel', 'steel_15', 1490, 2260, 730),
    (16, 'steel', 'steel_16', 1490, 2260, 900),
    (17, 'steel', 'steel_17', 1490, 2260, 1100),
    (18, 'steel', 'steel_18', 1490, 2260, 1460),
    (19, 'steel', 'steel_19', 1490, 2260, 1650),
    (20, 'steel', 'steel_20', 1490, 1130, 550),
    (21, 'steel', 'steel_21', 1490, 1130, 730),
    (22, 'steel', 'steel_22', 1490, 1130, 1100),
    (23, 'steel', 'steel_23', 1490, 1130, 1460),
    (24, 'steel', 'steel_24', 760, 1130, 550),
    (25, 'steel', 'steel_25', 760, 1130, 730),
    (26, 'steel', 'steel_26', 1990, 2260, 550),
    (27, 'steel', 'steel_27', 1990, 2260, 730),
    (28, 'steel', 'steel_28', 1990, 2260, 900),
    (29, 'steel', 'steel_29', 1990, 2260, 1100),
    (30, 'steel', 'steel_30', 1990, 2260, 1460),
    (31, 'steel', 'steel_31', 1990, 2260, 1650),
]

assert len(CATALOG_31) == 31
assert sum(1 for r in CATALOG_31 if r[1] == 'wood') == 10
assert sum(1 for r in CATALOG_31 if r[1] == 'steel') == 21
