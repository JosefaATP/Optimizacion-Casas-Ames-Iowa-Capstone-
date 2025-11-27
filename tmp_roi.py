from pathlib import Path
import csv, html

text = Path(r"Info proyecto/data_description (1).txt").read_text(encoding="utf-8", errors="ignore")
block = text[text.lower().find("neighborhood:"):]
block = block.split("Condition1:")[0]
mapping = {}
for line in block.splitlines():
    parts = line.strip().split("\t")
    if len(parts) == 2 and parts[0] and parts[1]:
        mapping[parts[0].strip()] = parts[1].strip()

rows = []
with open(r"bench_out/construction_benchmark/roi_by_neighborhood.csv") as f:
    rd = csv.DictReader(f)
    for r in rd:
        try:
            roi = float(r["avg_roi"]) * 100
        except:
            continue
        rows.append((mapping.get(r["neighborhood"], r["neighborhood"]), roi))

rows = sorted(rows, key=lambda x: x[1], reverse=True)[:10]
width, bar_h, gap, left, right, top = 900, 26, 10, 220, 40, 40
height = top + len(rows) * (bar_h + gap) + 20
max_val = max(v for _, v in rows)
lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
    '<rect width="100%" height="100%" fill="white"/>',
    f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" fill="#111" font-weight="bold">ROI promedio (%) - Top 10 barrios</text>',
]
for i, (name, val) in enumerate(rows):
    y = top + i * (bar_h + gap)
    w = (width - left - right) * (val / max_val if max_val else 0)
    name = html.escape(name)
    lines.append(f'<text x="10" y="{y + bar_h*0.72:.1f}" font-size="12" fill="#333">{name}</text>')
    lines.append(f'<rect x="{left}" y="{y:.1f}" width="{w:.1f}" height="{bar_h}" fill="#377eb8" rx="3" ry="3"/>')
    lines.append(f'<text x="{left + w + 6:.1f}" y="{y + bar_h*0.72:.1f}" font-size="12" fill="#333">{val:.1f}%</text>')
lines.append('</svg>')

out = Path(r"bench_out/construction_benchmark/figures/roi_by_neighborhood_top10_full.svg")
out.write_text("\n".join(lines), encoding="utf-8")
print("SVG regenerado:", out)
