import csv
from pathlib import Path
import argparse

NEW_HEADER = [
    'neigh','lot','budget','status','runtime_s','gap','y_price',
    'y_price_out','y_log_in','y_log_out','cost','obj','floor1','floor2',
    'area_1st','area_2nd','bsmt','beds','fullbath','halfbath','kitchen'
]

OLD_HEADER = [
    'neigh','lot','budget','status','runtime_s','gap','y_price','cost','obj',
    'floor1','floor2','area_1st','area_2nd','bsmt','beds','fullbath','halfbath','kitchen'
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', required=True)
    ap.add_argument('--outfile', required=True)
    args = ap.parse_args()

    src = Path(args.infile)
    dst = Path(args.outfile)
    rows_in = src.read_text(encoding='utf-8').splitlines()
    out_rows = []

    # skip header line from input; rows may have mixed schemas
    for line in rows_in[1:]:
        if not line.strip():
            continue
        toks = line.split(',')
        if len(toks) == len(NEW_HEADER):
            rec = dict(zip(NEW_HEADER, toks))
        elif len(toks) == len(OLD_HEADER):
            tmp = dict(zip(OLD_HEADER, toks))
            rec = {k: '' for k in NEW_HEADER}
            # copy common fields
            for k in OLD_HEADER:
                rec[k] = tmp.get(k, '')
            # diag fields missing in old schema
            rec['y_price_out'] = ''
            rec['y_log_in'] = ''
            rec['y_log_out'] = ''
        else:
            # try to coerce by padding; otherwise skip
            if len(toks) < len(NEW_HEADER):
                toks = toks + ['']*(len(NEW_HEADER)-len(toks))
                rec = dict(zip(NEW_HEADER, toks))
            else:
                # too many tokens; keep first N
                rec = dict(zip(NEW_HEADER, toks[:len(NEW_HEADER)]))
        out_rows.append(rec)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=NEW_HEADER)
        w.writeheader()
        w.writerows(out_rows)

if __name__ == '__main__':
    main()

