import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: remove_nulls.py <path-to-file>")
    sys.exit(2)

p = Path(sys.argv[1])
if not p.exists():
    print(f"File not found: {p}")
    sys.exit(1)

b = p.read_bytes()
cnt = b.count(b"\x00")
if cnt == 0:
    print(f"No null bytes in {p}")
    sys.exit(0)

# backup
bak = p.with_name(p.name + ".bak")
if not bak.exists():
    bak.write_bytes(b)

new = b.replace(b"\x00", b"")
p.write_bytes(new)
print(f"Removed {cnt} null byte(s) from {p}; backup at {bak}")
