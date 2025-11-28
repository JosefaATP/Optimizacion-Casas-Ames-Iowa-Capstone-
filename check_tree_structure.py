#!/usr/bin/env python
"""
Verificar cómo están estructurados los nodos XGBoost.
"""

import json
from optimization.remodel.xgb_predictor import XGBBundle

def main():
    bundle = XGBBundle()
    bst = bundle.reg.get_booster()
    dumps = bst.get_dump(with_stats=False, dump_format="json")
    
    print(f"Total trees: {len(dumps)}")
    
    # Chequear primeros árboles que tengan splits
    found = False
    for tree_idx in range(min(100, len(dumps))):
        js = dumps[tree_idx]
        node = json.loads(js)
        
        if "split" in node:  # Es un nodo con split (no es leaf)
            print(f"\nTree {tree_idx} root node (first with split):")
            # Print first 800 chars
            dumped = json.dumps(node, indent=2)
            print(dumped[:800])
            
            if "children" in node:
                print(f"\nChildren: {len(node['children'])} nodes")
                for i, ch in enumerate(node["children"]):
                    nodeid = ch.get("nodeid")
                    leaf_val = ch.get("leaf")
                    split = ch.get("split")
                    print(f"  Child {i}: nodeid={nodeid}, leaf={leaf_val is not None}, has_split={split is not None}")
            
            print(f"Node 'yes' field: {node.get('yes')}")
            print(f"Node 'no' field: {node.get('no')}")
            found = True
            break
    
    if not found:
        print("No trees found with splits - all are leaves?")

if __name__ == "__main__":
    main()
