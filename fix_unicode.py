#!/usr/bin/env python3
"""
Script to fix Unicode characters in Python files by replacing them with ASCII equivalents.
"""

import sys

def fix_unicode_in_file(filepath):
    """Replace Unicode characters with ASCII equivalents in a Python file."""
    try:
        # Read the file with UTF-8 encoding
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace Unicode characters with ASCII equivalents
        replacements = {
            '✔': '[OK]',      # Checkmark
            '→': '->',        # Right arrow
            '–': '-',         # En dash
            '—': '--',        # Em dash
        }
        
        original_content = content
        for old_char, new_char in replacements.items():
            count = content.count(old_char)
            if count > 0:
                print(f"Replacing {count} occurrence(s) of '{old_char}' with '{new_char}'")
                content = content.replace(old_char, new_char)
        
        # Write back with UTF-8 encoding
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n✓ Successfully fixed Unicode characters in {filepath}")
        return True
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_unicode.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    success = fix_unicode_in_file(filepath)
    sys.exit(0 if success else 1)
