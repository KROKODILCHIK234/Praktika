#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def fix_syntax_errors():
    """Fix specific syntax errors in app_fixed.py"""
    
    # Read the file
    with open('app_fixed.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} lines...")
    
    # Fix specific problematic sections
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix 1: Remove duplicate else statements around line 1204-1206
        if i >= 1203 and i <= 1207:
            if 'else:' in line and i == 1204:
                # Keep this else statement
                fixed_lines.append(line)
            elif 'else:' in line and i == 1205:
                # Skip this duplicate else
                print(f"Skipped duplicate else at line {i+1}")
            elif 'else:' in line and i == 1207:
                # Skip this duplicate else
                print(f"Skipped duplicate else at line {i+1}")
            else:
                fixed_lines.append(line)
        # Fix 2: Fix unterminated string literal
        elif 'key=" geometry_plot_loaded\\)' in line:
            fixed_lines.append(line.replace('key=" geometry_plot_loaded\\)', 'key="geometry_plot_1"'))
            print(f"Fixed unterminated string literal at line {i+1}")
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write the fixed content back
    with open('app_fixed.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("Syntax errors fixed!")

if __name__ == "__main__":
    fix_syntax_errors() 