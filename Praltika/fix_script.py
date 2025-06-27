try:
    # Read the file with UTF-8 encoding
    with open('app_fixed.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Create a new list of lines with fixed indentation
    fixed_lines = []
    
    # Track indentation level
    indentation_stack = []
    current_indentation = ""
    
    # Process each line
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        # Skip empty lines
        if not stripped_line:
            fixed_lines.append(line)
            continue
        
        # Calculate leading spaces
        leading_spaces = len(line) - len(line.lstrip())
        spaces = " " * leading_spaces
        
        # Check for indentation issues around line 1576
        if i+1 == 1576:
            # This is the problematic line - ensure it has correct indentation
            fixed_line = " " * 16 + stripped_line + "\n"
            fixed_lines.append(fixed_line)
            print(f"Fixed line {i+1}: '{fixed_line.strip()}' with {16} spaces")
        elif i+1 == 1577:
            # This is the line after the problematic line - ensure it has correct indentation
            fixed_line = " " * 20 + stripped_line + "\n"
            fixed_lines.append(fixed_line)
            print(f"Fixed line {i+1}: '{fixed_line.strip()}' with {20} spaces")
        else:
            # Keep the line as is
            fixed_lines.append(line)
            
        # Print lines in the problematic area for debugging
        if 1570 <= i+1 <= 1580:
            print(f"Line {i+1}: '{stripped_line}' with {leading_spaces} spaces")
    
    # Write the fixed content back to a new file
    with open('app_fixed_new.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("File processed successfully. Check app_fixed_new.py")
    
except Exception as e:
    print(f"Error: {e}")
