import re

def parse_momp_file(file_content):
    lines = file_content.strip().split('\n')
    results = []
    
    i = 0
    while i < len(lines):
        # Find start of block: "T is length"
        if re.match(r'T is length \d+, and m is set to 512', lines[i].strip()):
            length = re.search(r'T is length (\d+),', lines[i].strip()).group(1)
            i += 1  # Skip to next lines
            
            # Skip until MOMP section
            while i < len(lines) and not lines[i].strip().startswith('====== MOMP ======'):
                i += 1
            i += 1  # Skip header
            
            # Find Tpaa1in1 line
            while i < len(lines) and not (lines[i].strip().startswith('MOMP : Tpaa1in1 |')):
                i += 1
            
            if i < len(lines):
                line = lines[i].strip()
                curly_match = re.search(r'\{(\d+,\s*\d+)\}', line)
                time_match = re.search(r'Time:\s*([\d.]+)', line)
                
                curly = curly_match.group(1) if curly_match else None
                time = time_match.group(1) if time_match else None
                
                if curly and time:
                    results.append({
                        'length': length,
                        'curly': curly,
                        'time': time
                    })
        i += 1
    
    return results

# Example usage (replace with your file read)
data = """[paste your full file content here]"""
parsed = parse_momp_file(data)
for item in parsed:
    print(item)
