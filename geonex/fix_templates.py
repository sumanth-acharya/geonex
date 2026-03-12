import glob
import re

for filepath in glob.glob('templates/*.html'):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix URL_FOR spaces, eg: url_for(" about") -> url_for("about")
    fixed_content = re.sub(r'url_for\(\"\s+([^"]+)\"\)', r'url_for("\1")', content)
    
    if content != fixed_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"Fixed typos in {filepath}")
