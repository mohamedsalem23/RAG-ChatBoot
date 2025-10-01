import pandas as pd
import re

# Function to parse a Markdown table and convert it to a pandas DataFrame
def parse_markdown_table_to_df(markdown_text):
    # Search for a Markdown table (between '|' characters)
    table_pattern = r'\|(.+?)\|'
    tables = re.findall(table_pattern, markdown_text, re.DOTALL)
    if not tables:
        return None
    
    # Assume the first table
    table_lines = tables[0].strip().split('\n')
    if len(table_lines) < 2:
        return None
    
    # Extract rows
    rows = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if cells:
            rows.append(cells)
    
    if len(rows) < 2:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df
