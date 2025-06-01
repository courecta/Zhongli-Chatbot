"""
Zhongli Dialogue Parser
Processes dialogue data for chatbot training
"""

import os
import sys
import glob
import re
from pathlib import Path
from collections import Counter, defaultdict
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

def discover_source_files():
    """Find source text files for parsing"""
    patterns = [
        'ZhongliScripture.txt',
        'Zhongli*.txt',
        '*zhongli*.txt',
        '*dialogue*.txt',
        '*script*.txt',
        '*.txt'
    ]

    base_path = Path('.') # Assuming files are in the current or subdirectories
    found_files = []
    for pattern in patterns:
        found_files.extend(list(base_path.rglob(pattern)))

    found_files = list(set(found_files)) # Remove duplicates
    found_files.sort()

    main_file = None
    for file_path in found_files:
        if 'ZhongliScripture.txt' in file_path.name:
            main_file = file_path
            break

    if not main_file and found_files:
        main_file = found_files[0] # Fallback to the first found file

    return main_file

def detect_encoding(file_path):
    """Detect file encoding"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue

    return 'utf-8'

def parse_dialogue_patterns(content):
    """Parse dialogue using multiple regex patterns"""
    patterns = [
        r'^([^:]+):\s*(.+)$',
        r'^"([^"]+)"\s*-\s*(.+)$',
        r'^(.+?)\s*says?\s*[:\-]\s*(.+)$',
        r'^(.+?)\s*[:\-]\s*(.+)$'
    ]

    dialogues = []

    for line in content.split('\n'):
        line = line.strip()
        if not line or len(line) < 3:
            continue

        parsed = False
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                speaker, dialogue = match.groups()
                speaker = speaker.strip()
                dialogue = dialogue.strip()

                if len(dialogue) >= 5 and len(speaker) >= 2:
                    dialogues.append({
                        'speaker': speaker,
                        'dialogue': dialogue
                    })
                    parsed = True
                    break

        if not parsed and len(line) >= 10:
            dialogues.append({
                'speaker': 'Unknown',
                'dialogue': line
            })

    return dialogues

def validate_data(df):
    """Validate and clean parsed data"""
    if df.empty:
        return df

    # Remove duplicates
    df = df.drop_duplicates(subset=['dialogue'], keep='first')

    # Filter by length
    df = df[df['dialogue'].str.len().between(5, 1000)]

    # Clean speaker names
    df['speaker'] = df['speaker'].str.title().str.strip()

    # Filter common invalid speakers
    invalid_speakers = ['Unknown', 'Narrator', 'System', '']
    df = df[~df['speaker'].isin(invalid_speakers)]

    return df.reset_index(drop=True)

def analyze_dataset(df):
    """Analyze dataset characteristics"""
    if df.empty:
        return {}

    stats = {
        'total_dialogues': len(df),
        'unique_speakers': df['speaker'].nunique(),
        'avg_dialogue_length': df['dialogue'].str.len().mean(),
        'speaker_distribution': df['speaker'].value_counts().to_dict(),
        'length_stats': {
            'min': df['dialogue'].str.len().min(),
            'max': df['dialogue'].str.len().max(),
            'median': df['dialogue'].str.len().median()
        }
    }

    return stats

def export_data(df, filename='zhongli_parsed_dialogues.csv'):
    """Export processed data to CSV"""
    try:
        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)

        # Export basic format
        basic_df = df[['speaker', 'dialogue']].copy()
        basic_df.columns = ['name', 'line']

        basic_path = f'output/{filename}'
        basic_df.to_csv(basic_path, index=False, encoding='utf-8')

        # Export detailed format
        detailed_path = f'output/detailed_{filename}'
        df.to_csv(detailed_path, index=False, encoding='utf-8')

        return basic_path, detailed_path

    except Exception as e:
        print(f"Export failed: {e}")
        return None, None

def process_dialogue_file(file_path):
    """Process a single dialogue file."""
    if not file_path or not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame() # Return empty DataFrame

    print(f"Processing file: {file_path}")
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()

    dialogues = parse_dialogue_patterns(content)
    if not dialogues:
        print(f"No dialogues found in {file_path}")
        return pd.DataFrame()

    df = pd.DataFrame(dialogues, columns=['speaker', 'dialogue'])
    df = validate_data(df)
    return df

def main():
    """Main processing function"""
    main_file = discover_source_files()
    if main_file:
        df = process_dialogue_file(main_file)
        if not df.empty:
            analyze_dataset(df)
            export_data(df)
        else:
            print("No data processed.")
    else:
        print("No source files found.")

if __name__ == "__main__":
    main()
