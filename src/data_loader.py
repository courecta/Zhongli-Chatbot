"""
Data loading and preprocessing for DialoGPT training
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split

def detect_file_encoding(file_path: str) -> str:
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

def find_preprocessed_data() -> Optional[str]:
    """Find preprocessed CSV files"""
    possible_paths = [
        'zhongli_parsed_dialogues.csv',
        'ZhongliScript.csv',
        'output/zhongli_parsed_dialogues.csv'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None

def load_dialogue_data(file_path: Optional[str] = None):
    """Load and validate dialogue data"""
    if file_path is None:
        file_path = find_preprocessed_data()

    if not file_path or not os.path.exists(file_path):
        print("No data file found")
        return None

    # Detect encoding
    encoding = detect_file_encoding(file_path)

    try:
        data = pd.read_csv(file_path, encoding=encoding)
        print(f"Loaded {len(data)} rows from {file_path}")

        # Standardize column names
        column_mappings = {
            'line': 'dialogue',
            'text': 'dialogue',
            'content': 'dialogue',
            'message': 'dialogue',
            'name': 'speaker',
            'character': 'speaker',
            'who': 'speaker'
        }

        for old_name, new_name in column_mappings.items():
            if old_name in data.columns and new_name not in data.columns:
                data.rename(columns={old_name: new_name}, inplace=True)

        # Validate required columns
        required_columns = ['dialogue', 'speaker']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return None

        # Basic cleaning
        data = data.dropna(subset=['dialogue', 'speaker'])
        data = data[data['dialogue'].str.len() >= 5]
        data = data[data['speaker'].str.len() >= 2]

        print(f"After cleaning: {len(data)} valid dialogues")
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def validate_data_quality(data) -> dict:
    """Validate data quality metrics"""
    if data.empty:
        return {'score': 0, 'issues': ['No data']}

    issues = []
    score = 0

    # Check data volume
    if len(data) >= 100:
        score += 1
    else:
        issues.append(f"Low data volume: {len(data)} samples")

    # Check speaker diversity
    unique_speakers = data['speaker'].nunique()
    if unique_speakers >= 2:
        score += 1
    else:
        issues.append(f"Low speaker diversity: {unique_speakers} speakers")

    # Check dialogue length distribution
    dialogue_lengths = data['dialogue'].str.len()
    avg_length = dialogue_lengths.mean()
    if 20 <= avg_length <= 200:
        score += 1
    else:
        issues.append(f"Suboptimal dialogue length: {avg_length:.1f} chars")

    # Check for duplicates
    duplicates = data['dialogue'].duplicated().sum()
    if duplicates / len(data) < 0.1:
        score += 1
    else:
        issues.append(f"High duplicate rate: {duplicates/len(data):.1%}")

    return {'score': score, 'max_score': 4, 'issues': issues}

def create_train_validation_split(
    data,
    test_size: float = 0.1,
    random_state: int = 42
):
    """Create train-validation split"""
    if len(data) < 10:
        print("Dataset too small for split")
        return data, data.sample(min(2, len(data)))

    trn_df, val_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    print(f"Training samples: {len(trn_df)}")
    print(f"Validation samples: {len(val_df)}")

    return trn_df, val_df

def main():
    """Main data loading function"""
    # Load data
    data = load_dialogue_data()

    if data is None:
        print("Failed to load data")
        return None, None

    # Validate quality
    quality = validate_data_quality(data)
    print(f"Data quality score: {quality['score']}/{quality['max_score']}")

    if quality['issues']:
        print("Quality issues:")
        for issue in quality['issues']:
            print(f"  - {issue}")

    # Create split
    train_data, val_data = create_train_validation_split(data)

    return train_data, val_data

if __name__ == "__main__":
    train_df, val_df = main()
