"""
Dataset creation and conversation context management for DialoGPT training
"""

from typing import List, Dict, Any
import pandas as pd # Added import for pandas
import torch # Added import for torch

class ConversationDataset:
    """Dataset for conversation training"""

    def __init__(self, tokenizer, data, block_size: int = 512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = self._build_examples(data)

    def _build_examples(self, data):
        examples = []
        conversations = self._create_conversations(data)
        for conv in conversations:
            encoded_conv = self._encode_conversation(conv)
            if encoded_conv is not None:
                examples.append(encoded_conv)
        return examples

    def _create_conversations(self, data) -> List[List[str]]:
        """Helper to create list of conversation strings"""
        # Assuming data is a DataFrame with 'speaker' and 'dialogue' columns
        # This is a simplified example; adapt based on your data structure
        if not isinstance(data, pd.DataFrame) or 'dialogue' not in data.columns:
            # Or handle cases where data is not a DataFrame as expected
            return []

        # Example: Group by a conversation ID if available, or treat each row as part of a continuous dialogue
        # This part needs to be adapted to how conversations are structured in your data
        dialogues = data['dialogue'].astype(str).tolist()
        # For simplicity, treating all dialogues as one long conversation split into turns
        # A more sophisticated approach would identify separate conversations
        return [dialogues] # Returns a list containing one conversation (all dialogues)

    def _encode_conversation(self, conversation: List[str]):
        """Encode a single conversation with tokenizer."""
        if not conversation:
            return None

        # Join turns with EOS token, then tokenize the whole conversation
        # Add EOS token to each utterance before joining
        eos_token = self.tokenizer.eos_token
        if eos_token is None:
            # Fallback if eos_token is not set (though it should be for causal LMs)
            eos_token = "<|endoftext|>"

        full_text = eos_token.join(utterance + eos_token for utterance in conversation)

        tokens = self.tokenizer.encode(full_text, add_special_tokens=True, truncation=True, max_length=self.block_size)

        if not tokens:
            return None

        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def create_conversation_contexts(data, character_name: str = None):
    """Create conversation contexts from dialogue data"""
    # Ensure data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        print("Data is not a pandas DataFrame. Cannot create contexts.")
        return pd.DataFrame() # Return empty DataFrame

    if character_name:
        # Filter by character name if provided
        if 'speaker' in data.columns:
            char_data = data[data['speaker'].str.contains(character_name, case=False, na=False)]
        else:
            print(f"'speaker' column not found. Cannot filter by character_name.")
            return pd.DataFrame() # Return empty DataFrame if speaker column is missing
    else:
        char_data = data

    if char_data.empty:
        print("No data found for the specified character or in general.")
        return pd.DataFrame() # Return empty DataFrame if no data

    contexts = []

    # Create simple conversation pairs
    # Ensure 'dialogue' and 'speaker' columns exist
    if 'dialogue' not in char_data.columns or 'speaker' not in char_data.columns:
        print("'dialogue' or 'speaker' column not found in char_data.")
        return pd.DataFrame()

    dialogues = char_data['dialogue'].tolist()
    # speakers = char_data['speaker'].tolist() # speakers variable was unused

    for i in range(len(dialogues) - 1):
        # Simple context: previous line is input, current line is output
        # This is a basic approach; more complex context creation might be needed
        contexts.append({'input_text': dialogues[i], 'target_text': dialogues[i+1]})

    return pd.DataFrame(contexts)

def create_datasets(tokenizer, train_data, val_data, block_size: int = 512, character_name: str = None):
    """Create training and validation datasets"""

    print("Creating conversation contexts...")

    # Create conversation contexts
    train_contexts = create_conversation_contexts(train_data, character_name)
    val_contexts = create_conversation_contexts(val_data, character_name)

    print(f"Training contexts: {len(train_contexts)}")
    print(f"Validation contexts: {len(val_contexts)}")

    # Create datasets
    train_dataset = ConversationDataset(tokenizer, train_data, block_size)
    val_dataset = ConversationDataset(tokenizer, val_data, block_size)

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    return train_dataset, val_dataset

def validate_datasets(train_dataset, val_dataset, tokenizer):
    """Validate dataset creation"""
    if len(train_dataset) == 0:
        print("Warning: Empty training dataset")
        return False

    if len(val_dataset) == 0:
        print("Warning: Empty validation dataset")
        return False

    # Check a sample
    try:
        sample = train_dataset[0]
        decoded = tokenizer.decode(sample, skip_special_tokens=True)
        print(f"Sample length: {len(sample)} tokens")
        print(f"Sample preview: {decoded[:100]}...")
        return True
    except Exception as e:
        print(f"Dataset validation error: {e}")
        return False

def main():
    """Main dataset creation function"""
    try:
        from .data_loader import main as load_data
        from .model_setup import main as setup_model
    except ImportError:
        from data_loader import main as load_data
        from model_setup import main as setup_model

    # Load data
    train_data, val_data = load_data()
    if train_data is None:
        print("Failed to load data")
        return None, None

    # Load model components
    model, tokenizer, config, training_args = setup_model()
    if tokenizer is None:
        print("Failed to load tokenizer")
        return None, None

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        tokenizer, train_data, val_data,
        block_size=512, character_name="Zhongli"
    )

    # Validate
    if validate_datasets(train_dataset, val_dataset, tokenizer):
        print("Datasets created successfully")
        return train_dataset, val_dataset
    else:
        print("Dataset validation failed")
        return None, None

if __name__ == "__main__":
    train_dataset, val_dataset = main()
