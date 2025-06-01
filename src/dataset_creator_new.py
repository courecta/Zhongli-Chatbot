"""
Dataset creation and conversation context management for DialoGPT training
"""

from typing import List, Dict, Any
from .dependencies import pd, torch

class ConversationDataset:
    """Dataset for conversation training"""

    def __init__(self, tokenizer, data, block_size: int = 512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        self._build_examples(data)

    def _build_examples(self, data):
        """Build conversation examples from data"""
        if pd is None:
            raise ImportError("pandas is required")

        conversations = self._create_conversations(data)

        for conversation in conversations:
            encoded = self._encode_conversation(conversation)
            if encoded is not None:
                self.examples.append(encoded)

    def _create_conversations(self, data) -> List[List[str]]:
        """Create conversation pairs from dialogue data"""
        conversations = []

        # Simple approach: create context-response pairs
        dialogues = data['dialogue'].tolist()

        for i in range(len(dialogues) - 1):
            context = dialogues[i]
            response = dialogues[i + 1]

            # Filter by length
            if len(context) > 5 and len(response) > 5:
                conversations.append([context, response])

        return conversations

    def _encode_conversation(self, conversation: List[str]):
        """Encode conversation into tokens"""
        try:
            # Join conversation with EOS tokens
            full_text = ""
            for turn in conversation:
                full_text += turn + self.tokenizer.eos_token

            # Tokenize
            tokens = self.tokenizer.encode(full_text, max_length=self.block_size, truncation=True)

            if len(tokens) < 10:  # Skip very short examples
                return None

            return torch.tensor(tokens, dtype=torch.long)

        except Exception as e:
            print(f"Encoding error: {e}")
            return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def create_conversation_contexts(data, character_name: str = None):
    """Create conversation contexts from dialogue data"""
    if pd is None:
        raise ImportError("pandas is required")

    if character_name:
        # Filter for specific character
        char_data = data[data['speaker'].str.contains(character_name, case=False, na=False)]
        if len(char_data) < 10:
            print(f"Insufficient data for {character_name}, using all data")
            char_data = data
    else:
        char_data = data

    contexts = []

    # Create simple conversation pairs
    dialogues = char_data['dialogue'].tolist()
    speakers = char_data['speaker'].tolist()

    for i in range(len(dialogues) - 1):
        context = {
            'context': dialogues[i],
            'response': dialogues[i + 1],
            'context_speaker': speakers[i],
            'response_speaker': speakers[i + 1]
        }
        contexts.append(context)

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
