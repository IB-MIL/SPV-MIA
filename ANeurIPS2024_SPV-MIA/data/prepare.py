import os
import random
import datasets
from datasets import load_from_disk, DatasetDict # Ensure DatasetDict is imported
from attack.utils import create_folder # Assuming this is your utility

block_size = None
tokenizer_ = None
max_buff_size = None
text_column = None
# It's generally better to pass these as parameters or get them from args
# instead of using globals, but we'll keep them for now to match your structure.
chars_per_token = 3.6  # Assuming a default or ensure it's set via args
num_of_sequences = 1024 # Assuming a default or ensure it's set via args


def packing_texts(examples):
    more_examples = True
    packed_texts = []
    # packed_ids = [] # This was defined but not used in the return, consider if needed
    assert list(examples.keys()) == ["text"] , f"Expected 'text' key, got {list(examples.keys())}"
    iterator = iter(examples["text"])
    # total_num = 0 # Defined but not used
    # drop_num = 0  # Defined but not used

    # Ensure tokenizer_ and block_size are not None before using them
    if tokenizer_ is None or block_size is None:
        raise ValueError("tokenizer_ or block_size is not set for packing_texts")

    while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= max_buff_size: # max_buff_size also needs to be set
                break
            try:
                buffer.append(next(iterator))
                buffer_len += len(buffer[-1]) # Assumes text elements are strings
            except StopIteration:
                more_examples = False
                break
        
        if not buffer: # If buffer is empty, skip tokenization
            continue

        # The double tokenization and decoding seems unusual and might be inefficient.
        # Consider if tokenizer_(buffer, truncation=False)["input_ids"] is sufficient
        # if your goal is just to get all token IDs.
        tokenized_inputs_initial = tokenizer_(buffer, truncation=False)["input_ids"]
        # The following two lines might be redundant if the tokenizer handles batch decoding and re-encoding consistently.
        # inputs_decoded = tokenizer_.batch_decode(tokenized_inputs_initial) 
        # tokenized_inputs_final = tokenizer_(inputs_decoded, truncation=False)["input_ids"]
        
        all_token_ids = []
        # Using tokenized_inputs_initial directly assuming it's a list of lists of token IDs
        for tokenized_input in tokenized_inputs_initial: 
            all_token_ids.extend(tokenized_input) # Assumes tokenized_input is a list of IDs

        for i in range(0, len(all_token_ids), block_size):
            input_ids_chunk = all_token_ids[i: i + block_size]
            if len(input_ids_chunk) == block_size:
                # packed_ids.append(input_ids_chunk) # If you need to return IDs
                input_text = tokenizer_.decode(input_ids_chunk)
                # total_num += 1
                # The check len(tokenizer_.encode(input_text)) == block_size can sometimes fail
                # due to special tokens or slight variations in encoding/decoding.
                # It's often safer to trust the chunking if the goal is fixed-length blocks.
                packed_texts.append(input_text)
                # drop_num += 1 # This was incrementing for every packed text, not just dropped ones.
                                # If it's for dropped ones, the logic needs adjustment.
    return {
        "text": packed_texts
    }

def dataset_prepare(args, tokenizer=None):
    """
    Prepares the dataset either by loading from Hugging Face Hub or from local disk.
    Automatically detects whether to use `load_dataset` or `load_from_disk`.

    Args:
        args: Argument parser with dataset config
        tokenizer: Tokenizer to use (if any)

    Returns:
        train_dataset, valid_dataset: processed training and validation datasets
    """
    print(f"DEBUG: args object in dataset_prepare: {args}") # For debugging

    dataset_path_base = "/fred/oz413/LLM"

    # Use args.dataset_name as it's likely how -d is parsed
    if not hasattr(args, 'dataset_name'):
        raise AttributeError("The 'args' object is missing the 'dataset_name' attribute. Check your main script's argument parsing for -d.")
    if not hasattr(args, 'dataset_config_name'):
        raise AttributeError("The 'args' object is missing the 'dataset_config_name' attribute. Check your main script's argument parsing.")

    # Build full path
    dataset_root = os.path.join(dataset_path_base, args.dataset_name) # MODIFIED: args.dataset -> args.dataset_name
    dataset_path_with_config = os.path.join(dataset_root, args.dataset_config_name)

  
    # And if not, but dataset_root exists, that one is used.
    # This might need adjustment based on how you've saved your datasets.
    # For ag_news saved by the save_agnews_script, dataset_path_with_config is the correct one.
    if os.path.exists(dataset_path_with_config) and os.path.isdir(dataset_path_with_config):
        dataset_path = dataset_path_with_config
        print(f"Found dataset at specific config path: {dataset_path}")
    elif os.path.exists(dataset_root) and os.path.isdir(dataset_root):
        dataset_path = dataset_root
        print(f"Found dataset at root path (no specific config subdir): {dataset_path}")
    else:
        # This error should ideally not be hit if the save_agnews_script ran successfully.
        raise FileNotFoundError(
            f"Dataset path not found. Checked: {dataset_path_with_config} and {dataset_root}. "
            "Ensure the dataset was downloaded and saved correctly using the preparation script."
        )

    print(f"Loading dataset from disk: {dataset_path}")
    raw_dataset = load_from_disk(dataset_path)

    # If it's a DatasetDict, expect 'train' and 'test' splits
    # AG News, when loaded from Hub, is a DatasetDict with 'train' and 'test'.
    if isinstance(raw_dataset, DatasetDict):
        if 'train' not in raw_dataset:
            raise ValueError("Loaded DatasetDict is missing 'train' split.")
        train_dataset = raw_dataset['train']
        
        # Use 'test' split for validation if available, otherwise try 'validation'
        if 'test' in raw_dataset:
            valid_dataset = raw_dataset['test'] 
        elif 'validation' in raw_dataset:
            valid_dataset = raw_dataset['validation']
        else:
            raise ValueError("Loaded DatasetDict is missing 'test' or 'validation' split.")
    # If it's a single Dataset object (less likely for ag_news from Hub but possible if processed differently)
    elif isinstance(raw_dataset, datasets.Dataset):
        print("Warning: Loaded dataset is a single Dataset object. Manually splitting for train/validation.")
        # This manual split might not be what you intend if ag_news has predefined splits.
        # Consider adjusting if you always expect a DatasetDict from load_from_disk.
        if not hasattr(args, 'validation_split_percentage') or not (0 < args.validation_split_percentage < 1):
            raise ValueError("validation_split_percentage arg is missing or invalid for splitting a single Dataset.")
        train_size = int((1 - args.validation_split_percentage) * len(raw_dataset))
        train_dataset = raw_dataset.select(range(train_size))
        valid_dataset = raw_dataset.select(range(train_size, len(raw_dataset)))
    else:
        raise ValueError(f"Loaded dataset from {dataset_path} is not a DatasetDict or Dataset. Type: {type(raw_dataset)}")

    global text_column # Using global for text_column
    # Determine column names from one of the splits, e.g., train_dataset
    column_names_in_split = train_dataset.column_names 
    if "text" in column_names_in_split:
        text_column = "text"
    elif "document" in column_names_in_split: # Common in AG News
        text_column = "document"
    elif "content" in column_names_in_split:
        text_column = "content"
    else:
        raise ValueError(f"No suitable text column found in dataset columns: {column_names_in_split}")
    
    print(f"Identified text column as: '{text_column}'")

    train_dataset = train_dataset.select_columns([text_column]) # select_columns expects a list
    valid_dataset = valid_dataset.select_columns([text_column]) # select_columns expects a list
    
    if text_column != "text":
        print(f"Renaming column '{text_column}' to 'text'.")
        train_dataset = train_dataset.rename_column(text_column, "text")
        valid_dataset = valid_dataset.rename_column(text_column, "text")

    if args.packing:
        print("Applying text packing...")
        global block_size, tokenizer_, max_buff_size # Using globals
        
        # Ensure these are set, e.g., from args or defaults
        if not hasattr(args, 'block_size'):
             raise AttributeError("args missing 'block_size' for packing.")
        block_size = args.block_size

        # Make sure chars_per_token and num_of_sequences are available
        # For example, if they are expected to be in args:
        current_chars_per_token = getattr(args, 'chars_per_token', 3.6) # Default if not in args
        current_num_of_sequences = getattr(args, 'num_of_sequences', 1024) # Default if not in args

        max_buff_size = block_size * current_chars_per_token * current_num_of_sequences
        tokenizer_ = tokenizer # tokenizer is passed as an argument to dataset_prepare

        if not hasattr(args, 'cache_path') or not hasattr(args, 'preprocessing_num_workers') or not hasattr(args, 'use_dataset_cache'):
            raise AttributeError("args missing one or more of: cache_path, preprocessing_num_workers, use_dataset_cache for packing.")

        cache_dir_train = f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}"
        create_folder(cache_dir_train) # Ensure cache directory for map exists
        
        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=os.path.join(cache_dir_train, "train_dataset_packed"),
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing train texts in chunks of {block_size} tokens"
        )
        
        cache_dir_valid = f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}"
        create_folder(cache_dir_valid) # Ensure cache directory for map exists (can be same as train)

        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=os.path.join(cache_dir_valid, "valid_dataset_packed"),
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing valid texts in chunks of {block_size} tokens"
        )
    else:
        print("Skipping text packing.")

    return train_dataset, valid_dataset
