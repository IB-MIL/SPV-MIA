import os
import random
import datasets
from datasets import load_dataset, load_from_disk, DatasetDict
from attack.utils import create_folder  # Assuming this is your utility

block_size = None
tokenizer_ = None
max_buff_size = None
text_column = None
chars_per_token = 3.6  # Default, or set via args
num_of_sequences = 1024  # Default, or set via args


def packing_texts(examples):
    if tokenizer_ is None or block_size is None or max_buff_size is None:
        raise ValueError("tokenizer_, block_size, and max_buff_size must be set before calling packing_texts")

    packed_texts = []
    assert list(examples.keys()) == ["text"], f"Expected 'text' key, got {list(examples.keys())}"
    iterator = iter(examples["text"])

    more_examples = True
    while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= max_buff_size:
                break
            try:
                text = next(iterator)
                buffer.append(text)
                buffer_len += len(text)
            except StopIteration:
                more_examples = False
                break

        if not buffer:
            continue

        # Tokenize buffer without truncation to get all token ids
        tokenized_inputs = tokenizer_(buffer, truncation=False)["input_ids"]
        all_token_ids = []
        for token_list in tokenized_inputs:
            all_token_ids.extend(token_list)

        for i in range(0, len(all_token_ids), block_size):
            chunk = all_token_ids[i : i + block_size]
            if len(chunk) == block_size:
                text_chunk = tokenizer_.decode(chunk)
                packed_texts.append(text_chunk)

    return {"text": packed_texts}


def dataset_prepare(args, tokenizer=None):
    """
    Prepares the dataset either by loading from Hugging Face Hub or from local disk.
    Automatically detects whether to use `load_dataset` or `load_from_disk`.

    Args:
        args: Argument parser or config with dataset settings
        tokenizer: Tokenizer to use (if any)

    Returns:
        train_dataset, valid_dataset: processed datasets
    """
    print(f"DEBUG: args object in dataset_prepare: {args}")

    dataset_path_base = "/fred/oz413/LLM"

    # Check required args attributes
    for attr in ["dataset_name", "dataset_config_name"]:
        if not hasattr(args, attr):
            raise AttributeError(f"args object missing required attribute '{attr}'")

    dataset_root = os.path.join(dataset_path_base, args.dataset_name)
    dataset_path_with_config = os.path.join(dataset_root, args.dataset_config_name)

    # Decide dataset path (local disk)
    if os.path.exists(dataset_path_with_config) and os.path.isdir(dataset_path_with_config):
        dataset_path = dataset_path_with_config
        print(f"Found dataset at config path: {dataset_path}")
    elif os.path.exists(dataset_root) and os.path.isdir(dataset_root):
        dataset_path = dataset_root
        print(f"Found dataset at root path: {dataset_path}")
    else:
        # Path does not exist, fallback to loading from HF hub
        dataset_path = None
        print(f"No local dataset found at {dataset_path_with_config} or {dataset_root}, will try loading from Hub.")

    # Load dataset either from local disk or from Hub
    if dataset_path is not None:
        raw_dataset = load_from_disk(dataset_path)
    else:
        raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name)

    # Extract train and valid splits
    if isinstance(raw_dataset, DatasetDict):
        if "train" not in raw_dataset:
            raise ValueError("DatasetDict missing 'train' split")
        train_dataset = raw_dataset["train"]

        if "test" in raw_dataset:
            valid_dataset = raw_dataset["test"]
        elif "validation" in raw_dataset:
            valid_dataset = raw_dataset["validation"]
        else:
            raise ValueError("DatasetDict missing both 'test' and 'validation' splits")
    elif isinstance(raw_dataset, datasets.Dataset):
        # If single Dataset, split manually (requires validation_split_percentage)
        if not hasattr(args, "validation_split_percentage") or not (0 < args.validation_split_percentage < 1):
            raise ValueError("Missing or invalid validation_split_percentage for single Dataset split")
        split_idx = int(len(raw_dataset) * (1 - args.validation_split_percentage))
        train_dataset = raw_dataset.select(range(split_idx))
        valid_dataset = raw_dataset.select(range(split_idx, len(raw_dataset)))
    else:
        raise ValueError(f"Loaded dataset is not DatasetDict or Dataset: {type(raw_dataset)}")

    global text_column
    column_names = train_dataset.column_names
    # Find a suitable text column
    for col_candidate in ["text", "document", "content"]:
        if col_candidate in column_names:
            text_column = col_candidate
            break
    else:
        raise ValueError(f"No suitable text column found in dataset columns: {column_names}")

    print(f"Using text column: '{text_column}'")

    # Select only the text column to reduce memory, then rename to "text" if needed
    train_dataset = train_dataset.select_columns([text_column])
    valid_dataset = valid_dataset.select_columns([text_column])
    if text_column != "text":
        print(f"Renaming column '{text_column}' to 'text'")
        train_dataset = train_dataset.rename_column(text_column, "text")
        valid_dataset = valid_dataset.rename_column(text_column, "text")

    if getattr(args, "packing", False):
        print("Applying text packing...")

        global block_size, tokenizer_, max_buff_size
        if not hasattr(args, "block_size"):
            raise AttributeError("args missing 'block_size' for packing")
        block_size = args.block_size
        tokenizer_ = tokenizer

        chars_per_token_local = getattr(args, "chars_per_token", chars_per_token)
        num_of_sequences_local = getattr(args, "num_of_sequences", num_of_sequences)
        max_buff_size = int(block_size * chars_per_token_local * num_of_sequences_local)

        for required_attr in ["cache_path", "preprocessing_num_workers", "use_dataset_cache"]:
            if not hasattr(args, required_attr):
                raise AttributeError(f"args missing required attribute '{required_attr}' for packing")

        cache_dir = os.path.join(args.cache_path, args.dataset_name, args.dataset_config_name)
        create_folder(cache_dir)

        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=os.path.join(cache_dir, "train_dataset_packed"),
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing train texts in chunks of {block_size} tokens"
        )

        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=os.path.join(cache_dir, "valid_dataset_packed"),
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing valid texts in chunks of {block_size} tokens"
        )
    else:
        print("Skipping text packing.")

    return train_dataset, valid_dataset
