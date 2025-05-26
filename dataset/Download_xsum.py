from datasets import load_dataset

ds = load_dataset("EdinburghNLP/xsum")

print(f"Dataset cached files are located at: {ds.cache_files}")
print(ds)