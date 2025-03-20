import pandas as pd
from datasets import DatasetDict, Dataset

# Load dataset
df = pd.read_csv("data/phishing_site_urls.csv").dropna()

# Split data into two classes
df_safe = df[df['Label'] == "good"]
df_not_safe = df[df['Label'] == "bad"]

# Sample equal number of records from both classes
num_samples = 1500
df_safe_sample = df_safe.sample(num_samples, random_state=42)
df_not_safe_sample = df_not_safe.sample(num_samples, random_state=42)

# replace "Email Type" with Boolean flag "isPhising"
df_safe_sample = df_safe_sample.assign(isPhishing=False)
df_safe_sample = df_safe_sample.drop('Label',axis=1)
df_not_safe_sample = df_not_safe_sample.assign(isPhishing=True)
df_not_safe_sample = df_not_safe_sample.drop('Label',axis=1)

# Concatenate the samples to create a new balanced dataset
balanced_df = pd.concat([df_safe_sample, df_not_safe_sample])
balanced_df.columns = ['text', 'labels']

# convert labels column to int
balanced_df['labels'] = balanced_df['labels'].astype(int)

# Shuffle the balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train, validation, and test sets
train_size = int(0.7 * len(balanced_df))
valid_size = int(0.15 * len(balanced_df))

train_df = balanced_df[:train_size]
valid_df = balanced_df[train_size:train_size + valid_size]
test_df = balanced_df[train_size + valid_size:]

# Convert DataFrames to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)
test_ds = Dataset.from_pandas(test_df)

# Create DatasetDict and save to disk
dataset_dict = DatasetDict({'train': train_ds, 'validation': valid_ds, 'test': test_ds})
print(dataset_dict)

dataset_save_path = "data/phishing_dataset"
dataset_dict.save_to_disk(dataset_save_path)

print(f"Dataset saved to {dataset_save_path}")
