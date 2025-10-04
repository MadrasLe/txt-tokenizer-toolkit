import os
import json
import gc
from datetime import datetime
from itertools import chain

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

# ===== CONFIG =====
DATASET_PATH = "/data/YOURTEXT.txt" 
TOKENIZER_NAME = "gpt2"
SEED = 42
CHUNK_SIZE = 512
NUM_PROC = min(os.cpu_count() or 2, 4)
BATCH_SIZE = 6000

RUN_TAG = datetime.now().strftime("%Y%m%dT%H%M%SZ")
OUTPUT_DIR_DRIVE = f"/data/tokenized_dataset_{TOKENIZER_NAME}_{RUN_TAG}"

# ===== CHECK =====
assert os.path.exists(DATASET_PATH), f"ERRO: O arquivo {DATASET_PATH} não foi encontrado."

# ===== TOKENIZER =====
print(f"--- Carregando tokenizer '{TOKENIZER_NAME}' do Hugging Face ---")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer carregado. PAD ID: {tokenizer.pad_token_id}")

# ===== LOAD TEXT =====
print("\n--- Carregando arquivo de texto com load_dataset ---")
raw = load_dataset("text", data_files=DATASET_PATH, split="train")
print(f"Dataset carregado com {len(raw)} exemplos.")
print(raw)

# ===== MAP FUNCS =====
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=False, padding=False, 
        return_attention_mask=False, return_token_type_ids=False
    )

def group_texts(examples):
    all_input_ids = []
    for ids in examples["input_ids"]:
        all_input_ids.extend([tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id])

    concatenated_examples = {"input_ids": all_input_ids}
    total_length = (len(concatenated_examples["input_ids"]) // CHUNK_SIZE) * CHUNK_SIZE
    if total_length == 0:
        return {"input_ids": []} 
    
    result = {
        "input_ids": [all_input_ids[i : i + CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)]
    }
    return result

# ===== TOKENIZE & CHUNK =====
print("\n--- Tokenizando (Map 1/2) ---")
tokenized = raw.map(
    tokenize_function, batched=True, batch_size=BATCH_SIZE, 
    num_proc=NUM_PROC, remove_columns=["text"]
)
del raw; gc.collect()

print("\n--- Chunking (Map 2/2) ---")
chunked = tokenized.map(
    group_texts, batched=True, batch_size=BATCH_SIZE, 
    num_proc=NUM_PROC, remove_columns=tokenized.column_names
)
del tokenized; gc.collect()

# ===== SPLIT & TYPES =====
print("\n--- Split train/validation ---")
dd_split = chunked.train_test_split(test_size=0.01, seed=SEED)
del chunked; gc.collect()

print("\n--- Fixando dtypes para int32 ---")
features = Features({"input_ids": Sequence(Value("int32"))})
dd = DatasetDict({
    "train": dd_split["train"].cast(features),
    "validation": dd_split["test"].cast(features),
})

print("\n--- Embaralhando train ---")
dd["train"] = dd["train"].shuffle(seed=SEED)

# ===== SAVE =====
print(f"\n--- Salvando DatasetDict no Drive: {OUTPUT_DIR_DRIVE} ---")
os.makedirs(OUTPUT_DIR_DRIVE, exist_ok=True)

dd.save_to_disk(OUTPUT_DIR_DRIVE, max_shard_size="2GB")
print("Salvo com sucesso.")

# ===== MANIFEST =====
print("\n--- Gravando manifest.json ---")
manifest = {
    "tokenizer_name": TOKENIZER_NAME, "tokenizer_vocab_size": tokenizer.vocab_size,
    "special_tokens_used": {
        "unk_token": tokenizer.unk_token, "pad_token": tokenizer.pad_token,
        "bos_token": tokenizer.bos_token, "eos_token": tokenizer.eos_token,
    },
    "chunk_size": CHUNK_SIZE, "batch_size": BATCH_SIZE, "num_proc": NUM_PROC,
    "seed": SEED, "run_tag": RUN_TAG,
}
with open(os.path.join(OUTPUT_DIR_DRIVE, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("Manifesto salvo.")

# ===== CLEANUP =====
print("\n--- Limpando caches e RAM ---")
os.system('rm -rf ~/.cache/huggingface/* 2>/dev/null || true')
gc.collect()

print("\n" + "="*60)
print("PROCESSO 100% CONCLUÍDO")
print(f"Dataset pronto em: {OUTPUT_DIR_DRIVE}")
print("="*60)
