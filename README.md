# tokenizer_script_for_any_txt
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗-Transformers-orange)
![HF Datasets](https://img.shields.io/badge/🤗-Datasets-yellow)

- 🇧🇷 README em Português
- 
Script pra preparar **pré-treino de LLM** a partir de um `.txt`.

* Tokeniza com Hugging Face
* Concatena + adiciona `BOS/EOS`
* Corta em **chunks fixos** (`CHUNK_SIZE`)
* Salva **apenas `input_ids`** em **`int32`** (economiza disco) no formato `DatasetDict`
* Gera `train/validation` + `manifest.json`

> **Obs.:** não salva `attention_mask`/`token_type_ids` ou Labels.
> Para pré-treino, gere a máscara no **collate** do treino.

## Requisitos

```bash
pip install -U datasets transformers
```

## Uso

Edite o topo do script (caminho do `.txt`, tokenizer, `CHUNK_SIZE`) e rode:

```bash
python tokenizer_script_for_any_txt.py
```

Saída (exemplo):

```
tokenized_dataset_gpt2_YYYYMMDDTHHMMSSZ/
  ├─ train/
  ├─ validation/
  └─ manifest.json
```

## Dica (collate PyTorch)

```python
def collate(batch, pad_id):
    import torch
    ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.int32)
    attn = (ids != pad_id).to(torch.int64)
    return {"input_ids": ids.to(torch.int64), "attention_mask": attn, "labels": ids.to(torch.int64)}
```

## Licença

MIT.
