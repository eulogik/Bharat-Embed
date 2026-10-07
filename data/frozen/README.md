# data/frozen - freeze log

This dir holds hash frozen triplets. No train run starts without hashes here.

## Policy

- FLORES text never ships in train pack. Eval only. License CC-BY-SA.
- Train pack is Banking77 plus MS MARCO hard subset plus synthetic Hinglish plus synthetic legal.
- Synthetic slices need 500 human spot checks each before freeze.
- Write SHA256SUMS right after freeze. Train scripts print and check it.

## Target files

- triplets_40k.jsonl (15k Hinglish synth, 5k Banking77, 10k legal synth for shared eval, 10k MS MARCO hard)
- legal_10k.jsonl (statute query pairs, ours, Apache-2.0)
- SHA256SUMS

## Row shape

```json
{"query": "task: search result | query: ...", "pos": "title: ... | text: ...", "negs": ["title: ... | text: ..."]}
```

Queries and docs must use byte exact prefixes from src/embed/prefixes.py.
