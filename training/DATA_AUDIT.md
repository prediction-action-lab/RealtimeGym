# Initial dataset audit

Source: `SALT-NLP/RealtimeGym` at commit
`044d7e301056baa6b88134c6e7080579ca662af6`.

All 836 trajectory files were read: 50,590 steps across Freeway (12,697),
Snake (10,493), and Overcooked (27,400). The default preparation settings
retain all agent paradigms and hold out seed 6 for validation and seed 7 for
testing.

135 malformed action labels were excluded and recorded in `rejected.jsonl`:
109 Freeway, 10 Snake, and 16 Overcooked. They include full action sequences,
equations such as `a_{9} = U`, and unknown letters. No action was inferred
from these labels or from the `thinking` field.

After duplicate-pair removal and removal of prompts shared across splits:

| Split | Freeway | Snake | Overcooked | Total | Prompts with conflicting labels |
|---|---:|---:|---:|---:|---:|
| Train | 5,023 | 5,703 | 15,285 | 26,011 | 779 |
| Validation | 666 | 1,089 | 2,595 | 4,350 | 83 |
| Test | 848 | 1,107 | 2,793 | 4,748 | 131 |

Conflicts are retained, so a deterministic policy cannot match every label for
those inputs. Exact prompt overlap was checked across all three splits.
This does not establish that the underlying layouts/dynamics are novel.

Using the real `Qwen/Qwen3-8B` tokenizer at model revision
`b968826d9c46dd6066d109eabc6255188de91218`, every retained example was encoded
and its supervised labels decoded. Each has exactly two supervised tokens:
one action letter and `<|im_end|>`. Every masked prompt prefix was checked
against the inference chat template with thinking disabled.

| Split | Minimum tokens | 95th percentile | Maximum tokens |
|---|---:|---:|---:|
| Train | 319 | 688 | 768 |
| Validation | 320 | 697 | 771 |
| Test | 320 | 692 | 767 |

The generated files and `tokenization-report.json` are under `data/sft/`,
which is ignored by Git. The manifest records source hashes and removal
counts. A fresh machine can regenerate the prepared data using:

```bash
uv run --project training python -m training.prepare \
  --revision 044d7e301056baa6b88134c6e7080579ca662af6
```

Validation also included a tiny randomly initialized Qwen3 model trained with
LoRA on CPU, followed by adapter save/reload and generated-action evaluation.
No Qwen3-8B weights were downloaded and no full GPU training or game-performance
evaluation was performed. These checks validate the pipeline, not model quality.
