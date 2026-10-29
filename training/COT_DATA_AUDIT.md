# Reasoning-and-action dataset audit

Source: `SALT-NLP/RealtimeGym` at the same pinned commit used for action-only
training, `044d7e301056baa6b88134c6e7080579ca662af6`.

All 836 trajectories were read, containing 50,590 steps. Every `thinking`
field is a string. Of those, 6,226 are the literal placeholder
`Still thinking...`; they are excluded from this training set. The same 135
invalid action labels are excluded as in the first experiment. Existing
thinking delimiters are removed before wrapping each trace once; remaining
trace text and any appended teacher plan are retained. The recorded `action`
field supplies the final label, including when the trace describes a different
action or a plan begun on an earlier turn.

After the same prompt/action deduplication and cross-split overlap removal:

| Split | Freeway | Snake | Overcooked | Total |
|---|---:|---:|---:|---:|
| Train | 3,586 | 5,576 | 13,347 | 22,509 |
| Validation | 569 | 1,070 | 2,294 | 3,933 |
| Test | 585 | 1,083 | 2,435 | 4,103 |

The manifest counts 486 training prompts with conflicting action labels,
55 in validation, and 93 in test. These labels are retained, not resolved by
inventing an optimal action. Repeated traces for the same prompt/action are
deduplicated consistently with the first experiment. Exact overlapping prompt
hashes are removed across splits with priority test, validation, then train.
These splits do not establish unseen layouts or novel game dynamics.

## Real Qwen3 tokenizer preflight

Tokenizer: `Qwen/Qwen3-8B` at
`b968826d9c46dd6066d109eabc6255188de91218`, with thinking enabled.
The full train and validation splits were encoded with response-only labels;
all examples fit the 32,768-token configuration without truncation.

| Split | Minimum tokens | 95th percentile | Maximum tokens | Supervised tokens per epoch |
|---|---:|---:|---:|---:|
| Train | 395 | 15,718 | 31,247 | 94,714,452 |
| Validation | 409 | 15,817 | 30,822 | 15,349,994 |

For the first, last, and longest trace by character count in each split, the
encoded training sequence also matched Qwen's native full assistant-message
template through EOS. The masked prompt ends at the assistant header, and
loss includes `<think>`, the recorded trace body, `</think>`, the recorded
action, and `<|im_end|>`. Padding masks are verified in unit tests. Test examples
were not generated against held-out labels or scored by a model.

Prepared data, rejection records, the manifest with original file hashes, and
`tokenization-report.json` are under ignored `data/sft-cot/`. The action-only
`data/sft/` is unchanged. Source/config/data matching and smoke/resume paths
are covered by local tests, including training and reloading a tiny randomly
initialized Qwen3 LoRA on CPU. No full 8B training or GPU memory benchmark was
run locally; the CHTC smoke must verify the long-sequence memory requirement.
