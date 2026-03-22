"""Real data benchmarks — WikiText-2 through GAIA v2.

Optional: requires transformers + datasets packages.
Port of POC-022's approach into v2's TransitionCounter.
"""

from __future__ import annotations

import time

from gaia.modules.language import TransitionCounter


def _has_deps() -> bool:
    try:
        import transformers  # noqa: F401
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


def bench_wikitext2(
    max_sequences: int = 2000,
    seq_len: int = 64,
    max_context_len: int = 3,
    top_k: int = 10,
) -> dict[str, float]:
    """WikiText-2 next-token prediction benchmark.

    Loads WikiText-2 train split, tokenizes with GPT-2,
    learns transitions, measures top-k hit rate.

    v1 baseline (POC-022): 65% hit rate at 2K sequences.
    """
    from transformers import AutoTokenizer  # type: ignore[import-not-found]
    from datasets import load_dataset  # type: ignore[import-not-found]

    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Tokenize into sequences of seq_len
    print(f"Tokenizing (max {max_sequences} sequences of length {seq_len})...")
    sequences: list[list[int]] = []
    buffer: list[int] = []

    for item in dataset:
        text = item["text"]
        if not text or not text.strip():
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(token_ids)

        while len(buffer) >= seq_len:
            sequences.append(buffer[:seq_len])
            buffer = buffer[seq_len:]
            if len(sequences) >= max_sequences:
                break
        if len(sequences) >= max_sequences:
            break

    print(f"Got {len(sequences)} sequences, {len(sequences) * seq_len:,} tokens total")

    # Learn transitions
    counter = TransitionCounter(max_context_len=max_context_len)

    t0 = time.perf_counter()
    for seq in sequences:
        counter.learn_from_sequence(seq)
    learn_time = time.perf_counter() - t0

    total_tokens = len(sequences) * seq_len
    tokens_per_second = total_tokens / max(learn_time, 1e-9)

    # Measure top-k hit rate
    # For each position in each sequence, predict next token and check if
    # correct answer is in top-k predictions
    print(f"Measuring hit rate (top-{top_k})...")
    hits = 0
    total = 0

    for seq in sequences:
        for pos in range(max_context_len, len(seq) - 1):
            for ctx_len in range(1, min(max_context_len + 1, pos + 1)):
                context = tuple(seq[pos - ctx_len + 1 : pos + 1])
                expected = seq[pos + 1]

                pred_ids, probs = counter.predict(context, top_k=top_k)
                if len(pred_ids) > 0 and expected in pred_ids.tolist():
                    hits += 1
                    break  # Count as hit if any context length predicts correctly
            total += 1

    hit_rate = hits / max(total, 1)

    # Vocab coverage
    unique_tokens = set()
    for seq in sequences:
        unique_tokens.update(seq)
    vocab_coverage = len(unique_tokens)

    print(f"Hit rate: {hit_rate:.1%}, Tokens/sec: {tokens_per_second:,.0f}")

    return {
        "wikitext2_hit_rate": hit_rate,
        "wikitext2_tokens_per_second": tokens_per_second,
        "wikitext2_vocab_coverage": float(vocab_coverage),
        "wikitext2_sequences": float(len(sequences)),
    }
