"""Scorecard formatting and pass/fail thresholds."""

from __future__ import annotations


# (threshold_value, direction)  direction: "gt" = must be > threshold, "lt" = must be <
THRESHOLDS: dict[str, tuple[float, str]] = {
    # Efficiency
    "tokens_per_second": (50_000, "gt"),
    "learning_overhead_ratio": (2.0, "lt"),
    # Continuous Learning
    "domain_a_accuracy_before": (0.9, "gt"),
    "domain_b_accuracy": (0.9, "gt"),
    "domain_a_retention_after": (0.9, "gt"),
    "forgetting_rate": (0.05, "lt"),
    # Hallucination Detection
    "valid_concentration": (0.6, "gt"),
    "invalid_concentration": (0.3, "lt"),
    "detection_gap": (0.3, "gt"),
    # Memory
    "storage_ratio": (0.5, "lt"),
    "retrieval_precision_at_3": (0.5, "gt"),
    # E2E Hallucination (through process() pipeline)
    # Concentration measures depth agreement, not accuracy — absolute values
    # are high for both valid and invalid. The GAP is the key metric.
    "e2e_valid_concentration": (0.4, "gt"),
    "e2e_detection_gap": (0.05, "gt"),
}


def _check(value: float, threshold: float, direction: str) -> bool:
    if direction == "gt":
        return value > threshold
    return value < threshold


def _fmt(value: float, name: str) -> str:
    if "tokens_per_second" in name:
        return f"{value:>12,.0f}"
    if "ratio" in name or "rate" in name:
        return f"{value:>12.2f}"
    return f"{value:>12.3f}"


def print_scorecard(
    results: dict[str, float],
    thresholds: dict[str, tuple[float, str]] | None = None,
    title: str = "GAIA v2 BENCHMARK SCORECARD",
) -> bool:
    """Print formatted scorecard. Returns True if all pass."""
    if thresholds is None:
        thresholds = THRESHOLDS

    print(f"\n{title}")
    print("=" * len(title))

    # Group by section
    sections = {
        "EFFICIENCY": ["tokens_per_second", "learning_overhead_ratio"],
        "CONTINUOUS LEARNING": [
            "domain_a_accuracy_before",
            "domain_b_accuracy",
            "domain_a_retention_after",
            "forgetting_rate",
        ],
        "HALLUCINATION DETECTION": [
            "valid_concentration",
            "invalid_concentration",
            "detection_gap",
        ],
        "MEMORY": ["storage_ratio", "retrieval_precision_at_3"],
        "E2E HALLUCINATION": [
            "e2e_valid_concentration",
            "e2e_invalid_concentration",  # Reported but no threshold (informational)
            "e2e_detection_gap",
        ],
    }

    all_pass = True
    n_pass = 0
    n_total = 0

    for section, keys in sections.items():
        section_keys = [k for k in keys if k in results]
        if not section_keys:
            continue

        print(f"\n{section}")
        for key in section_keys:
            value = results[key]
            if key in thresholds:
                threshold, direction = thresholds[key]
                passed = _check(value, threshold, direction)
                sym = ">" if direction == "gt" else "<"
                status = "PASS" if passed else "FAIL"
                print(
                    f"  {key + ':':34s}{_fmt(value, key)}  {status}  "
                    f"(threshold: {sym}{threshold:g})"
                )
                if not passed:
                    all_pass = False
                n_total += 1
                if passed:
                    n_pass += 1
            else:
                print(f"  {key + ':':34s}{_fmt(value, key)}")

    # Also print any keys not in sections
    extra = [k for k in results if not any(k in v for v in sections.values())]
    if extra:
        print("\nADDITIONAL")
        for key in extra:
            value = results[key]
            print(f"  {key + ':':34s}{_fmt(value, key)}")

    print(f"\nRESULT: {n_pass}/{n_total} PASS")
    return all_pass
