# GAIA Proof of Concepts - Journal Schema

> **Standard format for documenting POC research journeys**

---

## File Naming Convention

```
YYYY-MM-DD_descriptive_slug.md
```

Examples:
- `2024-12-16_initial_encoding_tests.md`
- `2024-12-17_frequency_mapping_discovery.md`

---

## Required Sections

Every journal entry MUST include:

### 1. Summary (Top)
Brief 2-3 sentence summary of what was attempted and key outcomes.

### 2. Timeline
Chronological log of activities with status markers:

```markdown
### HH:MM - Activity Type

Description of what was done.

**Status:** ✅ Confirmed | ❌ Failed | 🔄 In Progress | 💡 Insight
```

Activity Types:
- **Setup**: Environment, dependencies, configuration
- **Experiment**: Running actual tests
- **Analysis**: Interpreting results
- **Discovery**: Unexpected findings
- **Bug Fix**: Resolving issues
- **Planning**: Next steps, pivots

### 3. Key Findings
Bullet list of most important discoveries.

### 4. Metrics Collected
Quantitative data gathered (tables preferred).

### 5. Challenges Encountered
What went wrong or was harder than expected.

### 6. Next Steps
Concrete actions for follow-up.

---

## Template

```markdown
# Journal: [Descriptive Title]

**Date:** YYYY-MM-DD  
**POC:** POC-XXX  
**Author:** [Name]  
**Status:** 🔄 In Progress | ✅ Complete | ❌ Blocked

---

## Summary

[2-3 sentences summarizing the session]

---

## Timeline

### HH:MM - Setup

[Description]

**Status:** ✅ | ❌ | 🔄 | 💡

### HH:MM - Experiment

[Description]

**Status:** ✅ | ❌ | 🔄 | 💡

---

## Key Findings

- Finding 1
- Finding 2
- Finding 3

---

## Metrics Collected

| Metric | Value | Notes |
|--------|-------|-------|
| Metric 1 | X | ... |
| Metric 2 | Y | ... |

---

## Challenges Encountered

1. Challenge 1
2. Challenge 2

---

## Next Steps

- [ ] Action 1
- [ ] Action 2
- [ ] Action 3

---

## Raw Notes

[Optional: Unstructured notes, observations, questions]
```

---

## Journal Rules

1. **One file per day** (exceptions for major discoveries)
2. **Timestamp all activities** for reproducibility
3. **Include failures** - they're as valuable as successes
4. **Link to code/data** when referencing specific files
5. **Be honest** about uncertainty and unknowns

---

## Cross-Linking

When referencing other documents:
- Other journals: `[2024-12-16 entry](./journals/2024-12-16_initial_tests.md)`
- Specs: `[Phase 4 Spec](../../.spec/phase4-transformers.spec.md)`
- Code: `[encoder.py](./scripts/exp_01_encoder.py)`
- Results: `[results](./results/exp_01_20241216.json)`
