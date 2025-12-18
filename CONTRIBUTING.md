# Contributing

Thanks for your interest in improving this baseline.

## Development workflow

1. Create a new branch:
   ```bash
   git checkout -b feature/my-change
   ```
2. Make changes and run notebook checks (at minimum, run all cells end-to-end).
3. Open a Pull Request with:
   - a brief description
   - what changed
   - how it was tested

## Data safety

Never commit:
- raw EBM production data
- taxpayer identifiers
- credentials (.env, keys)

Use synthetic or anonymized samples if examples are needed.
