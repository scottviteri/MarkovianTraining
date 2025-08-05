# Anonymous Submission Instructions

## Creating Anonymous Archive

To create an anonymous submission archive that excludes personal information and large unnecessary files:

```bash
# Create anonymous archive excluding files with personal information and large training artifacts
tar -czf markovian_training_submission.tar.gz \
    --exclude='.git' \
    --exclude='*.log' \
    --exclude='*.fls' \
    --exclude='*.fdb_latexmk' \
    --exclude='*.aux' \
    --exclude='*.out' \
    --exclude='*.synctex.gz' \
    --exclude='__pycache__' \
    --exclude='venv' \
    --exclude='results/wiki_continuation' \
    --exclude='results/gsm8k' \
    --exclude='results/arithmetic' \
    --exclude='results/mmlu' \
    --exclude='results/wiki_compression' \
    .
```

Or using zip:

```bash
# Create anonymous archive using zip
zip -r markovian_training_submission.zip . \
    -x '.git/*' \
    -x '*.log' \
    -x '*.fls' \
    -x '*.fdb_latexmk' \
    -x '*.aux' \
    -x '*.out' \
    -x '*.synctex.gz' \
    -x '__pycache__/*' \
    -x 'venv/*' \
    -x '*.pyc' \
    -x 'results/wiki_continuation/*' \
    -x 'results/gsm8k/*' \
    -x 'results/arithmetic/*' \
    -x 'results/mmlu/*' \
    -x 'results/wiki_compression/*'
```

## Files Excluded for Size and Anonymity

### Excluded for Anonymity:
- **LaTeX build files** (`*.log`, `*.fls`, `*.fdb_latexmk`): Contain local file paths with username
- **Git history** (`.git/`): Contains commit author information

### Excluded for Size (4.5GB → ~100MB):
- **Training checkpoints** (`results/wiki_continuation/`, `results/gsm8k/`, etc.): 
  - Adapter files are ~73MB each × 5 checkpoints × many runs = 4.3GB
  - Not needed for paper submission or reproducibility
- **Large training logs**: Full logs from complete training runs
- **Python cache files**: Not needed for submission
- **Virtual environment files**: Not needed for submission

### Kept for Reproducibility:
- **`results/samples/`**: Example training logs for visualization (88MB)
- **`results/evaluations/`**: Evaluation results (64KB)  
- **`results/figures/`**: Generated figures (4KB)
- **All source code**: Complete implementation
- **Paper and figures**: Complete LaTeX submission

## Anonymized Content

The following have been anonymized in the repository:
- `setup.py`: Author field set to "Anonymous"
- `LatexFolder/aaai2026.bib`: Author field for personal reference set to "Anonymous"
- All source code files contain no personal identifiers

## Verification

After creating the archive, you can verify anonymity by:

```bash
# Extract and search for personal information
tar -xzf markovian_training_submission.tar.gz -C /tmp/check
grep -r "scottviteri\|Scott Viteri" /tmp/check/ || echo "No personal information found"
```

The extracted archive should contain no personal identifiers.