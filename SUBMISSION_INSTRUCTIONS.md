# Anonymous Submission Instructions

## Creating Anonymous Archive

To create an anonymous submission archive that excludes personal information:

```bash
# Create anonymous archive excluding files with personal information
tar -czf markovian_training_submission.tar.gz \
    --exclude-from=.submission_exclude \
    --exclude='.git' \
    --exclude='*.log' \
    --exclude='*.fls' \
    --exclude='*.fdb_latexmk' \
    --exclude='__pycache__' \
    --exclude='venv' \
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
    -x '*.pyc'
```

## Files Excluded for Anonymity

- **LaTeX build files** (`*.log`, `*.fls`, `*.fdb_latexmk`): Contain local file paths with username
- **Git history** (`.git/`): Contains commit author information
- **Python cache files**: Not needed for submission
- **Virtual environment files**: Not needed for submission

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