# Release verification

Checked on 2026-10-13 using the original Linux / Python 3.10 / PyTorch 2.0.1 experiment environment, in an independent code-check directory.

Passed:

- Four unit tests: seven-threshold result parsing, duplicate-fold detection, WEAR assembly, and XRF V2 H5-to-TAL preparation.
- Resolution of all 81 primary configurations, retaining their original model, optimizer and loader settings.
- TinyHAR encoding for 12-channel WEAR and 30-channel XRF V2 windows.
- All five fusion implementations with finite outputs, correctly zeroed padded positions, and finite gradients.
- Fully masked LTCA padding without NaN outputs.
- End-to-end loss and backward for LTCA on WEAR RGB, WEAR Flow and XRF V2, each with ActionFormer, TemporalMaxer and TriDet.
- 81 manuscript table vectors matched to complete archived experiment logs.
- Unchanged SHA-256 hashes for the three model files and TinyHAR source recorded in SOURCE_PROVENANCE.md.

Synthetic backbone checks shorten padded sequence length to 256 for the check only. They run on CPU, do not enter an epoch/LOSO training loop, and do not save checkpoints. This verification establishes that the code and configuration paths operate together; it is not a repeat measurement of paper accuracy.

Commands:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tests/smoke_models.py
python scripts/run_suite.py --methods fusion --gpus 0 1 2 --seeds 1 --dry-run
```
