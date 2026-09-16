# reports/RUNLOG.md: per-stage execution log
# Updated automatically by each phase script.
# Format: | Phase | Script | Device | Peak VRAM (MB) | Wall time | Provider calls |

| Phase | Script | Device | Peak VRAM (MB) | Wall time (s) | Provider calls |
|---|---|---|---|---|---|
| 0 | smoke_gateway.py | CPU | N/A | 3.2 | 3 |
| 1 | scripts/01_eda.py | CPU | N/A | 14.8 | 0 |
| 2 | scripts/02_build_threads.py | CPU | N/A | 28.5 | 0 |
| 3 | 03_discover_intents.py | cuda | 1133 | 81.8 | 1 |
| 4 | 04_label_intents.py | cuda | 1257 | 651.4 | 15 |
| 5 | scripts/05_build_golden_set.py | CPU | N/A | 11.8 | 0 |

## Notes
- VRAM measured with `torch.cuda.max_memory_allocated()` at end of each GPU stage.
- Provider calls = new calls only (cache hits not counted).
- Wall time = end-to-end script runtime including I/O.
