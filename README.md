# psdl1 — DL1 product similarity

DL1 aims to find products with similar original shapes and electrode connectivity even when unit sizes differ. Real layouts and correspondence labels are available only in the closed network; the current evidence uses synthetic shapes.

G3 repository naming remains pending. G5 approval of whole-unit isotropic resize plus pad, electrode × actual-layer set aggregation, the real minimum linewidth and grid, and the source CNT/VIA endpoint format remains pending. The versioned manifest and three-channel tensor path here are **draft candidates**, not a confirmed user design. No real search performance or connectivity preservation has been established.

The input consists of original shape masks and source-supplied connection metadata. Hotspot, CA, and fab fields are rejected. Products share a physical pixel pitch (about 0.05 µm in the user data); their pixel counts differ, and all layers within each product share one frame. This contract expects a single whole unit; it does not infer or crop a unit from a 2×2 simulation domain.

See [docs/data_contract.md](docs/data_contract.md) for the manifest example, observation states, path rules, endpoint bindings, and validation evidence. Unversioned manifests must be explicitly converted with known lineage, frame, valid coverage, and observations; the loader does not guess them.

Validate and save the per-product grid evidence without training:

```bash
.venv/bin/python -m psdl1.data data/manifest.json --grid 256 --report grid-report.json
# Equivalent entry point after installation:
psdl1-validate data/manifest.json --grid 256 --report grid-report.json
```

The report records thin-line warnings, thresholded component changes, coverage, and G5 pending status. Its axis-run estimate is a diagnostic, not the true physical minimum linewidth. A warning requires review or a different candidate grid. Even a report without warnings does not certify connectivity. The existing training entry point does not automatically enforce this validator; consuming spatial valid masks, endpoint relations, grid reports, and safe augmentation decisions belongs to T9.

`prepare_product` preserves spatial validity and source endpoint relations alongside the candidate tensors. `product_tensors` is the existing three-channel compatibility interface (element / CNT-VIA union / other elements), and the existing model consumes only that interface. Each rest channel excludes its own element before composing the other masks, including XY overlaps across layers. A CNT image without source endpoint metadata leaves connectivity **unavailable**, never CNN-verified. Role and actual layer remain bound to each element; layers are never centred independently.

The existing model, training, loss, and search code is pending T9 review. Their set encoder, unlabeled negative policy, weights loader, and training augmentations are not approved by this input-contract task. `augment_with_check` supplies evidence for a single common transform and never approves a view as a connectivity-preserving positive label.

The checked runtime is Python 3.12. `requirements.lock` records exact installed dependency versions, including the optional timm stack; it is a runtime snapshot for this macOS environment, not a portable wheel/build lock. Obtain approved distributions and the build backend through the closed-network package process; no dependency updates or downloads were performed in this task.

```bash
# From an approved local wheel directory, after preparing a Python environment:
uv pip install --no-index --find-links /approved/wheels -r requirements.lock
uv pip install --no-index --find-links /approved/wheels -e .

.venv/bin/python -m pytest -p no:warnings tests/test_data.py tests/test_input_contract.py
.venv/bin/python -m pytest -p no:warnings
```

The full default suite skips the existing 40-epoch training/search test; its T9 coverage requires explicit `--run-training`. Test fixtures reject socket network connections. No training or pretrained weights were used for T8. Before importing any pretrained artifact, record its exact file hash, source, and applicable license; no artifact or license has been accepted here.
