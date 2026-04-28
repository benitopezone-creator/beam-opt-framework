# beam-opt-framework

A general-purpose **dimensional optimization framework** for complex beam structures
modelled with ANSYS MAPDL (BEAM188 / SHELL181 elements).

The framework is intentionally application-agnostic: any assembly of spline-connected
beams with fixed structural sub-components can be optimized by providing a configuration
module and, optionally, a custom stress-field evaluator.

---

## Repository layout

```
beam_opt_framework/
│
├── beam_opt/                   ← installable package
│   ├── __init__.py
│   ├── config.py               ← material constants, solver knobs (no hard-coded geometry)
│   ├── mapdl_manager.py        ← MAPDL singleton with guaranteed exit()
│   ├── section_law.py          ← CubicSpline section law (b(s), h(s))
│   ├── fixed_elements.py       ← FixedBeam / FixedShell base classes
│   ├── model.py                ← FEM model builder (geometry-agnostic)
│   ├── stress/
│   │   ├── __init__.py
│   │   ├── base.py             ← AbstractStressEvaluator interface
│   │   └── rect_beam.py        ← BEAM188 rectangular section evaluator (VM)
│   ├── analysis.py             ← FEM solve + post-processing orchestrator
│   ├── optimization.py         ← COBYLA optimizer wrapper
│   └── report.py               ← textual report + matplotlib plots
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/<your-org>/beam-opt-framework.git
cd beam-opt-framework

# 2. Install in editable mode (Python ≥ 3.10)
pip install -e .

python run_optimization.py --config v05 --n-ctrl 8 --nproc 4
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--config` | geometry config module |
| `--n-ctrl` | `8` | number of spline control points |
| `--nproc` | `4` | MAPDL parallel processes |
| `--run-location` | `./mapdl_tmp` | working directory for MAPDL files |
| `--save-plots` | *(none)* | path prefix for PNG output |
| `--no-plots` | *(flag)* | suppress matplotlib windows |

---

## Extending to a new cross-section shape

1. Create `beam_opt/stress/my_section.py` subclassing `AbstractStressEvaluator`.
2. Implement `__call__(mapdl, model, law) -> np.ndarray` (one VM value per optimisable element).
3. Pass an instance to `BeamOptimizer(…, stress_evaluator=MyEvaluator())`.

No other files need to be modified.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ansys-mapdl-core` | MAPDL Python client |
| `scipy` | COBYLA optimizer |
| `numpy` | numerical operations |
| `matplotlib` | result plots (optional) |

---

## License

MIT
