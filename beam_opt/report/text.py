"""
beam_opt.report.text
====================
Textual optimisation report.
"""

from __future__ import annotations

from ..optimization.problem import TON_TO_G, OptimizationProblem


def print_report(result, problem: OptimizationProblem) -> None:
    """
    Print a full optimisation report.

    Parameters
    ----------
    result : scipy.optimize.OptimizeResult
        Result of the driver.
    problem : OptimizationProblem
        The solved problem; the optimum is re-evaluated (from cache if possible).
    """
    SEP = "=" * 80
    SEP2 = "-" * 80

    sigma_adm = problem.sigma_adm

    print()
    print(SEP)
    print(" " * 20 + "BEAM-OPT FRAMEWORK – OPTIMISATION REPORT")
    print(SEP)

    # ── Convergence ──
    print("\n[CONVERGENCE]")
    status = "✓ CONVERGED" if result.success else "✗ NOT CONVERGED"
    print(f"  {status}")
    if not result.success:
        print(f"  Reason : {result.message}")
    print(f"  FEM evaluations : {result.nfev}")

    # ── Reconstruct solution ──
    x_opt = result.x
    ev = problem.evaluate(x_opt)

    # ── Objective ──
    print("\n[OBJECTIVE]")
    print(f"  {problem.objective_fn.describe(problem, ev)}")

    # ── Mass ──
    print("\n[STRUCTURAL MASS]")
    m_opt = ev.mass * TON_TO_G
    m_fixed = problem.mass_fixed * TON_TO_G
    m_total = m_opt + m_fixed
    print(f"  Optimisable elements : {m_opt:8.3f} g")
    print(f"  Fixed elements       : {m_fixed:8.3f} g")
    print(f"  {'─' * 35}")
    print(f"  TOTAL                : {m_total:8.3f} g")
    if problem.mass_max is not None:
        ok = "✓ OK" if m_opt <= problem.mass_max * (1 + 1e-4) else "✗ VIOLATED"
        print(f"  Budget (optimisable) : {problem.mass_max:8.3f} g   {ok}")

    # ── Stress ──
    sigma_vals = ev.sigma
    sigma_max = float(sigma_vals.max())
    mat = problem.model.material
    print("\n[STRESS VERIFICATION]")
    print(f"  σ_max    : {sigma_max:7.2f} MPa")
    if problem.stress_limited:
        print(
            f"  σ_adm    : {sigma_adm:7.2f} MPa  "
            f"(σ_y={mat.sigma_y} / FS={mat.sigma_y / sigma_adm:g})"
        )
        print(f"  Utilisation : {sigma_max / sigma_adm * 100:6.1f} %")
        print(f"  {'✓ OK' if sigma_max <= sigma_adm * (1 + 1e-4) else '✗ VIOLATED'}")
    else:
        print(
            f"  no stress constraint (σ_y = {mat.sigma_y} MPa, σ_max/σ_y = {sigma_max / mat.sigma_y:.0%})"
        )

    # ── Fatigue ──
    if problem.fatigue is not None:
        lives = problem.lives(ev)
        k = int(lives.argmin())
        fat = problem.fatigue
        if fat.spectrum is None:
            load = f"R = {fat.load_ratio:g}"
        else:
            load = f"{fat.spectrum.cycles:g} cycles per {fat.spectrum.name}, Miner's rule"
        print("\n[FATIGUE LIFE]")
        print(f"  N_min    : {lives[k]:.4g} {fat.unit}  (element {k + 1}, {load})")
        if problem.life_min is not None:
            ok = "✓ OK" if lives[k] >= problem.life_min * (1 - 1e-3) else "✗ VIOLATED"
            print(f"  Required : {problem.life_min:.4g} {fat.unit}   {ok}")

    # ── Reaction ──
    f_react = float(ev.reaction)
    if problem.reaction_nodes:
        print("\n[REACTION FORCE VERIFICATION]")
        print(f"  F_react  : {f_react:7.4f} N")
    if problem.f_react_max is not None:
        print(f"  F_max    : {problem.f_react_max:7.4f} N")
        print(f"  Utilisation : {f_react / problem.f_react_max * 100:6.1f} %")
        print(f"  {'✓ OK' if f_react <= problem.f_react_max * (1 + 1e-4) else '✗ VIOLATED'}")

    if problem.f_react_min is not None:
        print(f"  F_min    : {problem.f_react_min:7.4f} N")
        print(f"  {'✓ OK' if f_react >= problem.f_react_min * (1 - 1e-4) else '✗ VIOLATED'}")

    # ── Displacements ──
    if problem.displacement_limits:
        print("\n[DISPLACEMENT VERIFICATION]")
        for (node, dof, lim), value in zip(problem.displacement_limits, ev.displacements):
            ok = "✓" if abs(value) <= lim * (1 + 1e-4) else "✗"
            print(f"  {dof} at {node!r:<12}: {value:10.4f}   limit {lim:g}   {ok}")

    # ── Element table ──
    law = problem.law_from_x(x_opt)
    beam = problem.model.beam
    s_mids = beam.mesh.s_elem
    params = law.eval(s_mids)
    area = beam.sections_at(law, s_mids).area
    names = list(params)

    print()
    print(SEP2)
    print("  ELEMENT-WISE SECTION DISTRIBUTION AND STRESS")
    print(SEP2)
    head = "".join(f"  {f'{n} [mm]':>7}" for n in names)
    print(
        f"  {'Elem':>4}  {'s [mm]':>8}{head}  {'A [mm²]':>8}  {'σ_VM [MPa]':>11}  "
        f"{'util. [%]':>9}  Status"
    )
    print("  " + "-" * 72)

    for i, (s, sv) in enumerate(zip(s_mids, sigma_vals)):
        ut = sv / sigma_adm * 100.0
        ok = "✓" if sv <= sigma_adm else "✗"
        vals = "".join(f"  {params[n][i]:7.3f}" for n in names)
        print(f"  {i + 1:>4}  {s:8.1f}{vals}  {area[i]:8.3f}  {sv:11.2f}  {ut:9.1f}  {ok}")

    print("  " + "-" * 72)

    # ── Summary statistics ──
    print()
    print("  [SECTION STATISTICS]")
    for n in names:
        v = params[n]
        print(
            f"    {n:<2} — min: {v.min():.3f}  max: {v.max():.3f}  "
            f"mean: {v.mean():.3f}  idx_max: {v.argmax() + 1}"
        )
    print(
        f"    σ  — min: {sigma_vals.min():.2f}  max: {sigma_vals.max():.2f}  "
        f"idx_max: {sigma_vals.argmax() + 1}"
    )
    idx_c = int(sigma_vals.argmax())
    dims = ", ".join(f"{n}={params[n][idx_c]:.3f} mm" for n in names)
    print(
        f"\n    Critical element: #{idx_c + 1}  "
        f"(s={s_mids[idx_c]:.1f} mm, {dims}, σ={sigma_vals[idx_c]:.2f} MPa)"
    )

    print()
    print(SEP)
