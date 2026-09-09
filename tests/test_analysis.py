import numpy as np
import pandas as pd

from neural_factors.analysis import fit_canonical_basis, apply_canonical_basis


def _frame(f, b):
    data = {"date": pd.date_range("2000-01-31", periods=len(f), freq="ME")}
    data["portfolio_return"] = np.sum(f * b, axis=1)
    for j in range(f.shape[1]):
        data[f"factor_{j:02d}"] = f[:, j]
        data[f"allocator_{j:02d}"] = b[:, j]
    return pd.DataFrame(data)


def test_canonical_reconstruction_and_basis_invariance():
    rng = np.random.default_rng(4)
    f = rng.normal(size=(240, 5)) * 0.03
    b = rng.normal(size=(240, 5))
    frame = _frame(f, b)
    basis = fit_canonical_basis(frame)
    canonical = apply_canonical_basis(frame, basis)
    assert np.max(np.abs(canonical["reconstruction_error"])) < 1e-6

    a = rng.normal(size=(5, 5)) + 2.0 * np.eye(5)
    f2 = f @ a.T
    b2 = b @ np.linalg.inv(a)
    basis2 = fit_canonical_basis(_frame(f2, b2))
    p1 = basis.economic_eigenvalues / basis.economic_eigenvalues.sum()
    p2 = basis2.economic_eigenvalues / basis2.economic_eigenvalues.sum()
    assert np.allclose(p1, p2, rtol=2e-3, atol=2e-3)
