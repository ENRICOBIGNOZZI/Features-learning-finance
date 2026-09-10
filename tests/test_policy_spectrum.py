import numpy as np

from neural_factors.policy_spectrum import spectrum_from_grams


def test_policy_spectrum_is_basis_invariant():
    rng = np.random.default_rng(123)
    phi = rng.normal(size=(5000, 5))
    b = rng.normal(size=(800, 5))
    g_phi = phi.T @ phi / len(phi)
    g_b = b.T @ b / len(b)
    values = spectrum_from_grams(g_phi, g_b)

    a = rng.normal(size=(5, 5)) + 3.0 * np.eye(5)
    phi_rot = phi @ a.T
    b_rot = b @ np.linalg.inv(a)
    g_phi_rot = phi_rot.T @ phi_rot / len(phi_rot)
    g_b_rot = b_rot.T @ b_rot / len(b_rot)
    values_rot = spectrum_from_grams(g_phi_rot, g_b_rot)
    assert np.allclose(values, values_rot, rtol=1e-8, atol=1e-8)
