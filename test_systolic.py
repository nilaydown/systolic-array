"""
Tests for the systolic array. Run with: python3 test_systolic.py
Each test compares your systolic output against the reference (NumPy or manual).
"""

import sys
from systolic import matmul_systolic, stagger_inputs


def approx_equal(C, expected, eps=1e-6):
    if len(C) != len(expected):
        return False
    for i in range(len(C)):
        if len(C[i]) != len(expected[i]):
            return False
        for j in range(len(C[i])):
            if abs(C[i][j] - expected[i][j]) > eps:
                return False
    return True


def test_2x2():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]  # by hand
    got = matmul_systolic(A, B)
    assert approx_equal(got, expected), f"2x2 failed:\n  got      {got}\n  expected {expected}"
    print("PASS: 2x2 matmul")


def test_3x3_identity():
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    got = matmul_systolic(A, I)
    assert approx_equal(got, A), f"identity failed:\n  got {got}"
    print("PASS: A @ I == A")


def test_3x3_random():
    try:
        import numpy as np
    except ImportError:
        print("SKIP: 3x3 random (numpy not installed)")
        return
    np.random.seed(42)
    A = np.random.randint(0, 10, size=(3, 3)).astype(float)
    B = np.random.randint(0, 10, size=(3, 3)).astype(float)
    expected = (A @ B).tolist()
    got = matmul_systolic(A.tolist(), B.tolist())
    assert approx_equal(got, expected), (
        f"3x3 random failed:\n  A={A.tolist()}\n  B={B.tolist()}\n"
        f"  got      {got}\n  expected {expected}"
    )
    print("PASS: 3x3 random matmul matches numpy")


def test_stagger_rows():
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    streams = stagger_inputs(A, axis="rows")
    assert len(streams) == 3, "expected 3 row streams"
    assert streams[0][:3] == [1, 2, 3], f"row 0 should start with row values, got {streams[0]}"
    assert streams[1][0] is None and streams[1][1:4] == [4, 5, 6], \
        f"row 1 should have 1 leading None then row, got {streams[1]}"
    assert streams[2][0] is None and streams[2][1] is None and streams[2][2:5] == [7, 8, 9], \
        f"row 2 should have 2 leading Nones then row, got {streams[2]}"
    print("PASS: stagger_inputs rows")


def test_stagger_cols():
    B = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    streams = stagger_inputs(B, axis="cols")
    assert len(streams) == 3, "expected 3 col streams"
    # col 0 = [1, 4, 7]
    assert streams[0][:3] == [1, 4, 7], f"col 0 wrong: {streams[0]}"
    # col 1 = [2, 5, 8] with 1 leading None
    assert streams[1][0] is None and streams[1][1:4] == [2, 5, 8], f"col 1 wrong: {streams[1]}"
    print("PASS: stagger_inputs cols")


def main():
    tests = [test_stagger_rows, test_stagger_cols, test_2x2, test_3x3_identity, test_3x3_random]
    failed = 0
    for t in tests:
        try:
            t()
        except NotImplementedError as e:
            print(f"TODO: {t.__name__} — {e}")
            failed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passing")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
