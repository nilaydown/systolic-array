"""
Watch a 3x3 systolic matmul tick by tick. Run: python3 run.py
"""

from systolic import matmul_systolic


def main():
    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]

    B = [[1, 0, 2],
         [0, 1, 1],
         [3, 1, 0]]

    print(f"A = {A}")
    print(f"B = {B}")
    print(f"Expected A @ B = [[10, 5, 4], [22, 11, 13], [34, 17, 22]]")
    print("Running systolic matmul with per-tick grid dump...")

    C = matmul_systolic(A, B, verbose=True)

    print("\n=== FINAL ===")
    for row in C:
        print(" ", row)


if __name__ == "__main__":
    main()
