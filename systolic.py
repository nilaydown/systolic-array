"""Systolic array matrix multiplier."""

from typing import List, Optional


class Cell:
    def __init__(self):
        self.acc: float = 0.0
        self.a_in: Optional[float] = None
        self.b_in: Optional[float] = None
        self.a_out: Optional[float] = None
        self.b_out: Optional[float] = None

    def tick(self) -> None:
        if self.a_in is not None and self.b_in is not None:
            self.acc += self.a_in * self.b_in
        self.a_out = self.a_in
        self.b_out = self.b_in
        self.a_in = None
        self.b_in = None


class SystolicArray:
    def __init__(self, n: int):
        self.n = n
        self.cells: List[List[Cell]] = [[Cell() for _ in range(n)] for _ in range(n)]
        self.tick_count = 0

    def cell(self, i: int, j: int) -> Cell:
        return self.cells[i][j]

    def feed(self, a_streams: List[List[Optional[float]]],
             b_streams: List[List[Optional[float]]]) -> None:
        t = self.tick_count
        for i in range(self.n):
            if t < len(a_streams[i]):
                self.cells[i][0].a_in = a_streams[i][t]
        for j in range(self.n):
            if t < len(b_streams[j]):
                self.cells[0][j].b_in = b_streams[j][t]

    def tick(self) -> None:
        # Phase 1: every cell computes and stages its outputs
        for row in self.cells:
            for c in row:
                c.tick()

        # Phase 2: propagate staged outputs into neighbors' inputs
        for i in range(self.n):
            for j in range(self.n):
                c = self.cells[i][j]
                if j + 1 < self.n:
                    self.cells[i][j + 1].a_in = c.a_out
                if i + 1 < self.n:
                    self.cells[i + 1][j].b_in = c.b_out

        self.tick_count += 1

    def result(self) -> List[List[float]]:
        return [[self.cells[i][j].acc for j in range(self.n)] for i in range(self.n)]


def stagger_inputs(M: List[List[float]], axis: str) -> List[List[Optional[float]]]:
    n = len(M)
    total = 3 * n - 2
    streams: List[List[Optional[float]]] = []
    if axis == "rows":
        for i in range(n):
            row = list(M[i])
            stream = [None] * i + row + [None] * (total - i - n)
            streams.append(stream)
    elif axis == "cols":
        for j in range(n):
            col = [M[i][j] for i in range(n)]
            stream = [None] * j + col + [None] * (total - j - n)
            streams.append(stream)
    else:
        raise ValueError(f"axis must be 'rows' or 'cols', got {axis!r}")
    return streams


def matmul_systolic(A: List[List[float]], B: List[List[float]],
                    verbose: bool = False) -> List[List[float]]:
    n = len(A)
    assert len(A[0]) == n and len(B) == n and len(B[0]) == n, "expect square matrices"

    arr = SystolicArray(n)
    a_streams = stagger_inputs(A, axis="rows")
    b_streams = stagger_inputs(B, axis="cols")
    total_ticks = 3 * n - 2

    for _ in range(total_ticks):
        arr.feed(a_streams, b_streams)
        arr.tick()
        if verbose:
            print_grid(arr)

    return arr.result()


def print_grid(arr: SystolicArray) -> None:
    print(f"\n=== Clock tick {arr.tick_count} ===")
    for i in range(arr.n):
        row = []
        for j in range(arr.n):
            c = arr.cells[i][j]
            row.append(f"{c.acc:>6.1f}")
        print("  [" + ", ".join(row) + "]")
