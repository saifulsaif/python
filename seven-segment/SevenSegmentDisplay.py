class SevenSegmentDisplay:
    def __init__(self):
        # Seven segment digit patterns
        # (a, b, c, d, e, f, g)
        self._digit_map = {
            (1, 1, 1, 1, 1, 1, 0): 0,
            (0, 1, 1, 0, 0, 0, 0): 1,
            (1, 1, 0, 1, 1, 0, 1): 2,
            (1, 1, 1, 1, 0, 0, 1): 3,
            (0, 1, 1, 0, 0, 1, 1): 4,
            (1, 0, 1, 1, 0, 1, 1): 5,
            (1, 0, 1, 1, 1, 1, 1): 6,
            (1, 1, 1, 0, 0, 0, 0): 7,
            (1, 1, 1, 1, 1, 1, 1): 8,
            (1, 1, 1, 1, 0, 1, 1): 9,
        }

    def decode(self, signal):
        """
        Convert 7-segment signal to a digit (0–9)
        """
        self._validate(signal)

        signal_key = tuple(signal)

        if signal_key not in self._digit_map:
            raise ValueError("Unknown seven-segment pattern")

        return self._digit_map[signal_key]

    def _validate(self, signal):
        if not isinstance(signal, (list, tuple)):
            raise TypeError("Signal must be a list or tuple")

        if len(signal) != 7:
            raise ValueError("Signal must contain exactly 7 values")

        if any(bit not in (0, 1) for bit in signal):
            raise ValueError("Each segment must be 0 or 1")
