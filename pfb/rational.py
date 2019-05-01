import typing

__all__ = [
    "Rational"
]


class Rational:

    def __init__(self, numerator, denominator):
        self._numerator = int(numerator)
        self._denominator = int(denominator)

    def __float__(self):
        return self._numerator / self._denominator

    def __int__(self):
        return int(self.__float__())

    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

    @property
    def numerator(self):
        return self._numerator

    @property
    def nu(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    @property
    def de(self):
        return self._denominator

    def normalize(self, n):
        val = (self.de*n)/self.nu
        if int(val) == val:
            return int(val)
        else:
            raise ValueError(f"Couldn't normalize {n}")

    @classmethod
    def from_str(cls, rational_str: typing.Any, delimiter: str = "/"):
        """
        Return a new instance of Rational if `rational_str` is a str.
        If `rational_str` is already a Rational object, just return that.

        Args:
            rational_str (str/Rational)
            delimiter (str)
        Returns:
            Rational
        """
        if hasattr(rational_str, "format"):
            return cls(*rational_str.split(delimiter))
        elif hasattr(rational_str, "nu"):
            return rational_str
        else:
            raise RuntimeError(
                (f"Couldn't identify {rational_str} "
                 f"of type {type(rational_str)}"))
