"""Classes for use with IGOR Pro

Code is based on asetk module by Leopold Talirz
(https://github.com/ltalirz/asetk/blob/master/asetk/format/igor.py)
"""

import re

import numpy as np


def _format_igor_number(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Igor wave data must contain only finite numbers")
    return f"{value:.10g}"


def create_igor_text(wave_blocks, *, comments=()):
    """Return a standards-compliant Igor Text file containing multiple wave blocks.

    wave_blocks is an iterable of (names, columns) pairs. Each column becomes
    a one-dimensional Igor wave. Comments are emitted as Igor commands (X //),
    because bare comments are not valid top-level Igor Text keywords.
    """
    lines = ["IGOR"]
    lines.extend(
        f"X // {str(comment).replace(chr(13), ' ').replace(chr(10), ' ')}"
        for comment in comments
    )

    used_names = set()
    for names, columns in wave_blocks:
        names = list(names)
        columns = [np.asarray(column) for column in columns]
        if not names or len(names) != len(columns):
            raise ValueError("Each Igor wave block needs one name per column")

        lengths = {len(column) for column in columns if column.ndim == 1}
        if len(lengths) != 1 or any(column.ndim != 1 for column in columns):
            raise ValueError(
                "Igor wave columns must be one-dimensional and equally long"
            )

        for name in names:
            if len(name) > 31 or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None:
                raise ValueError(f"Invalid Igor wave name: {name!r}")
            if name in used_names:
                raise ValueError(f"Duplicate Igor wave name: {name}")
            used_names.add(name)

        lines.append("WAVES/D\t" + "\t".join(names))
        lines.append("BEGIN")
        for row in zip(*columns, strict=True):
            lines.append("\t" + "\t".join(_format_igor_number(value) for value in row))
        lines.append("END")

    if not used_names:
        raise ValueError("At least one Igor wave block is required")
    return "\r".join(lines) + "\r\r"


class FileNotBeginWithIgorError(OSError):
    def __init__(self):
        super().__init__("File does not begin with 'IGOR'")


class MissingBeginStatementError(OSError):
    def __init__(self):
        super().__init__("Missing 'BEGIN' statement of data block")


class UnknownParameterError(KeyError):
    def __init__(self, key):
        super().__init__(f"Unknown parameter {key}")


class Axis:
    """Represents an axis of an IGOR wave"""

    def __init__(self, symbol, minimum, delta, unit, wavename=None):
        self.symbol = symbol
        self.min = minimum
        self.delta = delta
        self.unit = unit
        self.wavename = wavename

    def __str__(self):
        """Prints axis in itx format
        Note: SetScale/P expects minimum value and step-size
        """
        delta = 0 if self.delta is None else self.delta
        s = 'X SetScale/P {symb} {min},{delta}, "{unit}", {name};\n'.format(
            symb=self.symbol,
            min=self.min,
            delta=delta,
            unit=self.unit,
            name=self.wavename,
        )
        return s

    def read(self, string):
        """Read axis from string
        Format:
        X SetScale/P x 0,2.01342281879195e-11,"m", data_00381_Up;
        SetScale d 0,0,"V", data_00381_Up
        """
        match = re.search(
            'SetScale/?P? (.) ([+-\\.\\de]+),([+-\\.\\de]+),"(\\w+)",\\s*(\\w+)', string
        )
        self.symbol = match.group(1)
        self.min = float(match.group(2))
        self.delta = float(match.group(3))
        self.unit = match.group(4)
        self.wavename = match.group(5)


class Wave:
    """A class template for IGOR waves of generic dimension"""

    def __init__(self, data, axes, name=None):
        """Initialize IGOR wave of generic dimension"""
        self.data = data
        self.name = "PYTHON_IMPORT" if name is None else name
        self.axes = axes

    def __str__(self):
        """Print IGOR wave"""
        s = ""
        s += "IGOR\n"

        dimstring = "("
        for i in range(len(self.data.shape)):
            dimstring += f"{self.data.shape[i]}, "
        dimstring = dimstring[:-2] + ")"

        s += f"WAVES/N={dimstring}  {self.name}\n"
        s += "BEGIN\n"
        s += self.print_data()
        s += "END\n"
        for ax in self.axes:
            s += str(ax)
        return s

    def read(self, fname):
        """Read IGOR wave

        Should work for any dimension.
        Tested so far only for 2d wave.
        """
        f = open(fname)
        content = f.read()
        f.close()

        lines = content.split("\r")

        line = lines.pop(0)
        if not line == "IGOR":
            raise FileNotBeginWithIgorError()

        line = lines.pop(0)
        while not re.match("WAVES", line):
            line = lines.pop(0)
        match = re.search(r"WAVES/N=\(([\d,]+)\)\s+(.+)", line)
        grid = match.group(1).split(",")
        grid = np.array(grid, dtype=int)
        self.name = match.group(2)

        line = lines.pop(0)
        if not line == "BEGIN":
            raise MissingBeginStatementError()

        # Read data.
        datastring = ""
        line = lines.pop(0)
        while not re.match("END", line):
            datastring += line
            line = lines.pop(0)
        data = np.array(datastring.split(), dtype=float)
        self.data = data.reshape(grid)

        # Read axes.
        line = lines.pop(0)
        matches = re.findall("SetScale.+?(?:;|$)", line)
        self.axes = []
        for match in matches:
            ax = Axis(None, None, None, None)
            ax.read(match)
            self.axes.append(ax)

    @property
    def extent(self):
        """Returns extent for plotting"""
        grid = self.data.shape
        extent = []
        for i in range(len(grid)):
            ax = self.axes[i]
            extent.append(ax.min)
            extent.append(ax.min + ax.delta * grid[i])

        return np.array(extent)

    def print_data(self):
        """Determines how to print the data block.

        To be implemented by subclasses."""

    def write(self, fname):
        f = open(fname, "w")
        f.write(str(self))
        f.close()


class Wave1d(Wave):
    """1d Igor wave"""

    default_parameters = {
        "xmin": 0.0,
        "xdelta": None,
        "xlabel": "x",
        "ylabel": "y",
    }

    def __init__(self, data=None, axes=None, name="1d", **kwargs):
        """Initialize 1d IGOR wave"""
        super().__init__(data, axes, name)

        self.parameters = self.default_parameters
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                raise UnknownParameterError(key)

        if axes is None:
            p = self.parameters
            x = Axis(
                symbol="x",
                minimum=p["xmin"],
                delta=p["xdelta"],
                unit=p["xlabel"],
                wavename=self.name,
            )
            self.axes = [x]

    def print_data(self):
        s = ""
        for line in self.data:
            s += f"{float(line):12.6e}\n"

        return s


class Wave2d(Wave):
    """2d Igor wave"""

    default_parameters = {
        "xmin": 0.0,
        "xdelta": None,
        "xmax": None,
        "xlabel": "x",
        "ymin": 0.0,
        "ydelta": None,
        "ymax": None,
        "ylabel": "y",
    }

    def __init__(self, data=None, axes=None, name=None, **kwargs):
        """Initialize 2d Igor wave
        Parameters

         * data
         * name
         * xmin, xdelta, xlabel
         * ymin, ydelta, ylabel
        """
        super().__init__(data, axes=axes, name=name)

        self.parameters = self.default_parameters
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                raise UnknownParameterError(key)

        if axes is None:
            p = self.parameters

            nx, ny = self.data.shape
            if p["xmax"] is None:
                p["xmax"] = p["xdelta"] * nx
            elif p["xdelta"] is None:
                p["xdelta"] = p["xmax"] / nx

            if p["ymax"] is None:
                p["ymax"] = p["ydelta"] * ny
            elif p["ydelta"] is None:
                p["ydelta"] = p["ymax"] / ny

            x = Axis(
                symbol="x",
                minimum=p["xmin"],
                delta=p["xdelta"],
                unit=p["xlabel"],
                wavename=self.name,
            )
            y = Axis(
                symbol="y",
                minimum=p["ymin"],
                delta=p["ydelta"],
                unit=p["ylabel"],
                wavename=self.name,
            )
            self.axes = [x, y]

    def print_data(self):
        """Determines how to print the data block"""
        s = ""
        for line in self.data:
            for x in line:
                s += f"{x:12.6e} "
            s += "\n"

        return s
