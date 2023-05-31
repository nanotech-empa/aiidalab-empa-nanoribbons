"""Utility functions for the nanoribbon workchain viewers."""

import colorsys

import ase
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from aiida import orm, plugins
from ase.data.colors import cpk_colors
from ase.neighborlist import NeighborList

# AiiDA data types.
ArrayData = plugins.DataFactory("array")


def get_calc_by_label(workcalc, label):
    calcs = get_calcs_by_label(workcalc, label)
    assert len(calcs) == 1
    return calcs[0]


def get_calcs_by_label(workcalc, label):
    """Get step calculation of a workchain by its name."""
    qbld = orm.QueryBuilder()
    qbld.append(orm.WorkChainNode, filters={"uuid": workcalc.uuid})
    qbld.append(
        orm.CalcJobNode, with_incoming=orm.WorkChainNode, filters={"label": label}
    )
    calcs = [c[0] for c in qbld.all()]
    for calc in calcs:
        assert calc.is_finished_ok
    return calcs


def from_cube_to_arraydata(cube_content):
    """Convert cube file to the AiiDA ArrayData object."""
    lines = cube_content.splitlines()
    natoms = int(lines[2].split()[0])  # The number of atoms listed in the file
    header = lines[
        : 6 + natoms
    ]  # Header of the file: comments, the voxel, and the number of atoms and datapoints

    # Parse the declared dimensions of the volumetric data
    x_line = header[3].split()
    xdim = int(x_line[0])
    y_line = header[4].split()
    ydim = int(y_line[0])
    z_line = header[5].split()
    zdim = int(z_line[0])

    # Get the vectors describing the basis voxel
    voxel_array = np.array(
        [
            [x_line[1], x_line[2], x_line[3]],
            [y_line[1], y_line[2], y_line[3]],
            [z_line[1], z_line[2], z_line[3]],
        ],
        dtype=np.float64,
    )
    atm_numbers = np.empty(natoms, int)
    coordinates = np.empty((natoms, 3))
    for i in range(natoms):
        line = header[6 + i].split()
        atm_numbers[i] = int(line[0])
        coordinates[i] = [float(s) for s in line[2:]]

    # Get the volumetric data
    data_array = np.empty(xdim * ydim * zdim, dtype=float)
    cursor = 0
    for line in lines[6 + natoms :]:  # The actual data: atoms and volumetric data
        lsplitted = line.split()
        data_array[cursor : cursor + len(lsplitted)] = lsplitted
        cursor += len(lsplitted)

    arraydata = ArrayData()
    arraydata.set_array("voxel", voxel_array)
    arraydata.set_array("data", data_array.reshape((xdim, ydim, zdim)))
    arraydata.set_array("data_units", np.array("e/bohr^3"))
    arraydata.set_array("coordinates_units", np.array("bohr"))
    arraydata.set_array("coordinates", coordinates)
    arraydata.set_array("atomic_numbers", atm_numbers)

    return arraydata


def adjust_lightness(color, amount=0.9):
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_struct_2d(ax_plt, atoms, alpha):
    """Plot structure on 2d matplotlib plot."""
    if alpha == 0:
        return

    # Plot overlayed structure.
    strct = atoms.repeat((4, 1, 1))
    strct.positions[:, 0] -= atoms.cell[0, 0]
    cov_radii = [ase.data.covalent_radii[a.number] for a in strct]
    nlist = NeighborList(cov_radii, bothways=True, self_interaction=False)
    nlist.update(strct)

    for atm in strct:
        # Circles.
        pos = atm.position
        nmbrs = ase.data.atomic_numbers[atm.symbol]
        ax_plt.add_artist(
            plt.Circle(
                (pos[0], pos[1]),
                ase.data.covalent_radii[nmbrs] * 0.4,
                color=adjust_lightness(cpk_colors[nmbrs], amount=0.90),
                fill=True,
                clip_on=True,
                alpha=alpha,
            )
        )

        # Bonds.
        for theneig in nlist.get_neighbors(atm.index)[0]:
            pos = (strct[theneig].position + atm.position) / 2
            pos0 = atm.position
            if (pos[0] - pos0[0]) ** 2 + (pos[1] - pos0[1]) ** 2 < 2:
                ax_plt.plot(
                    [pos0[0], pos[0]],
                    [pos0[1], pos[1]],
                    color=adjust_lightness(cpk_colors[nmbrs], amount=0.90),
                    linewidth=2,
                    linestyle="-",
                    solid_capstyle="butt",
                    alpha=alpha,
                )
