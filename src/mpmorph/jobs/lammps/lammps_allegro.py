import os
from jobflow import Maker, job

import pandas as pd
from pymatgen.core.structure import Structure
from .helpers import run_lammps, trajectory_from_lammps_dump

from mpmorph.schemas.lammps_calc import LammpsCalc

from pkg_resources import resource_filename


class BasicLammpsAllegro(Maker):
    """
    Run LAMMPS directly using allegro at a constant temperature.
    Required params:
        lammsps_cmd (str): lammps command to run sans the input file name.
            e.g. 'mpirun -n 4 lmp_mpi'
    """

    name = "LAMMPS_CALCULATION"

    @job(trajectory="trajectory", output_schema=LammpsCalc)
    def make(self, temperature: int, total_steps: int, structure: Structure = None):
        lammps_bin = os.environ.get("LAMMPS_CMD")
        allegro_path = os.environ.get("ALLEGRO_PATH")

        chem_sys_str = " ".join(
            el.symbol for el in structure.composition.elements
        )  # May cause elemtent ordering issues
        script_options = {
            "__temperature__": temperature,
            "__deployed_model_file__": allegro_path,
            "__species__": chem_sys_str,
            "__total_steps__": total_steps,
            "__print_every_n_step__": 10,
            "__precision_const__": 1e-6,
        }

        template_path = resource_filename(
            "mpmorph", "jobs/lammps/templates/basic_allergo.lammps"
        )

        run_lammps(structure, template_path, script_options, lammps_bin)

        trajectory = trajectory_from_lammps_dump("trajectory.lammpstrj")

        df = pd.read_csv(
            "lammps_allegro_data.txt",
            delimiter=" ",
            index_col="step",
            skiprows=1,
            names=[
                "step",
                "temp",
                "vol",
                "density",
                "etotal",
                "c_atomicenergies",
                "pe",
                "c_totalatomicenergy",
            ],
        )

        metadata = {
            "temperature": temperature,
            "total_steps": total_steps,
        }

        output = LammpsCalc(
            dir_name=os.getcwd(),
            trajectory=trajectory,
            composition=structure.composition,
            reduced_formula=structure.composition.reduced_formula,
            metadata=metadata,
            dump_data=df.to_dict(),
        )

        return output
