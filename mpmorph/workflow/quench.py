from fireworks import Firework, Workflow
from pymatgen import Structure, Composition
from mpmorph.fireworks import powerups
from atomate.vasp.fireworks.core import OptimizeFW
from mpmorph.fireworks.core import StaticFW, MDFW
from mpmorph.util import recursive_update
import numpy as np


def get_quench(structures, temp=None, priority=None, quench_type="simulated_anneal",
               cool_args=None, hold_args=None, quench_args=None, descriptor="", **kwargs):
    fw_list = []
    temp = {"start_temp": 3000, "end_temp": 500, "temp_step": 500} if temp is None else temp
    cool_args = {"md_params": {"nsteps": 200}} if cool_args is None else cool_args
    hold_args = {"md_params": {"nsteps": 500}} if hold_args is None else hold_args
    quench_args = {} if quench_args is None else quench_args

    for (i, structure) in enumerate(structures):
        _fw_list = []
        if quench_type == "simulated_anneal":
            for t in np.arange(temp["start_temp"], temp["end_temp"], -temp["temp_step"]):
                # get fw for cool step
                use_prev_structure = False
                if len(_fw_list) > 0:
                    use_prev_structure = True
                _fw = get_MDFW(structure, t, t - temp["temp_step"],
                               name="snap_" + str(i) + "_cool_" + str(t - temp["temp_step"]),
                               args=cool_args, parents=[_fw_list[-1]] if len(_fw_list) > 0 else [],
                               priority=priority, previous_structure=use_prev_structure,
                               insert_db=True, **kwargs)
                _fw_list.append(_fw)
                # get fw for hold step
                _fw = get_MDFW(structure, t - temp["temp_step"], t - temp["temp_step"],
                               name="snap_" + str(i) + "_hold_" + str(t - temp["temp_step"]),
                               args=hold_args, parents=[_fw_list[-1]], priority=priority,
                               previous_structure=True, insert_db=True, **kwargs)
                _fw_list.append(_fw)

        if quench_type in ["simulated_anneal", "mp_quench"]:
            # Relax OptimizeFW and StaticFW
            run_args = {"run_specs": {"vasp_input_set": None, "vasp_cmd": ">>vasp_cmd<<",
                                      "db_file": ">>db_file<<",
                                      "spec": {"_priority": priority}
                                      },
                        "optional_fw_params": {"override_default_vasp_params": {}}
                        }
            run_args = recursive_update(run_args, quench_args)
            _name = "snap_" + str(i)

            fw1 = OptimizeFW(structure=structure, name=_name + descriptor + "_optimize",
                             parents=[_fw_list[-1]] if len(_fw_list) > 0 else [],
                             **run_args["run_specs"], **run_args["optional_fw_params"],
                             max_force_threshold=None)
            if len(_fw_list) > 0:
                fw1 = powerups.add_cont_structure(fw1)
            fw1 = powerups.add_pass_structure(fw1)

            fw2 = StaticFW(structure=structure, name=_name + descriptor + "_static",
                           parents=[fw1], **run_args["run_specs"],
                           **run_args["optional_fw_params"])
            fw2 = powerups.add_cont_structure(fw2)
            fw2 = powerups.add_pass_structure(fw2)

            _fw_list.extend([fw1, fw2])

        fw_list.extend(_fw_list)

    name = structure.composition.reduced_formula + descriptor + "_quench"
    wf = Workflow(fw_list, name=name)
    return wf


def get_single_quench(structure, temp=None, priority=None, cool_args=None,
                      hold_args=None, quench_args=None, parents=None, descriptor="",
                      quench_type="simulated_anneal", add_static=False, **kwargs):
    temp = {"start_temp": 3000, "end_temp": 500, "temp_step": 500} if temp is None else temp
    cool_args = {"md_params": {"nsteps": 200}} if cool_args is None else cool_args
    hold_args = {"md_params": {"nsteps": 500}} if hold_args is None else hold_args
    quench_args = {} if quench_args is None else quench_args

    fws = []
    if quench_type == "simulated_anneal":
        for t in np.arange(temp["start_temp"], temp["end_temp"], -temp["temp_step"]):
            # get fw for cool step
            previous_structure = True if parents or fws else False
            parents = [fws[-1]] if len(fws) > 0 else parents
            end_temp = t - temp["temp_step"]
            fw = get_MDFW(structure, t, end_temp, name="cool_" + str(end_temp),
                          args=cool_args, parents=parents, priority=priority,
                          previous_structure=previous_structure,
                          insert_db=False, **kwargs)
            fws.append(fw)
            # get fw for hold step
            fw = get_MDFW(structure, end_temp, end_temp, name="hold_" + str(end_temp),
                          args=hold_args, parents=[fws[-1]], priority=priority,
                          previous_structure=True, insert_db=False, **kwargs)
            fws.append(fw)

    if quench_type in ["simulated_anneal", "mp_quench"]:
        # Relax OptimizeFW and StaticFW
        run_args = {"run_specs": {"vasp_input_set": None, "vasp_cmd": ">>vasp_cmd<<",
                                  "db_file": ">>db_file<<",
                                  "spec": {"_priority": priority}
                                  },
                    "optional_fw_params": {"override_default_vasp_params": {}}
                    }
        run_args = recursive_update(run_args, quench_args)
        parents = [fws[-1]] if len(fws) > 0 else parents
        name = '%s_optimize' % descriptor if descriptor else 'optimize'
        fw = OptimizeFW(structure=structure, name=name, parents=parents,
                        **run_args["run_specs"], **run_args["optional_fw_params"],
                        max_force_threshold=None)
        if len(fws) > 0:
            fw = powerups.add_cont_structure(fw)
        fw = powerups.add_pass_structure(fw)
        fws.append(fw)

        if add_static:
            name = '%s_static' % descriptor if descriptor else 'static'
            fw = StaticFW(structure=structure, name=name, parents=[fw],
                          **run_args["run_specs"], **run_args["optional_fw_params"])
            fw = powerups.add_cont_structure(fw)
            fw = powerups.add_pass_structure(fw)
            fws.append(fw)

    return fws


def get_MDFW(structure, start_temp, end_temp, name="molecular dynamics", priority=None,
             job_time=None, args={}, **kwargs):
    run_args = {"md_params": {"start_temp": start_temp, "end_temp": end_temp, "nsteps": 500},
                "run_specs": {"vasp_input_set": None, "vasp_cmd": ">>vasp_cmd<<",
                              "db_file": ">>db_file<<", "wall_time": 40000
                              },
                "optional_fw_params": {
                    "override_default_vasp_params": {'user_incar_settings': {'ISIF': 1, 'LWAVE': False}},
                    "copy_vasp_outputs": False,
                    "spec": {"_queueadapter": {'walltime': job_time}, "_priority": priority}
                }
                }

    run_args = recursive_update(run_args, args)
    _mdfw = MDFW(structure=structure, name=name, **run_args["md_params"],
                 **run_args["run_specs"], **run_args["optional_fw_params"], **kwargs)
    return _mdfw