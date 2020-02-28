import six
import numpy as np
import itertools

# AiiDA imports
from aiida.orm import Code, Computer, Dict, Int, Float, KpointsData, Str, StructureData, SinglefileData, load_node
from aiida.engine import WorkChain, ToContext, CalcJob, run, submit
#from aiida.orm.nodes.data.upf import get_pseudos_dict, get_pseudos_from_structure

# aiida_quantumespresso imports
from aiida.engine import ExitCode
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_quantumespresso.utils.pseudopotential import validate_and_prepare_pseudos_inputs

# aditional imports
from . import aux_script_strings


class KSWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(KSWorkChain, cls).define(spec)
        spec.input("pw_code", valid_type=Code)
        spec.input("pp_code", valid_type=Code)
        spec.input("workchain", valid_type=Int)
        spec.input("inpkpoints", valid_type=Str)
        spec.input("inpbands", valid_type=Str)
        # TODO: check why it does not work
        #spec.inputs("metadata.label", valid_type=six.string_types,
        #            default="NanoribbonWorkChain", non_db=True, help="Label of the work chain.")
        spec.outline(
            cls.run_scf,
            cls.run_bands,
            cls.run_export_orbitals
        )
        #spec.dynamic_output()
        spec.outputs.dynamic = True


    def run_scf(self):                
        self.ctx.orig_w = load_node(self.inputs.workchain.value)
        self.ctx.pseudo_family=self.ctx.orig_w.inputs.pseudo_family
        self.ctx.structure = self.ctx.orig_w.called_descendants[-2].outputs.output_structure
        kpoints = self.ctx.orig_w.called_descendants[-3].inputs.kpoints
        nkpt=kpoints.get_kpoints_mesh()[0][0]
        return self._submit_pw_calc(self.ctx.structure, label="scf", runtype='scf',
                                    kpoints=kpoints,nkpt=nkpt, wallhours=1)


    def run_bands(self):
        prev_calc = self.ctx.scf
        self._check_prev_calc(prev_calc)
        parent_folder = prev_calc.outputs.remote_folder
        kpoints=self.ctx.orig_w.called_descendants[-7].inputs.kpoints
        return self._submit_pw_calc(self.ctx.structure,
                                    label="bands",
                                    parent_folder=parent_folder,
                                    runtype='bands',
                                    kpoints=kpoints,
                                    nkpt=12,
                                    wallhours=1)


    # =========================================================================
    def run_export_orbitals(self):
        self.report("Running pp.x to export KS orbitals")
        builder = PpCalculation.get_builder()
        builder.code = self.inputs.pp_code
        nproc_mach = builder.code.computer.get_default_mpiprocs_per_machine()

        prev_calc = self.ctx.bands
        self._check_prev_calc(prev_calc)
        builder.parent_folder = prev_calc.outputs.remote_folder

        nel = prev_calc.res.number_of_electrons
        bands=list(map(int, self.inputs.inpbands.value.split()))
        kpoints=list(map(int, self.inputs.inpkpoints.value.split()))
        #nkpt = prev_calc.res.number_of_k_points
        #nbnd = prev_calc.res.number_of_bands
        #nspin = prev_calc.res.number_of_spin_components
        volume = prev_calc.res.volume
        nhours = int(1)
        
        nnodes=int(prev_calc.attributes['resources']['num_machines'])
        npools = int(prev_calc.inputs.settings['cmdline'][1])
        #nproc_mach=int(prev_calc.attributes['resources']['num_mpiprocs_per_machine'])
        for ib,ik in itertools.product(bands,kpoints):     
            builder.parameters = Dict(dict={
                  'inputpp': {
                      # contribution of a selected wavefunction
                      # to charge density
                      'plot_num': 7,
                      'kpoint(1)': ik,
                      'kpoint(2)': ik,
                      'kband(1)': ib+int(nel/2),
                      'kband(2)': ib+int(nel/2),
                  },
                  'plot': {
                      'iflag': 3,  # 3D plot
                      'output_format': 6,  # CUBE format
                      'fileout': '_orbital.cube',
                  },
            })

            builder.metadata.label = "export_orbitals"
            builder.metadata.options = {
                "resources": {
                    "num_machines": nnodes,
                    "num_mpiprocs_per_machine": nproc_mach,
                },
                "max_wallclock_seconds":  nhours * 60 * 60,  # 6 hours
                "append_text": aux_script_strings.cube_cutter,
                "withmpi": True,
            }

            
            builder.settings = Dict(
                     dict={'additional_retrieve_list': ['*.cube.gz'],
                           'cmdline':
                     ["-npools", str(npools)]                         
                          }
                   )
            running = self.submit(builder)
            label = 'export_orbitals_{}_{}'.format(ib,ik)
            self.to_context(**{label:running})
        return


    # =========================================================================
    def _check_prev_calc(self, prev_calc):
        error = None
        output_fname = prev_calc.attributes['output_filename']
        if not prev_calc.is_finished_ok:
            error = "Previous calculation failed" #in state: "+prev_calc.get_state()
        elif output_fname not in prev_calc.outputs.retrieved.list_object_names():
            error = "Previous calculation did not retrive {}".format(output_fname)
        else:
            content = prev_calc.outputs.retrieved.get_object_content(output_fname)
            if "JOB DONE." not in content:
                error = "Previous calculation not DONE."
        if error:
            self.report("ERROR: "+error)
            #self.abort(msg=error) ## ABORT WILL NOT WORK, not defined
            #raise Exception(error)
            return ExitCode(450)

    # =========================================================================
    def _submit_pw_calc(self, structure, label, runtype, 
                        kpoints=None,nkpt=None, wallhours=24, parent_folder=None):
        self.report("Running pw.x for "+label)
        builder = PwCalculation.get_builder()

        builder.code = self.inputs.pw_code
        builder.structure = structure
        builder.parameters = self._get_parameters(structure, runtype,label)
        builder.pseudos = validate_and_prepare_pseudos_inputs(structure, None, self.ctx.pseudo_family)

        
        if parent_folder:
            builder.parent_folder = parent_folder

        # kpoints
        use_symmetry = runtype != "bands"
        #kpoints = self._get_kpoints(nkpoints, use_symmetry=use_symmetry)
        builder.kpoints = kpoints

        # parallelization settings
        ## TEMPORARY double pools in case of spin
        spinpools=int(1)
        start_mag = self._get_magnetization(structure)
        if any([m != 0 for m in start_mag.values()]):
            spinpools = int(2)
        npools = spinpools*min(  int(nkpt/5), int(5)  )
        natoms = len(structure.sites)
        nnodes = (1 + int(natoms/60) ) * npools

        builder.metadata.label = label
        builder.metadata.options = {
            "resources": { "num_machines": nnodes },
            "withmpi": True,
            "max_wallclock_seconds": wallhours * 60 * 60,
            }

        builder.settings = Dict(dict={'cmdline': ["-npools", str(npools)]})

        future = self.submit(builder)
        return ToContext(**{label:future})

    # =========================================================================
    def _get_parameters(self, structure, runtype, label):
        params = {'CONTROL': {
                     'calculation': runtype,
                     'wf_collect': True,
                     'forc_conv_thr': 0.0001,
                     'nstep': 500,
                     },
                  'SYSTEM': {
                       'ecutwfc': 50.,
                       'ecutrho': 400.,
                       'occupations': 'smearing',
                       'degauss': 0.001,
                       },
                  'ELECTRONS': {
                       'conv_thr': 1.e-8,
                       'mixing_beta': 0.25,
                       'electron_maxstep': 50,
                       'scf_must_converge': False,
                      },
                  }

        if label == 'cell_opt1':
            params['CONTROL']['forc_conv_thr']=0.0005
        if runtype == "vc-relax":
            # in y and z direction there is only vacuum
            params['CELL'] = {'cell_dofree': 'x'}

        # if runtype == "bands":
        #     params['CONTROL']['restart_mode'] = 'restart'

        start_mag = self._get_magnetization(structure)
        if any([m != 0 for m in start_mag.values()]):
            params['SYSTEM']['nspin'] = 2
            params['SYSTEM']['starting_magnetization'] = start_mag

        return Dict(dict=params)

    # =========================================================================
    def _get_kpoints(self, nx, use_symmetry=True):
        nx = max(1, nx)

        kpoints = KpointsData()
        if use_symmetry:
            kpoints.set_kpoints_mesh([nx, 1, 1], offset=[0.0, 0.0, 0.0])
        else:
            # list kpoints explicitly
            points = [[r, 0.0, 0.0] for r in np.linspace(0, 0.5, nx)]
            kpoints.set_kpoints(points)

        return kpoints


    # =========================================================================
    def _get_magnetization(self, structure):
        start_mag = {}
        for i in structure.kinds:
            if i.name.endswith("1"):
                start_mag[i.name] = 1.0
            elif i.name.endswith("2"):
                start_mag[i.name] = -1.0
            else:
                start_mag[i.name] = 0.0
        return start_mag
