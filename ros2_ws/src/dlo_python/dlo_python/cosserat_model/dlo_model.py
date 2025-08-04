import elastica as ea
from elastica.rod.cosserat_rod import CosseratRod
from elastica.callback_functions import CallBackBaseClass

from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import extend_stepper_interface


from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

from dlo_python.cosserat_model.action import MoveAction, EndsMoveAction, Action
from dlo_python.cosserat_model.contact import RodPlaneContact

from typing import List


class DloSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,  # Enabled to use boundary conditions
    ea.Forcing,  # Enabled to use forcing 'GravityForces'
    ea.Connections,  # Enabled to use FixedJoint
    ea.CallBacks,  # Enabled to use callback
    ea.Damping,  # Enabled to use damping models on systems.
    ea.Contact,  # Enabled to use contact models
):
    pass


class DloCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            # Save time, step number, position, orientation and velocity
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            return


@dataclass
class DloModelParams:
    dt: float
    n_elem: int
    length: float
    radius: float
    density: float
    youngs_modulus: float
    nu: float
    action_velocity: float
    poission_ratio: float = 0.25
    plane_spring_constant: float = 1e2
    plane_damping_constant: float = 1e-1
    plane_slip_velocity_tol: float = 1e-4
    plane_kinetic_mu_array: np.ndarray = np.array([1.0, 2.0, 3.0])


class DloModel:

    def __init__(self, dlo_params, position=None, directors=None):

        self.dlo_params = dlo_params
        self.dt = dlo_params.dt
        self.action_vel = dlo_params.action_velocity

        self.action = None

        #######################
        # constants
        self.gravity = -9.80665

        # shear modulus
        self.shear_modulus = self.dlo_params.youngs_modulus / (2 * (1 + dlo_params.poission_ratio))

        self.simulator = DloSimulator()

        # Create rod
        self.rod = CosseratRod.straight_rod(
            n_elements=dlo_params.n_elem,
            start=np.array([0.0, 0.0, dlo_params.radius]),
            direction=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            base_length=dlo_params.length,
            base_radius=dlo_params.radius,
            density=dlo_params.density,
            youngs_modulus=dlo_params.youngs_modulus,
            shear_modulus=self.shear_modulus,
        )

        if position is not None and directors is not None:
            self.rod.position_collection[:] = position[:]
            self.rod.director_collection[:] = directors[:]

            self.rod.rest_kappa[:] = self.rod.kappa[:]
            self.rod.rest_sigma[:] = self.rod.sigma[:]

        # set bending stiffness
        self.set_bending_stiffness(young_modulus=dlo_params.youngs_modulus, shear_modulus=self.shear_modulus)
        self.set_shear_stiffness_inextensible(S=10e3)

        self.simulator.append(self.rod)

        # Hold data from callback function
        self.callback_data = ea.defaultdict(list)
        self.simulator.collect_diagnostics(self.rod).using(
            DloCallBack, step_skip=1000, callback_params=self.callback_data
        )

        # integration scheme
        self.timestepper = PositionVerlet()

    def set_bending_stiffness(self, young_modulus, shear_modulus):
        I_1 = I_2 = np.pi / 4 * self.dlo_params.radius**4
        I_3 = np.pi / 2 * self.dlo_params.radius**4
        self.rod.bend_matrix[0, 0, :] = I_1 * young_modulus
        self.rod.bend_matrix[1, 1, :] = I_2 * young_modulus
        self.rod.bend_matrix[2, 2, :] = I_3 * shear_modulus

    def set_shear_stiffness_inextensible(self, S=10e5):
        self.rod.shear_matrix[0, 0, :] = S
        self.rod.shear_matrix[1, 1, :] = S
        self.rod.shear_matrix[2, 2, :] = S

    def get_callback_data(self):
        return self.callback_data

    def fix_rod(self, indices: list):
        self.simulator.constrain(self.rod).using(ea.GeneralConstraint, constrained_position_idx=tuple(indices))
        # print("Fixed nodes: ", tuple(indices))

    def add_gravity(self):
        self.simulator.add_forcing_to(self.rod).using(ea.GravityForces, acc_gravity=np.array([0.0, 0.0, self.gravity]))
        # print("Gravity added")

    def add_damping(self):
        self.simulator.dampen(self.rod).using(
            ea.AnalyticalLinearDamper, damping_constant=self.dlo_params.nu, time_step=self.dt
        )
        # print("Damping added")

    def add_move_action(self):
        self.simulator.add_forcing_to(self.rod).using(
            MoveAction, action=self.action, dt=self.dt, velocity=self.action_vel
        )

    def add_multi_move_action(self):
        self.simulator.add_forcing_to(self.rod).using(
            EndsMoveAction, list_actions=self.action, dt=self.dt, velocity=self.action_vel
        )

    def add_plane(self, plane_normal, plane_origin):

        ground_plane = ea.Plane(plane_normal=plane_normal, plane_origin=plane_origin)
        self.simulator.append(ground_plane)

        self.simulator.detect_contact_between(self.rod, ground_plane).using(
            RodPlaneContact,
            k=self.dlo_params.plane_spring_constant,
            nu=self.dlo_params.plane_damping_constant,
            slip_velocity_tol=self.dlo_params.plane_slip_velocity_tol,
            kinetic_mu_array=self.dlo_params.plane_kinetic_mu_array,
        )

    def fixed_ends(self):
        self.simulator.constrain(self.rod).using(
            ea.FixedConstraint, constrained_position_idx=(0, -1), constrained_director_idx=(0, -1)
        )

    def build_model(
        self,
        action_move=None,
        action_multi_move=None,
        gravity=True,
        damping=True,
        plane=True,
        fix_indices=None,
        fix_ends=False,
    ):
        if action_move is not None:
            self.action = action_move
            self.add_move_action()
        elif action_multi_move is not None:
            self.action = action_multi_move
            self.add_multi_move_action()

        # fix the first and last node
        if fix_indices is not None:
            self.fix_rod(fix_indices)

        if gravity:
            # add gravity
            self.add_gravity()

        if damping:
            # add damping
            self.add_damping()

        if plane:
            # add plane
            self.add_plane(plane_normal=np.array([0.0, 0.0, 1.0]), plane_origin=np.array([0.0, 0.0, 0.0]))

        if fix_ends:
            self.fixed_ends()

        # finalize
        self.simulator.finalize()

    def compute_steps_from_action(self):

        tot_steps = 50000

        if not hasattr(self, "action") or self.action is None:
            return tot_steps

        if isinstance(self.action, Action):
            disp_norm = np.linalg.norm([self.action.x, self.action.y, self.action.z])
            if disp_norm > 0:
                tot_steps = int(disp_norm / (self.action_vel * self.dt))
        elif isinstance(self.action, List):
            max_disp_norm = 0
            for action in self.action:
                disp_norm = np.linalg.norm([action.x, action.y, action.z])
                if disp_norm > max_disp_norm:
                    max_disp_norm = disp_norm

            tot_steps = int(max_disp_norm / (self.action_vel * self.dt))

        return tot_steps

    def run_simulation(self, progress_bar=True):

        total_steps = self.compute_steps_from_action()
        # print("Total steps: ", total_steps)

        do_step, stages_and_updates = extend_stepper_interface(self.timestepper, self.simulator)

        init_shape = self.rod.position_collection.copy()
        init_directors = self.rod.director_collection.copy()

        time = 0
        if progress_bar:
            for i in tqdm(range(total_steps)):
                time = do_step(self.timestepper, stages_and_updates, self.simulator, time, self.dt)
        else:
            for i in range(total_steps):
                time = do_step(self.timestepper, stages_and_updates, self.simulator, time, self.dt)

        final_shape = self.rod.position_collection.copy()
        final_directors = self.rod.director_collection.copy()

        return {
            "dlo_params": self.dlo_params,
            "total_steps": total_steps,
            "action": self.action,
            "init_shape": init_shape,
            "init_directors": init_directors,
            "final_shape": final_shape,
            "final_directors": final_directors,
        }
