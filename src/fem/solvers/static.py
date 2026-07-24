"""Static finite element solvers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import numpy as np

from fem.assembly.boundary import apply_dirichlet_to_linear_system
from fem.assembly.elasticity import assemble_stiffness_matrix
from fem.preprocess.data import PreprocessBundle
from fem.solvers.data import FEMRunDef, StaticSolution
from fem.solvers.linear import solve_sparse_direct


def solve_linear_static(bundle: PreprocessBundle, run_def: FEMRunDef) -> StaticSolution:
    """Solve one linear elastic static problem."""
    stiffness = assemble_stiffness_matrix(bundle, run_def.plane_state)
    constrained_matrix, constrained_rhs = apply_dirichlet_to_linear_system(
        stiffness,
        run_def.external_force,
        run_def.dirichlet.dofs,
        run_def.dirichlet.values,
    )
    displacement = solve_sparse_direct(constrained_matrix, constrained_rhs)
    reaction = stiffness @ displacement - run_def.external_force

    return StaticSolution(
        displacement=np.asarray(displacement, dtype=np.float64),
        reaction=np.asarray(reaction, dtype=np.float64),
    )
