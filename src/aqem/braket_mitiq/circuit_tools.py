# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from amazon-braket-examples (examples/error_mitigation/tools/
# circuit_tools.py) under the Apache License, Version 2.0. See the NOTICE file
# at the repository root for attribution.
#
# Adaptation: only the metadata-free, qiskit-free helpers are kept. The upstream
# `QubitMap`, `restricted_circuit_layout`, and `fidelity_estimation` were dropped
# because they depend on `device.properties.standardized` (vendor calibration
# metadata) and qiskit — both excluded by the design's no-device-metadata
# constraint (design-doc §1.3 / §2.2).

import numpy as np

from braket.circuits import Circuit


def multiply_gates(circuit : Circuit, gates : list[str], repetitions : int = 1) -> Circuit:
    """ multiply a gate by the number of repetitions -> generally, not an identity preserving operation """
    new = Circuit()
    for ins in circuit.instructions:
        if ins.operator.name in gates:
            for _ in range(repetitions):
                new.add_instruction(ins)
        else:
            new.add_instruction(ins)
    return new


def strip_verbatim(circuit : Circuit) -> Circuit:
    """ strip verbatim instructions from a circuit """
    new = Circuit()
    for ins in circuit.instructions:
        if "Verbatim" not in ins.operator.name:
            new.add_instruction(ins)
    return new


def convert_paulis(circ : Circuit) -> Circuit:
    """ convert Paulis to rx and rz gates """
    new = Circuit()
    for ins in circ.instructions:
        match ins.operator.name:
            case "X":
                new.rx(ins.target,np.pi)
            case "Y":
                new.rz(ins.target,np.pi)
                new.rx(ins.target,np.pi)
            case "Z":
                new.rz(ins.target,np.pi)
            case "I":
                pass
            case _:
                new.add_instruction(ins)
    return new
