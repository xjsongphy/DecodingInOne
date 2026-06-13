"""Faithful surface-code geometry used by the Ising decoder pipeline."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from decoding_in_one.codes.base import PauliString, QuantumCode, QubitCoordinate


class SurfaceCode(QuantumCode):
    """Rotated surface code with the same qubit ordering as Ising-Decoding."""

    def __init__(self, distance: int, rotation: str = "XV"):
        if distance % 2 == 0:
            raise ValueError("Distance must be odd")
        rotation = rotation.upper()
        if rotation not in ("XV", "XH", "ZV", "ZH"):
            raise ValueError("rotation must be one of: XV, XH, ZV, ZH")

        self.distance = int(distance)
        self.rotation = rotation
        self.first_bulk_syndrome_type = rotation[0]
        self.rotated_type = rotation[1]
        self.logical_direction = self.first_bulk_syndrome_type + (
            "V" if self.rotated_type == "H" else "H"
        )
        self._build_code()

    def _build_code(self) -> None:
        code_dict = self._generate_code()
        self.code_dict = code_dict
        self.data_qubits_dict = code_dict["data"]
        self.xcheck_qubits_dict = code_dict["syndrome_X"]
        self.zcheck_qubits_dict = code_dict["syndrome_Z"]

        self._data_qubits = list(self.data_qubits_dict.keys())
        self._xcheck_qubits = list(self.xcheck_qubits_dict.keys())
        self._zcheck_qubits = list(self.zcheck_qubits_dict.keys())
        self._all_qubits = self._data_qubits + self._xcheck_qubits + self._zcheck_qubits

        self.hx = np.zeros((len(self._xcheck_qubits), len(self._data_qubits)), dtype=np.int8)
        self.hz = np.zeros((len(self._zcheck_qubits), len(self._data_qubits)), dtype=np.int8)
        for i, xcheck in enumerate(self._xcheck_qubits):
            for data_q in self.xcheck_qubits_dict[xcheck]["plaquette"]["qubit_id"]:
                if data_q != -1:
                    self.hx[i, data_q] = 1
        for i, zcheck in enumerate(self._zcheck_qubits):
            for data_q in self.zcheck_qubits_dict[zcheck]["plaquette"]["qubit_id"]:
                if data_q != -1:
                    self.hz[i, data_q] = 1

        self.lx = np.zeros((self.distance, self.distance), dtype=np.int8)
        self.lz = np.zeros((self.distance, self.distance), dtype=np.int8)
        self.lx[0, : self.distance] = 1
        self.lz[0, : self.distance] = 1
        if self.logical_direction == "XH":
            self.lx = self.lx.reshape(1, -1)
            self.lz = self.lz.T.reshape(1, -1)
        elif self.logical_direction == "XV":
            self.lx = self.lx.T.reshape(1, -1)
            self.lz = self.lz.reshape(1, -1)
        elif self.logical_direction == "ZH":
            self.lx = self.lx.T.reshape(1, -1)
            self.lz = self.lz.reshape(1, -1)
        elif self.logical_direction == "ZV":
            self.lx = self.lx.reshape(1, -1)
            self.lz = self.lz.T.reshape(1, -1)

        self._data_coords = {
            qid: (
                (self.data_qubits_dict[qid]["coord"][0] - 1) // 2,
                (self.data_qubits_dict[qid]["coord"][1] - 1) // 2,
            )
            for qid in self._data_qubits
        }
        self._full_qubit_coords = {
            qid: tuple(float(c) for c in self._data_coords[qid]) for qid in self._data_qubits
        }
        self._check_coords = {
            qid: (
                self.xcheck_qubits_dict[qid]["coord"][0] / 2.0 - 0.5,
                self.xcheck_qubits_dict[qid]["coord"][1] / 2.0 - 0.5,
            )
            for qid in self._xcheck_qubits
        }
        self._check_coords.update(
            {
                qid: (
                    self.zcheck_qubits_dict[qid]["coord"][0] / 2.0 - 0.5,
                    self.zcheck_qubits_dict[qid]["coord"][1] / 2.0 - 0.5,
                )
                for qid in self._zcheck_qubits
            }
        )
        self._full_qubit_coords.update(self._check_coords)

    def _generate_code(self) -> dict:
        code_dict = {
            "data": {q: {"coord": []} for q in range(self.distance**2)},
            "syndrome_X": {
                x: {"coord": [], "plaquette": {"coord": [], "qubit_id": []}, "type": ""}
                for x in range(self.distance**2, self.distance**2 + (self.distance**2 - 1) // 2)
            },
            "syndrome_Z": {
                z: {"coord": [], "plaquette": {"coord": [], "qubit_id": []}, "type": ""}
                for z in range(
                    self.distance**2 + (self.distance**2 - 1) // 2,
                    self.distance**2 + (self.distance**2 - 1),
                )
            },
        }

        qubit_coord_dict = {"data": [], "syndrome_X": [], "syndrome_Z": []}

        x_flag = False if self.first_bulk_syndrome_type == "X" else True
        for i in range(self.distance):
            x_flag = not x_flag
            for j in range(self.distance):
                x = 1 + 2 * i
                y = 1 + 2 * j
                qubit_coord_dict["data"].append([x, y])
                if i < self.distance - 1 and j < self.distance - 1:
                    if x_flag:
                        qubit_coord_dict["syndrome_X"].append([x + 1, y + 1])
                    else:
                        qubit_coord_dict["syndrome_Z"].append([x + 1, y + 1])
                    x_flag = not x_flag

        position = [0, 0]
        keep_flag = 0 if self.rotated_type == "H" else 1
        x_flag = (self.rotated_type == "H") * (self.first_bulk_syndrome_type == "X") + (
            self.rotated_type == "V"
        ) * (self.first_bulk_syndrome_type == "Z")
        for _ in range(4 * self.distance - 1):
            position = self._hop(position, self.distance)
            keep_flag = (keep_flag + 1) % 2
            keep = not (keep_flag % 2)
            if keep and position not in (
                [0, 0],
                [0, 2 * self.distance],
                [2 * self.distance, 0],
                [2 * self.distance, 2 * self.distance],
            ):
                if x_flag:
                    qubit_coord_dict["syndrome_X"].append(position)
                else:
                    qubit_coord_dict["syndrome_Z"].append(position)

            if position in (
                [0, 0],
                [0, 2 * self.distance],
                [2 * self.distance, 0],
                [2 * self.distance, 2 * self.distance],
            ):
                keep_flag = (keep_flag - 1) % 2
                x_flag = not x_flag

        qubit_coord_dict["data"] = sorted(qubit_coord_dict["data"])
        qubit_coord_dict["syndrome_X"] = sorted(qubit_coord_dict["syndrome_X"])
        temp = [[y, x] for x, y in qubit_coord_dict["syndrome_Z"]]
        temp = sorted(temp)
        qubit_coord_dict["syndrome_Z"] = [[y, x] for x, y in temp]
        self._qubit_coord_dict = qubit_coord_dict

        for i, coord in enumerate(qubit_coord_dict["data"]):
            code_dict["data"][i]["coord"] = coord

        for i, coord in enumerate(qubit_coord_dict["syndrome_X"]):
            qid = i + self.distance**2
            code_dict["syndrome_X"][qid]["coord"] = coord
            code_dict["syndrome_X"][qid]["type"] = (
                "boundary" if 0 in coord or 2 * self.distance in coord else "bulk"
            )

        z_offset = self.distance**2 + (self.distance**2 - 1) // 2
        for i, coord in enumerate(qubit_coord_dict["syndrome_Z"]):
            qid = i + z_offset
            code_dict["syndrome_Z"][qid]["coord"] = coord
            code_dict["syndrome_Z"][qid]["type"] = (
                "boundary" if 0 in coord or 2 * self.distance in coord else "bulk"
            )

        for qid in code_dict["syndrome_X"]:
            i, j = code_dict["syndrome_X"][qid]["coord"]
            if self.logical_direction in ("XH", "ZV"):
                candidates = [[i - 1, j + 1], [i + 1, j + 1], [i - 1, j - 1], [i + 1, j - 1]]
            else:
                candidates = [[i - 1, j + 1], [i - 1, j - 1], [i + 1, j + 1], [i + 1, j - 1]]
            for candidate in candidates:
                if candidate in qubit_coord_dict["data"]:
                    code_dict["syndrome_X"][qid]["plaquette"]["coord"].append(candidate)
                    code_dict["syndrome_X"][qid]["plaquette"]["qubit_id"].append(
                        qubit_coord_dict["data"].index(candidate)
                    )
                else:
                    code_dict["syndrome_X"][qid]["plaquette"]["coord"].append([-1, -1])
                    code_dict["syndrome_X"][qid]["plaquette"]["qubit_id"].append(-1)

        for qid in code_dict["syndrome_Z"]:
            i, j = code_dict["syndrome_Z"][qid]["coord"]
            if self.logical_direction in ("XH", "ZV"):
                candidates = [[i - 1, j + 1], [i - 1, j - 1], [i + 1, j + 1], [i + 1, j - 1]]
            else:
                candidates = [[i - 1, j + 1], [i + 1, j + 1], [i - 1, j - 1], [i + 1, j - 1]]
            for candidate in candidates:
                if candidate in qubit_coord_dict["data"]:
                    code_dict["syndrome_Z"][qid]["plaquette"]["coord"].append(candidate)
                    code_dict["syndrome_Z"][qid]["plaquette"]["qubit_id"].append(
                        qubit_coord_dict["data"].index(candidate)
                    )
                else:
                    code_dict["syndrome_Z"][qid]["plaquette"]["coord"].append([-1, -1])
                    code_dict["syndrome_Z"][qid]["plaquette"]["qubit_id"].append(-1)

        return code_dict

    @staticmethod
    def _hop(position: List[int], distance: int) -> List[int]:
        x, y = position
        if x == 0 and y == 0:
            return [0, 2]
        if x == 0:
            return [x + 2, y] if y == 2 * distance else [x, y + 2]
        if x == 2 * distance:
            return [x - 2, y] if y == 0 else [x, y - 2]
        if y == 0:
            return [x, y + 2] if x == 0 else [x - 2, y]
        if y == 2 * distance:
            return [x, y - 2] if x == 2 * distance else [x + 2, y]
        raise RuntimeError("Invalid boundary hop state")

    def get_n_physical(self) -> int:
        return len(self._data_qubits)

    def get_n_logical(self) -> int:
        return 1

    def get_stabilizers(self) -> List[PauliString]:
        stabilizers: List[PauliString] = []
        n = len(self._data_qubits)
        for qid in self._xcheck_qubits:
            ops = ["I"] * n
            for data_q in self.xcheck_qubits_dict[qid]["plaquette"]["qubit_id"]:
                if data_q != -1:
                    ops[data_q] = "X"
            stabilizers.append(PauliString("".join(ops)))
        for qid in self._zcheck_qubits:
            ops = ["I"] * n
            for data_q in self.zcheck_qubits_dict[qid]["plaquette"]["qubit_id"]:
                if data_q != -1:
                    ops[data_q] = "Z"
            stabilizers.append(PauliString("".join(ops)))
        return stabilizers

    def get_logical_operators(self) -> Dict[str, PauliString]:
        lx_ops = ["I"] * len(self._data_qubits)
        lz_ops = ["I"] * len(self._data_qubits)
        for idx, value in enumerate(self.lx.flatten().tolist()):
            if value == 1:
                lx_ops[idx] = "X"
        for idx, value in enumerate(self.lz.flatten().tolist()):
            if value == 1:
                lz_ops[idx] = "Z"
        return {"X": PauliString("".join(lx_ops)), "Z": PauliString("".join(lz_ops))}

    def get_qubit_topology(self) -> Dict[int, Tuple[int, int]]:
        return dict(self._data_coords)

    def get_qubit_coordinates(self) -> Dict[int, QubitCoordinate]:
        coords: Dict[int, QubitCoordinate] = {}
        for qid, coord in self._data_coords.items():
            coords[qid] = QubitCoordinate(
                coords=self._full_qubit_coords[qid],
                qubit_id=qid,
                qubit_type="data",
            )
        for qid in self._xcheck_qubits:
            coords[qid] = QubitCoordinate(
                coords=self._full_qubit_coords[qid],
                qubit_id=qid,
                qubit_type="check_X",
            )
        for qid in self._zcheck_qubits:
            coords[qid] = QubitCoordinate(
                coords=self._full_qubit_coords[qid],
                qubit_id=qid,
                qubit_type="check_Z",
            )
        return coords

    def get_data_qubits(self) -> List[int]:
        return list(self._data_qubits)

    def get_check_qubits(self, stabilizer_type: str) -> List[int]:
        stabilizer_type = stabilizer_type.upper()
        if stabilizer_type == "X":
            return list(self._xcheck_qubits)
        if stabilizer_type == "Z":
            return list(self._zcheck_qubits)
        raise ValueError("stabilizer_type must be 'X' or 'Z'")

    def get_stabilizer_supports(self, stabilizer_type: str) -> Dict[int, Tuple[int, ...]]:
        stabilizer_type = stabilizer_type.upper()
        if stabilizer_type == "X":
            source = self.xcheck_qubits_dict
            qubits = self._xcheck_qubits
        elif stabilizer_type == "Z":
            source = self.zcheck_qubits_dict
            qubits = self._zcheck_qubits
        else:
            raise ValueError("stabilizer_type must be 'X' or 'Z'")
        return {
            qid: tuple(q for q in source[qid]["plaquette"]["qubit_id"] if q != -1) for qid in qubits
        }

    def get_stabilizer_measurement_layers(
        self, stabilizer_type: str
    ) -> List[List[Tuple[int, int]]]:
        stabilizer_type = stabilizer_type.upper()
        if stabilizer_type == "X":
            source = self.xcheck_qubits_dict
            qubits = self._xcheck_qubits
            control_first = True
        elif stabilizer_type == "Z":
            source = self.zcheck_qubits_dict
            qubits = self._zcheck_qubits
            control_first = False
        else:
            raise ValueError("stabilizer_type must be 'X' or 'Z'")

        layers: List[List[Tuple[int, int]]] = [[], [], [], []]
        for qid in qubits:
            plaquette = source[qid]["plaquette"]["qubit_id"]
            for i, data_q in enumerate(plaquette):
                if data_q == -1:
                    continue
                if control_first:
                    layers[i].append((qid, data_q))
                else:
                    layers[i].append((data_q, qid))
        return layers

    def get_all_qubits(self) -> List[int]:
        return list(self._all_qubits)
