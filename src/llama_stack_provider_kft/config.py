# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Optional, Dict, Any

from pydantic import BaseModel


class InstructLabKubeFlowPostTrainingConfig(BaseModel):
    model_path: str = "/model"
    data_path: str
    gpu_identifier: str
    cpu_per_worker: str
    memory_per_worker: str
    tolerations: list
    node_selectors: dict
    pytorchjob_output_yaml: str
    model_pvc_name: str
    input_pvc_name: str
    output_pvc_name: str
    name_suffix: str
    phase_num: int
    base_image: str
    nproc_per_node: int = 3
    num_warmup_steps: int = 800
    save_samples: int = 0
    seed: int = 42
    job_timeout: int = 86400
    nnodes: int = 2
    delete_after_done: bool = False

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {}