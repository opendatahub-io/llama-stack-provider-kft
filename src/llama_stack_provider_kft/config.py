from typing import Dict, Any

from pydantic import BaseModel


class InstructLabKubeFlowPostTrainingConfig(BaseModel):
    model_path: str = "/model"
    data_path: str = "/data"
    gpu_identifier: str = "nvidia.com/gpu"
    cpu_per_worker: str = "1"
    memory_per_worker: str = "32Gi"
    tolerations: list
    node_selectors: dict
    pytorchjob_output_yaml: str
    model_pvc_name: str = "model"
    input_pvc_name: str = "data"
    output_pvc_name: str = "output"
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
