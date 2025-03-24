# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from llama_stack.distribution.datatypes import Api

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    JobStatus,
    ListPostTrainingJobsResponse,
    LoraFinetuningConfig,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)
from .config import (
    InstructLabKubeFlowPostTrainingConfig,
)
from llama_stack.log import get_logger
from llama_stack.providers.utils.scheduler import Scheduler
from llama_stack.providers.utils.scheduler import JobStatus as SchedulerJobStatus

_JOB_TYPE_SUPERVISED_FINE_TUNE = "supervised-fine-tune"

logger = get_logger(name=__name__, category="post_training")

class InstructLabKubeFlowPostTrainingImpl:
    def __init__(
        self,
        config: InstructLabKubeFlowPostTrainingConfig,
      #  datasetio_api: DatasetIO,
      #  datasets: Datasets,
    ) -> None:
        self.config = config
       # self.datasetio_api = datasetio_api
       # self.datasets_api = datasets
        self._scheduler = Scheduler()

        # TODO: assume sync job, will need jobs API for async scheduling
        self.jobs = {}
        self.checkpoints_dict = {}

    async def shutdown(self):
        pass

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig, # this is basically unused
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
        model: str,
        checkpoint_dir: Optional[str],
        algorithm_config: Optional[AlgorithmConfig],
    ) -> PostTrainingJob:
        if job_uuid in self.jobs:
            raise ValueError(f"Job {job_uuid} already exists")

        post_training_job = PostTrainingJob(job_uuid=job_uuid)

        job_status_response = PostTrainingJobStatusResponse(
            job_uuid=job_uuid,
            status=JobStatus.scheduled,
            scheduled_at=datetime.now(timezone.utc),
        )
        self.jobs[job_uuid] = job_status_response

        if isinstance(algorithm_config, LoraFinetuningConfig):
            raise NotImplementedError()
        else:
            async def handler(
                    on_log_message_cb, on_status_change_cb, on_artifact_collected_cb
                    # gpu_identifier: str,
                    # cpu_per_worker: str,
                    # memory_per_worker: str,
                    # tolerations: list,
                    # node_selectors: dict,
                    # pytorchjob_output_yaml: dsl.Output[dsl.Artifact],
                    # model_pvc_name: str,
                    # input_pvc_name: str,
                    # output_pvc_name: str,
                    # name_suffix: str,
                    # phase_num: int,
                    # base_image: str,
                    # nproc_per_node: int = self.config.nproc_per_node,
                    # nnodes: int = self.config.nnodes,
                    # num_epochs: int = training_config.n_epochs,
                    # effective_batch_size: int = training_config.effective_batch_size,
                    # learning_rate: float = training_config.learning_rate,
                    # num_warmup_steps: int = self.config.num_warmup_steps,
                    # save_samples: int = self.config.save_samples,
                    # max_batch_len: int = self.config.max_batch_len,
                    # seed: int = self.config.seed,
                    # job_timeout: int = 86400,
                    # delete_after_done: bool = False,
                ):
                    import logging
                    import os

                    from kubeflow.training import TrainingClient, models
                    from kubeflow.training.constants.constants import ISTIO_SIDECAR_INJECTION
                    from kubeflow.training.utils import utils as kfto_utils

                    on_log_message_cb("Starting InstructLab finetuning")

                    path_to_model = self.config.model_path # "/input_model"
                    path_to_data = self.config.data_path # "/input_data/knowledge/data.jsonl"
                    #path_to_model = list_phase1_final_model()
                    #path_to_data = "/input_data/skills/data.jsonl"

                    if self.config.gpu_identifier == "":
                        raise RuntimeError(f"GPU identifier cannot be empty")
                    resources_per_worker = {
                        "cpu": self.config.cpu_per_worker,
                        "memory": self.config.memory_per_worker,
                        self.config.gpu_identifier: self.config.nproc_per_node,
                    }

                    name = f"train-phase{self.config.phase_num}-{self.config.name_suffix.rstrip('-sdg')}"
                    command = ["/bin/sh", "-c", "--"]

                    master_args = [
                        f"""
                        echo "Running Training Phase"
                        echo "Using {path_to_model} model for training"
                        echo "Using {path_to_data} data for training"
                        mkdir -p /tmp/model;
                        torchrun --nnodes {self.config.nnodes} \
                        --nproc_per_node {self.config.nproc_per_node} \
                        --node_rank \$(RANK) \
                        --rdzv_endpoint \$(MASTER_ADDR):\$(MASTER_PORT) \
                        -m instructlab.training.main_ds \
                        --model_name_or_path={path_to_model} \
                        --data_path={path_to_data} \
                        --output_dir=/tmp/model \
                        --num_epochs={training_config.n_epochs} \
                        --effective_batch_size={training_config.effective_batch_size} \
                        --learning_rate={training_config.learning_rate} \
                        --num_warmup_steps={self.config.num_warmup_steps} \
                        --save_samples={self.config.save_samples} \
                        --log_level=INFO \
                        --max_batch_len={training_config.max_batch_len} \
                        --seed={self.config.seed} \
                        --cpu_offload_optimizer \
                        --cpu_offload_params_fsdp \
                        --distributed_training_framework fsdp \
                        --checkpoint_at_epoch
                        """
                    ]

                    # Set volumes
                    volumes = [
                        models.V1Volume(
                            name="input-data",
                            persistent_volume_claim=models.V1PersistentVolumeClaimVolumeSource(
                                claim_name=self.config.input_pvc_name
                            ),
                        ),
                        models.V1Volume(
                            name="model",
                            persistent_volume_claim=models.V1PersistentVolumeClaimVolumeSource(
                                claim_name=self.config.model_pvc_name
                            ),
                        ),
                        models.V1Volume(
                            name="output",
                            persistent_volume_claim=models.V1PersistentVolumeClaimVolumeSource(
                                claim_name=self.config.output_pvc_name
                            ),
                        ),
                    ]

                    # Set volume mounts
                    volume_mounts_master = [
                        models.V1VolumeMount(
                            mount_path="/input_data", name="input-data", read_only=True
                        ),
                        models.V1VolumeMount(mount_path="/input_model", name="model", read_only=True),
                        models.V1VolumeMount(mount_path="/output", name="output"),
                    ]
                    # Set env variables
                    env_vars = [
                        models.V1EnvVar(name="NNODES", value=f"{self.config.nnodes}"),
                        models.V1EnvVar(name="NPROC_PER_NODE", value=f"{self.config.nproc_per_node}"),
                        models.V1EnvVar(name="XDG_CACHE_HOME", value="/tmp"),
                        models.V1EnvVar(name="TRITON_CACHE_DIR", value="/tmp"),
                        models.V1EnvVar(name="HF_HOME", value="/tmp"),
                        models.V1EnvVar(name="TRANSFORMERS_CACHE", value="/tmp"),
                    ]

                    # Get container spec
                    master_container_spec = kfto_utils.get_container_spec(
                        base_image=self.config.base_image,
                        name="pytorch",
                        resources=resources_per_worker,
                        volume_mounts=volume_mounts_master,
                    )

                    # note from ilab-on-ocp:
                    # In the next release of kubeflow-training, the command
                    # and the args will be a part of kfto_utils.get_container_spec function
                    master_container_spec.command = command
                    master_container_spec.args = master_args
                    master_container_spec.env = env_vars


                    # create master pod spec
                    master_pod_template_spec = models.V1PodTemplateSpec(
                        metadata=models.V1ObjectMeta(annotations={ISTIO_SIDECAR_INJECTION: "false"}),
                        spec=models.V1PodSpec(
                            init_containers=None,
                            containers=[master_container_spec],
                            volumes=volumes,
                            tolerations=self.config.tolerations,
                            node_selector=self.config.node_selectors,
                        ),
                    )

                    logger.info("Generating job template.")
                    logger.info("Creating TrainingClient.")

                    # Initialize training client
                    # This also finds the namespace from /var/run/secrets/kubernetes.io/serviceaccount/namespace
                    # And it also loads the kube config
                    training_client = TrainingClient()
                    namespace = training_client.namespace
                    # Create pytorch job spec
                    job_template = kfto_utils.get_pytorchjob_template(
                        name=name,
                        namespace=namespace,
                    #    worker_pod_template_spec=worker_pod_template_spec,
                        master_pod_template_spec=master_pod_template_spec,
                        num_workers=self.config.nnodes,
                        num_procs_per_worker=self.config.nproc_per_node,
                    )
                    # Save the pytorch job yaml for record keeping and debugging
                    with open(self.config.pytorchjob_output_yaml, "w", encoding="utf-8") as f:
                        f.write(job_template.to_str())

                    # Run the pytorch job
                    logger.info(f"Creating PyTorchJob in namespace: {namespace}")
                    training_client.create_job(job_template, namespace=namespace)

                    expected_conditions = ["Succeeded", "Failed"]
                    logger.info(f"Monitoring job until status is any of {expected_conditions}.")

                    def get_logs(job):
                        _, _ = training_client.get_job_logs(name=job.metadata.name, follow=True)

                    training_client.wait_for_job_conditions(
                        name=name,
                        expected_conditions=set(expected_conditions),
                        wait_timeout=self.config.job_timeout,
                        timeout=self.config.job_timeout,
                        callback=get_logs,
                    )

                    on_status_change_cb(SchedulerJobStatus.completed)

                    if self.config.delete_after_done:
                        logger.info("Deleting job after completion.")
                        training_client.delete_job(name, namespace)
                    
                    on_log_message_cb("InstructLab finetuning completed")



            job_uuid = self._scheduler.schedule(_JOB_TYPE_SUPERVISED_FINE_TUNE, job_uuid, handler)


        return post_training_job

    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob:
        pass

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        return ListPostTrainingJobsResponse(data=[PostTrainingJob(job_uuid=uuid_) for uuid_ in self.jobs])

    async def get_training_job_status(self, job_uuid: str) -> Optional[PostTrainingJobStatusResponse]:
        return self.jobs.get(job_uuid, None)

    async def cancel_training_job(self, job_uuid: str) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    async def get_training_job_artifacts(self, job_uuid: str) -> Optional[PostTrainingJobArtifactsResponse]:
        if job_uuid in self.checkpoints_dict:
            checkpoints = self.checkpoints_dict.get(job_uuid, [])
            return PostTrainingJobArtifactsResponse(job_uuid=job_uuid, checkpoints=checkpoints)
        return None
    
async def get_adapter_impl(config: InstructLabKubeFlowPostTrainingConfig, _deps):
    impl = InstructLabKubeFlowPostTrainingImpl(
        config,
      #  _deps[Api.datasetio],
      #  _deps[Api.datasets]
    )
    return impl

