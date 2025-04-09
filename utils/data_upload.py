import click
import logging
import kubernetes
import time
import subprocess
import sys
import os

logger = logging.getLogger(__name__)


@click.option(
    "--data-path", required=True, type=click.Path(exists=True), help="Path to the data"
)
@click.option("--pvc-name", required=True, type=str, help="name of PVC")
@click.option("--namespace", type=str, required=True)
@click.command(
    name="data-upload", help="upload data to a PVC in the authenticated cluster"
)
@click.pass_context
def data_upload(
    ctx: click.Context,
    data_path: str,
    pvc_name: str,
    namespace: str,
):
    from kubernetes import config

    config.load_kube_config()  # Use ServiceAccount credentials

    # take local SDG data on disk
    # validate k8s connection
    # create PVC
    # upload data to PVC

    # Request the Kubernetes API
    v1 = kubernetes.client.CoreV1Api()

    pvc_manifest = kubernetes.client.V1PersistentVolumeClaim(
        api_version="v1",
        kind="PersistentVolumeClaim",
        metadata=kubernetes.client.V1ObjectMeta(name=pvc_name),
        spec=kubernetes.client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            resources=kubernetes.client.V1ResourceRequirements(
                requests={"storage": "25Gi"}
            ),
        ),
    )
    try:
        v1.create_namespaced_persistent_volume_claim(
            namespace=namespace, body=pvc_manifest
        )
    except kubernetes.client.exceptions.ApiException as exc:
        if "already exists" in str(exc):
            print("PVC already exists -- using existing PVC")
        else:
            click.secho("Error with Kube API", fg="red")
            click.exceptions.Exit(1)
    pod_manifest = kubernetes.client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=kubernetes.client.V1ObjectMeta(name=f"{pvc_name}-loader"),
        spec=kubernetes.client.V1PodSpec(
            containers=[
                kubernetes.client.V1Container(
                    name=f"{pvc_name}-loader-ctr",
                    image="alpine",
                    command=["/bin/sh", "-c", "sleep 3600"],
                    volume_mounts=[
                        kubernetes.client.V1VolumeMount(
                            mount_path="/mnt/storage", name="my-storage"
                        )
                    ],
                )
            ],
            volumes=[
                kubernetes.client.V1Volume(
                    name="my-storage",
                    persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=pvc_name
                    ),
                )
            ],
        ),
    )

    try:
        v1.create_namespaced_pod(namespace=namespace, body=pod_manifest)
    except kubernetes.client.exceptions.ApiException as exc:
        if "already exists" in str(exc):
            print("Pod already exists -- using existing pod")
        else:
            click.secho("Error with Kube API", fg="red")
            click.exceptions.Exit(1)
    check_pod_status(v1, namespace, f"{pvc_name}-loader")
    files_to_pod(data_path, namespace, f"{pvc_name}-loader", "/mnt/storage")

    # 3. Check the pod status until it starts running


def check_pod_status(v1, namespace, pod_name):
    pod_status = v1.read_namespaced_pod_status(name=pod_name, namespace=namespace)
    while pod_status.status.phase != "Running":
        print(f"Pod status: {pod_status.status.phase}")
        time.sleep(5)
        pod_status = v1.read_namespaced_pod_status(name=pod_name, namespace=namespace)

    print("Pod is running. Data upload in progress.")


def files_to_pod(local_path, namespace, pod_name, container_path):
    try:
        if os.path.isdir(local_path):
            # Build the tar command for local machine to pipe to pod
            tar_cmd = [
                "tar",
                "-C",
                os.path.dirname(local_path),
                "-cf",
                "-",
                os.path.basename(local_path),
            ]
            exec_cmd = [
                "oc",
                "exec",
                "-i",
                "-n",
                namespace,
                pod_name,
                "--",
                "tar",
                "-xf",
                "-",
                "-C",
                container_path,
            ]

            # Open the subprocess for both commands (pipe from tar to oc exec)
            tar_process = subprocess.Popen(
                tar_cmd, stdout=subprocess.PIPE, stderr=sys.stderr
            )
            exec_process = subprocess.Popen(
                exec_cmd, stdin=tar_process.stdout, stderr=sys.stderr
            )

            # Close stdout of tar_process as it is piped to exec_process
            tar_process.stdout.close()

            # Wait for both processes to finish
            tar_process.wait()
            exec_process.wait()
        else:
            command = [
                "oc",
                "cp",
                local_path,
                f"{namespace}/{pod_name}:{container_path}",
            ]
            process = subprocess.Popen(
                command, stdout=sys.stdout, stderr=sys.stderr, text=True
            )
            process.wait()
    except subprocess.CalledProcessError as e:
        print(f"Error during file sync: {e}", file=sys.stderr)
        raise
