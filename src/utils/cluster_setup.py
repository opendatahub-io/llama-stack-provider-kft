import click
import subprocess
import os
import logging

logger = logging.getLogger(__name__)


@click.command()
@click.pass_context
def cluster_setup(ctx: click.Context):
    # Copy current environment
    env = os.environ.copy()
    commands = [
        [
            "oc",
            "apply",
            "--server-side",
            "-k",
            "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=master",
        ],
        [
            "oc",
            "apply",
            "--server-side",
            "-k",
            "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=master",
        ],
        [
            "oc",
            "apply",
            "--server-side",
            "-k",
            "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1",
        ],  # this is necessary to get PyTorchJobs set up
    ]
    logger.info("Applying Kubeflow Resources...")
    for command in commands:
        p = subprocess.Popen(args=command, env=env)
        p.wait()
