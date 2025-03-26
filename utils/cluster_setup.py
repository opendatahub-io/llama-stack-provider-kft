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
    command = [
        "oc",
        "apply",
        "--server-side",
        "-k",
        "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.9.1",
    ]
    logger.info("Applying Kubeflow Resources...")
    p = subprocess.Popen(args=command, env=env)
    p.wait()
