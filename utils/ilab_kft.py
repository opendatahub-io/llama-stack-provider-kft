import click
from data_upload import data_upload
from cluster_setup import cluster_setup


@click.group()
@click.pass_context
def ilab_kft(ctx: click.Context):
    pass


# Register commands under the group
ilab_kft.add_command(data_upload)
ilab_kft.add_command(cluster_setup)

if __name__ == "__main__":
    ilab_kft()
