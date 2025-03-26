# llama-stack-provider-kft

Llama Stack Remote Post Training Provider for Distributed InstructLab Training using the Kubeflow Trainer

## Utilities

As a part of this package, the `ilab-kft` command line interface is available to get a cluster properly set up for distributed training.

**NOTE:** the `oc` cli is a pre-requisite to using this tool.

### How to

#### Upload your SDG data to the cluster

Currently `ilab-kft` allows you to upload a local directory of data to your cluster mounted in a PVC.

example: `python3.11 ilab-kft.py data-upload --data-path ~/.local/share/instructlab/datasets/ --pvc-name data --namespace default`

#### Upload your model

Using the same `data-upload` command, you can also upload models:

example: `python3.11 ilab-kft.py data-upload --data-path ~/.cache/instructlab/models/granite-7b-lab/ --pvc-name model --namespace default`


## Run training

Using llama-stack and the client SDK, one can spin up a llama stack server and run post-training using this provider

```
llama stack run run.yaml --image-type venv

python3.10 train.py
```

`train.py` utilizes the llama-stack-client python SDK to initialize training arguments, and pass the required arguments to `supervised_fine_tune` in order to kick off the provider implementation maintained externally in this repository.
