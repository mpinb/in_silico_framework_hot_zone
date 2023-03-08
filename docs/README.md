docs automatically created using sphinx (run in apptainer)

```console
ml apptainer/1.1.5
data
apptainer run -H /gpfs/soma_fs/scratch/meulemeester/ ../sphinx/sphinx_latest.sif sphinx-quickstart ./project_src/in_silico_framework/docs/
```