from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import AmlCompute

ws = Workspace.from_config()

experiment = Experiment(workspace=ws, name='mlflow_bootcamp')

compute_target_name = 'lowPriorityGpu'

aml_compute = AmlCompute(workspace=ws, name=compute_target_name)

env = Environment.from_conda_specification(name='cp-env', file_path='./src/conda.yml')
src = ScriptRunConfig(source_directory='./src', script='train.py', compute_target=aml_compute, environment=env)

run = experiment.submit(config=src)
print(run.get_portal_url())
