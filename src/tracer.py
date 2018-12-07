import os

import mlflow


IS_LOGABLE = True


class _Controller:
    def __init__(self, func):
        self._func = func

    def __call__(self, *args, **kwargs):
        if IS_LOGABLE:
            return self._func(*args, **kwargs)
        else:
            return


def _get_uri_of_run(run_id):
    path_mlrun = mlflow.tracking.get_tracking_uri()
    for exp_id in os.listdir(path_mlrun):
        for r in os.listdir(os.path.join(path_mlrun, exp_id)):
            if r == run_id:
                return os.path.join(path_mlrun, exp_id, run_id)
    return None


@_Controller
def start_trace(job_name=None):
    if job_name is not None:
        mlflow.set_experiment(job_name)
        mlflow.start_run()
    else:
        mlflow.start_run()


@_Controller
def end_trace():
    mlflow.end_run()


@_Controller
def log_param(key, val):
    mlflow.log_param(key, val)


@_Controller
def log_metric(key, val):
    mlflow.log_metric(key, val)


@_Controller
def log_artifact(fname):
    mlflow.log_artifact(fname)


@_Controller
def load_artifact(run_id, fname):
    return os.path.join(_get_uri_of_run(run_id), 'artifacts', fname)


@_Controller
def log_model(model_type, model, artifact_path):
    if model_type == 'sklearn':
        mlflow.sklearn.log_model(model, artifact_path)
    elif model_type == 'keras':
        mlflow.keras.log_model(model, artifact_path)
    elif model_type == 'tensorflow':
        mlflow.tensorflow.log_model(model, artifact_path)


@_Controller
def load_model(run_id, model_type, path):
    if model_type == 'sklearn':
        return mlflow.sklearn.load_model(path, run_id)
    elif model_type == 'keras':
        return mlflow.keras.load_model(path, run_id)
    elif model_type == 'tensorflow':
        return mlflow.tensorflow.load_model(path, run_id)
