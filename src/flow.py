import tempfile

import tracer
import process
from process import Datagroup


def _get_run_id_of_datagroup(train_start_year, valid_start_year):
    run_id = tracer.get_run_id_from_param('datagroup',
                                          {'train_start_year': train_start_year,
                                           'valid_start_year': valid_start_year})
    return run_id


def prepare_datagroup(train_start_year, valid_start_year):
    run_id = _get_run_id_of_datagroup(train_start_year, valid_start_year)
    if run_id is not None:
        return run_id

    tracer.start_trace('datagroup')
    tracer.log_param('train_start_year', train_start_year)
    tracer.log_param('valid_start_year', valid_start_year)

    datagroup = Datagroup(ratings=process.get_ratings(),
                          tags=process.get_tags(),
                          movies=process.get_movies(),
                          genome=process.get_genome())
    _, datagroup_after = process.split_datagroup(train_start_year, datagroup)
    train_group, valid_group = process.split_datagroup(valid_start_year, datagroup_after)

    temp_dir = tempfile.mkdtemp()
    paths = process.save_datagroup(temp_dir, train_group, 'train')
    for path in paths:
        tracer.log_artifact(path)
    paths = process.save_datagroup(temp_dir, valid_group, 'valid')
    for path in paths:
        tracer.log_artifact(path)

    run_id = tracer.get_current_run_id()
    tracer.end_trace()

    return run_id
