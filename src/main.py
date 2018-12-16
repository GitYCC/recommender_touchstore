import flow


def main():
    train_start_year = 1996
    valid_start_year = 2010
    convert_method = 'NoContentConverter'
    model_method = 'AverageModel'
    evaluate_methods = ['rms']

    datagroup_id = flow.prepare_datagroup(train_start_year, valid_start_year)
    flow.train(datagroup_id, convert_method, model_method, evaluate_methods)


if __name__ == '__main__':
    main()
