import flow


def main():
    train_start_year = 1996
    valid_start_year = 2010
    convert_method = 'NoContentConverter'
    model_method = 'AverageModel'
    evaluate_methods = ['rms']

    datagroup_id = flow.prepare_datagroup(train_start_year, valid_start_year)
    flow.train(datagroup_id, convert_method, model_method, evaluate_methods)

    flow.deploy(convert_method, model_method, evaluate_methods)

    flow.test_question1('d002a8e39a7046578a6a38e02962486a', model_method)


if __name__ == '__main__':
    main()
