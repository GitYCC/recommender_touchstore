import flow


def main():
    train_start_year = 2011
    valid_start_year = 2012
    convert_method = 'NoContentConverter'
    model_method = 'AverageModel'
    topic = 'question2'

    datagroup_id = flow.prepare_datagroup(train_start_year, valid_start_year)
    flow.train(datagroup_id, convert_method, model_method, topic)

    # flow.deploy(convert_method, model_method, topic)

    # flow.test('5095ac6b874743129f3bc38ce0c61e27')


if __name__ == '__main__':
    main()
