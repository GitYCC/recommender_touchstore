import flow


def main():
    train_start_year = 2011
    valid_start_year = 2012
    convert_method = 'NoContentConverter'
    model_method = 'PopularityModel'
    topic = 'question2'
    model_params = None
    # model_params = {
    #     'similarity_theshold': 0.5,
    # }

    datagroup_id = flow.prepare_datagroup(train_start_year, valid_start_year)
    flow.train(datagroup_id, convert_method, model_method, topic, model_params=model_params)

    # flow.deploy(convert_method, model_method, topic, model_params=model_params)

    # flow.test('5dcf04db701e47cea5d102547f0213be')


if __name__ == '__main__':
    main()
