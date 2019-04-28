import flow


def main():
    train_start_year = 2011
    valid_start_year = 2012
    convert_method = 'NoContentConverter'
    model_method = 'RealValuedMatrixFactorization'
    topic = 'question1'

    # model_params = None

    # model_params = {
    #     'similarity_theshold': 0.5,
    # }

    model_params = dict(dim=10, epoch=500, lr=0.05, l1=0.05, l2=0.2)

    datagroup_id = flow.prepare_datagroup(train_start_year, valid_start_year)
    flow.train(datagroup_id, convert_method, model_method, topic, model_params=model_params)

    # flow.deploy(convert_method, model_method, topic, model_params=model_params)

    # flow.test('d1531eb4f61a4f47bb611e27400e6933')


if __name__ == '__main__':
    main()
