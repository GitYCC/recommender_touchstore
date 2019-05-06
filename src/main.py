import flow


def main():
    train_start_year = 2011
    valid_start_year = 2012
    topic = 'question2'

    model_method = 'PopularityModel'
    model_params = None
    decorate_method = None
    decorate_params = None

    # model_method = 'ItemCosineSimilarity'
    # model_params = dict(similarity_theshold=0.5, use_mean_centering=False)
    # decorate_method = None
    # decorate_params = None

    # model_method = 'RealValuedMatrixFactorization'
    # model_params = dict(dim=10, epoch=500, lr=0.05, l1=0.05, l2=0.2)
    # decorate_method = 'NegativeDataDecorator'
    # decorate_params = dict(negative_data_ratio=1.0, negative_data_value=0)

    datagroup_id = flow.prepare_datagroup(train_start_year, valid_start_year)
    flow.train(datagroup_id, model_method, topic, model_params=model_params,
               decorate_method=decorate_method, decorate_params=decorate_params)

    # deploy_id = flow.deploy(model_method, topic, model_params=model_params,
    #                         decorate_method=decorate_method, decorate_params=decorate_params)
    # flow.test(deploy_id)


if __name__ == '__main__':
    main()
