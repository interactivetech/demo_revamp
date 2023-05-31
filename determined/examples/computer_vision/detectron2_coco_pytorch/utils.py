from determined.experimental import Determined

def check_model(model_name):

    if len(Determined().get_models(name=model_name)) > 0:
        model = Determined().get_models(name=model_name)[0]
    else:
        model = Determined().create_model(model_name)

    return model