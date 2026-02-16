class learningModel():
    
    def __init__(self, model, optimizer, optimizer_name, shape, learning_rate):
        self.model = model
        self.optimizer = optimizer
        self.optimizer_name = optimizer_name
        self.shape = shape
        self.learning_rate = learning_rate

    def instantiate_model_and_optimizer(self):
        input_dim = self.shape[0]
        output_dim = self.shape[-1]
        mid_dims = self.shape[1:-1]
        model = self.model(input_dim, output_dim, mid_dims)
        optimizer = self.optimizer(model.parameters(), lr = self.learning_rate)
        return model, optimizer
    
    def return_hypers(self):
        input_dim = self.shape[0]
        output_dim = self.shape[-1]
        mid_dims = self.shape[1:-1]
        modello = self.model(input_dim, output_dim, mid_dims)
        return {
            'model': {'name' : modello.get_name(),
                      'input_dim' : input_dim,
                      'output_dim' : output_dim,
                      'interim_dim' : mid_dims},
            'optimizer' : {'name' : self.optimizer_name,
                           'learning rate' : self.learning_rate}
        }
    

class combinedModel():

    def __init__(self, models):
        self.models = models

    def return_hypers(self):

        all_hypers = {}

        for model in self.models:

            current_model_hypers = self.models[model].return_hypers()
            all_hypers[model] = current_model_hypers

        return all_hypers

    def return_model(self, model):
        try:
            return self.models[model]
        except KeyError:
            raise KeyError(f'Expected key of {model} present keys: {self.models.keys()}')
