

class Exp():
    def __init__(self):
        self.model = None
        self.dataset = None
        self.loader = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None

    @property
    def model(self):
        return self.model

    @model.setter
    def model(self, name):
        self.model = name
