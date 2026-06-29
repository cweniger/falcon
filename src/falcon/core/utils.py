import importlib


class LazyLoader:
    """This class is used to lazily load a class from a string path."""

    def __init__(self, class_path):
        self.class_path = class_path

    def __call__(self, *args, **kwargs):
        if isinstance(self.class_path, str):
            module_name, class_name = self.class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
        else:
            model_class = self.class_path
        model_instance = model_class(*args, **kwargs)
        return model_instance
