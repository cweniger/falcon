class Recorder:
    def __init__(self, every=1000, base_path="logs", format='joblib'):
        self.every = every
        self.base_path = base_path
        self.format = format
        self.counter = -1

    def __call__(self, objs):
        # Increment call counter
        self.counter += 1
        # Check if it's time to record
        if self.counter % self.every != 0:
            return  # Not time to record yet

        # Determine object to save
        if callable(objs):
            obj = objs()
        else:
            obj = objs

        if self.format == 'joblib':
            import joblib
            joblib.dump(obj, f"{self.base_path}_{self.counter}.joblib")
        else:
            raise ValueError(f"Unsupported format: {self.format}")