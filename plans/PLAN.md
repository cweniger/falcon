* 
* Composable @dataclass-based structure for configuration arguments of SNPE_A.py. No arguments should be passed directly, just instances of those @dataclass objects.
* Separate general inference logic (parent) from specific inference (daughter)
* The parent class should have interrupt, pause and related functionality, and also a generic train function, and relevant initialization functions
* The daughter class should imlement, load/save logic, network initialization, and a train_step and val_step, used by the train function of the parent
* train responsibility
  * setting up train and val data loaders
  * early stopping logic, based on output from val_step
  * logging what can be logged at that level
* train_step respobility
  * taking as input batches in terms of numpy tensors
  * move to gpu where approipriate
  * do anything implementation specific, like rescaling to u
  * also do feedback to batch_obj by discarding irrelevant examples
  * return dictionary of loggable metric (train loss, aux train loss etc)
  * learning rate decay
* val_step responsibility
  * similar to train_step, just on validation data
  * NO batch_obj feedback
  * return dictionary of loggable metrics, including one that clearly is used always for early stopping
   
