Core Concepts
=============

This section explains the fundamental concepts of continual learning and how they are implemented in Mammoth Lite.

Continual Learning Fundamentals
-------------------------------

Continual Learning Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mammoth Lite supports only a subset of the common continual learning scenarios:

**Class-Incremental Learning (Class-IL)**
  New classes are introduced in each task, and the model must classify among all seen classes without knowing task boundaries at test time.

**Task-Incremental Learning (Task-IL)**  
  New classes are introduced in each task, but the task identity is known at test time, so the model only needs to classify among classes within that task.

Check out the awesome `Three scenarios for continual learning <https://arxiv.org/abs/1904.07734>`_ paper for more details on these scenarios.

The Catastrophic Forgetting Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When neural networks learn new tasks, they tend to forget previously learned information. This usually happens because:

1. **Weight Interference**: New task learning overwrites weights important for previous tasks  
2. **Feature Drift**: Shared representations shift to accommodate new data  
3. **Output Bias**: Decision boundaries change to favor recent training data  

Mammoth Lite Framework Components
---------------------------------

Models (Continual Learning Algorithms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models in Mammoth Lite implement different strategies to mitigate catastrophic forgetting:

**Regularization-based Methods**
    Add penalties to prevent important weights from changing too much. This includes methods such as Elastic Weight Consolidation (here, you can find it in its less heavy "online" version: `--model=ewc_on`) and Learning without Forgetting (`--model=lwf`).

**Replay-based Methods**  
    Store and replay examples from previous tasks while learning new ones. This includes methods like Experience Replay (`--model=er`) and Dark Experience Replay (`--model=der`).

**Architectural Methods**
    Allocate different network parameters for different tasks. This includes methods like Progressive Neural Networks (`--model=pnn`) and PackNet (not available yet).

**Pretrain-based Methods**
    Use pretraining on a large dataset before continual learning. This includes methods like Learning to Prompt for Continua (`--model=l2p`) and CODA-Prompt (`--model=coda_prompt`).

**Mixed Methods**
    Combine multiple strategies for better performance. The vast majority of methods found in literature are mixed, such as Slow Learner with Classifier Alignment (`--model=slca`), which combines regularization from a pretrained network and generative replay.

All models inherit from the ``ContinualModel`` base class and implement the ``observe`` method. Models can be registered with the framework using the ``@register_model`` decorator:

.. code-block:: python

    @register_model('my-model')
    class MyContinualModel(ContinualModel):
        def observe(self, inputs, labels, not_aug_inputs, epoch=None) -> float:
            """
            Process a batch of training data.
            
            Args:
                inputs: input data processed with data augmentation
                labels: target labels  
                not_aug_inputs: non-augmented input data
                epoch: current epoch number (optional)
            """
            # Implement your continual learning algorithm here
            pass

Datasets (Continual Learning Benchmarks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Datasets in Mammoth Lite split traditional datasets into sequential tasks:

**Sequential CIFAR-10**
CIFAR-10 split into 5 tasks of 2 classes each:

- Task 1: airplane, automobile
- Task 2: bird, cat  
- Task 3: deer, dog
- Task 4: frog, horse
- Task 5: ship, truck

Datasets inherit from ``ContinualDataset`` and define:

- **Task structure**: How data is split across tasks (e.g., number of classes per task and number of tasks)
- **Transformations**: Data augmentation and normalization
- **get_data_loaders**: Method to return the COMPLETE train and test datasets, which will be split into tasks automatically during training.

In addition, datasets must be registered with the framework using the ``@register_dataset`` decorator.

.. code-block:: python

    @register_dataset('my-dataset')
    class MyDataset(ContinualDataset):
        NAME = 'my-dataset'
        SETTING = 'class-il'  # or 'task-il', 'domain-il'
        N_CLASSES_PER_TASK = 2
        N_TASKS = 5
       
        def get_data_loaders(self):
            """Return train and test loaders for current task."""
            pass

Backbones (Neural Network Architectures)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Backbones define the neural network architecture used for feature extraction:

For example, Mammoth Lite includes:
- **ResNet18**: A popular convolutional neural network architecture
- **Vision Transformer**: A transformer-based architecture for image classification

Backbones must inherit from ``MammothBackbone`` and can be registered with the framework using the ``@register_backbone`` decorator.

.. code-block:: python

   class MyBackbone(MammothBackbone):
       def __init__(self, num_classes, param1, param2):
           super().__init__()
           # Define your architecture
           
       def forward(self, x, returnt=ReturnTypes.OUT):
           """
           Forward pass with flexible return types.
           
           Args:
               x: Input tensor
               returnt: What to return (OUT, FEATURES, etc.)
           """
           pass

    @register_backbone('my-backbone-v1')
    def my_backbone(num_classes):
        return MyBackbone(num_classes, param1=10, param2=20)

    @register_backbone('my-backbone-v2')
    def my_backbone(num_classes):
        return MyBackbone(num_classes, param1=50, param2=100)

Training Process
----------------

Task-by-Task Learning
~~~~~~~~~~~~~~~~~~~~~

Mammoth Lite trains models sequentially on each task:

1. **Task Setup**: Load data for current task
2. **Training Loop**: Train for specified epochs  
3. **Evaluation**: Test on all seen tasks
4. **Task Transition**: Move to next task

.. code-block:: python

    for task_id in range(dataset.N_TASKS):

        # Hook to initialize the model for the current task, if needed
        model.begin_task(dataset)
       
        # Training loop
        for epoch in range(args.n_epochs):
            for batch in dataset.train_loader:
                model.observe(batch.inputs, batch.labels, batch.not_aug_inputs)
       
        # Hook to perform some final operations after training the task 
        model.end_task(dataset)

        # Evaluate on all tasks
        accuracy = evaluate(model, dataset)

All this is taken care by the ``train`` function (in ``mammoth_lite/utils/training.py``), which orchestrates the entire training process:

.. code-block:: python

    from mammoth_lite import train

    # Train the model on the continual learning scenario
    train(model, dataset)


Evaluation Metrics
~~~~~~~~~~~~~~~~~~

In Mammoth Lite, performance is measured in terms of *Final Average Accuracy* (FAA), which is the average accuracy across all tasks at the end of training. 

More metrics, such as *forgetting*, are available in the full Mammoth framework.

Extending Mammoth Lite
----------------------

Adding Custom Models
~~~~~~~~~~~~~~~~~~~~

1. Inherit from ``ContinualModel``
2. Implement the ``observe`` method
3. Register with ``@register_model('name')``
4. Optionally define compatibility settings

Adding Custom Datasets  
~~~~~~~~~~~~~~~~~~~~~~

1. Create a data source class inheriting from ``MammothDataset`` (see `mammoth_lite/datasets/seq_cifar10.py` for an example)
   - Define the dataset name, setting (e.g., 'class-il'), number of classes per task, and number of tasks
   - Implement the ``get_data_loaders`` method to return train and test loaders
2. Create a continual dataset class inheriting from ``ContinualDataset``
3. Register with ``@register_dataset('name')``

Adding Custom Backbones
~~~~~~~~~~~~~~~~~~~~~~~

1. Inherit from ``MammothBackbone``  
2. Implement forward pass with flexible return types
3. Register with ``@register_backbone('name')``

Next Steps
----------

Now that you understand the core concepts:

1. **Explore Examples**: See :doc:`examples/index` for hands-on implementations
2. **Read API Docs**: Check :doc:`api/index` for detailed reference
3. **Experiment**: Try different model-dataset combinations
4. **Contribute**: Add your own models, datasets, or improvements

The examples section will show you how to implement these concepts in practice with detailed Jupyter notebooks.
