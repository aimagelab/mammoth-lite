Basic Usage
===========

This example demonstrates the fundamental concepts and API of Mammoth Lite. You'll learn how to load models and datasets, configure parameters, and run continual learning experiments.

.. note::
   This example is based on the Jupyter notebook ``examples/notebooks/basics.ipynb``. You can run it interactively or follow along here.

Learning Objectives
-------------------

By the end of this example, you'll understand:

* How to import and use the core Mammoth Lite functions  
* How to explore available arguments and their purposes    
* How to load and configure models and datasets  
* How to run a basic continual learning experiment  
* How to interpret the results  

Core Functions Overview
-----------------------

Mammoth Lite provides several key functions for continual learning experiments:

* ``get_avail_args()``: Discover available arguments and their descriptions  
* ``load_runner()``: Load and configure a model-dataset pair  
* ``train()``: Execute the continual learning training process  

Let's explore each of these in detail.

Exploring Available Arguments
-----------------------------

Before running experiments, it's helpful to understand what arguments are available:

.. code-block:: python

   from mammoth_lite import get_avail_args

   # Get available arguments
   required_args, optional_args = get_avail_args()

   print("Required arguments:")
   for arg, info in required_args.items():
       print(f"  {arg}: {info['description']}")

   print("\\nOptional arguments:")
   for arg, info in optional_args.items():
       print(f"  {arg}: {info['default']} - {info['description']}")

**Expected Output:**

.. code-block:: text

   Required arguments:
     model: The continual learning algorithm to use
     dataset: The continual learning benchmark dataset
     n_epochs: Number of training epochs per task
     lr: Learning rate for the optimizer
     batch_size: Number of samples per batch

   Optional arguments:
     optimizer: Optimizer, default is `sgd`
     savecheck: Save model checkpoints (`last` or `task`), default is `last`, meaning the model will be saved at the end of training
     ...

Understanding Argument Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Required Arguments**
  These must be provided for every experiment. They specify the fundamental components (model and dataset).

**Optional Arguments**  
  These have default values but can be customized to:
  
  * Control training behavior (learning rate, epochs, batch size)  
  * Specify computational resources (device, number of workers)    
  * Configure checkpointing  

Loading Models and Datasets
---------------------------

The ``load_runner`` function sets up your experiment by loading and configuring the specified model and dataset:

.. code-block:: python

   from mammoth_lite import load_runner

   # Load SGD model with Sequential CIFAR-10 dataset
   model, dataset = load_runner(
       model='sgd',
       dataset='seq-cifar10', 
       args={
           'lr': 0.1,           # Learning rate
           'n_epochs': 1,       # Epochs per task (reduced for quick demo)
           'batch_size': 32     # Batch size
       }
   )

   print(f"Model: {type(model).__name__}")
   print(f"Dataset: {type(dataset).__name__}")
   print(f"Number of tasks: {dataset.N_TASKS}")
   print(f"Classes per task: {dataset.N_CLASSES_PER_TASK}")

**Expected Output:**

.. code-block:: text

    Loading model:  sgd
    - Using ResNet as backbone
    Using device cuda
    Model: Sgd
    Dataset: SequentialCIFAR10
    Number of tasks: 5
    Classes per task: 2

What Happens During Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you call ``load_runner``:

1. **Model Instantiation**: Creates an instance of the specified continual learning algorithm
2. **Dataset Setup**: Loads and configures the continual learning benchmark
3. **Backbone Loading**: Sets up the neural network architecture (default: ResNet18)
4. **Optimizer Configuration**: Initializes the optimizer with specified parameters
5. **Device Assignment**: Moves model to appropriate device (GPU if available)

Running the Training Process
----------------------------

The ``train`` function executes the continual learning experiment:

.. code-block:: python

   from mammoth_lite import train

   # Run the continual learning experiment
   train(model, dataset)

**Expected Output:**

.. code-block:: text

    Task 1 - Epoch 1: 100%|██████████| 313/313 [00:07<00:00, 42.78it/s, loss=0.571, lr=0.1]
    Evaluating Task 1: 100%|██████████| 63/63 [00:00<00:00, 93.82it/s, acc_task_1=78.5]
    Accuracy for task 1     [Class-IL]: 78.45       [Task-IL]: 78.45
    Task 2 - Epoch 1: 100%|██████████| 313/313 [00:07<00:00, 43.03it/s, loss=0.398, lr=0.1]
    Evaluating Task 2: 100%|██████████| 126/126 [00:01<00:00, 95.85it/s, acc_task_2=69.9]
    Accuracy for task 2     [Class-IL]: 34.95       [Task-IL]: 68.05
    Task 3 - Epoch 1: 100%|██████████| 313/313 [00:07<00:00, 43.03it/s, loss=0.398, lr=0.1]
    ...

Understanding the Training Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During training, you'll observe:

1. **Task-by-Task Learning**: The model learns each task sequentially
2. **Progress Bars**: Show training progress for each task
3. **Evaluation**: After each task, the model is tested on all seen tasks and the accuracy in terms of Final Average Accuracy (FAA) is reported both for Class-IL and Task-IL
4. **Catastrophic Forgetting**: Performance drops on previous tasks

Understanding the Results
-------------------------

Key Observations
~~~~~~~~~~~~~~~~

**Initial Performance**: Accuracy starts reasonably high. 

**Catastrophic Forgetting**: Performance on previous tasks drops significantly
  
  * Task 1 accuracy drops from 68.4% to 23.1% after learning Task 2  
  * This demonstrates the fundamental challenge in continual learning  

**Task-IL vs Class-IL**: The accuracy drop is less significant for Task-IL compared to Class-IL. This is expected, as Task-IL assumes the model is given the task identity during inference, while Class-IL does not.

Complete Example Script
-----------------------

Here's the complete example in one script:

.. code-block:: python

   """
   Basic Mammoth Lite Example
   
   This script demonstrates:
   - How to explore available arguments
   - How to load models and datasets  
   - How to run continual learning experiments
   - How to interpret the results
   """

   from mammoth_lite import get_avail_args, load_runner, train

    # 1. Explore available arguments
    print("=== Available Arguments ===")
    required_args, optional_args = get_avail_args()

    print("Required:")
    for arg, info in required_args.items():
        print(f"  {arg}: {info['description']}")

    print("\\nOptional:")
    for i, (arg, info) in enumerate(optional_args.items()):
        print(f"  {arg}: {info['default']} - {info['description']}")

    # 2. Load model and dataset
    print("\\n=== Loading Model and Dataset ===")
    model, dataset = load_runner(
        model='sgd',
        dataset='seq-cifar10',
        args={
            'lr': 0.1,
            'n_epochs': 1,      # Reduced for quick demo
            'batch_size': 32
        }
    )

    print(f"Model: {type(model).__name__}")
    print(f"Dataset: {dataset.NAME}")
    print(f"Tasks: {dataset.N_TASKS}")
    print(f"Classes per task: {dataset.N_CLASSES_PER_TASK}")

    # 3. Run training
    print("\\n=== Training ===")
    train(model, dataset)

    print("\\n=== Experiment Complete ===")
    print("Notice how accuracy on previous tasks drops as new tasks are learned.")
    print("This demonstrates catastrophic forgetting in continual learning.")

Variations to Try
-----------------

**Different Hyperparameters**

.. code-block:: python

   # Try different learning rates
   model, dataset = load_runner('sgd', 'seq-cifar10', {'lr': 0.01, 'batch_size': 32, 'n_epochs': 1})
   
   # Try different batch sizes  
   model, dataset = load_runner('sgd', 'seq-cifar10', {'batch_size': 64, 'n_epochs': 1, 'lr': 0.1})
   
   # Try more epochs per task
   model, dataset = load_runner('sgd', 'seq-cifar10', {'n_epochs': 5, 'batch_size': 32, 'lr': 0.1})

**Different Models** (when available)  

.. code-block:: python

   # Try other continual learning models
   from mammoth_lite import get_model_names
   print("Available models:", get_model_names())

**Different Datasets** (when available)  

.. code-block:: python

   # Try other continual learning benchmarks
   from mammoth_lite import get_dataset_names  
   print("Available datasets:", get_dataset_names())

Next Steps
----------

Now that you understand the basics:

1. **Try Different Configurations**: Experiment with various hyperparameters
2. **Create Custom Models**: Learn to build your own algorithms in :doc:`custom_model`
3. **Understand Evaluation**: Dive deeper into metrics and analysis
4. **Read the Theory**: Review :doc:`../core_concepts` for deeper understanding

This example provided the foundation for using Mammoth Lite. The next examples will build on these concepts to show more advanced usage patterns.
