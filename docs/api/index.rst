API Reference
=============

.. autosummary::
   :toctree: generated
   :template: custom-base-template.rst
   :hidden:
   :recursive:
   
   mammoth_lite/models
   mammoth_lite/datasets
   mammoth_lite/backbone
   mammoth_lite/utils

This section provides detailed API documentation for Mammoth Lite's core components.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   utils
   models
   datasets
   backbones
   args

Overview
--------

The Mammoth Lite API is organized into several key modules:

- **utils**: Core utilities and main entry points
- **models**: Continual learning strategies and algorithms
- **datasets**: Dataset loaders and continual learning benchmarks
- **backbones**: Neural network architectures
- **args**: Configuration and argument handling

Quick Reference
---------------

Most Common Functions
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: mammoth.utils.main.get_avail_args()

   Get all available command-line arguments and their descriptions.
   
   :returns: Dictionary containing all available arguments
   :rtype: dict

.. py:function:: mammoth.utils.main.load_runner(args)

   Load and configure a continual learning runner.
   
   :param args: Configuration dictionary or namespace
   :type args: dict or argparse.Namespace
   :returns: Configured runner instance
   :rtype: Runner

.. py:function:: mammoth.utils.main.train(args)

   Train a continual learning model with the given configuration.
   
   :param args: Training configuration
   :type args: dict or argparse.Namespace
   :returns: Training results and metrics
   :rtype: dict

Core Classes
~~~~~~~~~~~~

.. py:class:: mammoth.models.ContinualModel

   Base class for all continual learning models.
   
   .. py:method:: observe(inputs, labels, not_aug_inputs)
   
      Process a batch of training data.
      
      :param inputs: Input tensors
      :param labels: Target labels
      :param not_aug_inputs: Non-augmented inputs
      :returns: Loss value
      :rtype: torch.Tensor

.. py:class:: mammoth.datasets.MammothDataset

   Base class for Mammoth datasets.
   
   .. py:method:: get_data_loaders()
   
      Get train and test data loaders.
      
      :returns: Training and testing data loaders
      :rtype: tuple

.. py:class:: mammoth.backbones.MammothBackbone

   Base class for neural network backbones.
   
   .. py:method:: forward(x)
   
      Forward pass through the backbone.
      
      :param x: Input tensor
      :returns: Output features
      :rtype: torch.Tensor

Module Details
--------------

For detailed documentation of each module, see the individual pages in this section.

Configuration Reference
------------------------

Common Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset Parameters**:

- ``dataset``: Dataset name (e.g., 'seq-cifar10', 'seq-mnist')
- ``n_tasks``: Number of tasks (auto-detected for most datasets)
- ``validation``: Fraction of training data to use for validation

**Model Parameters**:

- ``model``: Continual learning strategy (e.g., 'sgd', 'er', 'ewc')
- ``backbone``: Neural network architecture (e.g., 'resnet18', 'alexnet')
- ``buffer_size``: Memory buffer size for rehearsal-based methods

**Training Parameters**:

- ``n_epochs``: Number of training epochs per task
- ``batch_size``: Training batch size
- ``lr``: Learning rate
- ``optim``: Optimizer type ('sgd', 'adam')

**System Parameters**:

- ``device``: Computing device ('cuda', 'cpu')
- ``seed``: Random seed for reproducibility
- ``notes``: Experiment description

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = {
        # Dataset
        'dataset': 'seq-cifar10',
        'validation': 0.1,
        
        # Model
        'model': 'er',
        'backbone': 'resnet18',
        'buffer_size': 200,
        
        # Training
        'n_epochs': 10,
        'batch_size': 32,
        'lr': 0.1,
        'optim': 'sgd',
        
        # System
        'device': 'cuda',
        'seed': 42,
        'notes': 'Example experiment'
    }

Error Handling
--------------

Common Exceptions
~~~~~~~~~~~~~~~~~

.. py:exception:: mammoth.utils.exceptions.InvalidArgumentError

   Raised when invalid arguments are provided to functions.

.. py:exception:: mammoth.utils.exceptions.DatasetNotFoundError

   Raised when a requested dataset is not available.

.. py:exception:: mammoth.utils.exceptions.ModelNotFoundError

   Raised when a requested model is not available.

Error Handling Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    try:
        runner = load_runner(config)
        results = runner.train()
    except InvalidArgumentError as e:
        print(f"Configuration error: {e}")
    except DatasetNotFoundError as e:
        print(f"Dataset error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

Performance Considerations
--------------------------

Memory Usage
~~~~~~~~~~~~

- Use appropriate ``batch_size`` for your GPU memory
- Consider ``buffer_size`` impact on memory for rehearsal methods
- Monitor memory usage during training

Computational Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~

- Choose appropriate ``backbone`` for your computational budget
- Use mixed precision training when available
- Consider using fewer ``n_epochs`` for initial experiments

Best Practices
--------------

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Good: Use configuration dictionaries
    config = {
        'dataset': 'seq-cifar10',
        'model': 'er',
        'seed': 42
    }
    
    # Good: Save configurations for reproducibility
    import json
    with open('experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)

Experiment Organization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Good: Organize experiments systematically
    experiments = [
        {'model': 'sgd', 'lr': 0.1},
        {'model': 'er', 'buffer_size': 200},
        {'model': 'ewc', 'ewc_lambda': 1000}
    ]
    
    results = {}
    for i, exp_config in enumerate(experiments):
        config.update(exp_config)
        runner = load_runner(config)
        results[f'experiment_{i}'] = runner.train()

Version Information
-------------------

.. py:function:: mammoth.__version__

   Get the current version of Mammoth Lite.
   
   :returns: Version string
   :rtype: str

.. code-block:: python

    import mammoth
    print(f"Mammoth Lite version: {mammoth.__version__}")

Support and Contributing
------------------------

For API questions and contributions:

- Check the `GitHub repository <https://github.com/aimagelab/mammoth>`_
- Read the contributing guidelines
- Open issues for bugs or feature requests
