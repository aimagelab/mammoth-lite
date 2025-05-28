Quickstart Guide
================

This quickstart guide will walk you through running your first continual learning experiment with Mammoth Lite in just a few minutes.

Prerequisites
-------------

Make sure you have Mammoth Lite installed. If not, see the :doc:`installation` guide.

Your First Experiment
---------------------

Let's start with the simplest possible continual learning experiment:

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

The easiest way to run an experiment is using the command line interface:

.. code-block:: bash

    cd mammoth_lite
    python main.py --model sgd --dataset seq-cifar10 --n_epochs 5 --lr 0.1 --batch_size 32

This command:

* Uses the **SGD** (Stochastic Gradient Descent) continual learning model, which is a simple baseline that learns sequentially without any special mechanisms to prevent forgetting.  
* Runs on the **Sequential CIFAR-10** benchmark.  
* Uses a learning rate of **0.1**, a batch size of **32**, and trains for **5 epochs per task**.  

You should see output showing the training progress across different tasks.

.. note::

    If you are using UV, make sure to activate your virtual environment first:

    .. code-block:: bash

        uv sync
        source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

    or run the command with `uv run`:

    .. code-block:: bash

        uv run main.py --model sgd --dataset seq-cifar10 --n_epochs 5 --lr 0.1 --batch_size 32

Python API
~~~~~~~~~~

You can also run experiments programmatically (e.g., in a Jupyter notebook or Python script, see :doc:`examples/index` for more examples):

.. code-block:: python

   from mammoth_lite import load_runner, train

   # Load model and dataset
   model, dataset = load_runner('sgd', 'seq-cifar10', 
                                args={'lr': 0.1, 'n_epochs': 5, 'batch_size': 32})

   # Run training
   train(model, dataset)

Understanding the Output
------------------------

During training, you'll see output like this:

.. code-block:: text

    Task 1 - Epoch 5: 100%|██████████| 1563/1563 [02:15<00:00, 11.54it/s]
    Accuracy for task 1:     [Class-IL]: 97.05%       [Task-IL]: 97.05%
    Task 2 - Epoch 5: 100%|██████████| 1563/1563 [02:15<00:00, 11.54it/s]
    Accuracy for task 2:     [Class-IL]: 40.90%       [Task-IL]: 72.85% # Notice the drop - this is catastrophic forgetting!


This shows:

* **Progress bars** for each task's training  
* **Accuracy**, measured as both Class-IL (class-incremental learning) and Task-IL (task-incremental learning)  


Customizing Your Experiment
---------------------------

Let's try a more customized experiment:

.. code-block:: bash

   python mammoth_lite/main.py \
     --model sgd \
     --dataset seq-cifar10 \
     --lr 0.01 \
     --batch_size 64 \
     --n_epochs 10

This sets:

* Learning rate to 0.01  
* Batch size to 64  
* Number of epochs per task to 10  

Available Arguments
~~~~~~~~~~~~~~~~~~~

To see all available arguments:

.. code-block:: python

   from mammoth_lite import get_avail_args

   required_args, optional_args = get_avail_args()

   print("Required arguments:")
   for arg, info in required_args.items():
       print(f"  {arg}: {info['description']}")

   print("\\nOptional arguments:")  
   for arg, info in optional_args.items():
       print(f"  {arg}: {info['default']} - {info['description']}")

Common Arguments
~~~~~~~~~~~~~~~~

Here are some commonly used arguments:

**Training Parameters:**  
* ``--lr``: Learning rate (default: 0.1)  
* ``--batch_size``: Batch size (default: 32)  
* ``--n_epochs``: Epochs per task (default: 50)  

**Model Selection:**  
* ``--model``: Continual learning algorithm  
* ``--backbone``: Neural network architecture  

**Dataset Options:**
* ``--dataset``: Continual learning benchmark

**Checkpoint Saving and Loading:**  
* ``--savecheck``: Enable saving the model checkpoint at the end of training (if set to `last`) or after eacfh task (if set to `task`)  
* ``--loadcheck``: Path to load a saved model checkpoint  


Exploring Available Components
------------------------------

List Available Models
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mammoth_lite import get_model_names
   print("Available models:", get_model_names())

List Available Datasets  
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mammoth_lite import get_dataset_names
   print("Available datasets:", get_dataset_names())

List Available Backbones
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mammoth_lite import get_backbone_names
   print("Available backbones:", get_backbone_names())

Running Different Experiments
-----------------------------

Try Different Models
~~~~~~~~~~~~~~~~~~~~

Experiment with different continual learning approaches:

.. code-block:: bash

   # Compare different models
   python mammoth_lite/main.py --model sgd --dataset seq-cifar10 --n_epochs 5
   # python mammoth_lite/main.py --model other_model --dataset seq-cifar10 --n_epochs 5

Try Different Datasets
~~~~~~~~~~~~~~~~~~~~~~

Test on different benchmarks:

.. code-block:: bash

   # Try different datasets when available
   python mammoth_lite/main.py --model sgd --dataset seq-cifar10

Next Steps
----------

Now that you've run your first experiments, you can:

1. **Learn Core Concepts**: Understand the theory behind continual learning in :doc:`core_concepts`  
2. **Explore Examples**: See detailed examples in :doc:`examples/index`  
3. **Create Custom Components**: Build your own models, datasets, or backbones  
4. **Learn to use the complete Mammoth repo**: Check out the full Mammoth documentation at https://aimagelab.github.io/mammoth/  

The examples section contains Jupyter notebooks that show how to:

* Create custom continual learning models  
* Design new datasets  
* Implement custom neural network backbones  

Continue exploring to become proficient with continual learning and Mammoth Lite!
