Getting Started with Mammoth Lite
==================================

Welcome to Mammoth Lite! This guide will help you understand what continual learning is and how to use Mammoth Lite to implement and experiment with continual learning algorithms.

What is Continual Learning?
---------------------------

Continual Learning (also known as Lifelong Learning or Incremental Learning) is a machine learning paradigm where models learn from a sequence of tasks or experiences without forgetting previously acquired knowledge. This is in contrast to traditional machine learning approaches where models are trained on a fixed dataset all at once.

Key challenges in continual learning include:

* **Catastrophic Forgetting**: The tendency of neural networks to forget previously learned tasks when learning new ones  
* **Task Interference**: When learning a new task negatively affects performance on previous tasks  
* **Memory Constraints**: Limited storage for previous data or experiences  
* **Plasticity-Stability Dilemma**: Balancing the ability to learn new information while preserving old knowledge

What is Mammoth Lite?
---------------------

Mammoth Lite is a simplified framework designed to make continual learning research more accessible. It provides:

* **Simple API**: Easy-to-use functions for loading models, datasets, and running experiments  
* **Modular Design**: Easily extendable with custom models, datasets, and backbones  
* **Educational Focus**: Clear examples and documentation for learning continual learning concepts  
* **Minimal Dependencies**: Lightweight installation with essential dependencies only

Key Features
------------

🎯 **Easy Experimentation**
  Run continual learning experiments with just a few lines of code

🔧 **Extensible Framework**
  Add custom models, datasets, and neural network backbones

📚 **Educational Resources**
  Comprehensive examples for learning continual learning

🚀 **Quick Setup**
  Minimal dependencies and straightforward installation

Framework Architecture
----------------------

Mammoth Lite is built around four main components:

1. **Models**: Continual learning algorithms (e.g., SGD, replay-based methods)
2. **Datasets**: Continual learning benchmarks and custom datasets
3. **Backbones**: Neural network architectures (e.g., ResNet, custom CNNs)
4. **Utils**: Training utilities, evaluation metrics, and helper functions

Basic Workflow
--------------

The typical workflow in Mammoth Lite follows these steps:

1. **Choose a Model**: Select a continual learning algorithm
2. **Select a Dataset**: Pick a continual learning benchmark
3. **Configure Parameters**: Set hyperparameters and training options
4. **Run Training**: Execute the continual learning experiment
5. **Evaluate Results**: Analyze performance across tasks

Example Quick Start
-------------------

Here's a simple example to get you started:

.. code-block:: python

   from mammoth_lite import load_runner, train

   # Load a continual learning model and dataset
   model, dataset = load_runner('sgd', 'seq-cifar10',
       args={'lr': 0.1, 'n_epochs': 5, 'batch_size': 32}
   )

   # Train the model on the continual learning scenario
   train(model, dataset)

This example:

1. Loads an SGD-based continual learning model
2. Sets up the Sequential CIFAR-10 benchmark
3. Configures training parameters
4. Runs the continual learning experiment

Next Steps
----------

Now that you understand the basics, you can:

1. **Install Mammoth Lite**: Follow the :doc:`installation` guide
2. **Try the Quickstart**: Go through the :doc:`quickstart` tutorial  
3. **Explore Examples**: Check out the detailed :doc:`examples/index`
4. **Learn Core Concepts**: Read about :doc:`core_concepts`
5. **Build Custom Components**: Create your own models, datasets, or backbones

Continue to the next section to learn how to install Mammoth Lite on your system.
