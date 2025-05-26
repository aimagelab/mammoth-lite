First Continual Learning Experiment
===================================

This tutorial will guide you through setting up and running your first continual learning experiment with Mammoth Lite. You'll learn how to train a model on a sequence of tasks and observe the effects of catastrophic forgetting.

Learning Objectives
-------------------

By the end of this tutorial, you will:

- Set up a complete continual learning experiment
- Understand task sequences and data splitting
- Train a model using a continual learning strategy
- Analyze results and observe catastrophic forgetting
- Compare different continual learning approaches

Prerequisites
-------------

- Python 3.8+ installed
- Mammoth Lite installed (see :doc:`../installation`)
- Basic understanding of machine learning concepts
- Familiarity with PyTorch (helpful but not required)

Step 1: Environment Setup
-------------------------

First, let's ensure you have everything installed and import the necessary modules:

.. code-block:: python

    import torch
    import numpy as np
    from mammoth.utils.main import get_avail_args, load_runner
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

Step 2: Explore Available Configurations
-----------------------------------------

Let's see what arguments and configurations are available:

.. code-block:: python

    # Get all available arguments
    args = get_avail_args()
    
    # Print some key configuration options
    print("Available datasets:")
    # This will show datasets like seq-cifar10, seq-mnist, etc.
    
    print("\nAvailable models:")
    # This will show models like sgd, ewc, lwf, etc.

Step 3: Configure Your First Experiment
----------------------------------------

Now let's set up a simple continual learning experiment using Sequential CIFAR-10:

.. code-block:: python

    # Configure the experiment
    config = {
        'dataset': 'seq-cifar10',      # Sequential CIFAR-10 dataset
        'model': 'sgd',                # Simple SGD baseline (prone to forgetting)
        'backbone': 'resnet18',        # ResNet-18 architecture
        'lr': 0.1,                     # Learning rate
        'n_epochs': 5,                 # Epochs per task
        'batch_size': 32,              # Batch size
        'buffer_size': 200,            # Memory buffer size (for rehearsal methods)
        'seed': 42,                    # Random seed for reproducibility
        'device': str(device)          # Device to use
    }
    
    print("Experiment configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

Step 4: Load and Run the Experiment
------------------------------------

Load the runner and execute the continual learning experiment:

.. code-block:: python

    # Load the configured runner
    runner = load_runner(config)
    
    print(f"Loaded runner: {type(runner).__name__}")
    print(f"Dataset: {runner.dataset}")
    print(f"Model: {runner.model}")
    print(f"Number of tasks: {runner.dataset.N_TASKS}")
    print(f"Number of classes per task: {runner.dataset.N_CLASSES_PER_TASK}")

Step 5: Train the Model
-----------------------

Now let's train the model on the sequence of tasks:

.. code-block:: python

    # Start training
    print("Starting continual learning training...")
    
    # The runner will automatically:
    # 1. Train on each task sequentially
    # 2. Evaluate on all previous tasks after each new task
    # 3. Track metrics like accuracy and forgetting
    
    results = runner.train()
    
    print("Training completed!")
    print(f"Final results: {results}")

Expected Output
---------------

During training, you should see output similar to:

.. code-block:: text

    Task 0: Training...
    Task 0: Accuracy = 85.2%
    
    Task 1: Training...
    Task 0: Accuracy = 72.1% (forgetting = 13.1%)
    Task 1: Accuracy = 83.7%
    
    Task 2: Training...
    Task 0: Accuracy = 58.3% (forgetting = 26.9%)
    Task 1: Accuracy = 69.2% (forgetting = 14.5%)
    Task 2: Accuracy = 82.1%
    
    ...

Step 6: Analyze the Results
---------------------------

The key metrics to observe:

.. code-block:: python

    # Access detailed results
    if hasattr(runner, 'results'):
        print("Task-wise accuracies:")
        for task_id, accuracy in enumerate(runner.results['task_accuracies']):
            print(f"  Task {task_id}: {accuracy:.2f}%")
        
        print(f"\nAverage accuracy: {runner.results['average_accuracy']:.2f}%")
        print(f"Forgetting measure: {runner.results['forgetting']:.2f}%")

Understanding the Results
-------------------------

**Catastrophic Forgetting**: Notice how the accuracy on earlier tasks drops as new tasks are learned. This is the core problem that continual learning aims to solve.

**Key Metrics**:

- **Task Accuracy**: Performance on each individual task
- **Average Accuracy**: Mean performance across all tasks
- **Forgetting**: How much performance drops on previous tasks
- **Forward Transfer**: How learning new tasks helps with future tasks
- **Backward Transfer**: How new tasks affect previous task performance

Step 7: Compare with a Continual Learning Method
-------------------------------------------------

Now let's see how a continual learning strategy performs:

.. code-block:: python

    # Configure experiment with Experience Replay (ER)
    config_er = config.copy()
    config_er['model'] = 'er'  # Experience Replay method
    
    print("Running with Experience Replay...")
    runner_er = load_runner(config_er)
    results_er = runner_er.train()
    
    print("Comparison:")
    print(f"SGD - Average Accuracy: {results['average_accuracy']:.2f}%")
    print(f"ER  - Average Accuracy: {results_er['average_accuracy']:.2f}%")
    print(f"Improvement: {results_er['average_accuracy'] - results['average_accuracy']:.2f}%")

Step 8: Experiment with Different Configurations
-------------------------------------------------

Try modifying these parameters to see their effects:

.. code-block:: python

    # Different continual learning methods to try:
    methods_to_try = ['sgd', 'er', 'ewc', 'lwf', 'icarl', 'gdumb']
    
    # Different datasets to try:
    datasets_to_try = ['seq-mnist', 'seq-cifar10', 'seq-cifar100', 'seq-tinyimg']
    
    # Different backbones to try:
    backbones_to_try = ['resnet18', 'resnet34', 'alexnet', 'vgg16']

Complete Example Script
-----------------------

Here's a complete script that you can run:

.. code-block:: python

    import torch
    from mammoth.utils.main import get_avail_args, load_runner
    
    def run_continual_learning_experiment():
        """Run a complete continual learning experiment."""
        
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Configuration
        config = {
            'dataset': 'seq-cifar10',
            'model': 'sgd',
            'backbone': 'resnet18',
            'lr': 0.1,
            'n_epochs': 5,
            'batch_size': 32,
            'buffer_size': 200,
            'seed': 42,
            'device': str(device)
        }
        
        # Run experiment
        print("Starting continual learning experiment...")
        runner = load_runner(config)
        results = runner.train()
        
        # Print results
        print(f"Experiment completed!")
        print(f"Results: {results}")
        
        return results
    
    if __name__ == "__main__":
        results = run_continual_learning_experiment()

Troubleshooting
---------------

**CUDA Out of Memory**:
- Reduce ``batch_size`` to 16 or 8
- Use a smaller backbone like ``alexnet``
- Enable mixed precision training if available

**Slow Training**:
- Reduce ``n_epochs`` to 2-3 for initial experiments
- Use CPU for small experiments: ``device='cpu'``
- Use smaller datasets like ``seq-mnist``

**Import Errors**:
- Ensure Mammoth Lite is properly installed
- Check Python version compatibility (3.8+)
- Verify PyTorch installation

Next Steps
----------

Now that you've completed your first continual learning experiment:

1. Try the :doc:`training_on_custom_data` tutorial to use your own datasets
2. Explore :doc:`comparing_strategies` to systematically evaluate different methods
3. Check out :doc:`../examples/index` for more advanced usage patterns
4. Read about :doc:`../core_concepts` to deepen your understanding

Congratulations! You've successfully run your first continual learning experiment with Mammoth Lite.
