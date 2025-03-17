# DA6401_Assignment1
Assignment 1 of DA6401
Report: https://wandb.ai/da24m014-iit-madras/DLA1/reports/Rajat-Abhijit-Kambale-DA24M014-DA6401-Assignment-1--VmlldzoxMTcxODY3NA?accessToken=shibl4n9jo8hdicagrjkw913tla8ehwq5tkcgs47ycs5yir5fr1k6ld75kj6be5n
Github: https://github.com/Rajat26402/DA6401_Assignment1

Code is organised in several sections, each handles specific part of training and evaluating the neural network.Below is the structure:

1. Argument Parsing: The train.py script uses argparse to parse command-line arguments for various hyperparameters.

2. Dataset Loading and Preprocessing: Dataset is loaded based on the argument passed(minst or fashion-minst). Then the data is normalized and one-hot encoded to labels. Training data is been split into 90-10% into validation dataset.

3. Wandb Tracking: Here we have initialized wandb where we track our models, and parameters are logged.

4. Model Training: Creating the feedforward neural network, selecting optimizer and updated the weight and biases.

5. Model Evaluation and Logging: Trained model is tested on test dataset and confusion matrix is logged using wandb.

