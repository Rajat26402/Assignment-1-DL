import numpy as np
import wandb
from keras.datasets import mnist, fashion_mnist
import argparse

def one_hot_encode(y, num_classes):
    encoded = np.zeros((y.size, num_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded

def linear(Z):
    return Z

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def tanh(Z):
    return np.tanh(Z)

def sigmoid_derivative(self, A):
    return A * (1 - A)
    
def tanh_derivative(self, A):
    return 1 - A**2
        
def relu_derivative(self, A):
    return np.where(A > 0, 1, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def main():
    
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST or Fashion-MNIST dataset')
    parser.add_argument('-wp', '--wandb_project', default='myprojectname', help='Project name')
    parser.add_argument('-we', '--wandb_entity', default='myname', help='Wandb entity')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='loss function')
    parser.add_argument('-o', '--optimizer', default='sgd', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam'], help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='momentum')
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001, help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay used by optimizers')
    parser.add_argument('-w_i', '--weight_init', default='random', choices=['random', 'Xavier'], help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4, help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', default='sigmoid', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], help='Activation function')
    args = parser.parse_args()

    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        class_names = [str(i) for i in range(10)] 
    else:  
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    if args.weight_init == 'Xavier':
        args.weight_init = 'xavier'
    if args.activation == 'ReLU':
        args.activation = 'relu'
    elif args.activation == 'identity':
        args.activation = 'linear' 
    
    if args.optimizer == 'nag':
        args.optimizer = 'nesterov'
    
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
    num_classes = 10
    y_train1 = y_train.copy()
    y_test1 = y_test.copy()
    y_train = one_hot_encode(y_train, num_classes)
    y_test = one_hot_encode(y_test, num_classes)
    split_idx = int(0.9 * len(x_train))
    x_train, x_val = x_train[:split_idx], x_train[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]
    y_train1, y_val1 = y_train1[:split_idx], y_train1[split_idx:]

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "dataset": args.dataset,"learning_rate": args.learning_rate,"epochs": args.epochs,"batch_size": args.batch_size,"loss": args.loss,"optimizer": args.optimizer,
            "momentum": args.momentum,"beta": args.beta,"beta1": args.beta1,"beta2": args.beta2,"epsilon": args.epsilon,"weight_decay": args.weight_decay,"weight_init": args.weight_init,
            "num_layers": args.num_layers,"hidden_size": args.hidden_size,"activation": args.activation
        }
    )
    
    network = [x_train.shape[1]] + [args.hidden_size] * args.num_layers + [num_classes]
    best_weights, best_biases = train(
        x_train, y_train, x_val, y_val,
        layers=network,learning_rate=args.learning_rate,activation=args.activation,optimizer=args.optimizer,weight_init=args.weight_init,
        weight_decay=args.weight_decay,epochs=args.epochs,batch_size=args.batch_size,beta=args.beta,beta1=args.beta1,beta2=args.beta2,
        epsilon=args.epsilon,momentum_param=args.momentum,loss_function=args.loss)
    
    log_confusion_matrices(best_weights, best_biases, args.activation, x_test, y_test1, class_names)
    
    test_predictions = predict(x_test, best_weights, best_biases, args.activation)
    test_accuracy = np.mean(test_predictions == y_test1)
    wandb.log({"test_accuracy": test_accuracy})
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")


def sgd(weights, biases, grads_W, grads_b, learning_rate, v_W=None, v_b=None, moment2_W=None, moment2_b=None, **kwargs):
    for i in range(len(weights)):
        weights[i] -= learning_rate * grads_W[i]
        biases[i] -= learning_rate * grads_b[i]
    return weights, biases, [], [], [], []

def momentum(weights, biases, grads_W, grads_b, learning_rate, velocity_W=None, velocity_b=None, moment2_W=None, moment2_b=None, **kwargs):
    momentum_param = kwargs.get('momentum_param', 0.9)
    if not velocity_W:
        velocity_W = [np.zeros_like(W) for W in weights]
        velocity_b = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        velocity_W[i] = momentum_param * velocity_W[i] - learning_rate * grads_W[i]
        velocity_b[i] = momentum_param * velocity_b[i] - learning_rate * grads_b[i]
        
        weights[i] += velocity_W[i]
        biases[i] += velocity_b[i]
    return weights, biases, velocity_W, velocity_b, [], []

def nesterov(weights, biases, grads_W, grads_b, learning_rate, velocity_W=None, velocity_b=None, moment2_W=None, moment2_b=None, **kwargs):
    momentum_param = kwargs.get('momentum_param', 0.9)
    if not velocity_W:
        velocity_W = [np.zeros_like(W) for W in weights]
        velocity_b = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        lookahead_W = weights[i] + momentum_param * velocity_W[i]
        lookahead_b = biases[i] + momentum_param * velocity_b[i]
        
        velocity_W[i] = momentum_param * velocity_W[i] - learning_rate * grads_W[i]
        velocity_b[i] = momentum_param * velocity_b[i] - learning_rate * grads_b[i]
        
        weights[i] = lookahead_W + velocity_W[i]
        biases[i] = lookahead_b + velocity_b[i]
    return weights, biases, velocity_W, velocity_b, [], []

def rmsprop(weights, biases, grads_W, grads_b, learning_rate, velocity_W=None, velocity_b=None, moment2_W=None, moment2_b=None, **kwargs):
    beta = kwargs.get('beta', 0.9)
    epsilon = kwargs.get('epsilon', 1e-6)
    
    if not velocity_W:
        velocity_W = [np.zeros_like(W) for W in weights]
        velocity_b = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        velocity_W[i] = beta * velocity_W[i] + (1 - beta) * (grads_W[i] ** 2)
        velocity_b[i] = beta * velocity_b[i] + (1 - beta) * (grads_b[i] ** 2)
        
        weights[i] -= learning_rate * grads_W[i] / (np.sqrt(velocity_W[i]) + epsilon)        
        biases[i] -= learning_rate * grads_b[i] / (np.sqrt(velocity_b[i]) + epsilon)
    return weights, biases, velocity_W, velocity_b, [], []

def adam(weights, biases, grads_W, grads_b, learning_rate, velocity_W=None, velocity_b=None, moment2_W=None, moment2_b=None, **kwargs):
    beta1 = kwargs.get('beta1', 0.9)
    beta2 = kwargs.get('beta2', 0.999)
    epsilon = kwargs.get('epsilon', 1e-6)
    t = kwargs.get('t', 1)
    
    if not velocity_W:
        velocity_W = [np.zeros_like(W) for W in weights]
        velocity_b = [np.zeros_like(b) for b in biases]
        moment2_W = [np.zeros_like(W) for W in weights]
        moment2_b = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        velocity_W[i] = beta1 * velocity_W[i] + (1 - beta1) * grads_W[i]
        velocity_b[i] = beta1 * velocity_b[i] + (1 - beta1) * grads_b[i]
        
        moment2_W[i] = beta2 * moment2_W[i] + (1 - beta2) * (grads_W[i] ** 2)
        moment2_b[i] = beta2 * moment2_b[i] + (1 - beta2) * (grads_b[i] ** 2)
        
        velocity_W_corrected = velocity_W[i] / (1 - beta1 ** t)
        velocity_b_corrected = velocity_b[i] / (1 - beta1 ** t)
        
        moment2_W_corrected = moment2_W[i] / (1 - beta2 ** t)
        moment2_b_corrected = moment2_b[i] / (1 - beta2 ** t)
        
        if moment2_b_corrected.shape != biases[i].shape:
            moment2_b_corrected = np.reshape(moment2_b_corrected, biases[i].shape)
        
        weights[i] -= learning_rate * velocity_W_corrected / (np.sqrt(moment2_W_corrected) + epsilon)
        biases[i] -= learning_rate * velocity_b_corrected / (np.sqrt(moment2_b_corrected) + epsilon)
    
    return weights, biases, velocity_W, velocity_b, moment2_W, moment2_b


def init_weights(layers, method="random"):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        if method.lower() == "xavier":
            limit = np.sqrt(6 / (layers[i] + layers[i+1]))
        else:  
            limit = 0.1
        W = np.random.uniform(-limit, limit, (layers[i], layers[i+1]))
        weights.append(W)
        biases.append(np.zeros((1, layers[i+1])))
    return weights, biases

activation_functions = {
    "linear": linear,
    "relu": relu, 
    "sigmoid": sigmoid, 
    "tanh": tanh
}

optimizer_functions = {
    "sgd": sgd,
    "momentum": momentum,
    "nesterov": nesterov,
    "rmsprop": rmsprop,
    "adam": adam,
}

def forward(X, weights, biases, activation):
    A = [X]
    for i in range(len(weights) - 1):
        Z = A[-1] @ weights[i] + biases[i]
        A.append(activation_functions[activation](Z))
    Z = A[-1] @ weights[-1] + biases[-1]
    A.append(softmax(Z))
    return A

def compute_loss(y_true, y_pred, weights, weight_decay, loss_function="cross_entropy"):
    if loss_function == "cross_entropy":
        loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    else:  
        loss = np.mean(np.sum((y_true - y_pred) ** 2, axis=1)) / 2
    
    if weight_decay > 0: #l2 regularization
        loss += (weight_decay / 2) * sum(np.sum(W**2) for W in weights)
    
    return loss

def backward(X, y, A, weights, weight_decay, activation, loss_function="cross_entropy"):
    grads_W, grads_b = [], []
    
    if loss_function == "mean_squared_error":
        dA = 2*(A[-1] - y)
    else:  
        dA = A[-1] - y
    
    for i in reversed(range(len(weights))):
        dW = A[i].T @ dA / X.shape[0]
        db = np.sum(dA, axis=0, keepdims=True) / X.shape[0]
        
        if weight_decay > 0:
            dW += weight_decay * weights[i]
        
        grads_W.append(dW)
        grads_b.append(db)
        
        if i > 0:
            if activation == "relu":
                dA = (dA @ weights[i].T) * (A[i] > 0)
            elif activation == "sigmoid":
                dA = (dA @ weights[i].T) * (A[i] * (1 - A[i]))
            elif activation == "tanh":
                dA = (dA @ weights[i].T) * (1 - A[i]**2)
            elif activation == "linear":
                dA = dA @ weights[i].T
    
    return grads_W[::-1], grads_b[::-1]

def predict(X, weights, biases, activation):
    A = forward(X, weights, biases, activation)
    return np.argmax(A[-1], axis=1)

def train(X_train, y_train, X_val, y_val, layers, learning_rate, activation, optimizer, 
          weight_init, weight_decay, epochs, batch_size, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-6, momentum_param=0.9, loss_function="cross_entropy"):
    
    weights, biases = init_weights(layers, weight_init)    
    velocity_W = []
    velocity_b = []
    moment2_W = []
    moment2_b = []
    t = 1  
    
    num_samples = X_train.shape[0]
    best_val_acc = 0
    best_weights, best_biases = None, None
    
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        X_train2, y_train2 = X_train[indices], y_train[indices]        
        total_loss, total_acc = 0, 0
        # num_batches = num_samples // batch_size
        
        for i in range(0, num_samples, batch_size):
            X_batch = X_train2[i:i + batch_size]
            y_batch = y_train2[i:i + batch_size]
            
            A = forward(X_batch, weights, biases, activation)
            y_pred = A[-1]
            
            loss = compute_loss(y_batch, y_pred, weights, weight_decay, loss_function)
            acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
            
            total_loss += loss * len(X_batch)
            total_acc += acc * len(X_batch)
            
            grads_W, grads_b = backward(X_batch, y_batch, A, weights, weight_decay, activation, loss_function)
            
            weights, biases, velocity_W, velocity_b, moment2_W, moment2_b = optimizer_functions[optimizer](
                weights, biases, grads_W, grads_b, learning_rate, 
                velocity_W, velocity_b, moment2_W, moment2_b,
                beta=beta, beta1=beta1, beta2=beta2, epsilon=epsilon,
                momentum_param=momentum_param, t=t
            )           
            if optimizer in ["adam"]:
                t += 1
        
        avg_loss = total_loss / num_samples
        avg_acc = total_acc / num_samples        
        val_A = forward(X_val, weights, biases, activation)
        val_pred = val_A[-1]
        val_loss = compute_loss(y_val, val_pred, weights, weight_decay, loss_function)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
        
        wandb.log({
            "epoch": epoch + 1, 
            "loss": avg_loss, 
            "accuracy": avg_acc,
            "val_loss": val_loss, 
            "val_accuracy": val_acc
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = [w.copy() for w in weights]
            best_biases = [b.copy() for b in biases]
    
    return best_weights, best_biases

def log_confusion_matrices(weights, biases, activation,x_test, y_test_original, class_names):
    test_predictions = predict(x_test, weights, biases, activation)
    wandb.log({
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test_original, 
            preds=test_predictions,
            class_names=class_names
        )
    })

if __name__ == "__main__":
    main()    