import random
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
random.seed(42)
#initalized weights and biases randomly
input_size, hidden_size, output_size = 2, 2, 1
W1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
W2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
b2 = random.uniform(-1, 1)
#epochs and learning rate intialized
epochs, learning_rate = 10000, 0.1
#updated weights and biases for some epochs
for epoch in range(epochs):
    hidden_outputs, final_outputs, errors, d_outputs, d_hiddens = [], [], [], [], []
    for i in range(len(X)):
        
        hidden_output = []
        for j in range(hidden_size):
            sum_h = sum(X[i][k] * W1[k][j] for k in range(input_size)) + b1[j]
            hidden_output.append(sigmoid(sum_h))
        hidden_outputs.append(hidden_output)
        sum_o = sum(hidden_output[j] * W2[j] for j in range(hidden_size)) + b2
        final_output = sigmoid(sum_o)
        final_outputs.append(final_output)
        error = Y[i][0] - final_output
        errors.append(error)
        d_output = error * sigmoid_derivative(final_output)
        d_outputs.append(d_output)
        d_hidden = [d_output * W2[j] * sigmoid_derivative(hidden_output[j]) for j in range(hidden_size)]
        d_hiddens.append(d_hidden)

   
    for i in range(len(X)):
        for j in range(hidden_size):
            W2[j] += learning_rate * hidden_outputs[i][j] * d_outputs[i]
        b2 += learning_rate * d_outputs[i]

        for j in range(hidden_size):
            for k in range(input_size):
                W1[k][j] += learning_rate * X[i][k] * d_hiddens[i][j]
            b1[j] += learning_rate * d_hiddens[i][j]
print("Final Weights and biases:")
for i in range(len(W1)):
    for j in range(len(W1[i])):
        print("w"+str(i+1)+str(i+1)+": "+str(W1[i][j]))
for i in range(len(b1)):
    print("b"+str(i+1)+": "+str(b1[i]))
for i in range(len(W2)):
    print("w3"+str(i+1)+": "+str(W2[i]))

print("b3: "+str(b2))


print("\nFinal Predictions:")
for i in range(len(X)):
    hidden_output = []
    for j in range(hidden_size):
        sum_h = sum(X[i][k] * W1[k][j] for k in range(input_size)) + b1[j]
        hidden_output.append(sigmoid(sum_h))

    sum_o = sum(hidden_output[j] * W2[j] for j in range(hidden_size)) + b2
    final_output = sigmoid(sum_o)

    print(f"Input: {X[i]} -> Output: {final_output:.4f}")

