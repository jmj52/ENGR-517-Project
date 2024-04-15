#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "include/mnist_file.h"
#include "include/neural_network.h"

// Convert a pixel value from 0-255 to one from 0 to 1
#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)

// Returns a random value between 0 and 1
#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))


// Initialise the weights and bias vectors with values between 0 and 1
void neural_network_random_weights(neural_network_t * network)
{
    int i, j;
#pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (i = 0; i < MNIST_LABELS; i++) {
        network->b[i] = RAND_FLOAT();
        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            network->W[i][j] = RAND_FLOAT();
        }
    }
}


// Calculate the softmax vector from the activations.
void neural_network_softmax(float * activations, int length)
{
    int i;
    float sum, max;

    // Determine max activation value
#pragma omp parallel for num_threads(NUM_THREADS) collapse(1)
    for (i = 1, max = activations[0]; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    // Normalize probabilities 0-1
#pragma omp parallel for num_threads(NUM_THREADS) collapse(1)
    for (i = 0, sum = 0; i < length; i++) {
        activations[i] = exp(activations[i] - max);
        sum += activations[i];
    }
#pragma omp parallel for num_threads(NUM_THREADS) collapse(1)
    for (i = 0; i < length; i++) {
        activations[i] /= sum;
    }
}


// Use the weights and bias vector to forward propagate through the neural network and calculate the activations.
void neural_network_hypothesis(mnist_image_t * image, neural_network_t * network, float activations[MNIST_LABELS])
{
    int i, j;
#pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (i = 0; i < MNIST_LABELS; i++) {
        activations[i] = network->b[i];
        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            activations[i] += network->W[i][j] * PIXEL_SCALE(image->pixels[j]);
        }
    }
    neural_network_softmax(activations, MNIST_LABELS);
}


// Update the gradients for this step of gradient descent using the gradient contributions from a single training example (image).
// This function returns the loss contribution from this training example.
float neural_network_gradient_update(mnist_image_t * image, neural_network_t * network, neural_network_gradient_t * gradient, uint8_t label)
{
    float activations[MNIST_LABELS];
    float b_grad, W_grad;
    int i, j;

    // First forward propagate through the network to calculate activations
    neural_network_hypothesis(image, network, activations);

    // Then back propagate through the network to update gradient and bias
#pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (i = 0; i < MNIST_LABELS; i++) {
        // This is the gradient for a softmax bias input
        b_grad = (i == label) ? activations[i] - 1 : activations[i];

        // Update the bias gradient
        gradient->b_grad[i] += b_grad;
        
        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            // The gradient for the neuron weight is the bias multiplied by the input weight
            W_grad = b_grad * PIXEL_SCALE(image->pixels[j]);

            // Update the weight gradient
            gradient->W_grad[i][j] += W_grad;
        }
    }

    // Calculate cross entropy loss
    return 0.0f - log(activations[label]);
}


// Run one step of gradient descent and update the neural network.
float neural_network_training_step(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate)
{
    neural_network_gradient_t gradient;
    float total_loss;
    int i, j;

    // Zero initialize gradient for weights and bias vector
    memset(&gradient, 0, sizeof(neural_network_gradient_t));

    // Calculate the gradient and the loss by looping through the training set
#pragma omp parallel for num_threads(NUM_THREADS) collapse(1)
    for (i = 0, total_loss = 0; i < dataset->size; i++) {
        total_loss += neural_network_gradient_update(&dataset->images[i], network, &gradient, dataset->labels[i]);
    }

    // Apply gradient descent to the network to update bias and weights based on error and learning rate
#pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (i = 0; i < MNIST_LABELS; i++) {
        network->b[i] -= learning_rate * gradient.b_grad[i] / ((float) dataset->size);

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            network->W[i][j] -= learning_rate * gradient.W_grad[i][j] / ((float) dataset->size);
        }
    }

    return total_loss;
}
