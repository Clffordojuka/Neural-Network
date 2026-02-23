// Setup
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.0001
#define EPOCHS 1000

// Neural Network Structure
typedef struct {
    // Layer 1: Input to Hidden
    double weights_ih[HIDDEN_SIZE][INPUT_SIZE];  // 4×3 matrix
    double bias_h[HIDDEN_SIZE];                  // 4 biases
    
    // Layer 2: Hidden to Output
    double weights_ho[OUTPUT_SIZE][HIDDEN_SIZE]; // 1×4 matrix
    double bias_o[OUTPUT_SIZE];                  // 1 bias
    
    // Activations (saved for backprop)
    double hidden[HIDDEN_SIZE];      // After ReLU
    double output[OUTPUT_SIZE];      // Final output
    
    // Pre-activations (needed for gradients)
    double z_hidden[HIDDEN_SIZE];    // Before ReLU
    double z_output[OUTPUT_SIZE];    // Before output
} NeuralNetwork;

// Activation Functions
// ReLU: max(0, x)
double relu(double x) {
    return (x > 0) ? x : 0;
}

// ReLU derivative: 1 if x>0, else 0
double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

// Weight Initialization
void init_network(NeuralNetwork *nn) {
    srand(time(NULL));
    
    // Initialize input → hidden
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            // Random between -0.5 and 0.5
            nn->weights_ih[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
        nn->bias_h[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
    
    // Initialize hidden → output
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->weights_ho[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
        nn->bias_o[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
}

// Forward Propagation
void forward_propagation(NeuralNetwork *nn, double input[INPUT_SIZE]) {
    // LAYER 1: Input → Hidden
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        // Weighted sum: z = w·x + b
        nn->z_hidden[i] = nn->bias_h[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->z_hidden[i] += nn->weights_ih[i][j] * input[j];
        }
        // Activation: a = ReLU(z)
        nn->hidden[i] = relu(nn->z_hidden[i]);
    }
    
    // LAYER 2: Hidden → Output
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        // Weighted sum
        nn->z_output[i] = nn->bias_o[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->z_output[i] += nn->weights_ho[i][j] * nn->hidden[j];
        }
        // Linear activation (no function)
        nn->output[i] = nn->z_output[i];
    }
}

// // Neuron i=0, input=[0.25, 0.286, 0.375]
// nn->z_hidden[0] = nn->bias_h[0];                   // Start with bias
// nn->z_hidden[0] += nn->weights_ih[0][0] * 0.25;    // Add rooms contribution
// nn->z_hidden[0] += nn->weights_ih[0][1] * 0.286;   // Add area contribution  
// nn->z_hidden[0] += nn->weights_ih[0][2] * 0.375;   // Add distance contribution
// nn->hidden[0] = relu(nn->z_hidden[0]);             // Apply ReLU

// Backpropagation
void backward_propagation(NeuralNetwork *nn, double input[INPUT_SIZE], 
                         double target[OUTPUT_SIZE]) {
    
    // STEP 1: Calculate output layer error
    // For MSE: error = predicted - actual
    double output_error[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = nn->output[i] - target[i];
    }
    
    // STEP 2: Backpropagate to hidden layer
    // hidden_error = output_error × weight × ReLU'(z)
    double hidden_error[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_error[i] = 0.0;
        // Sum weighted errors from next layer
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_error[i] += output_error[j] * nn->weights_ho[j][i];
        }
        // Multiply by ReLU derivative
        hidden_error[i] *= relu_derivative(nn->z_hidden[i]);
    }
    
    // STEP 3: Update hidden → output weights
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            // gradient = error × previous_activation
            nn->weights_ho[i][j] -= LEARNING_RATE * output_error[i] * nn->hidden[j];
        }
        nn->bias_o[i] -= LEARNING_RATE * output_error[i];
    }
    
    // STEP 4: Update input → hidden weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->weights_ih[i][j] -= LEARNING_RATE * hidden_error[i] * input[j];
        }
        nn->bias_h[i] -= LEARNING_RATE * hidden_error[i];
    }
}

// Data Normalization
void normalize_data(double data[][INPUT_SIZE], int num_samples,
                   double min[INPUT_SIZE], double max[INPUT_SIZE]) {
    // Find min and max for each feature
    for (int j = 0; j < INPUT_SIZE; j++) {
        min[j] = data[0][j];
        max[j] = data[0][j];
        for (int i = 1; i < num_samples; i++) {
            if (data[i][j] < min[j]) min[j] = data[i][j];
            if (data[i][j] > max[j]) max[j] = data[i][j];
        }
    }
    
    // Normalize: (value - min) / (max - min)
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
        }
    }
}

// Training Function
void train(NeuralNetwork *nn, double inputs[][INPUT_SIZE],
          double targets[][OUTPUT_SIZE], int num_samples) {
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;
        
        // Train on each example
        for (int i = 0; i < num_samples; i++) {
            // Forward pass: make prediction
            forward_propagation(nn, inputs[i]);
            
            // Calculate loss (MSE)
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                double error = nn->output[j] - targets[i][j];
                total_loss += error * error;
            }
            
            // Backward pass: update weights
            backward_propagation(nn, inputs[i], targets[i]);
        }
        
        // Average loss over all samples
        total_loss /= num_samples;
        
        // Print progress
        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %.4f\n", epoch, total_loss);
        }
    }
}

// Main Function
int main() {
    // Dataset: rooms, area_sqft, distance_km
    double training_inputs[][INPUT_SIZE] = {
        {3, 1500, 2.0},
        {4, 2000, 1.5},
        {2, 1000, 3.0},
        {5, 2500, 1.0},
        {3, 1800, 2.5},
        {4, 2200, 1.2},
        {2, 900, 4.0},
        {6, 3000, 0.8}
    };
    
    // Prices in $1000s
    double training_targets[][OUTPUT_SIZE] = {
        {300}, {400}, {200}, {500},
        {350}, {450}, {180}, {600}
    };
    
    int num_samples = 8;
    
    // Step 1: Normalize inputs
    double min[INPUT_SIZE], max[INPUT_SIZE];
    normalize_data(training_inputs, num_samples, min, max);
    
    // Step 2: Normalize targets
    double target_min = training_targets[0][0];
    double target_max = training_targets[0][0];
    for (int i = 1; i < num_samples; i++) {
        if (training_targets[i][0] < target_min) 
            target_min = training_targets[i][0];
        if (training_targets[i][0] > target_max) 
            target_max = training_targets[i][0];
    }
    for (int i = 0; i < num_samples; i++) {
        training_targets[i][0] = 
            (training_targets[i][0] - target_min) / (target_max - target_min);
    }
    
    // Step 3: Create and initialize network
    NeuralNetwork nn;
    init_network(&nn);
    
    printf("Training Neural Network...\n");
    printf("Architecture: %d → %d → %d\n\n", 
           INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Step 4: Train
    train(&nn, training_inputs, training_targets, num_samples);
    
    // Step 5: Test predictions
    printf("\n=== Predictions ===\n");
    for (int i = 0; i < num_samples; i++) {
        forward_propagation(&nn, training_inputs[i]);
        
        // Denormalize: value_original = value_norm × (max-min) + min
        double predicted = nn.output[0] * (target_max - target_min) + target_min;
        double actual = training_targets[i][0] * (target_max - target_min) + target_min;
        
        printf("Sample %d: Predicted $%.2fk, Actual $%.2fk\n",
               i+1, predicted, actual);
    }
    
    // Step 6: Predict new house
    printf("\n=== New Prediction ===\n");
    double new_house[INPUT_SIZE] = {3, 1600, 1.8};
    
    // Normalize using training data's min/max
    for (int j = 0; j < INPUT_SIZE; j++) {
        new_house[j] = (new_house[j] - min[j]) / (max[j] - min[j]);
    }
    
    forward_propagation(&nn, new_house);
    double prediction = nn.output[0] * (target_max - target_min) + target_min;
    
    printf("House: 3 rooms, 1600 sqft, 1.8km\n");
    printf("Predicted: $%.2fk\n", prediction);
    
    return 0;
}