// engine.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_SAMPLES 30000
#define MAX_LINE 2048

#define INPUT_SIZE 8
#define OUTPUT_SIZE 1

#define LEARNING_RATE 0.001
#define EPOCHS 50
#define BATCH_SIZE 32
#define L2_LAMBDA 0.0001
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

// ----------------------------
// Layer Structure
// ----------------------------

typedef struct {
    int input_size;
    int output_size;

    double *weights;
    double *bias;

    double *z;
    double *a;

    double *m_w;
    double *v_w;
    double *m_b;
    double *v_b;

} Layer;

// ----------------------------
// Network Structure
// ----------------------------

typedef struct {
    int num_layers;
    Layer *layers;
} NeuralNetwork;

// ----------------------------
// Utility
// ----------------------------

double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

double random_uniform() {
    return ((double)rand() / RAND_MAX) - 0.5;
}

// ----------------------------
// Layer Initialization (He Init)
// ----------------------------

void init_layer(Layer *layer, int in, int out) {
    layer->input_size = in;
    layer->output_size = out;

    layer->weights = malloc(sizeof(double) * in * out);
    layer->bias = calloc(out, sizeof(double));
    layer->z = malloc(sizeof(double) * out);
    layer->a = malloc(sizeof(double) * out);

    layer->m_w = calloc(in * out, sizeof(double));
    layer->v_w = calloc(in * out, sizeof(double));
    layer->m_b = calloc(out, sizeof(double));
    layer->v_b = calloc(out, sizeof(double));

    double scale = sqrt(2.0 / in);

    for (int i = 0; i < in * out; i++)
        layer->weights[i] = random_uniform() * 2.0 * scale;
}

// ----------------------------
// Create Network
// ----------------------------

NeuralNetwork create_network(int *sizes, int num_layers) {
    NeuralNetwork nn;
    nn.num_layers = num_layers - 1;
    nn.layers = malloc(sizeof(Layer) * nn.num_layers);

    for (int i = 0; i < nn.num_layers; i++)
        init_layer(&nn.layers[i], sizes[i], sizes[i + 1]);

    return nn;
}

// ----------------------------
// Forward Pass
// ----------------------------

void forward(NeuralNetwork *nn, double *input) {
    double *current_input = input;

    for (int l = 0; l < nn->num_layers; l++) {
        Layer *layer = &nn->layers[l];

        for (int i = 0; i < layer->output_size; i++) {
            double sum = layer->bias[i];

            for (int j = 0; j < layer->input_size; j++) {
                sum += layer->weights[i * layer->input_size + j] *
                       current_input[j];
            }

            layer->z[i] = sum;

            if (l == nn->num_layers - 1)
                layer->a[i] = sum;
            else
                layer->a[i] = relu(sum);
        }

        current_input = layer->a;
    }
}

// ----------------------------
// Adam Update
// ----------------------------

void adam_update(double *w, double *m, double *v,
                 double grad, int index,
                 int t) {

    m[index] = BETA1 * m[index] + (1 - BETA1) * grad;
    v[index] = BETA2 * v[index] + (1 - BETA2) * grad * grad;

    double m_hat = m[index] / (1 - pow(BETA1, t));
    double v_hat = v[index] / (1 - pow(BETA2, t));

    w[index] -= LEARNING_RATE *
                m_hat / (sqrt(v_hat) + EPSILON);
}

// ----------------------------
// Backward Pass
// ----------------------------

void backward(NeuralNetwork *nn,
              double *input,
              double *target,
              int timestep) {

    int L = nn->num_layers;
    double *delta = NULL;

    for (int l = L - 1; l >= 0; l--) {

        Layer *layer = &nn->layers[l];
        double *new_delta = malloc(sizeof(double) * layer->output_size);

        if (l == L - 1) {
            for (int i = 0; i < layer->output_size; i++)
                new_delta[i] = layer->a[i] - target[i];
        } else {
            Layer *next = &nn->layers[l + 1];

            for (int i = 0; i < layer->output_size; i++) {
                double sum = 0;
                for (int j = 0; j < next->output_size; j++)
                    sum += delta[j] *
                           next->weights[j * next->input_size + i];

                new_delta[i] =
                    sum * relu_derivative(layer->z[i]);
            }
        }

        double *prev_a = (l == 0) ? input
                                  : nn->layers[l - 1].a;

        for (int i = 0; i < layer->output_size; i++) {

            for (int j = 0; j < layer->input_size; j++) {

                int idx = i * layer->input_size + j;
                double grad =
                    new_delta[i] * prev_a[j] +
                    L2_LAMBDA * layer->weights[idx];

                adam_update(layer->weights,
                            layer->m_w,
                            layer->v_w,
                            grad, idx, timestep);
            }

            adam_update(layer->bias,
                        layer->m_b,
                        layer->v_b,
                        new_delta[i], i, timestep);
        }

        free(delta);
        delta = new_delta;
    }

    free(delta);
}

// ----------------------------
// CSV Loader
// ----------------------------

int load_csv(const char *filename,
             double inputs[][INPUT_SIZE],
             double targets[][OUTPUT_SIZE]) {

    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Cannot open file\n");
        exit(1);
    }

    char line[MAX_LINE];
    fgets(line, sizeof(line), file);

    int row = 0;

    while (fgets(line, sizeof(line), file) &&
           row < MAX_SAMPLES) {

        char *token = strtok(line, ",");

        for (int i = 0; i < INPUT_SIZE; i++) {
            inputs[row][i] = atof(token);
            token = strtok(NULL, ",");
        }

        targets[row][0] = atof(token);
        row++;
    }

    fclose(file);
    return row;
}

// ----------------------------
// Normalize
// ----------------------------

void normalize(double data[][INPUT_SIZE],
               int n,
               double *min,
               double *max) {

    for (int j = 0; j < INPUT_SIZE; j++) {
        min[j] = data[0][j];
        max[j] = data[0][j];

        for (int i = 1; i < n; i++) {
            if (data[i][j] < min[j])
                min[j] = data[i][j];
            if (data[i][j] > max[j])
                max[j] = data[i][j];
        }

        for (int i = 0; i < n; i++)
            data[i][j] =
                (data[i][j] - min[j]) /
                (max[j] - min[j] + 1e-9);
    }
}

// ----------------------------
// Save / Load Model
// ----------------------------

void save_model(NeuralNetwork *nn, const char *file) {
    FILE *f = fopen(file, "wb");

    fwrite(&nn->num_layers, sizeof(int), 1, f);

    for (int l = 0; l < nn->num_layers; l++) {
        Layer *layer = &nn->layers[l];

        fwrite(&layer->input_size, sizeof(int), 1, f);
        fwrite(&layer->output_size, sizeof(int), 1, f);

        fwrite(layer->weights,
               sizeof(double),
               layer->input_size * layer->output_size, f);

        fwrite(layer->bias,
               sizeof(double),
               layer->output_size, f);
    }

    fclose(f);
}

void load_model(NeuralNetwork *nn, const char *file) {
    FILE *f = fopen(file, "rb");

    int layers;
    fread(&layers, sizeof(int), 1, f);

    for (int l = 0; l < layers; l++) {
        Layer *layer = &nn->layers[l];

        fread(layer->weights,
              sizeof(double),
              layer->input_size * layer->output_size, f);

        fread(layer->bias,
              sizeof(double),
              layer->output_size, f);
    }

    fclose(f);
}

// ----------------------------
// MAIN
// ----------------------------

int main() {

    srand(time(NULL));

    static double inputs[MAX_SAMPLES][INPUT_SIZE];
    static double targets[MAX_SAMPLES][OUTPUT_SIZE];

    int n = load_csv("housing.csv",
                     inputs, targets);

    printf("Loaded %d samples\n", n);

    double min[INPUT_SIZE], max[INPUT_SIZE];
    normalize(inputs, n, min, max);

    int train_size = n * 0.8;

    int sizes[] = {INPUT_SIZE, 32, 16, 8, 1};
    NeuralNetwork nn =
        create_network(sizes, 5);

    int timestep = 1;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        double loss = 0;

        for (int i = 0; i < train_size; i++) {

            forward(&nn, inputs[i]);

            double error =
                nn.layers[nn.num_layers - 1].a[0]
                - targets[i][0];

            loss += error * error;

            backward(&nn,
                     inputs[i],
                     targets[i],
                     timestep++);

        }

        printf("Epoch %d | Loss: %.6f\n",
               epoch, loss / train_size);
    }

    save_model(&nn, "model.bin");

    printf("Model saved.\n");

    return 0;
}