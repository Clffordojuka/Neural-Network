// tiny_nn.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define MAX_SAMPLES 30000
#define MAX_LINE 2048

#define INPUT_SIZE 8
#define OUTPUT_SIZE 1

#define EPOCHS 50
#define BATCH_SIZE 32

#define LEARNING_RATE 0.001
#define L2_LAMBDA 0.0001

#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

//--------------------------------
// Layer
//--------------------------------

typedef struct {

    int in;
    int out;

    double *W;
    double *b;

    double *z;
    double *a;

    double *gradW;
    double *gradb;

    double *mW;
    double *vW;
    double *mb;
    double *vb;

} Layer;

//--------------------------------
// Network
//--------------------------------

typedef struct {

    int layers;
    Layer *layer;

} NeuralNetwork;

//--------------------------------
// Utility
//--------------------------------

double rand_uniform() {
    return ((double)rand() / RAND_MAX) - 0.5;
}

double relu(double x) { return x > 0 ? x : 0; }

double relu_derivative(double x) { return x > 0 ? 1 : 0; }

//--------------------------------
// Matrix Vector Multiply
//--------------------------------

void matvec(double *W, double *x, double *y, int rows, int cols)
{
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {

        double sum = 0;

        for (int j = 0; j < cols; j++)
            sum += W[i * cols + j] * x[j];

        y[i] = sum;
    }
}

//--------------------------------
// Layer Initialization
//--------------------------------

void init_layer(Layer *l, int in, int out)
{
    l->in = in;
    l->out = out;

    l->W = malloc(sizeof(double)*in*out);
    l->b = calloc(out,sizeof(double));

    l->z = malloc(sizeof(double)*out);
    l->a = malloc(sizeof(double)*out);

    l->gradW = calloc(in*out,sizeof(double));
    l->gradb = calloc(out,sizeof(double));

    l->mW = calloc(in*out,sizeof(double));
    l->vW = calloc(in*out,sizeof(double));

    l->mb = calloc(out,sizeof(double));
    l->vb = calloc(out,sizeof(double));

    double scale = sqrt(2.0/in);

    for(int i=0;i<in*out;i++)
        l->W[i] = rand_uniform()*2*scale;
}

//--------------------------------
// Create Network
//--------------------------------

NeuralNetwork create_network(int *sizes, int count)
{
    NeuralNetwork nn;

    nn.layers = count-1;
    nn.layer = malloc(sizeof(Layer)*nn.layers);

    for(int i=0;i<nn.layers;i++)
        init_layer(&nn.layer[i],sizes[i],sizes[i+1]);

    return nn;
}

//--------------------------------
// Forward Pass
//--------------------------------

void forward(NeuralNetwork *nn,double *input)
{
    double *x=input;

    for(int l=0;l<nn->layers;l++)
    {
        Layer *layer=&nn->layer[l];

        matvec(layer->W,x,layer->z,layer->out,layer->in);

        for(int i=0;i<layer->out;i++)
        {
            layer->z[i]+=layer->b[i];

            if(l==nn->layers-1)
                layer->a[i]=layer->z[i];
            else
                layer->a[i]=relu(layer->z[i]);
        }

        x=layer->a;
    }
}

//--------------------------------
// Adam Update
//--------------------------------

void adam(double *w,double *m,double *v,double g,int idx,int t)
{
    m[idx]=BETA1*m[idx]+(1-BETA1)*g;
    v[idx]=BETA2*v[idx]+(1-BETA2)*g*g;

    double mh=m[idx]/(1-pow(BETA1,t));
    double vh=v[idx]/(1-pow(BETA2,t));

    w[idx]-=LEARNING_RATE*mh/(sqrt(vh)+EPSILON);
}

//--------------------------------
// Backprop (gradient accumulate)
//--------------------------------

void backward(NeuralNetwork *nn,double *input,double *target)
{
    int L=nn->layers;

    double *delta=NULL;

    for(int l=L-1;l>=0;l--)
    {
        Layer *layer=&nn->layer[l];

        double *new_delta=malloc(sizeof(double)*layer->out);

        if(l==L-1)
        {
            for(int i=0;i<layer->out;i++)
                new_delta[i]=layer->a[i]-target[i];
        }
        else
        {
            Layer *next=&nn->layer[l+1];

            for(int i=0;i<layer->out;i++)
            {
                double sum=0;

                for(int j=0;j<next->out;j++)
                    sum+=delta[j]*next->W[j*next->in+i];

                new_delta[i]=sum*relu_derivative(layer->z[i]);
            }
        }

        double *prev=(l==0)?input:nn->layer[l-1].a;

        for(int i=0;i<layer->out;i++)
        {
            for(int j=0;j<layer->in;j++)
            {
                int idx=i*layer->in+j;

                layer->gradW[idx]+=new_delta[i]*prev[j];
            }

            layer->gradb[i]+=new_delta[i];
        }

        free(delta);
        delta=new_delta;
    }

    free(delta);
}

//--------------------------------
// Apply Batch Update
//--------------------------------

void apply_batch(Layer *layer,int t)
{
    int size=layer->in*layer->out;

    for(int i=0;i<size;i++)
    {
        double g=layer->gradW[i]/BATCH_SIZE+
                 L2_LAMBDA*layer->W[i];

        adam(layer->W,layer->mW,layer->vW,g,i,t);

        layer->gradW[i]=0;
    }

    for(int i=0;i<layer->out;i++)
    {
        double g=layer->gradb[i]/BATCH_SIZE;

        adam(layer->b,layer->mb,layer->vb,g,i,t);

        layer->gradb[i]=0;
    }
}

//--------------------------------
// CSV Loader
//--------------------------------

int load_csv(const char *file,
             double X[][INPUT_SIZE],
             double y[][OUTPUT_SIZE])
{
    FILE *f=fopen(file,"r");

    if(!f){
        printf("File not found\n");
        exit(1);
    }

    char line[MAX_LINE];

    fgets(line,sizeof(line),f);

    int row=0;

    while(fgets(line,sizeof(line),f)&&row<MAX_SAMPLES)
    {
        char *tok=strtok(line,",");

        for(int i=0;i<INPUT_SIZE;i++)
        {
            X[row][i]=atof(tok);
            tok=strtok(NULL,",");
        }

        y[row][0]=atof(tok);

        row++;
    }

    fclose(f);
    return row;
}

//--------------------------------
// Normalize
//--------------------------------

void normalize(double X[][INPUT_SIZE],int n)
{
    double min[INPUT_SIZE];
    double max[INPUT_SIZE];

    for(int j=0;j<INPUT_SIZE;j++)
    {
        min[j]=X[0][j];
        max[j]=X[0][j];

        for(int i=1;i<n;i++)
        {
            if(X[i][j]<min[j])min[j]=X[i][j];
            if(X[i][j]>max[j])max[j]=X[i][j];
        }

        for(int i=0;i<n;i++)
            X[i][j]=(X[i][j]-min[j])/(max[j]-min[j]+1e-9);
    }
}

//--------------------------------
// Shuffle Dataset
//--------------------------------

void shuffle(double X[][INPUT_SIZE],
             double y[][OUTPUT_SIZE],
             int n)
{
    for(int i=n-1;i>0;i--)
    {
        int j=rand()%(i+1);

        for(int k=0;k<INPUT_SIZE;k++)
        {
            double tmp=X[i][k];
            X[i][k]=X[j][k];
            X[j][k]=tmp;
        }

        double t=y[i][0];
        y[i][0]=y[j][0];
        y[j][0]=t;
    }
}

//--------------------------------
// Save Model
//--------------------------------

void save_model(NeuralNetwork *nn,const char *file)
{
    FILE *f=fopen(file,"wb");

    fwrite(&nn->layers,sizeof(int),1,f);

    for(int l=0;l<nn->layers;l++)
    {
        Layer *layer=&nn->layer[l];

        fwrite(layer->W,sizeof(double),
               layer->in*layer->out,f);

        fwrite(layer->b,sizeof(double),
               layer->out,f);
    }

    fclose(f);
}

//--------------------------------
// MAIN
//--------------------------------

int main()
{
    srand(time(NULL));

    static double X[MAX_SAMPLES][INPUT_SIZE];
    static double y[MAX_SAMPLES][OUTPUT_SIZE];

    int n=load_csv("housing.csv",X,y);

    printf("Loaded %d samples\n",n);

    normalize(X,n);

    int train=n*0.8;

    int sizes[]={INPUT_SIZE,32,16,8,1};

    NeuralNetwork nn=create_network(sizes,5);

    int t=1;

    for(int epoch=0;epoch<EPOCHS;epoch++)
    {
        shuffle(X,y,train);

        double loss=0;

        for(int i=0;i<train;i++)
        {
            forward(&nn,X[i]);

            double err=
            nn.layer[nn.layers-1].a[0]-y[i][0];

            loss+=err*err;

            backward(&nn,X[i],y[i]);

            if((i+1)%BATCH_SIZE==0)
            {
                for(int l=0;l<nn.layers;l++)
                    apply_batch(&nn.layer[l],t);

                t++;
            }
        }

        printf("Epoch %d | Loss %.6f\n",
               epoch,loss/train);
    }

    save_model(&nn,"model.bin");

    printf("Model saved\n");

    return 0;
}