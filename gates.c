#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define GRAPH

//#define train_count (sizeof(train) / sizeof(train[0]))

// training data set : value pair {input, output}
// predict the relationship between input and output
// OR-gate / AND-gate
typedef float sample[3];

sample or_train[] = {
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1},
};
sample and_train[] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 1},
};

sample nand_train[] = {
        {0, 0, 1},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0},
};


sample *train = nand_train;
size_t train_count = 4;

int main(int, char **);
float rand_float(void);
float cost(float, float, float);

loat sigmoid(float);

float sigmoidf(float x) // sigmoid function
{
        // squeeze everything between 0 and 1
        return 1.f / (1.f + expf(-x));
}

float rand_float()
{
        // gives a random number between 0 and 1
        return (float) rand() / (float) RAND_MAX;
}

float cost(float w1, float w2, float b)
{
        // how well the model perform
        // the closer to 0 the better the model
        float result = 0.0f;
        for (size_t i = 0; i < train_count; ++i) {
                float x1 = train[i][0];
                float x2 = train[i][1];
                float y = sigmoidf( x1*w1 + x2*w2 + b);         // rergression model : y = f(x) + c
                float d = y - train[i][2];
                result += d*d;
        }
        result /= train_count;
        return result;
}

int main(int argc, char** argv)
{
// the initial model based on training data has the form y = x * w, where w is the parameter (const)

        FILE *fp;
        // srand(time(0));
        srand(100);
        float w1 = rand_float();        // weight
        float w2 = rand_float();        // weight
        float b = rand_float(); // bias

        float eps = 1e-1; // shift by a margin of error
        float rate = 1e-1;// learning rate

#ifdef GRAPH
        fp = fopen("cost.txt", "w+");
#endif
        // train the model with more data by increasing i
        for (size_t i = 0; i<100*1000; ++i) {
                float c = cost(w1, w2, b);
#ifdef GRAPH
        fprintf(fp, "%f\n", c);
#endif
                printf("w1 = %f, w2 = %f, b = %f, c = %f\n", w1, w2, b, c);
                float dw1 = ( cost(w1+eps, w2, b) - c ) / eps;  // difference in weight
                float dw2 = ( cost(w1, w2+eps , b) - c ) / eps;  // difference in weight
                float db  = ( cost(w1, w2, b + eps) - c ) / eps; // difference in bias
                w1 -= rate*dw1;
                w2 -= rate*dw2;
                b -= rate*db;
        }

        printf("w1 = %f, w2 = %f, b = %f, c = %f \n", w1, w2, b, cost(w1, w2, b));
        for (size_t i=0; i < 2; i++)
                for (size_t j=0; j < 2; j++)
                        printf("%zu <OP> %zu = %f \n", i, j, sigmoidf(i*w1 + j*w2 + b));

#ifdef GRAPH
        fclose(fp);
#endif
        return 0;
}

