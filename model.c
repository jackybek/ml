#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define train_count (sizeof(train) / sizeof(train[0]));

// training data set : value pair {input, output}
// predict the relationship between input and output
float train[][2] = {
        {0, 0},
        {1, 2},
        {2, 4},
        {3, 6},
        {4, 8},
};

int main(void);
float rand_float(void);

float rand_float()
{
        // gives a random number between 0 and 1
        return (float) rand() / (float) RAND_MAX;
}

float cost(float w, float b)
{
        // how well the model perform
        // the closer to 0 the better the model
        float result = 0.0f;
        for (size_t i=0; i<500; ++i) {
                float x = train[i][0];
                float y = x*w + b;              // rergression model : y = f(x) + c
                float d = y - train[i][1];
                result += d*d;
        }
        result /= train_count;
        return result;
}

int main()
{
// the initial model based on training data has the form y = x * w, where w is the parameter (const)

        // srand(time(0));
        srand(100);
        float w = rand_float() * 10.0f; // weight
        float b = rand_float() * 10.0f; // bias

        float eps = 1e-3; // shift by a margin of error
        float rate = 1e-3;// learning rate

        printf("%f\n", cost(w, b));
        for (size_t i = 0; i<1500; ++i) {
                float c = cost(w, b);
                float dw = ( cost(w + eps, b) - c ) / eps;      // difference in weight
                float db = ( cost(w, b + eps) - c ) / eps;      // difference in bias
                w -= rate*dw;
                b -= rate*db;
                printf("cost = %f, w = %f, b(bias) = %f\n", cost(w, b), w, b);
        }
        printf("--------------------------------\n");
        printf("w = %f, b = %f \n", w, b);

        return 0;
}

