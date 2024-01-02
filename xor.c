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

sample xor_train[] = {
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0},
};

sample nor_train[] ={
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 0},
};

typedef struct {
        // layer 1 neuron
        float or_w1;
        float or_w2;
        float or_b;

        // layer 2 neuron
        float nand_w1;
        float nand_w2;
        float nand_b;

        // layer 3 neuron
        float and_w1;
        float and_w2;
        float and_b;
} Xor;

sample *train = xor_train;
size_t train_count = 4;
int main(int, char **);
float forward(Xor, float, float);
float rand_float(void);
float cost(Xor);
float sigmoidf(float);
Xor finite_diff(Xor, float);

float forward(Xor m, float x1, float x2)
{
        // define the model : XOR
        // step 1 : x1 OR x2 = Y
        // step 2 : x1 NAND x2 = W
        // step 3 : Y AND W
        float a = sigmoidf(m.or_w1*x1 + m.or_w2*x2 + m.or_b); // OR model
        float b = sigmoidf(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b); // nand model

        return sigmoidf(a*m.and_w1 + b*m.and_w2 + m.and_b);

}

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

float cost(Xor m)
{
        // how well the model perform
        // the closer to 0 the better the model
        float result = 0.0f;
        for (size_t i = 0; i < train_count; ++i) {
                float x1 = train[i][0];
                float x2 = train[i][1];
                float y = forward(m, x1,x2);            // rergression model : y = f(x) + c
                float d = y - train[i][2];
                result += d*d;
        }
        result /= train_count;
        return result;
}

Xor rand_xor()
{
        Xor m;
        // layer 1 neuron
        m.or_w1 = rand_float();
        m.or_w2 = rand_float();
        m.or_b = rand_float();

        // layer 2 neuron
        m.nand_w1 = rand_float();
        m.nand_w2 = rand_float();
        m.nand_b = rand_float();

        // layer 3 neuron
        m.and_w1 = rand_float();
        m.and_w2 = rand_float();
        m.and_b = rand_float();

        return m;
}

void print_xor(Xor m)
{
        // layer 1 neuron
        printf("or_w1 = %f\n", m.or_w1);
        printf("or_w2 = %f\n", m.or_w2);
        printf("or_b  = %f\n", m.or_b);

        // layer 2 neuron
        printf("nand_w1 = %f\n", m.nand_w1);
        printf("nand_w2 = %f\n", m.nand_w2);
        printf("nand_b  = %f\n", m.nand_b);

        // layer 3 neuron
        printf("and_w1 = %f\n", m.and_w1);
        printf("and_w1 = %f\n", m.and_w2);
        printf("and_b  = %f\n", m.and_b);
}

Xor finite_diff(Xor m, float eps)
{
        Xor g;
        float c = cost(m);
        float saved;

        // OR
        saved = m.or_w1;
        m.or_w1 += eps;
        g.or_w1 = (cost(m) - c) / eps;
        m.or_w1 = saved;

        saved = m.or_w2;
        m.or_w2 += eps;
        g.or_w2 = (cost(m) - c) / eps;
        m.or_w2 = saved;

        saved = m.or_b;
        m.or_b += eps;
        g.or_b = (cost(m) - c) / eps;
        m.or_b = saved;

        // NAND
        saved = m.nand_w1;
        m.nand_w1 += eps;
        g.nand_w1 = (cost(m) - c) / eps;
        m.nand_w1 = saved;

        saved = m.nand_w2;
        m.nand_w2 += eps;
        g.nand_w2 = (cost(m) - c) / eps;
        m.nand_w2 = saved;

        saved = m.nand_b;
        m.nand_b += eps;
        g.nand_b = (cost(m) - c) / eps;
        m.nand_b = saved;

        // AND
        saved = m.and_w1;
        m.and_w1 += eps;
        g.and_w1 = (cost(m) - c) / eps;
        m.and_w1 = saved;

        saved = m.and_w2;
        m.and_w2 += eps;
        g.and_w2 = (cost(m) - c) / eps;
        m.and_w2 = saved;

        saved = m.and_b;
        m.and_b += eps;
        g.and_b = (cost(m) - c) / eps;
        m.and_b = saved;

        return g;
}

Xor learn(Xor m, Xor g, float rate)
{
        m.or_w1 -= rate * g.or_w1;
        m.or_w2 -= rate * g.or_w2;
        m.or_b  -= rate * g.or_b;

        m.nand_w1 -= rate * g.nand_w1;
        m.nand_w2 -= rate * g.nand_w2;
        m.nand_b -= rate * g.nand_b;

        m.and_w1 -= rate * g.and_w1;
        m.and_w2 -= rate * g.and_w2;
        m.and_b  -= rate * g.and_b;

        return m;
}

int main(int argc, char** argv)
{
// the initial model based on training data has the form y = x * w, where w is the parameter (const)

        FILE *fp;
        float eps = 1e-1;
        float rate = 1e-1;

        srand(time(0));
        Xor m = rand_xor();
        //print_xor(m);
        //printf("---------------------------------\n");
        //print_xor(finite_diff(m, eps));
        for (size_t i = 0; i < 1000*1000; i++)
        {
                Xor g = finite_diff(m, eps);
                m = learn(m, g, rate);
        }
        printf("cost = %f\n", cost(m));

        for (size_t i = 0; i < 2; i++)
                for (size_t j=0; j < 2; j++)
                        printf("%zu XOR %zu = %f\n", i,j, forward(m,i,j)); 
//      printf("cost = %f \n", cost(m));

        printf("-------------------------------------\n");
        printf("truth table of -OR- neuron: \n");
        for (size_t i = 0; i < 2; i++)
                for (size_t j=0; j < 2; j++)
                        printf("%zu OP %zu = %f\n", i, j, sigmoidf(m.or_w1*i + m.or_w2*j + m.or_b));

        printf("-------------------------------------\n");
        printf("truth table of -AND- neuron: \n");
        for (size_t i = 0; i < 2; i++)
                for (size_t j=0; j < 2; j++)
                        printf("%zu OP %zu = %f\n", i, j, sigmoidf(m.and_w1*i + m.and_w2*j + m.and_b));

        printf("-------------------------------------\n");
        printf("truth table of -NAND- neuron: \n");
        for (size_t i = 0; i < 2; i++)
                for (size_t j=0; j < 2; j++)
                        printf("%zu OP %zu = %f\n", i, j, sigmoidf(m.nand_w1*i + m.nand_w2*j + m.nand_b));



        return 0;

}
