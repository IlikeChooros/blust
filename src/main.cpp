#include <blust/blust.hpp>
#include <chrono>
#include <iostream>

using namespace blust;

int main(int argc, char** argv)
{
    init(argc, argv, "");

	Input input     = Input({ 1, 768 });
    Dense hidden    = Dense(2048, relu)(input);
    Dense hidden2   = Dense(512, relu)(hidden);
    Dense hidden3   = Dense(128, relu)(hidden2);
    Dense hidden4   = Dense(512, relu)(hidden3);
    Dense hidden5   = Dense(64, relu)(hidden4);
    Dense feature   = Dense(2, softmax)(hidden5);
    matrix_t inputs({1, 768 }, 0.5f);

	utils::randomize(inputs.begin(), inputs.end(), inputs.size());

    hidden.randomize();
    feature.randomize();

    Model model(&input, &feature);
    model.compile(0.1);

    matrix_t expected{{0, 1}};
    batch_t batch_input = {inputs};
    batch_t batch_expected = {expected};

	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 10; i++)
		model.train_on_batch(batch_input, batch_expected);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;
	printf("avg time: %f ms\n", duration.count() / 10.0f);

    return 0;
}