#include <blust/models/Model.hpp>


START_BLUST_NAMESPACE

/**
 * @brief Prepare the model for training
 * @brief learning_rate: The learning rate to be used in the model
 */
void Model::compile(number_t learning_rate, error_funcs loss)
{
    m_learning_rate = learning_rate;
    m_error_func.reset(get_error_functor(loss));


    // Assert correct layer connections and types
    BLUST_ASSERT(dynamic_cast<Input*>(m_input_layer) != nullptr);
    BLUST_ASSERT(m_input_layer->m_next != nullptr);
    
    BaseLearningLayer* layer = dynamic_cast<BaseLearningLayer*>(m_input_layer->m_next);
    BLUST_ASSERT(layer != nullptr);

    while(true)
    {
        // If that's the last layer, it has to be the output layer
        if (layer->m_next == nullptr)
        {
            BLUST_ASSERT(layer == m_output_layer);
            break;
        }
        
        // Traverse through the layer list
        auto next = dynamic_cast<BaseLearningLayer*>(layer->m_next);
        BLUST_ASSERT(next != nullptr);
        BLUST_ASSERT(next->m_prev == layer);

        layer = next;
    }
}

void Model::call(matrix_t& inputs)
{
    BaseLayer* next = m_input_layer;
    matrix_t* p_inputs = &inputs; // avoid too much copying

    while (next != nullptr) {
        p_inputs = &next->feed_forward(*p_inputs);
        next = next->m_next;
    }
}

void Model::backprop(matrix_t& expected)
{
    auto layer      = dynamic_cast<BaseLearningLayer*>(m_output_layer);
    auto prev       = dynamic_cast<BaseLearningLayer*>(m_output_layer->m_prev);
    auto prev_input = &prev->m_activations;

    // Calculate the output gradient
    layer->gradient(*prev_input, expected, m_error_func);

    while (prev != nullptr)
    {
        layer       = prev;
        prev        = dynamic_cast<BaseLearningLayer*>(layer->m_prev);
        prev_input  = &layer->m_prev->m_activations;
        layer->gradient(*prev_input);
    }
}

void Model::fit()
{
    
}

void Model::apply_gradients()
{
	BaseLearningLayer* layer =
		dynamic_cast<BaseLearningLayer*>(m_input_layer->m_next);
	while (layer != nullptr)
	{
		layer->apply(m_learning_rate);
		layer = dynamic_cast<BaseLearningLayer*>(layer->m_next);
	}
}

void Model::train_on_batch(batch_t& inputs, batch_t& expected)
{
    // Backpropagate the gradients
    for (size_t i = 0; i < inputs.size(); i++) {
        call(inputs[i]);
        backprop(expected[i]);
    }

    // Apply the gradients
	apply_gradients();

    std::cout << "cost=" << dynamic_cast<BaseLearningLayer*>(m_output_layer)->cost(expected[expected.size() - 1], m_error_func) << '\n';
}

END_BLUST_NAMESPACE