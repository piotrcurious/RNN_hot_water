// ... (previous code remains the same up to the LSTMCell class)

// Complete LSTM cell implementation
void LSTMCell::forward(const std::vector<float>& input, 
                      std::vector<float>& hidden_state,
                      std::vector<float>& cell_state,
                      bool training) {
    std::vector<float> combined_input(input_size_ + hidden_size_);
    std::copy(input.begin(), input.end(), combined_input.begin());
    std::copy(hidden_state.begin(), hidden_state.end(), combined_input.begin() + input_size_);

    // Gate activations
    std::vector<float> forget_gate(hidden_size_);
    std::vector<float> input_gate(hidden_size_);
    std::vector<float> cell_gate(hidden_size_);
    std::vector<float> output_gate(hidden_size_);

    // Calculate gates
    for (int h = 0; h < hidden_size_; h++) {
        float f = 0.0f, i = 0.0f, c = 0.0f, o = 0.0f;
        
        for (int j = 0; j < input_size_ + hidden_size_; j++) {
            f += combined_input[j] * weights_.Wf[j][h];
            i += combined_input[j] * weights_.Wi[j][h];
            c += combined_input[j] * weights_.Wc[j][h];
            o += combined_input[j] * weights_.Wo[j][h];
        }

        forget_gate[h] = ActivationFunctions::sigmoid(f + weights_.bf[h]);
        input_gate[h] = ActivationFunctions::sigmoid(i + weights_.bi[h]);
        cell_gate[h] = ActivationFunctions::tanh(c + weights_.bc[h]);
        output_gate[h] = ActivationFunctions::sigmoid(o + weights_.bo[h]);
    }

    // Update cell state
    for (int h = 0; h < hidden_size_; h++) {
        cell_state[h] = forget_gate[h] * cell_state[h] + 
                       input_gate[h] * cell_gate[h];
    }

    // Calculate hidden state with dropout
    std::vector<float> raw_hidden(hidden_size_);
    for (int h = 0; h < hidden_size_; h++) {
        raw_hidden[h] = output_gate[h] * ActivationFunctions::tanh(cell_state[h]);
        if (training) {
            if (Utils::random_uniform() < Config::Training::DROPOUT_RATE) {
                raw_hidden[h] = 0;
            } else {
                raw_hidden[h] *= (1.0f - Config::Training::DROPOUT_RATE);
            }
        }
    }

    // Apply layer normalization
    layer_norm_.normalize(raw_hidden, hidden_state);

    // Store states for backward pass if training
    if (training) {
        cache_ = {
            combined_input,
            forget_gate,
            input_gate,
            cell_gate,
            output_gate,
            raw_hidden
        };
    }
}

void LSTMCell::backward(const std::vector<float>& input,
                       const std::vector<float>& gradient,
                       std::vector<float>& input_gradient) {
    // Initialize weight gradients
    WeightGradients grads;
    
    // Backpropagate through layer normalization
    std::vector<float> pre_norm_gradient(hidden_size_);
    layer_norm_.backward(cache_.raw_hidden, gradient, pre_norm_gradient);

    // Backpropagate through gates
    for (int h = 0; h < hidden_size_; h++) {
        // Output gate gradient
        float do_h = pre_norm_gradient[h] * 
                    ActivationFunctions::tanh(cache_.cell_state[h]) *
                    ActivationFunctions::sigmoid_derivative(cache_.output_gate[h]);

        // Cell state gradient
        float dc_h = pre_norm_gradient[h] * cache_.output_gate[h] *
                    ActivationFunctions::tanh_derivative(cache_.cell_state[h]);
        
        // Add gradient from next timestep if available
        if (!next_cell_gradient_.empty()) {
            dc_h += next_cell_gradient_[h];
        }

        // Input gate gradient
        float di_h = dc_h * cache_.cell_gate[h] *
                    ActivationFunctions::sigmoid_derivative(cache_.input_gate[h]);

        // Forget gate gradient
        float df_h = dc_h * prev_cell_state_[h] *
                    ActivationFunctions::sigmoid_derivative(cache_.forget_gate[h]);

        // Cell gate gradient
        float dc_tilde_h = dc_h * cache_.input_gate[h] *
                          ActivationFunctions::tanh_derivative(cache_.cell_gate[h]);

        // Accumulate weight gradients
        for (int i = 0; i < input_size_ + hidden_size_; i++) {
            grads.dWf[i][h] = df_h * cache_.combined_input[i];
            grads.dWi[i][h] = di_h * cache_.combined_input[i];
            grads.dWc[i][h] = dc_tilde_h * cache_.combined_input[i];
            grads.dWo[i][h] = do_h * cache_.combined_input[i];
        }

        // Bias gradients
        grads.dbf[h] = df_h;
        grads.dbi[h] = di_h;
        grads.dbc[h] = dc_tilde_h;
        grads.dbo[h] = do_h;
    }

    // Apply gradient clipping
    clip_gradients(grads);

    // Update weights using Adam optimizer
    update_weights(grads);

    // Calculate input gradients for previous layer
    input_gradient.resize(input_size_);
    std::fill(input_gradient.begin(), input_gradient.end(), 0.0f);
    
    for (int i = 0; i < input_size_; i++) {
        for (int h = 0; h < hidden_size_; h++) {
            input_gradient[i] += 
                grads.dWf[i][h] * weights_.Wf[i][h] +
                grads.dWi[i][h] * weights_.Wi[i][h] +
                grads.dWc[i][h] * weights_.Wc[i][h] +
                grads.dWo[i][h] * weights_.Wo[i][h];
        }
    }
}

// Network implementation
float Network::forward(const std::vector<std::vector<float>>& sequence, bool training) {
    std::vector<float> cell_state(Config::Network::HIDDEN_SIZE, 0.0f);
    std::vector<float> hidden_state(Config::Network::HIDDEN_SIZE, 0.0f);
    
    // Process sequence through LSTM layers
    for (const auto& input : sequence) {
        std::vector<float> layer_input = input;
        
        for (auto& layer : layers_) {
            layer->forward(layer_input, hidden_state, cell_state, training);
            layer_input = hidden_state;
        }
    }

    // Calculate final output
    float output = output_bias_;
    for (int h = 0; h < Config::Network::HIDDEN_SIZE; h++) {
        output += hidden_state[h] * output_weights_[h];
    }

    return ActivationFunctions::sigmoid(output);
}

void Network::backward(const std::vector<std::vector<float>>& sequence, float target) {
    float output = forward(sequence, true);
    float output_error = output - target;
    
    // Output layer gradients
    std::vector<float> gradient(Config::Network::HIDDEN_SIZE);
    float output_delta = output_error * ActivationFunctions::sigmoid_derivative(output);
    
    for (int h = 0; h < Config::Network::HIDDEN_SIZE; h++) {
        gradient[h] = output_delta * output_weights_[h];
        
        // Update output weights
        float weight_grad = output_delta * last_hidden_[h];
        update_adam(output_weights_[h], weight_grad, output_momentum_[h], output_velocity_[h]);
    }

    // Update output bias
    update_adam(output_bias_, output_delta, bias_momentum_, bias_velocity_);

    // Backpropagate through LSTM layers
    for (int i = layers_.size() - 1; i >= 0; i--) {
        std::vector<float> layer_gradient;
        layers_[i]->backward(sequence[i], gradient, layer_gradient);
        gradient = layer_gradient;
    }
}

// Trainer implementation
void Trainer::train_batch(const std::vector<std::vector<std::vector<float>>>& batch,
                         const std::vector<bool>& targets) {
    if (batch.size() != targets.size()) {
        throw NetworkException("Batch size mismatch with targets");
    }

    float total_presence_loss = 0.0f;
    float total_service_loss = 0.0f;

    for (size_t i = 0; i < batch.size(); i++) {
        // Train presence network
        float presence_pred = presence_net_.forward(batch[i], true);
        float presence_loss = -targets[i] * log(presence_pred) - 
                            (1 - targets[i]) * log(1 - presence_pred);
        total_presence_loss += presence_loss;
        presence_net_.backward(batch[i], targets[i]);

        // Train service network
        float service_pred = service_net_.forward(batch[i], true);
        float service_loss = -targets[i] * log(service_pred) - 
                           (1 - targets[i]) * log(1 - service_pred);
        total_service_loss += service_loss;
        service_net_.backward(batch[i], targets[i]);
    }

    // Log training progress
    if (++training_steps_ % 100 == 0) {
        Serial.printf("Step %lu: Presence Loss: %.4f, Service Loss: %.4f\n",
                     training_steps_,
                     total_presence_loss / batch.size(),
                     total_service_loss / batch.size());
    }
}

// HotWaterManager implementation
void HotWaterManager::update(unsigned long current_time) {
    auto time_features = Utils::TimeFeatures::extract();
    bool pir_detected = HardwareInterface::read_pir();
    bool service_requested = HardwareInterface::read_service();

    std::vector<float> current_input = {
        time_features.time_of_day,
        time_features.day_of_week,
        static_cast<float>(pir_detected),
        time_features.sin_time,
        time_features.cos_time
    };

    // Create sequence from recent history
    std::vector<std::vector<float>> sequence;
    if (history_.size() >= Config::Network::SEQUENCE_LENGTH) {
        sequence.assign(
            history_.end() - Config::Network::SEQUENCE_LENGTH,
            history_.end()
        );
    } else {
        sequence = std::vector<std::vector<float>>(
            Config::Network::SEQUENCE_LENGTH,
            current_input
        );
    }

    // Make predictions
    float presence_prob = presence_net_.forward(sequence);
    float service_prob = service_net_.forward(sequence);

    // Apply business logic
    if (service_requested) {
        presence_prob = std::min(1.0f, presence_prob * 1.5f);
    }

    // Update outputs
    HardwareInterface::set_presence_pwm(presence_prob);
    HardwareInterface::set_service_pwm(service_prob);
}

void HotWaterManager::sample_and_train(unsigned long current_time) {
    static unsigned long last_sample_time = 0;

    if (current_time - last_sample_time >= Config::Training::SAMPLE_INTERVAL) {
        last_sample_time = current_time;

        auto features = Utils::TimeFeatures::extract();
        std::vector<float> current_input = {
            features.time_of_day,
            features.day_of_week,
            static_cast<float>(HardwareInterface::read_pir()),
            features.sin_time,
            features.cos_time
        };

        history_.push_back(current_input);

        // Maintain history size
        if (history_.size() > Config::Network::SEQUENCE_LENGTH * Config::Network::BATCH_SIZE * 2) {
            history_.erase(history_.begin());
        }

        // Create training batch when enough data is available
        if (history_.size() >= Config::Network::SEQUENCE_LENGTH) {
            create_training_batch();
        }
    }
}

void HotWaterManager::debug_output(unsigned long current_time) {
    static unsigned long last_debug_time = 0;

    if (current_time - last_debug_time >= Config::Training::DEBUG_INTERVAL) {
        last_debug_time = current_time;

        auto features = Utils::TimeFeatures::extract();
        bool pir_detected = HardwareInterface::read_pir();
        bool service_requested = HardwareInterface::read_service();

        std::vector<std::vector<float>> sequence(1, std::vector<float>{
            features.time_of_day,
            features.day_of_week,
            static_cast<float>(pir_detected),
            features.sin_time,
            features.cos_time
        });

        float presence_prob = presence_net_.forward(sequence);
        float service_prob = service_net_.forward(sequence);

        Serial.printf("Time: %.2f Day: %.1f PIR: %d SRV: %d P: %.3f S: %.3f\n",
                     features.time_of_day,
                     features.day_of_week,
                     pir_detected,
                     service_requested,
                     presence_prob,
                     service_prob);
    }
}

// ... (rest of the code remains the same)
