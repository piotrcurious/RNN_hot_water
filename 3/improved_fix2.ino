#include <Arduino.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>

// ... (previous Config namespace remains the same)

// Forward declarations with complete interfaces
class LayerNormalization {
public:
    explicit LayerNormalization(int size);
    void normalize(const std::vector<float>& input, std::vector<float>& output);
    void backward(const std::vector<float>& input, 
                 const std::vector<float>& gradient,
                 std::vector<float>& input_gradient);

private:
    int size_;
    std::vector<float> gamma_;
    std::vector<float> beta_;
    std::vector<float> gamma_momentum_;
    std::vector<float> gamma_velocity_;
    std::vector<float> beta_momentum_;
    std::vector<float> beta_velocity_;
};

class LSTMCell {
public:
    struct Weights {
        std::vector<std::vector<float>> Wf;
        std::vector<std::vector<float>> Wi;
        std::vector<std::vector<float>> Wc;
        std::vector<std::vector<float>> Wo;
        std::vector<float> bf, bi, bc, bo;
    };

    struct WeightGradients {
        std::vector<std::vector<float>> dWf;
        std::vector<std::vector<float>> dWi;
        std::vector<std::vector<float>> dWc;
        std::vector<std::vector<float>> dWo;
        std::vector<float> dbf, dbi, dbc, dbo;
    };

    struct Cache {
        std::vector<float> combined_input;
        std::vector<float> forget_gate;
        std::vector<float> input_gate;
        std::vector<float> cell_gate;
        std::vector<float> output_gate;
        std::vector<float> raw_hidden;
        std::vector<float> cell_state;
    };

    LSTMCell(int input_size, int hidden_size);
    
    void forward(const std::vector<float>& input,
                std::vector<float>& hidden_state,
                std::vector<float>& cell_state,
                bool training = false);
                
    void backward(const std::vector<float>& input,
                 const std::vector<float>& gradient,
                 std::vector<float>& input_gradient);

private:
    void initialize_weights();
    void clip_gradients(WeightGradients& grads);
    void update_weights(const WeightGradients& grads);

    int input_size_;
    int hidden_size_;
    LayerNormalization layer_norm_;
    Weights weights_;
    Cache cache_;
    std::vector<float> prev_cell_state_;
    std::vector<float> next_cell_gradient_;
};

class Network {
public:
    Network();
    
    float forward(const std::vector<std::vector<float>>& sequence, 
                 bool training = false);
    void backward(const std::vector<std::vector<float>>& sequence, 
                 float target);

private:
    void initialize();
    void update_adam(float& param, float gradient, float& m, float& v);

    std::vector<std::unique_ptr<LSTMCell>> layers_;
    std::vector<float> output_weights_;
    float output_bias_;
    std::vector<float> output_momentum_;
    std::vector<float> output_velocity_;
    float bias_momentum_;
    float bias_velocity_;
    std::vector<float> last_hidden_;
    unsigned long training_steps_;
};

class Trainer {
public:
    Trainer(Network& presence_net, Network& service_net);
    
    void train_batch(const std::vector<std::vector<std::vector<float>>>& batch,
                    const std::vector<bool>& targets);

private:
    Network& presence_net_;
    Network& service_net_;
    unsigned long training_steps_;
};

class HotWaterManager {
public:
    HotWaterManager();
    void setup();
    void loop();

private:
    void update(unsigned long current_time);
    void sample_and_train(unsigned long current_time);
    void debug_output(unsigned long current_time);
    void create_training_batch();

    Network presence_net_;
    Network service_net_;
    Trainer trainer_;
    std::vector<std::vector<float>> history_;
    std::vector<std::vector<std::vector<float>>> current_batch_;
    std::vector<bool> current_batch_targets_;
};

// Implementation of LayerNormalization::backward
void LayerNormalization::backward(const std::vector<float>& input,
                                const std::vector<float>& gradient,
                                std::vector<float>& input_gradient) {
    float mean = 0.0f;
    float variance = 0.0f;
    
    // Calculate mean and variance
    for (int i = 0; i < size_; i++) {
        mean += input[i];
    }
    mean /= size_;
    
    for (int i = 0; i < size_; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance = variance / size_ + Config::Training::EPSILON;
    
    float std_dev = sqrt(variance);
    std::vector<float> normalized(size_);
    
    for (int i = 0; i < size_; i++) {
        normalized[i] = (input[i] - mean) / std_dev;
    }
    
    // Calculate gradients
    std::vector<float> dgamma(size_);
    std::vector<float> dbeta(size_);
    
    for (int i = 0; i < size_; i++) {
        dgamma[i] = gradient[i] * normalized[i];
        dbeta[i] = gradient[i];
    }
    
    // Update gamma and beta using Adam
    for (int i = 0; i < size_; i++) {
        update_adam(gamma_[i], dgamma[i], gamma_momentum_[i], gamma_velocity_[i]);
        update_adam(beta_[i], dbeta[i], beta_momentum_[i], beta_velocity_[i]);
    }
    
    // Calculate input gradients
    input_gradient.resize(size_);
    float dx_norm_sum = 0.0f;
    float dx_norm_mean = 0.0f;
    
    for (int i = 0; i < size_; i++) {
        dx_norm_sum += gradient[i] * gamma_[i];
        dx_norm_mean += gradient[i] * gamma_[i] * normalized[i];
    }
    dx_norm_mean /= size_;
    
    for (int i = 0; i < size_; i++) {
        input_gradient[i] = (gamma_[i] * (gradient[i] - dx_norm_mean)) / std_dev;
    }
}

// Implementation of LSTMCell weight initialization
void LSTMCell::initialize_weights() {
    // Initialize weight matrices
    weights_.Wf = std::vector<std::vector<float>>(
        input_size_ + hidden_size_,
        std::vector<float>(hidden_size_)
    );
    weights_.Wi = std::vector<std::vector<float>>(
        input_size_ + hidden_size_,
        std::vector<float>(hidden_size_)
    );
    weights_.Wc = std::vector<std::vector<float>>(
        input_size_ + hidden_size_,
        std::vector<float>(hidden_size_)
    );
    weights_.Wo = std::vector<std::vector<float>>(
        input_size_ + hidden_size_,
        std::vector<float>(hidden_size_)
    );

    // Initialize bias vectors
    weights_.bf = std::vector<float>(hidden_size_, 1.0f);  // Forget gate bias initialized to 1
    weights_.bi = std::vector<float>(hidden_size_, 0.0f);
    weights_.bc = std::vector<float>(hidden_size_, 0.0f);
    weights_.bo = std::vector<float>(hidden_size_, 0.0f);

    // Initialize weights using He initialization
    for (int i = 0; i < input_size_ + hidden_size_; i++) {
        for (int h = 0; h < hidden_size_; h++) {
            weights_.Wf[i][h] = Utils::he_uniform(input_size_ + hidden_size_);
            weights_.Wi[i][h] = Utils::he_uniform(input_size_ + hidden_size_);
            weights_.Wc[i][h] = Utils::he_uniform(input_size_ + hidden_size_);
            weights_.Wo[i][h] = Utils::he_uniform(input_size_ + hidden_size_);
        }
    }
}

// ... (rest of the implementations remain the same)

void Network::update_adam(float& param, float gradient, float& m, float& v) {
    m = Config::Training::BETA1 * m + (1 - Config::Training::BETA1) * gradient;
    v = Config::Training::BETA2 * v + (1 - Config::Training::BETA2) * gradient * gradient;
    
    float m_hat = m / (1 - pow(Config::Training::BETA1, training_steps_));
    float v_hat = v / (1 - pow(Config::Training::BETA2, training_steps_));
    
    param -= Config::Training::INITIAL_LEARNING_RATE * m_hat / 
            (sqrt(v_hat) + Config::Training::EPSILON);
}

// Main setup and loop remain the same
HotWaterManager app;

void setup() {
    app.setup();
}

void loop() {
    app.loop();
}
