#include <Arduino.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Pin definitions remain the same
const int PIR_PIN = 4;
const int SERVICE_PIN = 5;
const int PRESENCE_PWM = 18;
const int SERVICE_PWM = 19;

// Enhanced network architecture
const int INPUT_SIZE = 5;  // time, weekday, sensor, sin(time), cos(time)
const int HIDDEN_LAYERS = 2;
const int HIDDEN_SIZE = 32;
const int SEQUENCE_LENGTH = 4;  // Consider previous time steps

// Advanced training parameters
const float INITIAL_LEARNING_RATE = 0.01f;
const float MIN_LEARNING_RATE = 0.0001f;
const float BETA1 = 0.9f;      // Adam optimizer parameter
const float BETA2 = 0.999f;    // Adam optimizer parameter
const float EPSILON = 1e-8f;   // Adam optimizer parameter
const float L2_LAMBDA = 0.001f;  // L2 regularization
const float GRADIENT_CLIP = 5.0f;
const float DROPOUT_RATE = 0.2f;
const int BATCH_SIZE = 32;

// Layer Normalization parameters
struct LayerNorm {
    float gamma[HIDDEN_SIZE];
    float beta[HIDDEN_SIZE];
    float epsilon = 1e-5f;
};

// LSTM Cell with Layer Normalization
struct LSTMLayer {
    // Gates and states
    struct {
        float forget[HIDDEN_SIZE];
        float input[HIDDEN_SIZE];
        float cell[HIDDEN_SIZE];
        float output[HIDDEN_SIZE];
        float cell_state[HIDDEN_SIZE];
        float hidden_state[HIDDEN_SIZE];
    } gates;

    // Weights and biases
    struct {
        float Wf[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];  // Combined input and recurrent weights
        float Wi[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float Wc[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float Wo[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        
        float bf[HIDDEN_SIZE];
        float bi[HIDDEN_SIZE];
        float bc[HIDDEN_SIZE];
        float bo[HIDDEN_SIZE];
    } weights;

    // Adam optimizer states
    struct {
        float m_Wf[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float v_Wf[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float m_Wi[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float v_Wi[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float m_Wc[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float v_Wc[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float m_Wo[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float v_Wo[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        
        float m_bf[HIDDEN_SIZE];
        float v_bf[HIDDEN_SIZE];
        float m_bi[HIDDEN_SIZE];
        float v_bi[HIDDEN_SIZE];
        float m_bc[HIDDEN_SIZE];
        float v_bc[HIDDEN_SIZE];
        float m_bo[HIDDEN_SIZE];
        float v_bo[HIDDEN_SIZE];
    } adam;

    LayerNorm layer_norm;
};

// Network structure
struct Network {
    LSTMLayer layers[HIDDEN_LAYERS];
    float output_weights[HIDDEN_SIZE];
    float output_bias;
    
    // Output layer optimizer states
    float m_output[HIDDEN_SIZE];
    float v_output[HIDDEN_SIZE];
    float m_bias, v_bias;
    
    unsigned long training_steps;
};

Network presence_net;
Network service_net;

// Circular buffer for sequence learning
struct SequencePoint {
    float inputs[INPUT_SIZE];
    bool presence;
    bool service;
    unsigned long timestamp;
};

std::vector<SequencePoint> history;
std::vector<std::vector<SequencePoint>> batch_sequences;

// Advanced activation functions
float gelu(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f/M_PI) * (x + 0.044715f * x * x * x)));
}

float swish(float x) {
    return x * sigmoid(x);
}

// Layer Normalization
void layerNormalize(const float* input, float* output, const LayerNorm& norm) {
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        mean += input[i];
    }
    mean /= HIDDEN_SIZE;
    
    // Calculate variance
    float variance = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance = variance / HIDDEN_SIZE + norm.epsilon;
    
    // Normalize and scale
    float std_dev = sqrt(variance);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output[i] = norm.gamma[i] * ((input[i] - mean) / std_dev) + norm.beta[i];
    }
}

// Enhanced initialization with scaled variants
float glorot_uniform(int fan_in, int fan_out) {
    float limit = sqrt(6.0f / (fan_in + fan_out));
    return (random(1000000) / 500000.0f - 1.0f) * limit;
}

float he_uniform(int fan_in) {
    float limit = sqrt(6.0f / fan_in);
    return (random(1000000) / 500000.0f - 1.0f) * limit;
}

void initNetwork(Network& net) {
    net.training_steps = 0;
    
    for (int l = 0; l < HIDDEN_LAYERS; l++) {
        auto& layer = net.layers[l];
        int fan_in = (l == 0) ? INPUT_SIZE : HIDDEN_SIZE;
        
        // Initialize weights with He initialization
        for (int i = 0; i < INPUT_SIZE + HIDDEN_SIZE; i++) {
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                layer.weights.Wf[i][h] = he_uniform(fan_in);
                layer.weights.Wi[i][h] = he_uniform(fan_in);
                layer.weights.Wc[i][h] = he_uniform(fan_in);
                layer.weights.Wo[i][h] = he_uniform(fan_in);
                
                // Initialize Adam accumulators
                layer.adam.m_Wf[i][h] = 0.0f;
                layer.adam.v_Wf[i][h] = 0.0f;
                layer.adam.m_Wi[i][h] = 0.0f;
                layer.adam.v_Wi[i][h] = 0.0f;
                layer.adam.m_Wc[i][h] = 0.0f;
                layer.adam.v_Wc[i][h] = 0.0f;
                layer.adam.m_Wo[i][h] = 0.0f;
                layer.adam.v_Wo[i][h] = 0.0f;
            }
        }
        
        // Initialize layer normalization parameters
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            layer.layer_norm.gamma[h] = 1.0f;
            layer.layer_norm.beta[h] = 0.0f;
            
            // Initialize biases
            layer.weights.bf[h] = 1.0f;  // Forget gate bias initialized to 1
            layer.weights.bi[h] = 0.0f;
            layer.weights.bc[h] = 0.0f;
            layer.weights.bo[h] = 0.0f;
            
            // Initialize states
            layer.gates.cell_state[h] = 0.0f;
            layer.gates.hidden_state[h] = 0.0f;
            
            // Initialize Adam bias accumulators
            layer.adam.m_bf[h] = 0.0f;
            layer.adam.v_bf[h] = 0.0f;
            layer.adam.m_bi[h] = 0.0f;
            layer.adam.v_bi[h] = 0.0f;
            layer.adam.m_bc[h] = 0.0f;
            layer.adam.v_bc[h] = 0.0f;
            layer.adam.m_bo[h] = 0.0f;
            layer.adam.v_bo[h] = 0.0f;
        }
    }
    
    // Initialize output layer
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        net.output_weights[h] = glorot_uniform(HIDDEN_SIZE, 1);
        net.m_output[h] = 0.0f;
        net.v_output[h] = 0.0f;
    }
    net.output_bias = 0.0f;
    net.m_bias = 0.0f;
    net.v_bias = 0.0f;
}

// Forward pass with advanced features
float forwardPass(Network& net, const std::vector<SequencePoint>& sequence, bool training = false) {
    float combined_input[INPUT_SIZE + HIDDEN_SIZE];
    float layer_output[HIDDEN_SIZE];
    
    // Process each timestep in the sequence
    for (const auto& point : sequence) {
        // Copy input features
        memcpy(combined_input, point.inputs, INPUT_SIZE * sizeof(float));
        
        // Process each layer
        for (int l = 0; l < HIDDEN_LAYERS; l++) {
            auto& layer = net.layers[l];
            
            // Combine input with previous hidden state
            if (l > 0) {
                memcpy(combined_input, layer_output, HIDDEN_SIZE * sizeof(float));
            }
            memcpy(combined_input + INPUT_SIZE, layer.gates.hidden_state, HIDDEN_SIZE * sizeof(float));
            
            // Calculate gates
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                float f = 0.0f, i = 0.0f, c = 0.0f, o = 0.0f;
                
                for (int j = 0; j < INPUT_SIZE + HIDDEN_SIZE; j++) {
                    f += combined_input[j] * layer.weights.Wf[j][h];
                    i += combined_input[j] * layer.weights.Wi[j][h];
                    c += combined_input[j] * layer.weights.Wc[j][h];
                    o += combined_input[j] * layer.weights.Wo[j][h];
                }
                
                // Add biases and apply activation functions
                layer.gates.forget[h] = sigmoid(f + layer.weights.bf[h]);
                layer.gates.input[h] = sigmoid(i + layer.weights.bi[h]);
                layer.gates.cell[h] = tanh(c + layer.weights.bc[h]);
                layer.gates.output[h] = sigmoid(o + layer.weights.bo[h]);
                
                // Update cell state
                layer.gates.cell_state[h] = layer.gates.forget[h] * layer.gates.cell_state[h] + 
                                          layer.gates.input[h] * layer.gates.cell[h];
                
                // Calculate hidden state with dropout during training
                float hidden = layer.gates.output[h] * tanh(layer.gates.cell_state[h]);
                if (training && random(100) < DROPOUT_RATE * 100) {
                    layer.gates.hidden_state[h] = 0;
                } else {
                    layer.gates.hidden_state[h] = hidden * (training ? (1.0f - DROPOUT_RATE) : 1.0f);
                }
            }
            
            // Apply layer normalization
            layerNormalize(layer.gates.hidden_state, layer_output, layer.layer_norm);
        }
    }
    
    // Calculate final output
    float output = net.output_bias;
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        output += layer_output[h] * net.output_weights[h];
    }
    
    return sigmoid(output);
}

// Backpropagation with Adam optimization
void backpropagate(Network& net, const std::vector<SequencePoint>& sequence, float target) {
    net.training_steps++;
    
    // Calculate learning rate with decay
    float lr = INITIAL_LEARNING_RATE * 
               sqrt(1.0f - pow(BETA2, net.training_steps)) / 
               (1.0f - pow(BETA1, net.training_steps));
    lr = std::max(lr, MIN_LEARNING_RATE);
    
    float output = forwardPass(net, sequence, true);
    float output_error = output - target;
    
    // Calculate gradients and update weights using Adam
    // (Implementation continues with detailed gradient calculations,
    // Adam optimization updates, and L2 regularization)
    // ...

    // Note: The full implementation of backpropagation would be quite lengthy,
    // including detailed gradient calculations for all layers, Adam optimization
    // steps, and L2 regularization. Would you like me to continue with that
    // specific part of the implementation?
}

void setup() {
    Serial.begin(115200);
    
    // Initialize GPIO
    pinMode(PIR_PIN, INPUT);
    pinMode(SERVICE_PIN, INPUT_PULLUP);
    pinMode(PRESENCE_PWM, OUTPUT);
    pinMode(SERVICE_PWM, OUTPUT);
    
    // Initialize PWM
    ledcSetup(0, 5000, 8);
    ledcSetup(1, 5000, 8);
    ledcAttachPin(PRESENCE_PWM, 0);
    ledcAttachPin(SERVICE_PWM, 1);
    
    // Initialize networks
    initNetwork(presence_net);
    initNetwork(service_net);
    
    // Reserve memory
    history.reserve(SEQUENCE_LENGTH * BATCH_SIZE * 2);
    batch_sequences.reserve(BATCH_SIZE);
}

void loop() {
    static unsigned long lastSampleTime = 0;
    static unsigned long lastDebugTime = 0;
    unsigned long currentTime = millis();
    
    // Get current time
    time_t now;
    time(&now);
    struct tm timeinfo;
    localtime_r(&now, &timeinfo);
    
    // Create enhanced input features
    float timeOfDay = (timeinfo.tm_hour * 3600 + timeinfo.tm_min * 60 + timeinfo.tm_sec) / 86400.0f;
    float inputs[INPUT_SIZE] = {
        timeOfDay,                          // Normalized time of day
        (float)timeinfo.tm_wday / 6.0f,    // Normalized day of week
        0.0f,                              // Sensor value (set later)
        sin(2 * M_PI * timeOfDay),         // Cyclical time encoding
        cos(2 * M_PI * timeOfDay)          // Cyclical time encoding
    };
    
    // Read sensors
    bool pirDetected = digitalRead(PIR_PIN);
    bool serviceRequested = !digitalRead(SERVICE_PIN);
    inputs[2] = pirDetected ? 1.0f : 0.0f;
    
    // Sample and train every 15 minutes
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
        lastSampleTime = currentTime;
        
        // Create sequence point
        SequencePoint point = {
            .timestamp = currentTime
        };
        memcpy(point.inputs, inputs, sizeof(inputs));
        point.presence = pirDetected;
        point.service = serviceRequested;
        
        // Add to history
        history.push_back(point);
        
        // Maintain history size
        if (history.size() > SEQUENCE_LENGTH * BATCH_SIZE * 2) {
            history.erase(history.begin());
        }
        
        // Create training sequences when enough history is available
        if (history.size() >= SEQUENCE_LENGTH) {
            std::vector<SequencePoint> sequence;
            for (size_t i = history.size() - SEQUENCE_LENGTH; i < history.size(); i++) {
                sequence.push_back(history[i]);
            }
            batch_sequences.push_back(sequence);
            
            // Train when batch is ready
            if (batch_sequences.size() >= BATCH_SIZE) {
                trainNetworks();
                batch_sequences.clear();
            }
        }
    }
    
    // Make predictions using sequence
    std::vector<SequencePoint> current_sequence;
    if (history.size() >= SEQUENCE_LENGTH) {
        for (size_t i = history.size() - SEQUENCE_LENGTH; i < history.size(); i++) {
            current_sequence.push_back(history[i]);
        }
    } else {
        current_sequence.push_back({});
        memcpy(current_sequence.back().inputs, inputs, sizeof(inputs));
    }
    
    float presence_prob = forwardPass(presence_net, current_sequence);
    float service_prob = forwardPass(service_net, current_sequence);
    
    // Boost presence probability if service is requested
    if (serviceRequested) {
        presence_prob = std::min(1.0f, presence_prob * 1.5f);
    }
    
    // Update PWM outputs
    ledcWrite(0, (int)(presence_prob * 255));
    ledcWrite(1, (int)(service_prob * 255));
    
    // Debug output every second
    if (currentTime - lastDebugTime >= 1000) {
        lastDebugTime = currentTime;
        float current_lr = INITIAL_LEARNING_RATE * 
                          sqrt(1.0f - pow(BETA2, presence_net.training_steps)) / 
                          (1.0f - pow(BETA1, presence_net.training_steps));
        current_lr = std::max(current_lr, MIN_LEARNING_RATE);
        
        Serial.printf("Time: %02d:%02d Day: %d PIR: %d SRV: %d P: %.3f S: %.3f LR: %.5f Steps: %lu\n",
                     timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_wday,
                     pirDetected, serviceRequested, presence_prob, service_prob,
                     current_lr, presence_net.training_steps);
    }
}

// Implement detailed backpropagation with Adam optimization
void backpropagateLSTM(Network& net, const std::vector<SequencePoint>& sequence, float target) {
    const float lr = std::max(
        INITIAL_LEARNING_RATE * sqrt(1.0f - pow(BETA2, net.training_steps)) / 
        (1.0f - pow(BETA1, net.training_steps)),
        MIN_LEARNING_RATE
    );
    
    // Store intermediate values for backpropagation
    struct LayerState {
        float combined_input[INPUT_SIZE + HIDDEN_SIZE];
        float gate_inputs[4][HIDDEN_SIZE];  // For f, i, c, o gates
        float cell_candidates[HIDDEN_SIZE];
        float cell_state[HIDDEN_SIZE];
        float hidden_state[HIDDEN_SIZE];
    };
    
    std::vector<std::vector<LayerState>> states(HIDDEN_LAYERS, std::vector<LayerState>(sequence.size()));
    
    // Forward pass storing intermediate values
    float output = forwardPass(net, sequence, true);
    float output_error = output - target;
    
    // Output layer gradients
    float output_delta = output_error * output * (1.0f - output);
    
    // Initialize gradient accumulators
    struct LayerGradients {
        float dWf[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float dWi[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float dWc[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float dWo[INPUT_SIZE + HIDDEN_SIZE][HIDDEN_SIZE];
        float dbf[HIDDEN_SIZE];
        float dbi[HIDDEN_SIZE];
        float dbc[HIDDEN_SIZE];
        float dbo[HIDDEN_SIZE];
    };
    
    std::vector<LayerGradients> layer_gradients(HIDDEN_LAYERS);
    
    // Initialize gradients to zero
    for (int l = 0; l < HIDDEN_LAYERS; l++) {
        memset(&layer_gradients[l], 0, sizeof(LayerGradients));
    }
    
    // Backward pass through time
    for (int t = sequence.size() - 1; t >= 0; t--) {
        float next_layer_delta[HIDDEN_SIZE] = {0};
        
        // Propagate through layers
        for (int l = HIDDEN_LAYERS - 1; l >= 0; l--) {
            auto& layer = net.layers[l];
            auto& state = states[l][t];
            
            // Calculate gate deltas
            float cell_delta[HIDDEN_SIZE] = {0};
            float hidden_delta[HIDDEN_SIZE];
            
            if (l == HIDDEN_LAYERS - 1) {
                // Last hidden layer
                for (int h = 0; h < HIDDEN_SIZE; h++) {
                    hidden_delta[h] = output_delta * net.output_weights[h];
                }
            } else {
                // Hidden layers
                memcpy(hidden_delta, next_layer_delta, HIDDEN_SIZE * sizeof(float));
            }
            
            // Calculate gradients for gates
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                // Output gate gradient
                float do_h = hidden_delta[h] * tanh(state.cell_state[h]) * 
                            state.gate_inputs[3][h] * (1.0f - state.gate_inputs[3][h]);
                
                // Cell state gradient
                cell_delta[h] += hidden_delta[h] * layer.gates.output[h] * 
                                (1.0f - tanh(state.cell_state[h]) * tanh(state.cell_state[h]));
                
                if (t < sequence.size() - 1) {
                    cell_delta[h] += state.cell_state[h + 1];
                }
                
                // Input gate gradient
                float di_h = cell_delta[h] * state.cell_candidates[h] * 
                            state.gate_inputs[1][h] * (1.0f - state.gate_inputs[1][h]);
                
                // Forget gate gradient
                float df_h = cell_delta[h] * state.cell_state[h] * 
                            state.gate_inputs[0][h] * (1.0f - state.gate_inputs[0][h]);
                
                // Cell candidate gradient
                float dc_h = cell_delta[h] * state.gate_inputs[1][h] * 
                            (1.0f - state.cell_candidates[h] * state.cell_candidates[h]);
                
                // Accumulate gradients for this timestep
                for (int i = 0; i < INPUT_SIZE + HIDDEN_SIZE; i++) {
                    layer_gradients[l].dWf[i][h] += df_h * state.combined_input[i];
                    layer_gradients[l].dWi[i][h] += di_h * state.combined_input[i];
                    layer_gradients[l].dWc[i][h] += dc_h * state.combined_input[i];
                    layer_gradients[l].dWo[i][h] += do_h * state.combined_input[i];
                }
                
                layer_gradients[l].dbf[h] += df_h;
                layer_gradients[l].dbi[h] += di_h;
                layer_gradients[l].dbc[h] += dc_h;
                layer_gradients[l].dbo[h] += do_h;
                
                // Calculate delta for next layer
                if (l > 0) {
                    for (int i = 0; i < HIDDEN_SIZE; i++) {
                        next_layer_delta[i] = 
                            df_h * layer.weights.Wf[INPUT_SIZE + h][i] +
                            di_h * layer.weights.Wi[INPUT_SIZE + h][i] +
                            dc_h * layer.weights.Wc[INPUT_SIZE + h][i] +
                            do_h * layer.weights.Wo[INPUT_SIZE + h][i];
                    }
                }
            }
        }
    }
    
    // Apply gradients using Adam optimization
    for (int l = 0; l < HIDDEN_LAYERS; l++) {
        auto& layer = net.layers[l];
        
        // Apply gradient clipping
        float grad_norm = 0.0f;
        for (int i = 0; i < INPUT_SIZE + HIDDEN_SIZE; i++) {
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                grad_norm += 
                    layer_gradients[l].dWf[i][h] * layer_gradients[l].dWf[i][h] +
                    layer_gradients[l].dWi[i][h] * layer_gradients[l].dWi[i][h] +
                    layer_gradients[l].dWc[i][h] * layer_gradients[l].dWc[i][h] +
                    layer_gradients[l].dWo[i][h] * layer_gradients[l].dWo[i][h];
            }
        }
        grad_norm = sqrt(grad_norm);
        
        if (grad_norm > GRADIENT_CLIP) {
            float scale = GRADIENT_CLIP / grad_norm;
            for (int i = 0; i < INPUT_SIZE + HIDDEN_SIZE; i++) {
                for (int h = 0; h < HIDDEN_SIZE; h++) {
                    layer_gradients[l].dWf[i][h] *= scale;
                    layer_gradients[l].dWi[i][h] *= scale;
                    layer_gradients[l].dWc[i][h] *= scale;
                    layer_gradients[l].dWo[i][h] *= scale;
                }
            }
        }
        
        // Update weights and biases using Adam
        for (int i = 0; i < INPUT_SIZE + HIDDEN_SIZE; i++) {
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                // Update forget gate weights
                updateAdam(layer.weights.Wf[i][h], layer_gradients[l].dWf[i][h],
                          layer.adam.m_Wf[i][h], layer.adam.v_Wf[i][h], lr);
                          
                // Update input gate weights
                updateAdam(layer.weights.Wi[i][h], layer_gradients[l].dWi[i][h],
                          layer.adam.m_Wi[i][h], layer.adam.v_Wi[i][h], lr);
                          
                // Update cell gate weights
                updateAdam(layer.weights.Wc[i][h], layer_gradients[l].dWc[i][h],
                          layer.adam.m_Wc[i][h], layer.adam.v_Wc[i][h], lr);
                          
                // Update output gate weights
                updateAdam(layer.weights.Wo[i][h], layer_gradients[l].dWo[i][h],
                          layer.adam.m_Wo[i][h], layer.adam.v_Wo[i][h], lr);
            }
        }
        
        // Update biases
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            updateAdam(layer.weights.bf[h], layer_gradients[l].dbf[h],
                      layer.adam.m_bf[h], layer.adam.v_bf[h], lr);
            updateAdam(layer.weights.bi[h], layer_gradients[l].dbi[h],
                      layer.adam.m_bi[h], layer.adam.v_bi[h], lr);
            updateAdam(layer.weights.bc[h], layer_gradients[l].dbc[h],
                      layer.adam.m_bc[h], layer.adam.v_bc[h], lr);
            updateAdam(layer.weights.bo[h], layer_gradients[l].dbo[h],
                      layer.adam.m_bo[h], layer.adam.v_bo[h], lr);
        }
    }
}

// Adam optimizer update function
void updateAdam(float& param, float gradient, float& m, float& v, float learning_rate) {
    m = BETA1 * m + (1 - BETA1) * gradient;
    v = BETA2 * v + (1 - BETA2) * gradient * gradient;
    
    float m_hat = m / (1 - pow(BETA1, presence_net.training_steps));
    float v_hat = v / (1 - pow(BETA2, presence_net.training_steps));
    
    param -= learning_rate * m_hat / (sqrt(v_hat) + EPSILON);
}

void trainNetworks() {
    for (const auto& sequence : batch_sequences) {
        backpropagateLSTM(presence_net, sequence, sequence.back().presence);
        backpropagateLSTM(service_net, sequence, sequence.back().service);
    }
}
