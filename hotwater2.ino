#include <Arduino.h>
#include <vector>
#include <cmath>

// Pin definitions (same as before)
const int PIR_PIN = 4;
const int SERVICE_PIN = 5;
const int PRESENCE_PWM = 18;
const int SERVICE_PWM = 19;

// Time constants
const int SAMPLES_PER_DAY = 96;
const int DAYS_PER_WEEK = 7;
const unsigned long SAMPLE_INTERVAL = 900000; // 15 minutes

// Improved RNN parameters
const int INPUT_SIZE = 3;  // time, weekday, sensor
const int HIDDEN_SIZE = 16;  // Increased for better pattern recognition
const float LEARNING_RATE = 0.005;
const float MOMENTUM = 0.9;
const float GRADIENT_CLIP = 1.0;
const float FORGET_GATE_BIAS = 1.0;  // LSTM optimization
const float DROPOUT_RATE = 0.1;
const int BATCH_SIZE = 24;  // Process one day of samples at a time

// LSTM Cell structure
struct LSTMCell {
    // State vectors
    float cell_state[HIDDEN_SIZE];
    float hidden_state[HIDDEN_SIZE];
    
    // Gates
    struct {
        float Wf[INPUT_SIZE][HIDDEN_SIZE];  // Forget gate
        float Wi[INPUT_SIZE][HIDDEN_SIZE];  // Input gate
        float Wc[INPUT_SIZE][HIDDEN_SIZE];  // Cell gate
        float Wo[INPUT_SIZE][HIDDEN_SIZE];  // Output gate
        
        float Uf[HIDDEN_SIZE][HIDDEN_SIZE];  // Recurrent weights
        float Ui[HIDDEN_SIZE][HIDDEN_SIZE];
        float Uc[HIDDEN_SIZE][HIDDEN_SIZE];
        float Uo[HIDDEN_SIZE][HIDDEN_SIZE];
        
        float bf[HIDDEN_SIZE];  // Biases
        float bi[HIDDEN_SIZE];
        float bc[HIDDEN_SIZE];
        float bo[HIDDEN_SIZE];
    } weights;
    
    // Output layer
    float Why[HIDDEN_SIZE];
    float by;
    
    // Gradient accumulation for batch processing
    struct {
        float dWf[INPUT_SIZE][HIDDEN_SIZE];
        float dWi[INPUT_SIZE][HIDDEN_SIZE];
        float dWc[INPUT_SIZE][HIDDEN_SIZE];
        float dWo[INPUT_SIZE][HIDDEN_SIZE];
        
        float dUf[HIDDEN_SIZE][HIDDEN_SIZE];
        float dUi[HIDDEN_SIZE][HIDDEN_SIZE];
        float dUc[HIDDEN_SIZE][HIDDEN_SIZE];
        float dUo[HIDDEN_SIZE][HIDDEN_SIZE];
        
        float dbf[HIDDEN_SIZE];
        float dbi[HIDDEN_SIZE];
        float dbc[HIDDEN_SIZE];
        float dbo[HIDDEN_SIZE];
        
        float dWhy[HIDDEN_SIZE];
        float dby;
    } gradients;
};

LSTMCell presence_lstm;
LSTMCell service_lstm;

// Circular buffer for batched learning
struct DataPoint {
    float inputs[INPUT_SIZE];
    bool presence;
    bool service;
    unsigned long timestamp;
};

std::vector<DataPoint> history;
std::vector<DataPoint> batch;

// Activation functions
float sigmoid(float x) {
    x = std::max(-6.0f, std::min(6.0f, x));  // Prevent overflow
    return 1.0f / (1.0f + exp(-x));
}

float tanh_activation(float x) {
    x = std::max(-6.0f, std::min(6.0f, x));
    return tanh(x);
}

// Xavier/Glorot initialization
float xavier_init(int fan_in, int fan_out) {
    float limit = sqrt(6.0f / (fan_in + fan_out));
    return (random(1000000) / 500000.0f - 1.0f) * limit;
}

void initLSTM(LSTMCell &cell) {
    // Initialize weights with Xavier initialization
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            cell.weights.Wf[i][h] = xavier_init(INPUT_SIZE, HIDDEN_SIZE);
            cell.weights.Wi[i][h] = xavier_init(INPUT_SIZE, HIDDEN_SIZE);
            cell.weights.Wc[i][h] = xavier_init(INPUT_SIZE, HIDDEN_SIZE);
            cell.weights.Wo[i][h] = xavier_init(INPUT_SIZE, HIDDEN_SIZE);
        }
    }
    
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        for (int h2 = 0; h2 < HIDDEN_SIZE; h2++) {
            cell.weights.Uf[h][h2] = xavier_init(HIDDEN_SIZE, HIDDEN_SIZE);
            cell.weights.Ui[h][h2] = xavier_init(HIDDEN_SIZE, HIDDEN_SIZE);
            cell.weights.Uc[h][h2] = xavier_init(HIDDEN_SIZE, HIDDEN_SIZE);
            cell.weights.Uo[h][h2] = xavier_init(HIDDEN_SIZE, HIDDEN_SIZE);
        }
        
        // Initialize biases
        cell.weights.bf[h] = FORGET_GATE_BIAS;  // Initialize forget gate bias to 1
        cell.weights.bi[h] = 0.0f;
        cell.weights.bc[h] = 0.0f;
        cell.weights.bo[h] = 0.0f;
        
        cell.Why[h] = xavier_init(HIDDEN_SIZE, 1);
        
        // Initialize states
        cell.cell_state[h] = 0.0f;
        cell.hidden_state[h] = 0.0f;
    }
    
    cell.by = 0.0f;
}

// Forward pass with dropout
float forwardPass(LSTMCell &cell, const float* input, bool training = false) {
    float gates_f[HIDDEN_SIZE] = {0};
    float gates_i[HIDDEN_SIZE] = {0};
    float gates_c[HIDDEN_SIZE] = {0};
    float gates_o[HIDDEN_SIZE] = {0};
    
    // Calculate gate values
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        // Input transformations
        for (int i = 0; i < INPUT_SIZE; i++) {
            gates_f[h] += input[i] * cell.weights.Wf[i][h];
            gates_i[h] += input[i] * cell.weights.Wi[i][h];
            gates_c[h] += input[i] * cell.weights.Wc[i][h];
            gates_o[h] += input[i] * cell.weights.Wo[i][h];
        }
        
        // Recurrent transformations
        for (int h2 = 0; h2 < HIDDEN_SIZE; h2++) {
            gates_f[h] += cell.hidden_state[h2] * cell.weights.Uf[h2][h];
            gates_i[h] += cell.hidden_state[h2] * cell.weights.Ui[h2][h];
            gates_c[h] += cell.hidden_state[h2] * cell.weights.Uc[h2][h];
            gates_o[h] += cell.hidden_state[h2] * cell.weights.Uo[h2][h];
        }
        
        // Add biases and apply activation functions
        gates_f[h] = sigmoid(gates_f[h] + cell.weights.bf[h]);
        gates_i[h] = sigmoid(gates_i[h] + cell.weights.bi[h]);
        gates_c[h] = tanh_activation(gates_c[h] + cell.weights.bc[h]);
        gates_o[h] = sigmoid(gates_o[h] + cell.weights.bo[h]);
        
        // Update cell state
        cell.cell_state[h] = gates_f[h] * cell.cell_state[h] + gates_i[h] * gates_c[h];
        
        // Update hidden state with dropout during training
        float hidden = gates_o[h] * tanh_activation(cell.cell_state[h]);
        if (training && random(100) < DROPOUT_RATE * 100) {
            cell.hidden_state[h] = 0;
        } else {
            cell.hidden_state[h] = hidden * (training ? (1.0f - DROPOUT_RATE) : 1.0f);
        }
    }
    
    // Calculate output
    float output = cell.by;
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        output += cell.hidden_state[h] * cell.Why[h];
    }
    
    return sigmoid(output);
}

void backPropagate(LSTMCell &cell, const float* input, float target, float learning_rate) {
    float output = forwardPass(cell, input, true);
    float output_error = output - target;
    
    // Initialize gradient accumulators if this is the first sample in batch
    if (batch.size() == 1) {
        memset(&cell.gradients, 0, sizeof(cell.gradients));
    }
    
    // Output layer gradients
    float dOutput = output_error * output * (1 - output);
    
    // Accumulate gradients for batch
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        cell.gradients.dWhy[h] += dOutput * cell.hidden_state[h];
    }
    cell.gradients.dby += dOutput;
    
    // If batch is complete, apply updates with gradient clipping
    if (batch.size() == BATCH_SIZE) {
        // Apply gradient clipping
        float grad_norm = 0.0f;
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            grad_norm += cell.gradients.dWhy[h] * cell.gradients.dWhy[h];
        }
        grad_norm = sqrt(grad_norm);
        
        if (grad_norm > GRADIENT_CLIP) {
            float scale = GRADIENT_CLIP / grad_norm;
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                cell.gradients.dWhy[h] *= scale;
            }
        }
        
        // Apply accumulated gradients
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            cell.Why[h] -= learning_rate * cell.gradients.dWhy[h] / BATCH_SIZE;
        }
        cell.by -= learning_rate * cell.gradients.dby / BATCH_SIZE;
    }
}

void setup() {
    Serial.begin(115200);
    
    // Initialize pins
    pinMode(PIR_PIN, INPUT);
    pinMode(SERVICE_PIN, INPUT_PULLUP);
    pinMode(PRESENCE_PWM, OUTPUT);
    pinMode(SERVICE_PWM, OUTPUT);
    
    // Initialize PWM
    ledcSetup(0, 5000, 8);
    ledcSetup(1, 5000, 8);
    ledcAttachPin(PRESENCE_PWM, 0);
    ledcAttachPin(SERVICE_PWM, 1);
    
    // Initialize LSTMs
    initLSTM(presence_lstm);
    initLSTM(service_lstm);
    
    // Reserve memory
    history.reserve(SAMPLES_PER_DAY * 7);  // One week of history
    batch.reserve(BATCH_SIZE);
}

void loop() {
    static unsigned long lastSampleTime = 0;
    unsigned long currentTime = millis();
    
    // Get current time and format inputs
    time_t now;
    time(&now);
    struct tm timeinfo;
    localtime_r(&now, &timeinfo);
    
    float inputs[INPUT_SIZE] = {
        (float)(timeinfo.tm_hour * 60 + timeinfo.tm_min) / 1440.0f,  // Normalized time of day
        (float)timeinfo.tm_wday / 6.0f,  // Normalized day of week
        0  // Will be set to sensor value
    };
    
    // Read sensors
    bool pirDetected = digitalRead(PIR_PIN);
    bool serviceRequested = !digitalRead(SERVICE_PIN);
    
    // Sample and train every 15 minutes
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
        lastSampleTime = currentTime;
        
        // Update inputs with sensor values
        inputs[2] = pirDetected ? 1.0f : 0.0f;
        
        // Add to batch
        DataPoint point = {
            .timestamp = currentTime
        };
        memcpy(point.inputs, inputs, sizeof(inputs));
        point.presence = pirDetected;
        point.service = serviceRequested;
        
        batch.push_back(point);
        
        // Train when batch is full
        if (batch.size() >= BATCH_SIZE) {
            float current_lr = LEARNING_RATE;
            
            // Implement learning rate decay
            if (history.size() > SAMPLES_PER_DAY * 7) {
                current_lr *= 0.95f;  // Decay learning rate over time
            }
            
            // Train on batch
            for (const auto &p : batch) {
                backPropagate(presence_lstm, p.inputs, p.presence, current_lr);
                backPropagate(service_lstm, p.inputs, p.service, current_lr);
            }
            
            // Add batch to history and clear
            history.insert(history.end(), batch.begin(), batch.end());
            batch.clear();
            
            // Trim history if needed
            while (history.size() > SAMPLES_PER_DAY * 7) {
                history.erase(history.begin());
            }
        }
    }
    
    // Make predictions
    inputs[2] = pirDetected ? 1.0f : 0.0f;
    float presence_prob = forwardPass(presence_lstm, inputs);
    float service_prob = forwardPass(service_lstm, inputs);
    
    // Boost presence probability if service is requested
    if (serviceRequested) {
        presence_prob = std::min(1.0f, presence_prob * 1.5f);
    }
    
    // Update PWM outputs
    ledcWrite(0, (int)(presence_prob * 255));
    ledcWrite(1, (int)(service_prob * 255));
    
    // Debug output every second
    static unsigned long lastDebugTime = 0;
    if (currentTime - lastDebugTime >= 1000) {
        lastDebugTime = currentTime;
        Serial.printf("Time: %02d:%02d Day: %d PIR: %d SRV: %d Presence: %.3f Service: %.3f LR: %.5f\n",
                     timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_wday,
                     pirDetected, serviceRequested, presence_prob, service_prob,
                     LEARNING_RATE * (history.size() > SAMPLES_PER_DAY * 7 ? 0.95f : 1.0f));
    }
    
    delay(100);  // Basic rate limiting
}
