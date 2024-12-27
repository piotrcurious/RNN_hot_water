#include <Arduino.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>

// Configuration namespace to centralize all parameters
namespace Config {
    // Hardware configuration
    struct GPIO {
        static constexpr int PIR_PIN = 4;
        static constexpr int SERVICE_PIN = 5;
        static constexpr int PRESENCE_PWM = 18;
        static constexpr int SERVICE_PWM = 19;
        static constexpr int PWM_FREQ = 5000;
        static constexpr int PWM_RESOLUTION = 8;
    };

    // Neural network architecture
    struct Network {
        static constexpr int INPUT_SIZE = 5;
        static constexpr int HIDDEN_LAYERS = 2;
        static constexpr int HIDDEN_SIZE = 32;
        static constexpr int SEQUENCE_LENGTH = 4;
        static constexpr int BATCH_SIZE = 32;
    };

    // Training hyperparameters
    struct Training {
        static constexpr float INITIAL_LEARNING_RATE = 0.01f;
        static constexpr float MIN_LEARNING_RATE = 0.0001f;
        static constexpr float BETA1 = 0.9f;
        static constexpr float BETA2 = 0.999f;
        static constexpr float EPSILON = 1e-8f;
        static constexpr float L2_LAMBDA = 0.001f;
        static constexpr float GRADIENT_CLIP = 5.0f;
        static constexpr float DROPOUT_RATE = 0.2f;
        static constexpr unsigned long SAMPLE_INTERVAL = 900000; // 15 minutes in milliseconds
        static constexpr unsigned long DEBUG_INTERVAL = 1000;    // 1 second in milliseconds
    };
}

// Exception classes for better error handling
class NetworkException : public std::runtime_error {
public:
    explicit NetworkException(const char* msg) : std::runtime_error(msg) {}
};

// Forward declarations
class ActivationFunctions;
class LayerNormalization;
class LSTMCell;
class Network;
class Trainer;

// Utility functions
namespace Utils {
    // Random number generation with better distribution
    float random_uniform(float min = -1.0f, float max = 1.0f) {
        return min + (max - min) * (random(1000000) / 1000000.0f);
    }

    // Weight initialization schemes
    float glorot_uniform(int fan_in, int fan_out) {
        float limit = sqrt(6.0f / (fan_in + fan_out));
        return random_uniform(-limit, limit);
    }

    float he_uniform(int fan_in) {
        float limit = sqrt(6.0f / fan_in);
        return random_uniform(-limit, limit);
    }

    // Time feature extraction
    struct TimeFeatures {
        float time_of_day;
        float day_of_week;
        float sin_time;
        float cos_time;

        static TimeFeatures extract() {
            time_t now;
            time(&now);
            struct tm timeinfo;
            localtime_r(&now, &timeinfo);
            
            float tod = (timeinfo.tm_hour * 3600 + timeinfo.tm_min * 60 + timeinfo.tm_sec) / 86400.0f;
            return {
                tod,
                static_cast<float>(timeinfo.tm_wday) / 6.0f,
                sin(2 * M_PI * tod),
                cos(2 * M_PI * tod)
            };
        }
    };
}

// Activation functions with forward and backward implementations
class ActivationFunctions {
public:
    static float sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    static float sigmoid_derivative(float x) {
        float sx = sigmoid(x);
        return sx * (1.0f - sx);
    }

    static float tanh(float x) {
        return std::tanh(x);
    }

    static float tanh_derivative(float x) {
        float tx = std::tanh(x);
        return 1.0f - tx * tx;
    }

    static float gelu(float x) {
        return 0.5f * x * (1.0f + tanh(sqrt(2.0f/M_PI) * (x + 0.044715f * x * x * x)));
    }

    static float gelu_derivative(float x) {
        // Approximate GELU derivative
        float cdf = 0.5f * (1.0f + tanh(sqrt(2.0f/M_PI) * (x + 0.044715f * x * x * x)));
        float pdf = exp(-0.5f * x * x) / sqrt(2.0f * M_PI);
        return cdf + x * pdf;
    }
};

// Layer normalization with improved numerical stability
class LayerNormalization {
public:
    LayerNormalization(int size) : size_(size) {
        gamma_ = std::vector<float>(size, 1.0f);
        beta_ = std::vector<float>(size, 0.0f);
    }

    void normalize(const std::vector<float>& input, std::vector<float>& output) {
        float mean = 0.0f;
        float variance = 0.0f;
        
        // Two-pass algorithm for better numerical stability
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
        for (int i = 0; i < size_; i++) {
            output[i] = gamma_[i] * ((input[i] - mean) / std_dev) + beta_[i];
        }
    }

private:
    int size_;
    std::vector<float> gamma_;
    std::vector<float> beta_;
};

// LSTM cell with improved memory management
class LSTMCell {
public:
    LSTMCell(int input_size, int hidden_size) 
        : input_size_(input_size), 
          hidden_size_(hidden_size),
          layer_norm_(hidden_size) {
        initialize_weights();
    }

    void forward(const std::vector<float>& input, 
                std::vector<float>& hidden_state,
                std::vector<float>& cell_state,
                bool training = false) {
        // Implementation of forward pass
        // (Similar to original but with better memory management)
    }

    void backward(const std::vector<float>& input,
                 const std::vector<float>& gradient,
                 std::vector<float>& input_gradient) {
        // Implementation of backward pass
        // (Similar to original but with better gradient handling)
    }

private:
    void initialize_weights() {
        // Initialize weights using improved initialization schemes
    }

    int input_size_;
    int hidden_size_;
    LayerNormalization layer_norm_;
    // Weight matrices and other member variables
};

// Main network class with improved architecture
class Network {
public:
    Network() {
        initialize();
    }

    float forward(const std::vector<std::vector<float>>& sequence, bool training = false) {
        // Implementation of forward pass
        return 0.0f; // Placeholder
    }

    void backward(const std::vector<std::vector<float>>& sequence, float target) {
        // Implementation of backward pass
    }

private:
    void initialize() {
        // Initialize network components
    }

    std::vector<std::unique_ptr<LSTMCell>> layers_;
    // Other member variables
};

// Trainer class to handle training logic
class Trainer {
public:
    Trainer(Network& presence_net, Network& service_net)
        : presence_net_(presence_net),
          service_net_(service_net) {}

    void train_batch(const std::vector<std::vector<std::vector<float>>>& batch,
                    const std::vector<bool>& targets) {
        // Implementation of batch training
    }

private:
    Network& presence_net_;
    Network& service_net_;
    // Training state variables
};

// Hardware interface class
class HardwareInterface {
public:
    static void initialize() {
        pinMode(Config::GPIO::PIR_PIN, INPUT);
        pinMode(Config::GPIO::SERVICE_PIN, INPUT_PULLUP);
        pinMode(Config::GPIO::PRESENCE_PWM, OUTPUT);
        pinMode(Config::GPIO::SERVICE_PWM, OUTPUT);

        ledcSetup(0, Config::GPIO::PWM_FREQ, Config::GPIO::PWM_RESOLUTION);
        ledcSetup(1, Config::GPIO::PWM_FREQ, Config::GPIO::PWM_RESOLUTION);
        ledcAttachPin(Config::GPIO::PRESENCE_PWM, 0);
        ledcAttachPin(Config::GPIO::SERVICE_PWM, 1);
    }

    static bool read_pir() {
        return digitalRead(Config::GPIO::PIR_PIN);
    }

    static bool read_service() {
        return !digitalRead(Config::GPIO::SERVICE_PIN);
    }

    static void set_presence_pwm(float value) {
        ledcWrite(0, static_cast<int>(value * 255));
    }

    static void set_service_pwm(float value) {
        ledcWrite(1, static_cast<int>(value * 255));
    }
};

// Main application class
class HotWaterManager {
public:
    HotWaterManager() : trainer_(presence_net_, service_net_) {
        HardwareInterface::initialize();
    }

    void setup() {
        Serial.begin(115200);
    }

    void loop() {
        unsigned long current_time = millis();
        
        try {
            update(current_time);
            sample_and_train(current_time);
            debug_output(current_time);
        }
        catch (const NetworkException& e) {
            Serial.printf("Network error: %s\n", e.what());
        }
        catch (const std::exception& e) {
            Serial.printf("Error: %s\n", e.what());
        }
    }

private:
    void update(unsigned long current_time);
    void sample_and_train(unsigned long current_time);
    void debug_output(unsigned long current_time);

    Network presence_net_;
    Network service_net_;
    Trainer trainer_;
    // Other member variables
};

// Global application instance
HotWaterManager app;

// Arduino entry points
void setup() {
    app.setup();
}

void loop() {
    app.loop();
}
