#include <Arduino.h>
#include <vector>
#include <cmath>

// Pin definitions
const int PIR_PIN = 4;        // PIR sensor input
const int SERVICE_PIN = 5;     // Service request switch
const int PRESENCE_PWM = 18;   // Presence probability output
const int SERVICE_PWM = 19;    // Service probability output

// Time constants
const int SAMPLES_PER_DAY = 96;  // 15-minute intervals
const int DAYS_PER_WEEK = 7;
const unsigned long SAMPLE_INTERVAL = 900000; // 15 minutes in milliseconds

// RNN parameters
const int HIDDEN_SIZE = 8;
const float LEARNING_RATE = 0.01;
const float MOMENTUM = 0.9;

struct RNNCell {
    float hidden[HIDDEN_SIZE];
    float output;
    
    // Weights
    float Wxh[3][HIDDEN_SIZE];  // Input to hidden weights (time, weekday, sensor)
    float Whh[HIDDEN_SIZE][HIDDEN_SIZE];  // Hidden to hidden weights
    float Why[HIDDEN_SIZE];  // Hidden to output weights
    
    // Previous weight updates (for momentum)
    float prev_Wxh[3][HIDDEN_SIZE];
    float prev_Whh[HIDDEN_SIZE][HIDDEN_SIZE];
    float prev_Why[HIDDEN_SIZE];
};

// Two separate RNN cells for presence and service prediction
RNNCell presence_rnn;
RNNCell service_rnn;

// Circular buffer for storing historical data
struct DataPoint {
    bool presence;
    bool service;
    int timeSlot;
    int weekDay;
};

std::vector<DataPoint> history;
const int HISTORY_SIZE = SAMPLES_PER_DAY * DAYS_PER_WEEK * 4;  // 4 weeks of history

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void initRNN(RNNCell &cell) {
    // Initialize weights with small random values
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            cell.Wxh[i][j] = (random(1000) - 500) / 5000.0;
            cell.prev_Wxh[i][j] = 0;
        }
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            cell.Whh[i][j] = (random(1000) - 500) / 5000.0;
            cell.prev_Whh[i][j] = 0;
        }
        cell.Why[i] = (random(1000) - 500) / 5000.0;
        cell.prev_Why[i] = 0;
        cell.hidden[i] = 0;
    }
    
    cell.output = 0;
}

float forwardPass(RNNCell &cell, float timeNorm, float weekDayNorm, float sensorInput) {
    float new_hidden[HIDDEN_SIZE] = {0};
    
    // Input to hidden
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        new_hidden[h] += cell.Wxh[0][h] * timeNorm;
        new_hidden[h] += cell.Wxh[1][h] * weekDayNorm;
        new_hidden[h] += cell.Wxh[2][h] * sensorInput;
        
        // Hidden to hidden
        for (int h2 = 0; h2 < HIDDEN_SIZE; h2++) {
            new_hidden[h] += cell.Whh[h2][h] * cell.hidden[h2];
        }
        new_hidden[h] = sigmoid(new_hidden[h]);
    }
    
    // Update hidden state
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        cell.hidden[h] = new_hidden[h];
    }
    
    // Hidden to output
    float output = 0;
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        output += cell.Why[h] * cell.hidden[h];
    }
    cell.output = sigmoid(output);
    
    return cell.output;
}

void backPropagate(RNNCell &cell, float timeNorm, float weekDayNorm, float sensorInput, float target) {
    float output_delta = (cell.output - target) * cell.output * (1 - cell.output);
    float hidden_delta[HIDDEN_SIZE];
    
    // Calculate hidden layer deltas
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        hidden_delta[h] = output_delta * cell.Why[h] * cell.hidden[h] * (1 - cell.hidden[h]);
    }
    
    // Update weights with momentum
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        float why_update = LEARNING_RATE * output_delta * cell.hidden[h] + MOMENTUM * cell.prev_Why[h];
        cell.Why[h] -= why_update;
        cell.prev_Why[h] = why_update;
        
        // Input to hidden weights
        float wxh_update = LEARNING_RATE * hidden_delta[h] * timeNorm + MOMENTUM * cell.prev_Wxh[0][h];
        cell.Wxh[0][h] -= wxh_update;
        cell.prev_Wxh[0][h] = wxh_update;
        
        wxh_update = LEARNING_RATE * hidden_delta[h] * weekDayNorm + MOMENTUM * cell.prev_Wxh[1][h];
        cell.Wxh[1][h] -= wxh_update;
        cell.prev_Wxh[1][h] = wxh_update;
        
        wxh_update = LEARNING_RATE * hidden_delta[h] * sensorInput + MOMENTUM * cell.prev_Wxh[2][h];
        cell.Wxh[2][h] -= wxh_update;
        cell.prev_Wxh[2][h] = wxh_update;
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
    ledcSetup(0, 5000, 8);  // Channel 0, 5kHz, 8-bit resolution
    ledcSetup(1, 5000, 8);  // Channel 1, 5kHz, 8-bit resolution
    ledcAttachPin(PRESENCE_PWM, 0);
    ledcAttachPin(SERVICE_PWM, 1);
    
    // Initialize RNNs
    initRNN(presence_rnn);
    initRNN(service_rnn);
    
    // Reserve space for history
    history.reserve(HISTORY_SIZE);
}

void loop() {
    static unsigned long lastSampleTime = 0;
    unsigned long currentTime = millis();
    
    // Get current time slot and week day
    time_t now;
    time(&now);
    struct tm timeinfo;
    localtime_r(&now, &timeinfo);
    
    int currentTimeSlot = (timeinfo.tm_hour * 4) + (timeinfo.tm_min / 15);
    int currentWeekDay = timeinfo.tm_wday;
    
    // Normalize inputs
    float timeNorm = (float)currentTimeSlot / SAMPLES_PER_DAY;
    float weekDayNorm = (float)currentWeekDay / DAYS_PER_WEEK;
    
    // Read sensors
    bool pirDetected = digitalRead(PIR_PIN);
    bool serviceRequested = !digitalRead(SERVICE_PIN);  // Active low
    
    // Sample and train every 15 minutes
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
        lastSampleTime = currentTime;
        
        // Add new data point to history
        DataPoint newPoint = {
            .presence = pirDetected,
            .service = serviceRequested,
            .timeSlot = currentTimeSlot,
            .weekDay = currentWeekDay
        };
        
        if (history.size() >= HISTORY_SIZE) {
            history.erase(history.begin());
        }
        history.push_back(newPoint);
        
        // Train RNNs with historical data
        for (const auto &point : history) {
            float time_norm = (float)point.timeSlot / SAMPLES_PER_DAY;
            float day_norm = (float)point.weekDay / DAYS_PER_WEEK;
            
            // Train presence RNN
            float presence_pred = forwardPass(presence_rnn, time_norm, day_norm, point.presence);
            backPropagate(presence_rnn, time_norm, day_norm, point.presence, point.presence);
            
            // Train service RNN
            float service_pred = forwardPass(service_rnn, time_norm, day_norm, point.service);
            backPropagate(service_rnn, time_norm, day_norm, point.service, point.service);
        }
    }
    
    // Make predictions
    float presence_prob = forwardPass(presence_rnn, timeNorm, weekDayNorm, pirDetected);
    float service_prob = forwardPass(service_rnn, timeNorm, weekDayNorm, serviceRequested);
    
    // Boost presence probability if service is requested
    if (serviceRequested) {
        presence_prob = presence_prob * 1.5;
        if (presence_prob > 1.0) presence_prob = 1.0;
    }
    
    // Update PWM outputs
    ledcWrite(0, (int)(presence_prob * 255));
    ledcWrite(1, (int)(service_prob * 255));
    
    // Debug output
    Serial.printf("Time: %02d:%02d Day: %d PIR: %d SRV: %d Presence: %.2f Service: %.2f\n",
                 timeinfo.tm_hour, timeinfo.tm_min, currentWeekDay,
                 pirDetected, serviceRequested, presence_prob, service_prob);
    
    delay(1000);  // Basic rate limiting for the main loop
}
