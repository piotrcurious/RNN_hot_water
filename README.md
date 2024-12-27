# RNN_hot_water
AI generated predictor trying to guess when to turn on hot water boiler

Two inputs determine presence of human (f.e. PIR sensor) and service request (turning on the water tap, f.e. from pump , pressostat or switch)


System predicts probability of human present and service request. 


it can be directly used to pre heat water in small boiler under the sink .
Two outputs for more versatility (f.e. ability to pre-heat also bigger boiler, drive pump , infer temperature settings etc)


Purpose - usually solar hot water systems have long pipe so users waste water waiting for hot water to arrive.

Adding small electric heater just under the sink or shower can help, as when running frequently it does not turn on because bulk hot water from solar heater keeps it warm, but it wastes electricity if running when f.e. no one is present. 


Ideally the electric heater should only turn on when "expecting" someone will use it, and just presence of a human is misleading. 


This device solves the problem. 
