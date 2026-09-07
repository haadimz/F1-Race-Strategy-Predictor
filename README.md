# F1-Race-Strategy-Predictor
A Python-based data analysis tool that simulates and optimises Formula 1 race strategies. It processes historical timing and telemetry data to model tire degradation, pit stop timing, and optimal compound selection for any given race

**WORK IN PROGRESS:**
The core combinatorial optimisation logic is currently complete. The engine successfully generates optimal race strategies based on tire degradation data, fuel consumption calculations and mandatory F1 stint regulations

**CORE FEATURES:**
* Extract and process historical API data, telemetry, and official timing data using the fastf1 Python library
* Model non-linear tire wear and predict lap pace degradation over time by implementing polynomial regression via scikit-learn
* Simulate complex race scenarios by evaluating thousands of potential pit stop windows and tire compound permutations
* Compute the mathematically optimal race strategy by utilising itertools for combinatorial optimisation of total race times
* Generate comparative data visualisations to analyse team pace, stint lengths, and strategic variations across different drivers 

**PLANNED FEATURES:**
* Factoring in safety car probabilities to dynamically adjust pit-window risk
* Integrating track temperature and historical weather data to calculate higher fidelity degradation curves
