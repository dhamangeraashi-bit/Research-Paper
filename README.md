Adaptive Bayesian Trust Scoring (ABT) Framework 
Welcome to the repository for my computer science engineering research project. As a first-year student, I built this framework to tackle a major challenge in security: detecting and stopping session hijacking attacks in real-time.

The Core Problem
Most standard security systems only verify who you are once—right when you type your password to log in. However, if a hacker gains access to your active session after that initial check, standard systems will not notice.
The ABT framework fixes this flaw by continuously monitoring two separate data streams simultaneously to verify that the authorized user is still the one operating the machine:
Network Layer Traffic: It tracks background communication metrics like transmission speeds, protocol behaviors, and connection rates.
Behavioral Biometrics: It continuously measures user typing rhythms, specifically looking at how many milliseconds a user holds down a key and the transition delay between typing characters.

How the Code Works Under the Hood
The system handles data and analyzes threats using a three-part pipeline:
The Sliding Window: Instead of analyzing a single data packet completely isolated from context, the data loader bundles incoming packets into rolling timelines of ten consecutive steps. 
This allows the system to look at a small history of behavior over time.
The Hybrid Neural Network: The model runs two distinct types of deep learning layers back-to-back. First, a Convolutional layer sifts through forty-three separate data markers to instantly find patterns connecting network
signals with typing behaviors. Second, an LSTM recurrent layer takes those patterns and tracks how they evolve across the ten-step timeline to identify anomalies.
The Adaptive Bayesian Decision Engine: The system maintains a running trust score for the active user that starts at a perfect value of one. If behaviors remain normal, the score stays high. 
If minor anomalies occur, the score drops slightly. If a severe threat pattern is recognized, a mathematical decay equation causes the trust score to drop sharply.
Once the score falls below the critical threshold of zero point four, the system automatically revokes access and boots the session.

Project Structure
main.py: The complete python script handling data loading, the hybrid network layers, the model training engine, and the live trust evaluation.
.gitignore: A simple configuration file telling Git to exclude large text dataset benchmarks so they don't bloat my cloud repository.