## Robotaxi Relocation Optimization
This repository contains the code and project report for the Robotaxi Relocation Optimization project, completed as a part of the IEOR 290 course project.

## 📝 Project Overview

This project tackles the challenge of efficiently managing autonomous taxi (robotaxi) fleets in urban environments by optimizing the dispatch of idle vehicles. The primary goal is to maximize the trip fulfillment rate by strategically repositioning empty robotaxis to meet dynamic demand, thereby minimizing passenger wait times and vehicle idle time.

We investigate a fluid-based relocation policy, which uses continuum approximations to optimize the distribution of vehicles at a large scale. The performance of this policy is validated through simulations grounded in real-world demand data from New York City's for-hire vehicle trips.

## Repository Structure

The repository is organized into the following components:

`nyc_trip`

This folder contains the code for learning model parameters (arrival rate, etc.) and the relocation strategy matrix.

- Notebooks:
`1Learning Model Params.ipynb`: This notebook processes the NYC FHVHV trip data to learn the key parameters for our model, including time-varying rider arrival rates, trip completion rates, and rider transition matrices.

`2Solve Relocation Matrix.ipynb`: Implements the fluid-based optimization model to solve for the optimal static empty-car relocation matrix.

`3Constructing NYC Taxi Grid.ipynb`: This notebook is used for constructing the NYC taxi grid from shapefiles and visualizing the resulting relocation policies on the map of New York City.

- Python scripts:
`solve_relocation_matrix.py`: A Python script version of the optimization model for solving the relocation matrix.
`constants.py`: Contains all the constants used throughout the project, such as taxi zone IDs and excluded locations.

`simulate`

This folder contains the code to simulate the NYC taxi environment and backtest the relocation strategies.

- Notebooks:

`1simulate.ipynb`: This notebook is used to run the simulations. It is configured with parameters such as the length of time blocks, total simulation duration, number of taxis, and different relocation policies. It runs simulations in parallel across multiple seeds and saves the output logs.

`2analyze_simulation.ipynb`: This notebook is used to analyze the simulation outputs. It produces time series plots comparing the performance of each relocation strategy and computes system-level performance metrics.

- Python Scripts:

`simulate.py`: This file contains the core of the discrete-event taxi simulator. It defines classes for events, the event queue, and vehicles, and manages the simulation clock and event handling.

`relocation_policies.py`: This script defines the different relocation policies that are tested in the simulation, including blind sampling, Join-the-Least-Congested-Region (JLCR-eta), and the Shortest Wait policy.

`utils.py`: Contains utility functions for preparing real-world demand data from NYC trips, plotting simulation results, and computing various performance metrics from the simulation logs.

`constants.py`: Contains constants used in the simulation, such as event types, vehicle statuses, and location IDs.


## References

Braverman, A., Dai, J. G., Liu, X., & Ying, L. (2019). Empty-Car Routing in ridesharing systems. Operations Research, 67(5), 1437–1452. doi:10.1287/opre.2018.1822

New York City Taxi and Limousine Commission. TLC Trip Record Data. Accessed: 2025-05-02. 2025. url: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.