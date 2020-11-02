<!-- ABOUT THE PROJECT -->
## About The Project

A Q learning Reinforcement Learning Project to learn to play the 2048 game. https://play2048.co/
The agent is able to achieve a highest tile of 2048 while training and 1024 while testing.
  
 


### Built With
This project relies mainly on the following `python 3.6` modules. 

* `gym-2048` - 2048 environment for gym
* `pytorch` - For model training
* `tensorboard` - For logging



<!-- GETTING STARTED -->
## Getting Started

To get a copy up and running follow these steps.

### Prerequisites

* virtualenv
https://gist.github.com/Geoyi/d9fab4f609e9f75941946be45000632b 

* requirements
    ```shell script
    source env/bin/activate
    pip install -r requirements.txt
    ```

### Usage

1. Clone the repo
    ```shell script
    git clone https://github.com/karkinissan/q_learning_2048.git
    ```
2. Navigate into project directory
    ```shell script
    cd q_learning_2048
    ```
3. Create and activate a virtual environment
    ```shell script
    python3 -m venv env
    source env/bin/activate
    ```
4. Install requirements
    ```shell script
    pip install -r requrements.txt
    ```
5. Run the application:  
    ```shell script
    python dqn_gym.py
   ```
### Testing 
The models are saved to ./runs/<log_directory>.  
Copy the model files and the record csv to "player_notebooks" directory and run the notebooks. 

