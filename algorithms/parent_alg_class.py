# Pass in the hyperparameters as a part of the init.
# Look into a way to save classes lol or something like that


import time
import numpy as np
import matplotlib.pyplot as plt
from .data_saving import check_for_id_already, read_list_of_lists_from_csv, save_data

def rolling_avg(arr, dist):
    new = []
    for i in range(len(arr)-dist):
        bef = i
        after = i + 5
        new_val = arr[bef:after].mean()
        new.append(new_val)
    
    return new


class algParent:
    """
    Parent class for algorithm implementations.
    Provides common functionality for running tests, tracking performance metrics,
    and displaying results.
    """

    
    def __init__(self, env, hyper_params, learning_model):
        """
        Initialize the algorithm parent class.
        
        Args:
            env: Environment object that the algorithm will interact with
        """
        self.env = env  # Store the environment reference
        self.hypers = hyper_params
        self.loss_lists = []  # List to store loss values from each test
        self.rewards_lists = []  # List to store reward values from each test
        self.loss_lists_temp = []  
        self.rewards_lists_temp = [] 
        self.test_times = []  # List to store execution times for each test
        self.learning_model = learning_model
        self.alg = 'None'
        

    def return_params(self):
        params = {
            'model_opt' : self.learning_model.return_hypers(),
            'hypers' : self.hypers,
            'env' : self.env.spec.id,
            'alg_name' : self.alg
        }
        return params
    

    def return_results(self):

        params = self.return_params()

        losses = self.loss_lists.copy()
        rewards = self.rewards_lists.copy()

        self.loss_lists = []
        self.rewards_lists = []
        return params, losses, rewards



    def avg_test_time(self):
        """
        Calculate the average time taken per test.
        
        Returns:
            float: Average test time in seconds, or 0 if no tests have been run
        """
        if not self.test_times:  # Check if test_times list is empty
            return 0  # Return 0 if no tests have been completed yet
        return sum(self.test_times)/len(self.test_times)  # Calculate and return average
    
    def single_test(self):
        """
        Placeholder method for a single test execution.
        This should be implemented by child classes with specific test logic.
        """
        pass  # To be implemented by child classes
    
    def run(self, tests, test_hypers, sound):
        """
        Run multiple tests and track performance metrics.
        
        Args:
            tests (int): Number of tests to run
            test_hypers (dict): Hyperparameters to use for the tests
        """
        
        for i in range(tests):  # Loop through the specified number of tests
            
            time1 = time.time()  # Record start time of current test
            
            self.single_test(test_hypers, sound)  # Execute a single test with given hyperparameters
            
            time2 = time.time()  # Record end time of current test
            self.test_times.append(time2-time1)  # Store the duration of this test

            self.save_results()
            
            avg = self.avg_test_time()  # Calculate average test time so far
            remaining_tests = tests - (i+1)  # Calculate how many tests are left
            
            if sound:
                if remaining_tests > 0:  # If there are more tests to run
                    estimated_time_left = avg * remaining_tests  # Estimate total remaining time
                    print(f'Completed {i+1}/{tests}. Estimated time left: {estimated_time_left:.2f} seconds')
                else:  # If this was the last test
                    print(f'Completed all {tests} tests!')
                
    def save_results(self):

        alg_data, loss_lists, rewards_lists = self.return_results()

        save_data(loss_lists, rewards_lists, alg_data)

    def display_results(self):

        self.save_results()

        alg_id = check_for_id_already("results/model_alg_data.ndjson", self.return_params())
        print(f"Id: {alg_id}")
        rewards = read_list_of_lists_from_csv(f'results/rewards_lists/{alg_id}.csv')

        rewards_ar = np.array(rewards).astype(np.int_)

        low = np.quantile(rewards_ar, 0.25, axis = 0)
        high = np.quantile(rewards_ar, 0.75, axis = 0) 
        median = np.quantile(rewards_ar, 0.5, axis = 0)

        '''smooth_low = rolling_avg(low, 10)
        smooth_high = rolling_avg(high, 10)
        smooth_median = rolling_avg(median, 10)'''

        plt.plot(median[0:500], linestyle='-', color='blue')
        plt.plot(high[0:500], linestyle='--', color='green')
        plt.plot(low[0:500], linestyle='--', color='red')
        plt.show()