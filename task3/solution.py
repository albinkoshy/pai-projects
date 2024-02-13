"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF, ConstantKernel
import matplotlib.pyplot as plt


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
np.random.seed(21)

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        
        objective_kernel = Matern(length_scale=0.225, nu=2.5)
        self.objective_model = GaussianProcessRegressor(kernel=objective_kernel)
        
        constraint_kernel = ConstantKernel(4) + (DotProduct() + Matern(length_scale=0.225, nu=2.5))
        self.constraint_model = GaussianProcessRegressor(kernel=constraint_kernel)
        
        self.x = []
        self.yf = []
        self.yv = []
        
        self.init_point_unpacked = False

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        mean, std = self.objective_model.predict(x, return_std=True)
        mean_constraint, std_constraint = self.constraint_model.predict(x, return_std=True)
        return mean + 2.2*std - 10*max(mean_constraint + 0.225*std_constraint, 0)

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.        
        self.x.append(x)
        self.yf.append(f)
        self.yv.append(v)
        
        if not self.init_point_unpacked:
            self.x[0] = self.x[0][0]
            self.init_point_unpacked = True
        
        self.objective_model.fit(np.array(self.x).reshape(-1, 1), np.array(self.yf))
        self.constraint_model.fit(np.array(self.x).reshape(-1, 1), np.array(self.yv))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        def objective(x):
            x = np.atleast_2d(x)
            return -(self.objective_model.predict(x) - 10*max(self.constraint_model.predict(x), 0))

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        print("Plotting...")
         
        X = np.linspace(start=0, stop=10, num=1000).reshape(-1, 1)
        mean_obj, std_obj = self.objective_model.predict(X, return_std=True)
        mean_con, std_con = self.constraint_model.predict(X, return_std=True)

        plt.subplot(2, 2, 1)
        plt.scatter(self.x, self.yf)
        plt.plot(X, mean_obj)
        plt.fill_between(
            X.ravel(),
            mean_obj - 1.96 * std_obj,
            mean_obj + 1.96 * std_obj,
            alpha=0.5,
        )
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        _ = plt.title("Objective Model")
        
        plt.subplot(2, 2, 2)
        plt.scatter(self.x, self.yv)
        plt.plot(X, mean_con)
        plt.fill_between(
            X.ravel(),
            mean_con - 1.96 * std_con,
            mean_con + 1.96 * std_con,
            alpha=0.5,
        )
        plt.xlabel("$x$")
        plt.ylabel("$v(x)$")
        _ = plt.title("Constraint Model")
        
        plt.show()

# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    return 10*np.sin(x) + 20


def v(x: float):
    """Dummy SA"""
    return np.sin(x) + 4


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    import warnings
    warnings.filterwarnings("ignore")
    
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point([np.array(x_init)], obj_val, cost_val)

    # Loop until budget is exhausted
    for _ in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()*0.0015
        cost_val = v(x) + np.random.randn()*0.0001
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    agent.plot()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
