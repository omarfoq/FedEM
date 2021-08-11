"""
Generates synthetic dataset following a mixture model.
"""
import os
import pickle
import argparse
import numpy as np

from sklearn.utils import shuffle

BOX = (-1.0, 1.0)
PATH = "all_data/"


class SyntheticMixtureDataset:
    def __init__(
            self,
            n_classes,
            n_components,
            n_tasks,
            dim,
            noise_level,
            alpha,
            box,
            uniform_marginal,
            seed
    ):

        if n_classes != 2:
            raise NotImplementedError("Only binary classification is supported for the moment")

        self.n_classes = n_classes
        self.n_components = n_components
        self.n_tasks = n_tasks
        self.dim = dim
        self.noise_level = noise_level
        self.alpha = alpha * np.ones(n_components)
        self.box = box
        self.uniform_marginal = uniform_marginal
        self.seed = seed

        np.random.seed(self.seed)
        self.num_samples = get_num_samples(self.n_tasks)

        self.theta = np.zeros((self.n_components, self.dim))
        self.mixture_weights = np.zeros((self.n_tasks, self.n_components))

        self.generate_mixture_weights()
        self.generate_components()

    def generate_mixture_weights(self):
        for task_id in range(self.n_tasks):
            self.mixture_weights[task_id] = np.random.dirichlet(alpha=self.alpha)

    def generate_components(self):
        self.theta = np.random.uniform(BOX[0], BOX[1], size=(self.n_components, self.dim))

    def generate_data(self, task_id, n_samples=10_000):
        latent_variable_count = np.random.multinomial(n_samples, self.mixture_weights[task_id])
        y = np.zeros(n_samples)

        if self.uniform_marginal:
            x = np.random.uniform(self.box[0], self.box[1], size=(n_samples, self.dim))
        else:
            raise NotImplementedError("Only uniform marginal is available for the moment")

        current_index = 0
        for component_id in range(self.n_components):
            y_hat = x[current_index:current_index + latent_variable_count[component_id]] @ self.theta[component_id]
            noise = np.random.normal(size=latent_variable_count[component_id], scale=self.noise_level)
            y[current_index: current_index + latent_variable_count[component_id]] = \
                np.round(sigmoid(y_hat + noise)).astype(int)

        return shuffle(x, y)

    def save_metadata(self, path_):
        metadata = dict()
        metadata["mixture_weights"] = self.mixture_weights
        metadata["theta"] = self.theta

        with open(path_, 'wb') as f:
            pickle.dump(metadata, f)


def save_data(x, y, path_):
    data = list(zip(x, y))
    with open(path_, 'wb') as f:
        pickle.dump(data, f)


def get_num_samples(num_tasks, min_num_samples=50, max_num_samples=1000):
    num_samples = np.random.lognormal(4, 2, num_tasks).astype(int)
    num_samples = [min(s + min_num_samples, max_num_samples) for s in num_samples]
    return num_samples


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        required=True,
        type=int
    )
    parser.add_argument(
        '--n_classes',
        help='number of classes;',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of classes;',
        type=int,
        default=3
    )
    parser.add_argument(
        '--dimension',
        help='data dimension;',
        type=int,
        default=150
    )
    parser.add_argument(
        '--noise_level',
        help='Noise level;',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--n_test',
        help='size of test set;',
        type=int,
        default=5_000
    )
    parser.add_argument(
        '--train_tasks_frac',
        help='fraction of tasks / clients  participating to the training; default is 0.0',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar;',
        type=float,
        default=0.4
    )
    parser.add_argument(
        '--uniform_marginal',
        help='flag indicating if the all tasks should have the same marginal;',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=12345,
        required=False
    )
    return parser.parse_args()


def main():
    os.makedirs(PATH, exist_ok=True)
    args = parse_args()

    np.random.seed(args.seed)
    dataset = \
        SyntheticMixtureDataset(
            n_components=args.n_components,
            n_classes=args.n_classes,
            n_tasks=args.n_tasks,
            dim=args.dimension,
            noise_level=args.noise_level,
            alpha=args.alpha,
            uniform_marginal=args.uniform_marginal,
            seed=args.seed,
            box=BOX,
        )

    dataset.save_metadata(os.path.join(PATH, "meta.pkl"))

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    for task_id in range(dataset.n_tasks):
        if task_id < int(args.train_tasks_frac*args.n_tasks):
            mode = "train"
        else:
            mode = "test"

        client_path = os.path.join(PATH, mode, "task_{}".format(task_id))
        os.makedirs(client_path, exist_ok=True)

        x_train, y_train = dataset.generate_data(task_id, dataset.num_samples[task_id])
        x_test, y_test = dataset.generate_data(task_id, args.n_test)

        save_data(x_train, y_train, os.path.join(client_path, "train.pkl"))
        save_data(x_test, y_test, os.path.join(client_path, "test.pkl"))


if __name__ == '__main__':
    main()
