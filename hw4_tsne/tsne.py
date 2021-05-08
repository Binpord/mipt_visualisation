import numpy as np


def tsne(
    input_points: np.ndarray,
    target_dimensions: int = 2,
    perplexity: float = 30.0,
    perplexity_tolerance: float = 1e-5,
    gradient_descent_num_iterations: int = 1000,
    gradient_descent_learning_rate: float = 5e-15,
) -> np.ndarray:
    """
    t-SNE or t-distributed stochastic neighbor embedding

    Statistical method for high-dimensional data visualization via dimensions reduction.
    Works via assigning a gaussian to each original data point and minimizing KL-divergence
    from that distribution with the t-Student distribution over the low-dimensional
    (typically, 2 or 3 for the purpose of the visualization) points.

    Parameters:
        input_points (np.ndarray, shape [N, d]): input data, N d-dimensional data points
        target_dimensions (int): number of dimensions to reduce input data to
        perplexity (float): predefined perplexity to choose data points' gaussians
                            standard deviation, typically a value between 5 and 50
        perplexity_tolerance (float): tolerance for the standard deviation bisection process

    Returns:
        output_points (np.ndarray, shape [N, target_dimensions]): result of dimensionality
                                                                  reduction
    """

    # First I approximate input points probabilities distributions p_{ij}, where
    #
    #                p_{ij} = (p_{j|i} + p_{i|j}) / (2 * N)
    #
    #                  exp(- norm(x_i - x_j) ** 2 / (2 * sigma_i ** 2))
    # p_{j|i} = ---------------------------------------------------------------
    #            sum_{k != i} exp(- norm(x_i - x_k) ** 2 / (2 * sigma_i ** 2))
    #
    # where norm is the Frobenius vector norm (np.linalg.norm) and sigma_i is the
    # standard deviation for the i-th point, which is selected so that perplexity
    # of the p_{.|i} distribution is close to the predefined perplexity given as
    # one of the function parameters.
    input_probabilities = calculate_input_probabilities(
        input_points, perplexity, perplexity_tolerance
    )

    # Now, given the probability distribution of input points, we want to choose
    # random initial output points and then minimize the KL-divergence between
    # their probability distribution and the one from input points.
    output_points = find_output_points(
        input_probabilities,
        target_dimensions,
        gradient_descent_num_iterations,
        gradient_descent_learning_rate,
    )

    return output_points


def calculate_input_probabilities(
    input_points: np.ndarray, perplexity: float, tolerance: float
) -> np.ndarray:
    """
    Computes input points probabilities distribution via assigning a gaussian to each point
    and calculating probabilities of all other points based on that. Standard deviation is
    selected so that the perplexity of the normal distribution is close to the predefined
    (in terms of tolerance).

    Parameters:
        input_points (np.ndarray, shape [N, d]): input data, N d-dimensional data points
        perplexity (float): predefined perplexity to match
        tolerance (float): perplexity matching tolerance

    Returns:
        input_probabilities (np.ndarray, shape [N, N]): input points probability distribution
    """

    # First I compute element wise squared distances. To do that I first compute
    # element wise differences between points and then square them so that each
    # D[i, j] = (x[i] - x[j]).T @ (x[i] - x[j]).

    # Point wise differences. This one-liner magic works because input_points
    # has a shape of [N, d], hence input_points[:, None, :] has shape [N, 1, d]
    # and input_points[None, :, :] has shape [1, N, d].
    # For the subtraction, numpy will broadcast shapes of these arrays, hence it
    # will repeat input_points[:, None, :] N times by the second axis and repeat
    # input_points[None, :, :] N times by the first one, effectively yielding
    # array of N arrays with N times repeated point vectors as the first term
    # and array of N repeating arrays of all the points as the second.
    # Hence the difference will be array of point wise differences.
    point_differences = input_points[:, None, :] - input_points[None, :, :]

    # Now I have point wise differences as vectors. Shape of the array is
    # [N, N, d]. Now I want to produce the array of shape [N, N] with the
    # squared distances, i.e. squared norms of these vectors.
    # I do so by flattening array of arrays of differences into array of
    # differences via .reshape(N * N, d) and then I unsqueeze second dim
    # in one copy and last dim in another and compute dot product, which
    # yields array of shape [N * N, 1], which I can reshape back to [N, N]
    # to obtain the result.
    num_points, point_dimensions = input_points.shape
    point_differences = point_differences.reshape(num_points * num_points, point_dimensions)
    squared_distances = (point_differences[:, None, :] @ point_differences[:, :, None]).reshape(
        num_points, num_points
    )

    # First I select standard deviations (aka normal distribution sigmas) so that each
    # gaussian has perplexity close to the predefined in terms of tolerance.
    sigmas = select_standard_deviations(squared_distances, perplexity, tolerance)

    # Then I calculate probabilities of other points based on the selected gaussians.
    # Probabilities are proportional to the gaussian exponent of negative distance
    # divided by the gaussian sigma.
    point_probabilities = np.exp(-squared_distances / sigmas)

    # Now we actually want conditional probabilities of the point given itself to be zero.
    np.fill_diagonal(point_probabilities, 0.0)

    # Now we have N arrays with probabilities, which sum up to anything and therefore
    # cannot be treated as the probability distribution. That is why I normalize them by
    # dividing them by their sums.
    point_probabilities /= point_probabilities.sum(axis=1)

    # Now we have N probability distributions. However we want one distribution, plus we
    # will be fitting the t-Student distribution to it and t-Student distribution is symmetric
    # hence, we want to symmetrize the distribution via
    # p_{ij} = (p_{j|i} + p_{i|j}) / 2
    # and then divide all probabilities by N so that all probabilities sum up to 1
    # and hence form one distribution.
    input_probabilities = (point_probabilities + point_probabilities.T) / (2 * len(input_points))

    return input_probabilities


def find_output_points(
    input_probabilities: np.ndarray,
    target_dimensions: int,
    num_iterations: int = 1000,
    learning_rate: float = 5e-15,
) -> np.ndarray:
    """
    Finds output points by minimizing KL-divergence between t-Student distribution over the output
    points and the input probabilities, which is given by gaussians.

    Parameters:
        input_probabilities (np.ndarray, shape [N, N]): input probability distribution
        target_dimensions (int): number of dimensions in output variables
        num_iterations (int): number of gradient descent iterations
        learning_rate (float): learning rate for the gradient descent

    Returns:
        output_points (np.ndarray, shape [N, target_dimensions]): output points
    """

    # Start with random initialization.
    num_points = len(input_probabilities)
    output_points = np.random.rand(num_points, target_dimensions)

    for iteration in range(num_iterations):
        # First we calculate point wise differences and inverted squared distances as they will
        # serve as basis for the t-Student distribution and hence will be used much in gradient
        # computation.
        point_differences = output_points[:, None, :] - output_points[None, :, :]
        flat_point_differences = point_differences.reshape(
            num_points * num_points, target_dimensions
        )
        squared_distances = (
            flat_point_differences[:, None, :] @ flat_point_differences[:, :, None]
        ).reshape(num_points, num_points)
        inverted_squared_distances = 1.0 / (1.0 + squared_distances)

        # Fill diagonal with zeros so that it won't be messing with computations and compute
        # output probabilities.
        np.fill_diagonal(inverted_squared_distances, 0.0)
        output_probabilities = inverted_squared_distances / inverted_squared_distances.sum()

        if iteration % 100 == 0:
            print(
                f"Iteration {iteration}:\tKL-div equals",
                kl_divergence(input_probabilities, output_probabilities),
            )

        # Now we compute gradient.
        # Formula for the gradient can be found in the original article or on many other sources,
        # such as https://opentsne.readthedocs.io/en/latest/tsne_algorithm.html.
        gradient = 4 * np.sum(
            ((input_probabilities - output_probabilities) * inverted_squared_distances).reshape(
                num_points, num_points, 1
            )
            * point_differences,
            axis=1,
        )

        # Update output points and go to the next iteration.
        output_points -= learning_rate * gradient

    return output_points


def select_standard_deviations(
    squared_distances: np.ndarray,
    predefined_perplexity: float,
    tolerance: float,
    max_tries: int = 50,
) -> np.ndarray:
    """
    Selects standard deviations for each sigmoid centered on each data point so that
    probability distribution given by it has perplexity close to the predefined
    in terms of tolerance.

    Parameters:
        squared_distances (np.ndarray, shape [N, N]): squared distances between data points
        predefined_perplexity (float): predefined perplexity
        tolerance (float): tolerance
        max_tries (int): maximum amount of tries for the bisection process

    Returns:
        sigmas (np.ndarray, shape [N]): standard deviations
    """

    # Each row of squared_distances contains 0.0 for distance between the point and itself.
    # More specifically diagonal consists of zeros. And we don't want them to spoil our
    # calculation of the perplexity. That is why I delete the diagonal elements.
    num_points = len(squared_distances)
    squared_distances = squared_distances[~np.eye(num_points, dtype=bool)].reshape(
        num_points, num_points - 1
    )

    # The process of selection of the sigmas for the gaussians cannot be effectively
    # parallelized and has to be done point by point.
    sigmas = []
    for point_squared_distances in squared_distances:
        # Each sigma is selected via the bisection process.

        # Initial search value is 1.0.
        sigma = 1.0

        # Now I do the bisection search.
        # Initialize min sigma with 0.0 and max sigma with None (i.e. unknown).
        min_sigma = 0.0
        max_sigma = None
        for _ in range(max_tries):
            # Given current sigma, calculate the perplexity.
            perplexity = calculate_perplexity(point_squared_distances, sigma)

            # If we found such sigma that resulting perplexity is close to
            # the predefined one, we end the bisection process.
            if abs(perplexity - predefined_perplexity) < tolerance:
                break

            # Else we update the min, max and current sigmas.
            if perplexity > predefined_perplexity:
                # If current perplexity is bigger than we want, we increase the sigma,
                # effectively smearing the distribution and lowering the perplexity.
                min_sigma = sigma
                if max_sigma is not None:
                    sigma = (sigma + max_sigma) / 2
                else:
                    sigma *= 2
            else:
                # If on the other hand we actually ended up with lower perplexity, than
                # we expected, then we need to decrease sigma in order to increase perplexity.
                max_sigma = sigma
                sigma = (sigma + min_sigma) / 2

        sigmas.append(sigma)

    return np.array(sigmas)


def calculate_perplexity(squared_distances: np.ndarray, sigma: float) -> float:
    """
    Calculates the perplexity of normal distribution given the squared distances to
    other points and standard deviation.

    Parameters:
        squared_distances (np.ndarray, shape [N - 1]): squared distances from current point
                                                       to all others except itself
        sigma (float): standard deviation

    Returns:
        perplexity (float): resulting perplexity
    """

    # First I calculate point probabilities. Probabilities are proportional to the gaussian
    # exponent of negative distance divided by the gaussian sigma.
    point_probabilities = np.exp(-squared_distances / sigma)

    # Now we need to normalize probabilities such that their sum equals to 1.

    # Because of the numerical instabilities we may end up with array of zeros at this point.
    # In order to address this problem, when I detect such case, I assign uniform distribution
    # to the point_probabilities.
    sum_probabilities = point_probabilities.sum()
    if abs(sum_probabilities) < 1e-12:
        point_probabilities = np.full_like(point_probabilities, 1.0 / len(point_probabilities))
    else:
        point_probabilities /= sum_probabilities

    # Now we are ready to compute perplexity. It is computed via exp(entropy(P)), and
    # entropy(P) equals sum p(x) * log p(x).
    # perplexity = np.exp(-np.sum(point_probabilities * np.log(point_probabilities)))
    perplexity = np.exp(-np.sum(point_probabilities * np.ma.log(point_probabilities)))

    return perplexity


def kl_divergence(input_probabilities: np.ndarray, output_probabilities: np.ndarray) -> float:
    """
    Computes KL-divergence of two discrete probability distributions given by the numpy arrays.
    Normally, distributions should have same same probability space and all the probabilities
    should be higher than zero. However, due to numerical instabilities, input and output
    probabilities can have zeros in them, which is not a good thing for division and logarithm
    operations.
    That is why I use numpy masked operations to mitigate this fact.

    Parameters:
        input_probabilities (np.ndarray, shape [N, N]): input probabilities
        output_probabilities (np.ndarray, shape [N, N]): output probabilities

    Returns:
        kl_div (float): KL-divergence value
    """
    return np.sum(
        input_probabilities * np.ma.log(np.ma.divide(input_probabilities, output_probabilities))
    )
