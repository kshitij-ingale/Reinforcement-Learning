""" Script to generate neural networks for policy network and state value function estimator """

import math
import tensorflow as tf


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, units, norm, activation):
        """ Fully connected layer block consisting of dense -> (batch) norm -> activation

        Parameters
        ----------
        units : int
            Number of hidden units in fully connected layer
        norm : str
            Normalization layer to be used (batch/instance/None) 
        activation : function from tf.nn module
            Activation function to be used for output

        Raises
        ------
        NotImplementedError
            Instance norm requires tensorflow-addons as of now, so that is not 
            implemented yet
        SyntaxError
            If normalization is not specified from (batch, instance, None), error is raised
        """
        super().__init__()
        dense = tf.keras.layers.Dense(units)
        self.functions = [dense]
        if norm is not None:
            if norm == "batch":
                self.functions.append(tf.keras.layers.BatchNormalization())
            elif norm == "instance":
                raise NotImplementedError(
                    "Instance norm requires tensorflow-addons (to be added later)"
                )
            else:
                raise SyntaxError(f"{norm} normalization not found")
        if activation:
            self.functions.append(activation)

    def call(self, x):
        for function in self.functions:
            x = function(x)
        return x


class Net(tf.keras.Model):
    def __init__(
        self, state_dim, action_dim, net_params, name, continuous_action_space=False
    ):
        """ Creates tf.keras model by building neural network from 
        configuration specified in yml file

        Parameters
        ----------
        state_dim : int
            Dimension of state vector obtained from environment
        action_dim : int
            Number of possible (discrete) actions
        net_params : OD (dict)
            Model configuration specifed in yml file in the form of 
            object dictionary (inherits from dict)
        name : str
            Name for the tf.keras model
        continuous_action_space : bool, optional
            Environment has continuous action space, which requires policy net
            to model action probabilities with gaussian distributions
            , by default False
        """
        super().__init__(name=name)
        self.continuous_action_space = continuous_action_space
        self.hidden_layers = []
        for num_hidden_unit in net_params.hidden_units:
            activation = getattr(tf.nn, net_params.activation) if net_params.activation else None
            self.hidden_layers.append(
                DenseBlock(
                    num_hidden_unit,
                    norm=net_params.normalization,
                    activation=activation,
                )
            )
        if self.continuous_action_space:
            self.mean = tf.keras.layers.Dense(action_dim)
            self.logvar = tf.Variable(tf.zeros((1, action_dim)), name="logvar")
        else:
            self.output_layer = tf.keras.layers.Dense(action_dim)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        if self.continuous_action_space:
            mean = self.mean(x)
            return mean, self.logvar
        else:
            x = self.output_layer(x)
            return x

    def get_action(self, inputs, deterministic_policy=False):
        """ Policy network outputs logits by default, this function can be used to
        get action from the action probabilities (or distribution) predicted by 
        model

        Parameters
        ----------
        inputs : tf.Tensor
            State vector obtained from environment
        deterministic_policy : bool, optional
            In order to use determinisitic policy, action is selected by argmax (discrete)
            or mean vector of action distribution (continuous)
            by default False

        Returns
        -------
        tf.Tensor
            Probabilities after applying softmax to model output for discrete case, or
            continuous action vector for continuous action space environment
        """
        if self.continuous_action_space:
            action_logits = self.call(inputs)
            mean, logvar = action_logits
            if deterministic_policy:
                action = mean
            else:
                eps = tf.random.normal(shape=mean.shape)
                action = eps * tf.exp(logvar) + mean
            action = action.numpy()[0]
            action_likelihood = calculate_likelihood(
                self.continuous_action_space, action_logits, action
            )
        else:
            action_logits = self.call(inputs)
            if deterministic_policy:
                action = tf.argmax(
                    tf.nn.softmax((action_logits), axis=1), axis=1
                ).numpy()[0]
            else:
                action = tf.random.categorical(action_logits, 1).numpy()[0][0]
            action_likelihood = tf.math.log(tf.nn.softmax(action_logits))[:, action]
        return action, action_likelihood

    def print_summary(self, state_dim):
        """ Prints keras model summary (only used for debug)

        Parameters
        ----------
        state_dim : int
            Dimension of state vector obtained from environment
        """
        x = tf.keras.Input(shape=(state_dim,))
        print(tf.keras.Model(inputs=[x], outputs=self.call(x)).summary())


def calculate_likelihood(continuous_action_space, action_logits, actions):
    if continuous_action_space:
        mean, logvar = action_logits
        logvar = tf.clip_by_value(logvar, -10, 5)
        likelihood = -0.5 * (
            logvar
            + tf.math.log(2.0 * math.pi)
            + (tf.math.square(actions - mean) / (tf.math.exp(logvar)))
        )
    else:
        likelihood = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=action_logits, labels=actions
        )
    return likelihood


def calculate_entropy(logvar):
    return tf.reduce_mean(0.5 * (1 + tf.math.log(2.0 * math.pi) + logvar))

def policy_gradient_loss(
    action_logits,
    old_actions,
    old_actions_likelihood,
    advantage_functions,
    continuous_action_space,
    entropy_weight,
    pi_ratio,
):
    likelihood = calculate_likelihood(
        continuous_action_space, action_logits, old_actions
    )
    new_to_old_policy_ratio = tf.exp(likelihood - old_actions_likelihood + 1e-6)
    clipped_ratio = tf.clip_by_value(
        new_to_old_policy_ratio, 1 - pi_ratio, 1 + pi_ratio
    )
    loss = -tf.reduce_mean(
        tf.math.minimum(
            new_to_old_policy_ratio * advantage_functions,
            clipped_ratio * advantage_functions,
        )
    )
    entropy = 0.0
    if continuous_action_space:
        entropy = calculate_entropy(action_logits[1])
        loss -= entropy_weight * entropy
    return loss, entropy, new_to_old_policy_ratio


def state_function_estimator_loss(PredictedQvalues, targets):
    """ Loss function for state value function estimator network

    Parameters
    ----------
    PredictedQvalues : tf.Tensor
        Value functions predicted by model
    targets : tf.Tensor
        Targets for model obtained by Monte Carlo or n-step returns

    Returns
    -------
    tf.Tensor
        MSE between estimated and target value functions
    """
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(PredictedQvalues, targets))


@tf.function(experimental_relax_shapes=True)
def policy_net_train_step(
    model,
    optimizer,
    states,
    advantage_functions,
    old_actions,
    old_actions_likelihood,
    entropy_weight,
    pi_ratio,
):

    with tf.GradientTape() as tape:
        action_logits = model(states)
        loss_value, entropy, new_to_old_policy_ratio = policy_gradient_loss(
            action_logits,
            old_actions,
            old_actions_likelihood,
            advantage_functions,
            model.continuous_action_space,
            entropy_weight,
            pi_ratio,
        )
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value, entropy, new_to_old_policy_ratio


@tf.function(experimental_relax_shapes=True)
def state_function_estimator_train_step(model, optimizer, states, targets):
    """ Training step for state value function estimator network

    Parameters
    ----------
    model : tf.keras.Model
        State value function estimator network model
    optimizer : tf.keras.optimizers
        Optimizer for state value function estimator network gradient update step
    states : tf.Tensor
        Batch of state vectors obtained from environment
    targets : tf.Tensor
        Targets for state value function estimator model obtained by Monte Carlo or
         n-step returns

    Returns
    -------
    tf.Tensor
        MSE between estimated and target value functions
    """
    with tf.GradientTape() as tape:
        PredictedQvalues = model(states)
        loss_value = state_function_estimator_loss(PredictedQvalues, targets)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value
