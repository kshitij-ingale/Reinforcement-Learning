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
            self.hidden_layers.append(
                DenseBlock(
                    num_hidden_unit,
                    norm=net_params.normalization,
                    activation=getattr(tf.nn, net_params.activation),
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
            mean, logvar = self.call(inputs)
            if deterministic_policy:
                action = mean
            else:
                eps = tf.random.normal(shape=mean.shape)
                action = eps * tf.exp(logvar) + mean
            action = action.numpy()[0]
        else:
            action_logits = self.call(inputs)
            if deterministic_policy:
                action = tf.argmax(
                    tf.nn.softmax((action_logits), axis=1), axis=1
                ).numpy()[0]
            else:
                action = tf.random.categorical(action_logits, 1).numpy()[0][0]
        return action

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
    entropy = None
    if continuous_action_space:
        mean, logvar = action_logits
        logvar = tf.clip_by_value(logvar, -10, 5)
        likelihood = -0.5 * (
            logvar
            + tf.math.log(2.0 * math.pi)
            + (tf.math.square(actions - mean) / (tf.math.exp(logvar)))
        )
        entropy = tf.reduce_mean(0.5 * (1 + tf.math.log(2.0 * math.pi) + logvar))
    else:
        likelihood = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=action_logits, labels=actions
        )
    return likelihood, entropy


def policy_gradient_loss(
    action_logits,
    actions,
    discounted_returns,
    continuous_action_space,
    pi_theta_old,
    entropy_weight,
    pi_ratio,
):
    """ Calculates loss for obtaining policy gradients. For this loss, we want predicted action probabilities
    to correspond to actual actions performed. So, we maximize likelihood of the predicted actions to be from 
    actual action distribution. Finally, advantage function is multiplied so that agent can learn which actions
    are better

    Parameters
    ----------
    action_logits : tf.Tensor
        Logits (or distribution parameters for continuous actions)corresponding
         to actions output from model
    actions : tf.Tensor
        Actions corresponding to model output, executed by agent in environment
    discounted_returns : tf.Tensor
        Future discounted rewards obtained by agent while performing the actions
    continuous_action_space : bool, optional
        Environment has continuous action space modeled with gaussian distributions, 
        accordingly loss will be computed with gaussian pdf, by default False

    Returns
    -------
    tf.Tensor
        Policy gradients in the form of loss for the policy network
    """
    likelihood, entropy = calculate_likelihood(
        continuous_action_space, action_logits, actions
    )
    new_to_old_policy_ratio = tf.exp(likelihood - pi_theta_old + 1e-6)
    loss = -tf.reduce_mean(
        tf.math.minimum(
            new_to_old_policy_ratio * discounted_returns,
            tf.clip_by_value(new_to_old_policy_ratio, 1 - pi_ratio, 1 + pi_ratio)
            * discounted_returns,
        )
    )
    if entropy is not None:
        loss += entropy_weight * entropy
    return loss, entropy


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
    discounted_returns,
    actions,
    pi_theta_old,
    entropy_weight,
    pi_ratio,
):
    """ Training step for policy network

    Parameters
    ----------
    model : tf.keras.Model
        Policy network model
    optimizer : tf.keras.optimizers
        Optimizer for policy network gradient update step
    states : tf.Tensor
        Batch of state vectors obtained from environment
    discounted_returns : tf.Tensor
        Batch of future discounted rewards obtained by agent while performing the actions
    actions : tf.Tensor
        Batch of actions executed by agent in environment

    Returns
    -------
    tf.Tensor
        Policy gradients in the form of loss for the policy network
    """
    with tf.GradientTape() as tape:
        action_logits = model(states)
        loss_value, entropy = policy_gradient_loss(
            action_logits,
            actions,
            discounted_returns,
            model.continuous_action_space,
            pi_theta_old,
            entropy_weight,
            pi_ratio,
        )
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value, entropy


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
