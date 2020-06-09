""" Script to generate neural networks for policy network and state value function estimator """

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
    def __init__(self, state_dim, action_dim, net_params, name):
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
        """
        super().__init__(name=name)
        self.hidden_layers = []
        for num_hidden_unit in net_params.hidden_units:
            self.hidden_layers.append(
                DenseBlock(
                    num_hidden_unit,
                    norm=net_params.normalization,
                    activation=getattr(tf.nn, net_params.activation),
                )
            )
        output_layer = tf.keras.layers.Dense(action_dim)
        self.hidden_layers.append(output_layer)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x

    def get_action_probabilities(self, inputs):
        """ Policy network outputs logits by default, this function can be used to
        apply softmax to obtain probabilities

        Parameters
        ----------
        inputs : tf.Tensor
            State vector obtained from environment

        Returns
        -------
        tf.Tensor
            Probabilities after applying softmax to model output
        """
        return tf.nn.softmax(self.call(inputs))

    def print_summary(self, state_dim):
        """ Prints keras model summary (only used for debug)

        Parameters
        ----------
        state_dim : int
            Dimension of state vector obtained from environment
        """
        x = tf.keras.Input(shape=(state_dim,))
        print(tf.keras.Model(inputs=[x], outputs=self.call(x)).summary())


def policy_gradient_loss(action_logits, actions, discounted_returns):
    """ Calculates loss for obtaining policy gradients 

    Parameters
    ----------
    action_logits : tf.Tensor
        Logits corresponding to actions output from model
    actions : tf.Tensor
        Actions corresponding to model output, executed by agent in environment
    discounted_returns : tf.Tensor
        Future discounted rewards obtained by agent while performing the actions

    Returns
    -------
    tf.Tensor
        Policy gradients in the form of loss for the policy network
    """
    probabilities_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=action_logits, labels=actions
    )
    return tf.reduce_mean(discounted_returns * probabilities_sum)


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
def policy_net_train_step(model, optimizer, states, discounted_returns, actions):
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
        loss_value = policy_gradient_loss(action_logits, actions, discounted_returns)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value


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
