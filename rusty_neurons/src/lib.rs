#![deny(missing_docs)]

//! A library aimed at creating Neural Networks easily.

mod utils;

/// Initialization types for use in [`NeuralNetwork::initialize()`] when randomizing weights and biases
///
/// # Example
///
/// ```rust
/// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
///
/// let mut network = NeuralNetwork::new(1, vec![], 1, ActivationFunction::Normal, ActivationFunction::Normal);
///
/// // Fills weights and biases with random numbers from -0.7 to 0.4
/// network.initialize(InitType::Random {min: -0.7, max: 0.4});
/// ```
pub enum InitType {
    /// [`InitType`] that creates random values based on the range:
    /// ±√(6.0 / (# of inputs + # of outputs))
    ///
    /// # Example
    ///
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// let mut network = NeuralNetwork::new(1, vec![], 1, ActivationFunction::Normal, ActivationFunction::Normal);
    ///
    /// // Fills weights and biases with random numbers using Uniform Xavier
    /// // The range is -3 to 3 in this case (±√(6 / (1 + 1)))
    /// network.initialize(InitType::UniformXavier);
    /// ```
    UniformXavier,
    /// [`InitType`] that creates random values based on the range:
    /// ±√(2.0 / (# of inputs + # of outputs))
    ///
    /// # Example
    ///
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// let mut network = NeuralNetwork::new(1, vec![], 1, ActivationFunction::Normal, ActivationFunction::Normal);
    ///
    /// // Fills weights and biases with random numbers using Normal Xavier
    /// // The range is -1 to 1 in this case (±√(2 / (1 + 1)))
    /// network.initialize(InitType::NormalXavier);
    /// ```
    NormalXavier,
    /// [`InitType`] that creates random values based on inputted range
    ///
    /// # Example
    ///
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// let mut network = NeuralNetwork::new(1, vec![], 1, ActivationFunction::Normal, ActivationFunction::Normal);
    ///
    /// // Fills weights and biases with random numbers from -0.7 to 0.4
    /// network.initialize(InitType::Random {min: -0.7, max: 0.4});
    /// ```
    Random {
        /// Minimum value in the range used to generate random weights and biases
        min: f64,
        /// Maximum value in the range used to generate random weights and biases
        max: f64
    },
}

/// [Activation Functions](https://en.wikipedia.org/wiki/Activation_function) used for nodes in the network.
///
/// # Example
/// ```rust
/// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
///
/// let mut network = NeuralNetwork::new(
///     1,
///     vec![],
///     1,
///     ActivationFunction::Normal,
///     ActivationFunction::Normal
/// );
/// ```
pub enum ActivationFunction {
    /// Does not change the nodes in any way:
    /// y = x
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// let mut network = NeuralNetwork::new(
    ///     1,
    ///     vec![],
    ///     1,
    ///     // Uses no function for hidden layer nodes
    ///     ActivationFunction::Normal,
    ///     // Uses no function for output nodes
    ///     ActivationFunction::Normal
    /// );
    /// ```
    Normal,
    /// Changes the value of nodes using the function:
    /// y = 1 / (1 + e^x)
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// let mut network = NeuralNetwork::new(
    ///     1,
    ///     vec![],
    ///     1,
    ///     // Uses the sigmoid function for hidden layer nodes
    ///     ActivationFunction::Sigmoid,
    ///     // Uses the sigmoid function for output nodes
    ///     ActivationFunction::Sigmoid
    /// );
    /// ```
    /// Usually used in the output layer nodes
    Sigmoid,
    /// Changes the value of nodes using the function:
    /// y = max(0, x)
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// let mut network = NeuralNetwork::new(
    ///     1,
    ///     vec![],
    ///     1,
    ///     // Uses the ReLU function for hidden layer nodes
    ///     ActivationFunction::ReLU,
    ///     // Uses the ReLU function for output nodes
    ///     ActivationFunction::ReLU
    /// );
    /// ```
    /// Usually used in the hidden layer nodes
    ReLU,
    /// Changes the value of nodes using the function:
    /// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// let mut network = NeuralNetwork::new(
    ///     1,
    ///     vec![],
    ///     1,
    ///     // Uses the GELU Tanh function for hidden layer nodes
    ///     ActivationFunction::GELUTanh,
    ///     // Uses the GELU Tanh function for output nodes
    ///     ActivationFunction::GELUTanh
    /// );
    /// ```
    GELUTanh,
    /// Changes the value of nodes using the function:
    /// y = x / 1 + e^(-1.702x)
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// let mut network = NeuralNetwork::new(
    ///     1,
    ///     vec![],
    ///     1,
    ///     // Uses the GELU Sigmoid function for hidden layer nodes
    ///     ActivationFunction::GELUSigmoid,
    ///     // Uses the GELU Sigmoid function for output nodes
    ///     ActivationFunction::GELUSigmoid
    /// );
    /// ```
    GELUSigmoid
}

// pub enum LossFunction {
//     MSE,
//     MAE,
//     RMSE,
//     Huber,
//     BinaryCrossEntropy,
//     CategorialCrossEntropy,
//     SparseCategorialCrossEntropy,
//     SVM,
//     FocalLoss,
//     KLDivergence,
//     TripletLoss,
// }

/// Structure used for the neural network. Use [`NeuralNetwork::new()`] to create a neural network.
///
/// # Example
/// ```rust
/// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
///
/// // Create network
/// let mut network = NeuralNetwork::new(
///     1, // 1 Input
///     vec![4, 2], // Vector with the number of nodes in each column
///     1, // 1 Output
///     ActivationFunction::ReLU, // Uses ReLU function to calculate hidden layer nodes
///     ActivationFunction::Sigmoid // Uses Sigmoid function to calculate output nodes
/// );
/// ```
pub struct NeuralNetwork {
    /// Number of inputs for a neural network
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// // Create network
    /// let mut network = NeuralNetwork::new(
    ///     1, // 1 Input
    ///     vec![4, 2], // Vector with the number of nodes in each column
    ///     1, // 1 Output
    ///     ActivationFunction::ReLU, // Uses ReLU function to calculate hidden layer nodes
    ///     ActivationFunction::Sigmoid // Uses Sigmoid function to calculate output nodes
    /// );
    /// ```
    inputs: i32,
    /// Number of [hidden layers](https://en.wikipedia.org/wiki/Hidden_layer) for a neural network
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// // Create network
    /// let mut network = NeuralNetwork::new(
    ///     1, // 1 Input
    ///     vec![4, 2], // Vector with the number of nodes in each column
    ///     1, // 1 Output
    ///     ActivationFunction::ReLU, // Uses ReLU function to calculate hidden layer nodes
    ///     ActivationFunction::Sigmoid // Uses Sigmoid function to calculate output nodes
    /// );
    /// ```
    hidden_layers: Vec<i32>,
    /// Number of outputs for a neural network
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// // Create network
    /// let mut network = NeuralNetwork::new(
    ///     1, // 1 Input
    ///     vec![4, 2], // Vector with the number of nodes in each column
    ///     1, // 1 Output
    ///     ActivationFunction::ReLU, // Uses ReLU function to calculate hidden layer nodes
    ///     ActivationFunction::Sigmoid // Uses Sigmoid function to calculate output nodes
    /// );
    /// ```
    outputs: i32,
    /// A vector containing the [weights](https://www.geeksforgeeks.org/deep-learning/the-role-of-weights-and-bias-in-neural-networks/) for the neural network.
    ///
    /// # Structure
    /// ```text
    /// Network
    ///  └── Column
    ///        └── Node
    ///             └── Weight
    /// ```
    weights: Vec<Vec<Vec<f64>>>,
    /// A vector containing the [biases](https://www.geeksforgeeks.org/deep-learning/the-role-of-weights-and-bias-in-neural-networks/) for the neural network.
    ///
    /// # Structure
    /// ```text
    /// Network
    ///  └── Column
    ///        └── Node
    ///             └── Bias
    /// ```
    biases: Vec<Vec<Vec<f64>>>,
    /// The [`ActivationFunction`] used on hidden layer nodes for [`NeuralNetwork::initialize`]
    layer_activation_function: ActivationFunction,
    /// The [`ActivationFunction`] used on output nodes for [`NeuralNetwork::initialize`]
    output_activation_function: ActivationFunction,
}

impl NeuralNetwork {
    /// A method in [`NeuralNetwork`] that creates a new network.
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// // Create network
    /// let mut network = NeuralNetwork::new(
    ///     1, // 1 Input
    ///     vec![4, 2], // Vector with the number of nodes in each column
    ///     1, // 1 Output
    ///     ActivationFunction::ReLU, // Uses ReLU function to calculate hidden layer nodes
    ///     ActivationFunction::Sigmoid // Uses Sigmoid function to calculate output nodes
    /// );
    /// ```
    pub fn new(inputs: i32, hidden_layers: Vec<i32>, outputs: i32, layer_activation_function: ActivationFunction, output_activation_function: ActivationFunction) -> NeuralNetwork {
        if inputs <= 0 {
            panic!("NeuralNetwork inputs cannot be less than 1.")
        } else if outputs <= 0 {
            panic!("NeuralNetwork outputs cannot be less than 1.")
        }

        let network = Self {
            inputs,
            hidden_layers,
            outputs,
            weights: Vec::new(),
            biases: Vec::new(),
            layer_activation_function,
            output_activation_function
        };

        network
    }

    fn random_vector(&self, size: i32, init_type: &InitType) -> Vec<f64> {
        match init_type {
            InitType::UniformXavier => {
                let range = (6.0 / (self.inputs + self.outputs) as f64).sqrt();
                utils::random_vector_from_range(size, -range, range)
            }
            InitType::NormalXavier => {
                let range = (2.0 / (self.inputs + self.outputs) as f64).sqrt();
                utils::random_vector_from_range(size, -range, range)
            }
            InitType::Random {min, max} => {
                utils::random_vector_from_range(size, *min, *max)
            }
        }
    }

    fn randomize(&self, init_type: &InitType) -> Vec<Vec<Vec<f64>>> {
        let mut value = Vec::new();
        if self.hidden_layers.is_empty() {
            value.push(
                (0..self.outputs).map(|_| Self::random_vector(self, self.inputs, init_type)).collect()
            );
        } else {
            // First layer (After Input Layer)
            value.push(
                (0..self.hidden_layers[0]).map(|_| Self::random_vector(self, self.inputs, init_type)).collect()
            );

            // Second layer to Second to last
            (1..self.hidden_layers.len() as i32).for_each(|i| {
                value.push(
                    (0..self.hidden_layers[i as usize])
                        .map(|_| Self::random_vector(self, self.hidden_layers[(i as usize) - 1], init_type))
                        .collect()
                )
            });

            // Last Layer (Output layer)
            value.push(
                (0..self.outputs).map(|_| Self::random_vector(self, *self.hidden_layers.last().unwrap(), init_type)).collect()
            );
        }
        value
    }

    /// Randomize weights and biases with an initiation type from [`InitType`]
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// // Create network
    /// let mut network = NeuralNetwork::new(1, vec![4, 2], 1, ActivationFunction::ReLU, ActivationFunction::Sigmoid );
    ///
    /// // Fills weights and biases with random numbers from -0.7 to 0.4
    /// network.initialize(InitType::Random {min: -0.7, max: 0.4});
    /// ```
    pub fn initialize(&mut self, init_type: InitType) {
        self.weights = Self::randomize(self, &init_type);
        self.biases = Self::randomize(self, &init_type);
    }

    fn check_initialized(&self) {
        if self.weights.is_empty() || self.biases.is_empty() {
            panic!("Neural network has not been initialized. Call initialize() first.");
        }
    }

    /// Runs the network with a vector of inputs and returns a vector of outputs.
    ///
    /// # Example
    /// ```rust
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// // Create network
    /// let mut network = NeuralNetwork::new(1, vec![4, 2], 1, ActivationFunction::ReLU, ActivationFunction::Sigmoid );
    ///
    /// // Fills weights and biases with random numbers from -0.7 to 0.4
    /// network.initialize(InitType::Random {min: -0.7, max: 0.4});
    ///
    /// // Runs Network with the input 2.0
    /// let result = network.run(vec![2.0]);
    /// ```
    /// # Panics
    /// * When network has not been initialized
    pub fn run(&self, inputs: Vec<f64>) -> Vec<f64> {
        self.check_initialized();
        if inputs.len() != self.inputs as usize {
            panic!("Incorrect amount of inputs to run NeuralNetwork. ({} of {} required)", inputs.len(), self.inputs);
        }

        let mut last_layer = inputs;
        let mut new_layer: Vec<f64> = vec![];
        for (i, (layer_weights, layer_biases)) in self.weights.iter().zip(&self.biases).enumerate() {
            for (node_weights, node_biases) in layer_weights.iter().zip(layer_biases) {
                new_layer.push(
                    node_weights.iter().zip(node_biases).enumerate().map(|(k, (weight, bias))| {
                        let value = last_layer[k] * weight + bias;
                        if i == self.hidden_layers.len() {
                            utils::activate(value, &self.output_activation_function)
                        } else {
                            utils::activate(value, &self.layer_activation_function)
                        }
                    }).sum()
                )
            }

            last_layer = new_layer.clone();
            new_layer.clear();
        }

        last_layer
    }

    // fn calculate_loss(predictions: Vec<f64>, answers: Vec<f64>, loss_function: LossFunction) -> Vec<f64> {
    //     match loss_function {
    //         LossFunction::MSE => {
    //             vec![]
    //         }
    //         LossFunction::MAE => {
    //             vec![]
    //         }
    //         LossFunction::RMSE => {
    //             vec![]
    //         }
    //         LossFunction::Huber => {
    //             vec![]
    //         }
    //         LossFunction::BinaryCrossEntropy => {
    //             vec![]
    //         }
    //         LossFunction::CategorialCrossEntropy => {
    //             vec![]
    //         }
    //         LossFunction::SparseCategorialCrossEntropy => {
    //             vec![]
    //         }
    //         LossFunction::SVM => {
    //             vec![]
    //         }
    //         LossFunction::FocalLoss => {
    //             vec![]
    //         }
    //         LossFunction::KLDivergence => {
    //             vec![]
    //         }
    //         LossFunction::TripletLoss => {
    //             vec![]
    //         }
    //     }
    // }

    /// Trains the network using [backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
    /// for a specified amount of [epochs](https://www.geeksforgeeks.org/machine-learning/epoch-in-machine-learning/).
    /// Takes a Vector of Vectors with f64(s) that aligns with the inputs specified.
    /// Also takes a Vector of Vectors with f64(s) that aligns with the outputs specified (The answers to the first set).
    ///
    /// # Example
    /// ```rust
    /// use rand::{rng, Rng};
    /// use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};
    ///
    /// // Create network
    /// let mut network = NeuralNetwork::new(2, vec![4, 2], 1, ActivationFunction::ReLU, ActivationFunction::Sigmoid );
    ///
    /// // Fills weights and biases with random numbers from -0.7 to 0.4
    /// network.initialize(InitType::Random {min: -0.7, max: 0.4});
    ///
    ///
    /// let x = (0..10).map(|_| vec![rng().random_range(-5.0..5.0)]).collect::<Vec<Vec<f64>>>();
    /// let y = x.map(|i| i[0] + i[1]);
    ///
    /// // Trains the network with x and y for 50 epochs.
    /// network.train(x, y, 50);
    /// ```
    /// # Panics
    /// * When network has not been initialized
    pub fn train(&mut self, x: Vec<f64>, y: Vec<f64>, epochs: i32) {
        self.check_initialized();
        for epoch in 0..epochs {

        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_network() {
        let mut network = NeuralNetwork::new(
            3,
            vec![4, 3],
            1,
            ActivationFunction::ReLU,
            ActivationFunction::Sigmoid,
        );

        network.initialize(InitType::UniformXavier);

        println!("{:#?}", network.run(vec![5.0, 0.0, 0.0]));
        println!("{:#?}", network.run(vec![1.0, 2.0, 0.0]));
    }
}