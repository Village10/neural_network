use rand::Rng;

pub enum InitType {
    UniformXavier,
    NormalXavier,
    Random {min: f64, max: f64},
}

pub enum ActivationFunction {
    Normal,
    Sigmoid,
    ReLU,
}

pub struct NeuralNetwork {
    inputs: i32,
    hidden_layers: Vec<i32>,
    outputs: i32,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<Vec<f64>>>,
    layer_activation_function: ActivationFunction,
    output_activation_function: ActivationFunction,
}

impl NeuralNetwork {

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

    fn random_vector_from_range(size: i32, start: f64, end: f64) -> Vec<f64> {
        (0..size)
            .map(|_| rand::rng().random_range(start..end)).collect()
    }

    fn random_vector(&self, size: i32, init_type: &InitType) -> Vec<f64> {
        match init_type {
            InitType::UniformXavier => {
                let range = (6.0 / (self.inputs + self.outputs) as f64).sqrt();
                Self::random_vector_from_range(size, -range, range)
            }
            InitType::NormalXavier => {
                let range = (2.0 / (self.inputs + self.outputs) as f64).sqrt();
                Self::random_vector_from_range(size, -range, range)
            }
            InitType::Random {min, max} => {
                Self::random_vector_from_range(size, *min, *max)
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

    pub fn initialize(&mut self, init_type: InitType) {
        self.weights = Self::randomize(self, &init_type);
        self.biases = Self::randomize(self, &init_type);
    }

    fn check_initialized(&self) {
        if self.weights.is_empty() || self.biases.is_empty() {
            panic!("Neural network has not been initialized. Call initialize() first.");
        }
    }

    pub fn add_hidden(&mut self, nodes: i32) {
        self.hidden_layers.push(nodes);
    }

    fn activate(value: f64, activation_function: &ActivationFunction) -> f64 {
        match activation_function {
            ActivationFunction::Normal => {
                value
            }
            ActivationFunction::ReLU => {
                value.max(0.0)
            }
            ActivationFunction::Sigmoid => {
                1.0 / (1.0 + (-value).exp())
            }
        }
    }

    pub fn run(&self, inputs: Vec<f64>) -> Vec<f64> {
        self.check_initialized();
        if inputs.len() != self.inputs as usize {
            panic!("Incorrect amount of inputs to run NeuralNetwork. ({} != {})", inputs.len(), self.inputs);
        }

        let mut last_layer = inputs;
        let mut new_layer: Vec<f64> = vec![];
        for (i, (layer_weights, layer_biases)) in self.weights.iter().zip(&self.biases).enumerate() {
            for (node_weights, node_biases) in layer_weights.iter().zip(layer_biases) {
                new_layer.push(
                    node_weights.iter().zip(node_biases).enumerate().map(|(k, (weight, bias))| {
                        let value = last_layer[k] * weight + bias;
                        if i == self.hidden_layers.len() {
                            NeuralNetwork::activate(value, &self.output_activation_function)
                        } else {
                            NeuralNetwork::activate(value, &self.layer_activation_function)
                        }
                    }).sum()
                )
            }

            last_layer = new_layer.clone();
            new_layer.clear();
        }

        last_layer
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