use rand;
use rand::Rng;



pub struct NeuralNetwork {
    inputs: i32,
    hidden_layers: Vec<i32>,
    outputs: i32,
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
}

impl NeuralNetwork {

    pub fn new(inputs: i32, hidden_layers: Vec<i32>, outputs: i32) -> NeuralNetwork {
        let mut network = Self {
            inputs,
            hidden_layers,
            outputs,
            weights: Vec::new(),
            biases: Vec::new(),
        };

        network.randomize_weights_and_biases();

        network
    }

    fn random_vector(size: i32) -> Vec<f64> {
        (0..size)
            .map(|_| rand::rng().random_range(-1.0..1.0))
            .collect()
    }

    fn randomize_weights_and_biases(&mut self) {
        if self.hidden_layers.is_empty() {

        } else {
            // First layer (After Input Layer)
            (0..self.hidden_layers[0]).for_each(|i| {
                self.weights.push(Self::random_vector(self.inputs));
            });

            // Second layer to Second to last
            (1..self.hidden_layers.len() as i32).for_each(|i| {
                (i..self.hidden_layers[i as usize]).for_each(|k| {
                    self.weights.push(Self::random_vector(k));
                });
            });

            // Last Layer (Output layer)
            (0..self.outputs).for_each(|i| {
                self.weights.push(Self::random_vector(*self.hidden_layers.last().unwrap()));
            });
        }

    }

    pub fn add_hidden(&mut self, nodes: i32) {
        self.hidden_layers.push(nodes);
    }

    pub fn run(&self, inputs: Vec<i32>) -> Vec<i32> {
        inputs
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
            1
        );

        network.add_hidden(5);

        println!("{:#?}", network.weights);
    }
}
