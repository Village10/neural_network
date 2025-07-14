# Rusty Neurons
A library aimed at creating neural networks easily.

## Installation

Run:
```shell
cargo add rusty_neurons
```
Or add this to your Cargo.toml:
```toml
[Dependencies]
rusty_neurons = "0.1.0"
```

## Example Network

```rust
use rand::{rng, Rng};
use rusty_neurons::{ActivationFunction, InitType, NeuralNetwork};

// Creates the network 
let mut network = NeuralNetwork::new(2, vec![4, 2], 1, ActivationFunction::ReLU, ActivationFunction::Sigmoid );

// Fills weights and biases with random numbers from -0.7 to 0.4
network.initialize(InitType::Random {min: -0.7, max: 0.4});

// Creates training data
let x = (0..10).map(|_| vec![rng().random_range(-5.0..5.0)]).collect::<Vec<Vec<f64>>>();
let y = x.map(|i| i[0] + i[1]);

// Trains the network with x and y for 50 epochs.
network.train(x, y, 50);

// Adds 2 floats together
let result = network.run(vec![1.0, 2.0])
```