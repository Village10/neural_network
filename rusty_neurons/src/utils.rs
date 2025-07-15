use std::f64::consts::PI;
use rand::Rng;
use crate::{utils, ActivationFunction};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn activate(x: f64, activation_function: &ActivationFunction) -> f64 {
    match activation_function {
        ActivationFunction::Normal => {
            x
        }
        ActivationFunction::ReLU => {
            x.max(0.0)
        }
        ActivationFunction::Sigmoid => {
            sigmoid(x)
        }
        ActivationFunction::GELUTanh => {
            0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        }
        ActivationFunction::GELUSigmoid => {
            x * sigmoid(1.702 * x)
        }
    }
}


pub fn random_vector_from_range(size: i32, start: f64, end: f64) -> Vec<f64> {
    (0..size)
        .map(|_| rand::rng().random_range(start..end)).collect()
}

pub fn forward_pass(inputs: Vec<f64>, weights: Vec<Vec<Vec<f64>>>, biases: Vec<Vec<Vec<f64>>>, layer_activation_function: &ActivationFunction, output_activation_function: &ActivationFunction) -> Vec<Vec<f64>> {

    let mut layers = vec![inputs];
    let mut new_layer: Vec<f64> = vec![];
    for (i, (layer_weights, layer_biases)) in weights.iter().zip(&biases).enumerate() {
        for (node_weights, node_biases) in layer_weights.iter().zip(layer_biases) {
            new_layer.push(
                node_weights.iter().zip(node_biases).enumerate().map(|(k, (weight, bias))| {
                    let value = layers[i][k] * weight + bias;
                    if i == weights.len() {
                        activate(value, &layer_activation_function)
                    } else {
                        activate(value, &output_activation_function)
                    }
                }).sum()
            )
        }

        layers.push(new_layer.clone());
        new_layer.clear();
    }
    layers
}