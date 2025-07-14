use std::f64::consts::PI;
use rand::Rng;
use crate::ActivationFunction;

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