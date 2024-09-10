use model::Model;
use eyre::Result;

mod config;
mod model;
mod session;

static CONFIG_CONTENT: &str = include_str!("../../nakdimon_ort/config.json");

pub struct Nakdimon {
    model: Model,
}

impl Nakdimon {
    pub fn new() -> Result<Self> {
        let model = model::Model::new(model_path, config_path)
        Self {}
    }

    pub fn compute(text: &str) -> Result<String> {

    }
}
