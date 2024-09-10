use eyre::bail;
use eyre::Result;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub rafe: String,
    pub niqqud: Vec<String>,
    pub dagesh: Vec<String>,
    pub sin: Vec<String>,
    pub hebrew_letters: Vec<String>,
    pub valid: Vec<String>,
    pub special: Vec<String>,
    pub normalize_map: std::collections::HashMap<String, String>,
    pub normalize_default_value: String,
    pub can_dagesh: Vec<String>,
    pub can_sin: Vec<String>,
    pub can_niqqud: Vec<String>,
    pub all_tokens: Vec<String>,
    pub remove_niqqud_range: Vec<u32>,
    pub max_len: usize,
}

impl Config {
    pub fn new(model_path: &str, config_path: &str) -> Result<Self> {
        let config = Self::load(model_path, config_path)?;

        let mut all_tokens = vec!["".to_string()];
        all_tokens.extend(config.special.clone());
        all_tokens.extend(config.valid.clone());

        Ok(Config {
            rafe: config.rafe,
            niqqud: config.niqqud,
            dagesh: config.dagesh,
            sin: config.sin,
            hebrew_letters: config.hebrew_letters.clone(),
            valid: [config.valid.clone(), config.hebrew_letters].concat(),
            special: config.special,
            normalize_default_value: config.normalize_map["default"].clone(),
            normalize_map: config.normalize_map,
            can_dagesh: config.can_dagesh,
            can_sin: config.can_sin,
            can_niqqud: config.can_niqqud,
            all_tokens,
            remove_niqqud_range: config.remove_niqqud_range,
            max_len: config.max_len,
        })
    }

    fn load(model_path: &str, config_path: &str) -> Result<Config> {
        if !Path::new(model_path).exists() {
            bail!(
                "Model file not found: {}\nPlease download the Nakdimon model before executing.\nYou can download it using the following command:\nwget https://github.com/thewh1teagle/nakdimon-ort/releases/download/v0.1.0/nakdimon.onnx",
                model_path
            );
        }

        if !Path::new(config_path).exists() {
            bail!(
                "Configuration file not found: {}\nPlease download the Nakdimon configuration file before executing.\nYou can download it using the following command:\nwget https://github.com/thewh1teagle/nakdimon-ort/raw/main/nakdimon_ort/config.json",
                config_path
            );
        }

        let config_content = fs::read_to_string(config_path)?;

        let config: Config = serde_json::from_str(&config_content)?;

        Ok(config)
    }
}
