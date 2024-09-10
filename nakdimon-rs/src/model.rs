use eyre::{Context, ContextCompat, Result};
use ndarray::Array;
use ort::Session;
use std::collections::HashMap;

use crate::{config::Config, session};

pub struct Model {
    config: Config,
    session: Session,
}

impl Model {
    pub fn new(model_path: &str, config_path: &str) -> Result<Self> {
        let config = Config::new(model_path, config_path)?;
        let session = session::create_session(model_path)?;

        Ok(Model { config, session })
    }

    pub fn normalize(&self, c: &str) -> String {
        if self.config.valid.contains(&c.to_string()) {
            c.to_string()
        } else {
            self.config
                .normalize_map
                .get(c)
                .unwrap_or(&self.config.normalize_default_value)
                .clone()
        }
    }

    pub fn split_to_rows(&self, text: &str) -> Vec<Vec<u32>> {
        let word_ids_rows: Vec<Vec<u32>> = text
            .split_whitespace()
            .map(|word| {
                word.chars()
                    .map(|c| {
                        self.config
                            .all_tokens
                            .iter()
                            .position(|x| x == &c.to_string())
                            .unwrap_or(0) as u32
                    })
                    .collect()
            })
            .collect();

        let mut rows: Vec<Vec<u32>> = Vec::new();
        let mut cur_row: Vec<u32> = Vec::new();

        for word_ids in word_ids_rows {
            if cur_row.len() + word_ids.len() + 1 > self.config.max_len {
                let padding = vec![0; self.config.max_len - cur_row.len()];
                rows.push([cur_row.clone(), padding].concat());
                cur_row.clear();
            }
            cur_row.extend(word_ids);
            cur_row.push(
                self.config
                    .all_tokens
                    .iter()
                    .position(|x| x == " ")
                    .unwrap_or(0) as u32,
            );
        }

        let padding = vec![0; self.config.max_len - cur_row.len()];
        rows.push([cur_row, padding].concat());

        rows
    }

    pub fn from_categorical(
        &self,
        input_tensor: &Array<f32, ndarray::IxDyn>,
        arr: Vec<Array<f32, ndarray::IxDyn>>,
    ) -> Vec<usize> {
        arr.into_iter()
            .filter_map(|a| {
                if input_tensor.iter().any(|&v| v > 0.0) {
                    Some(a.argmax().unwrap())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn prediction_to_text(
        &self,
        input_tensor: Array<u32, ndarray::IxDyn>,
        prediction: Vec<Array<f32, ndarray::IxDyn>>,
        undotted_text: &str,
    ) -> Vec<HashMap<String, String>> {
        let (niqqud, dagesh, sin) = (&prediction[0], &prediction[1], &prediction[2]);
        let niqqud_result = self.from_categorical(&input_tensor, vec![niqqud.clone()]);
        let dagesh_result = self.from_categorical(&input_tensor, vec![dagesh.clone()]);
        let sin_result = self.from_categorical(&input_tensor, vec![sin.clone()]);

        let mut output = Vec::new();
        for (i, c) in undotted_text.chars().enumerate() {
            let mut fresh = HashMap::new();
            fresh.insert("char".to_string(), c.to_string());
            fresh.insert("niqqud".to_string(), "".to_string());
            fresh.insert("dagesh".to_string(), "".to_string());
            fresh.insert("sin".to_string(), "".to_string());

            if self.config.hebrew_letters.contains(&c.to_string()) {
                if self.config.can_niqqud.contains(&c.to_string()) {
                    fresh.insert(
                        "niqqud".to_string(),
                        self.config.niqqud[niqqud_result[i]].clone(),
                    );
                }
                if self.config.can_dagesh.contains(&c.to_string()) {
                    fresh.insert(
                        "dagesh".to_string(),
                        self.config.dagesh[dagesh_result[i]].clone(),
                    );
                }
                if self.config.can_sin.contains(&c.to_string()) {
                    fresh.insert("sin".to_string(), self.config.sin[sin_result[i]].clone());
                }
            }
            output.push(fresh);
        }
        output
    }

    pub fn remove_niqqud(&self, text: &str) -> String {
        text.chars()
            .filter(|&c| {
                !(self.config.remove_niqqud_range[0] as u32
                    ..=self.config.remove_niqqud_range[1] as u32)
                    .contains(&(c as u32))
            })
            .collect()
    }

    pub fn to_text(&self, item: &HashMap<String, String>) -> String {
        format!(
            "{}{}{}{}",
            item.get("char").unwrap_or(&"".to_string()),
            item.get("dagesh").unwrap_or(&"".to_string()),
            item.get("sin").unwrap_or(&"".to_string()),
            item.get("niqqud").unwrap_or(&"".to_string())
        )
    }

    pub fn update_dotted(&self, items: Vec<HashMap<String, String>>) -> String {
        items
            .into_iter()
            .map(|item| self.to_text(&item))
            .collect::<String>()
    }

    pub fn compute(&self, text: &str) -> Result<String> {
        let undotted = self.remove_niqqud(text);
        let normalized: String = undotted
            .chars()
            .map(|c| self.normalize(&c.to_string()))
            .collect();
        let input = self.split_to_rows(&normalized);
        let input_tensor = Array::from_shape_vec(
            ndarray::IxDyn(&[input.len(), input[0].len()]),
            input.concat(),
        )?;

        let inputs = ort::inputs![input_tensor.into_dyn()]?;
        let outs = self.session.run(inputs)?;
        let outs = outs
            .get("output")
            .context("Output tensor not found")?
            .try_extract_tensor::<f32>()
            .context("Failed to extract tensor")?;

        let res = self.prediction_to_text(input_tensor, prediction, &undotted);
        Ok(self.update_dotted(res))
    }
}
