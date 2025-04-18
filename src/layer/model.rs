use candle_core::{DType, Device, Result, Tensor};

use candle_nn::{Embedding, embedding,
    LayerNorm, layer_norm,
    Linear, linear,
    Dropout,
    VarBuilder,Module};
use candle_nn::Activation::Gelu;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct RobertaConfig {
    pub architectures: Vec<String>,
    pub attention_probs_dropout_prob: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub pad_token_id: u32,
    pub type_vocab_size: usize,
    pub vocab_size: usize
}

fn cumsum_2d(mask: &Tensor, dim: u8, device: &Device) -> Result<Tensor> {
    let mask = mask.to_vec2::<u32>()?;

    let rows: usize = mask.len();
    let cols: usize = mask[0].len();

    let mut result = mask.clone();

    match dim {
        0 => {
            // Cumulative sum along rows
            for i in 0..rows {
                for j in 1..cols {
                    result[i][j] += result[i][j - 1];
                }
            }
        }
        1 => {
            // Cumulative sum along columns
            for i in 1..rows {
                for j in 0..cols {
                    result[i][j] += result[i - 1][j];
                }
            }
        }
        _ => panic!("Dimension not supported"),
    }

    let result = Tensor::new(result, &device)?;

    Ok(result)
}

pub fn create_position_ids_from_input_ids(
    input_ids: &Tensor,
    padding_idx: u32,
    past_key_values_length: u8,
) -> Result<Tensor> {
    let mask: Tensor = input_ids.ne(padding_idx)?.to_dtype(DType::U32)?;
    let incremental_indices: Tensor = cumsum_2d(&mask, 0, input_ids.device())?;

    let past_key_tensor: Tensor = Tensor::new(&[past_key_values_length],input_ids.device())?.to_dtype(input_ids.dtype())?;
    let pad_tensor: Tensor = Tensor::new(&[padding_idx],input_ids.device())?.to_dtype(input_ids.dtype())?;
    
    let incremental_indices: Tensor = incremental_indices
        .broadcast_add(&past_key_tensor)?
        .mul(&mask)?
        .broadcast_add(&pad_tensor)?;
    Ok(incremental_indices)
}

pub struct RobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    pub padding_idx: u32,
}

impl RobertaEmbeddings {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let word_embeddings: Embedding = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings: Embedding = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings: Embedding = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm: LayerNorm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let padding_idx: u32 = config.pad_token_id;

        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            padding_idx,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
        train: bool
    ) -> Result<Tensor> {
        let position_ids: Tensor = match position_ids {
            Some(ids) => ids.to_owned(),
            None => {
                match inputs_embeds{
                    Some(embed) => self.create_position_ids_from_input_embeds(embed)?,
                    None => create_position_ids_from_input_ids(input_ids, self.padding_idx, 0)?
                }
            }
        };
        
        let inputs_embeds: Tensor = match inputs_embeds {
            Some(embeds) => embeds.to_owned(),
            None => {
                let embeds: Tensor = self.word_embeddings.forward(input_ids)?;
                embeds
            }
        };

        let token_type_embeddings:Tensor = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings: Tensor = (inputs_embeds + token_type_embeddings)?;

        if let Some(position_embeddings) = &self.position_embeddings {
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
        }

        let embeddings: Tensor = self.layer_norm.forward(&embeddings)?;
        let embeddings: Tensor = self.dropout.forward(&embeddings,train)?;

        Ok(embeddings)
    }

    pub fn create_position_ids_from_input_embeds(&self, input_embeds: &Tensor) -> Result<Tensor> {
        // (batch, seq_length, embedding_dim)
        let input_shape: (usize, usize, usize) = input_embeds.dims3()?;
        let seq_length: u32 = input_shape.1 as u32;

        let mut position_ids: Tensor = Tensor::arange(
            self.padding_idx + 1,
            seq_length + self.padding_idx + 1,
            input_embeds.device(),
        )?;

        position_ids = position_ids
            .unsqueeze(0)?
            .expand((input_shape.0, input_shape.1))?;
        Ok(position_ids)
    }
}

pub struct RobertaSdpaSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    key: Linear,
    query: Linear,
    value: Linear,
    dropout: Dropout,
}

impl RobertaSdpaSelfAttention{
    pub fn load(vb: VarBuilder,config:&RobertaConfig) -> Result<RobertaSdpaSelfAttention>{

        assert_eq!(config.hidden_size % config.num_attention_heads, 0);
        let num_attention_heads: usize = config.num_attention_heads;
        let attention_head_size: usize = config.hidden_size / config.num_attention_heads;
        let all_head_size: usize = config.num_attention_heads * attention_head_size;

        let key: Linear = linear(config.hidden_size, all_head_size, vb.pp("key"))?;
        let query: Linear = linear(config.hidden_size, all_head_size, vb.pp("query"))?;
        let value: Linear = linear(config.hidden_size, all_head_size, vb.pp("value"))?;
        let dropout: Dropout = Dropout::new(config.attention_probs_dropout_prob);
        Ok(RobertaSdpaSelfAttention { 
            num_attention_heads: num_attention_heads, 
            attention_head_size: attention_head_size,
            key: key,
            query: query, 
            value: value, 
            dropout: dropout
        })  
    }

    pub fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape: Vec<usize> = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs: Tensor = xs.reshape(new_x_shape.as_slice())?;
        let xs: Tensor = xs.transpose(1, 2)?;
        xs.contiguous()
    }

    pub fn forward(&self, hidden_states: &Tensor,attention_mask: &Tensor, train:bool) -> Result<Tensor> {
        let query_layer: Tensor = self.query.forward(hidden_states)?;
        let key_layer: Tensor = self.key.forward(hidden_states)?;
        let value_layer: Tensor = self.value.forward(hidden_states)?;
        let query_layer: Tensor = self.transpose_for_scores(&query_layer)?;
        let key_layer: Tensor = self.transpose_for_scores(&key_layer)?;
        let value_layer: Tensor = self.transpose_for_scores(&value_layer)?;

        let attention_scores: Tensor = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores: Tensor = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores: Tensor = attention_scores.broadcast_add(attention_mask)?;
        let attention_probs: Tensor =
            { candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)? };
        let attention_probs: Tensor = self.dropout.forward(&attention_probs, train)?;

        let context_layer: Tensor = attention_probs.matmul(&value_layer)?;
        let context_layer: Tensor = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer: Tensor = context_layer.flatten_from(candle_core::D::Minus2)?;
        Ok(context_layer)
    }
}

struct RobertaSelfOutput{
    dense: Linear,
    layernorm: LayerNorm,
    dropout: Dropout
}

impl RobertaSelfOutput{
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<RobertaSelfOutput>{
        let dense: Linear = linear(config.hidden_size,config.hidden_size,vb.pp("dense"))?;
        let layernorm: LayerNorm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        let dropout: Dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(RobertaSelfOutput{
            dense: dense,
            layernorm: layernorm,
            dropout: dropout
        })
    }
    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor,train: bool) -> Result<Tensor>{
        let hidden_states: Tensor = self.dense.forward(hidden_states)?;
        let hidden_states: Tensor = self.dropout.forward(&hidden_states, train)?;
        let hidden_states: Tensor = hidden_states.add(&input_tensor)?;
        let hidden_states: Tensor = self.layernorm.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

struct RobertaAttention{
    self_attention: RobertaSdpaSelfAttention,
    output: RobertaSelfOutput
}

impl RobertaAttention{
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<RobertaAttention>{
        let self_attention: RobertaSdpaSelfAttention = RobertaSdpaSelfAttention::load(vb.pp("self"),config)?;
        let output: RobertaSelfOutput = RobertaSelfOutput::load(vb.pp("output"),config)?;
        Ok(RobertaAttention { 
            self_attention: self_attention, 
            output: output 
        })
    }

    fn forward(&self,hidden_states: &Tensor,attention_mask: &Tensor, train: bool) -> Result<Tensor>{
        let self_outputs: Tensor = self.self_attention.forward(hidden_states,attention_mask, train)?;
        let attention_output: Tensor = self.output.forward(&self_outputs, hidden_states, train)?;
        Ok(attention_output)
    }
}

struct RobertaIntermediate {
    dense: Linear,
}

impl RobertaIntermediate {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let dense: Linear = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        assert_eq!(config.hidden_act,String::from("gelu"));
        Ok(Self {
            dense,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states: Tensor = self.dense.forward(hidden_states)?;
        let hidden_states: Tensor = Gelu.forward(&hidden_states)?;
        // let hidden_states: Tensor = hidden_states.gelu()?;
        Ok(hidden_states)
    }
}

struct RobertaOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl RobertaOutput {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states,train)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

struct RobertaLayer {
    attention: RobertaAttention,
    intermediate: RobertaIntermediate,
    output: RobertaOutput,
}

impl RobertaLayer {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let attention: RobertaAttention = RobertaAttention::load(vb.pp("attention"), config)?;
        let intermediate: RobertaIntermediate = RobertaIntermediate::load(vb.pp("intermediate"), config)?;
        let output: RobertaOutput = RobertaOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor,attention_mask: &Tensor, train: bool) -> Result<Tensor> {
        let attention_output: Tensor = self.attention.forward(hidden_states, attention_mask,train)?;
        let intermediate_output: Tensor = self.intermediate.forward(&attention_output)?;
        let layer_output: Tensor = self
            .output
            .forward(&intermediate_output, &attention_output, train)?;
        Ok(layer_output)
    }
}

struct RobertaEncoder {
    layers: Vec<RobertaLayer>,
}

impl RobertaEncoder {
    fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let layers: Vec<RobertaLayer> = (0..config.num_hidden_layers)
            .map(|index| RobertaLayer::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<Result<Vec<RobertaLayer>>>()?;
        Ok(RobertaEncoder { layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor,train: bool) -> Result<Tensor> {
        let mut hidden_states: Tensor = hidden_states.clone();
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask, train)?;
        }
        Ok(hidden_states)
    }
}

pub struct RobertaModel {
    embeddings: RobertaEmbeddings,
    encoder: RobertaEncoder,
    pub device: Device,
}

impl RobertaModel {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let (embeddings, encoder) = match (
            RobertaEmbeddings::load(vb.pp("embeddings"), config),
            RobertaEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                return Err(err);
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor,attention_mask :&Tensor, train: bool) -> Result<Tensor> {
        let embedding_output = self
            .embeddings
            .forward(input_ids, token_type_ids, None, None, train)?;
        let sequence_output = self.encoder.forward(&embedding_output, attention_mask, train)?;
        Ok(sequence_output)
    }
}

pub struct RobertaClassificationHead{
    dense: Linear,
    dropout: Dropout,
    out_proj: Linear
}

impl RobertaClassificationHead{
    fn load(vb: VarBuilder,config: &RobertaConfig, num_labels: usize) -> Result<RobertaClassificationHead>{
        let dense: Linear = linear(config.hidden_size,config.hidden_size,vb.pp("dense"))?;
        let dropout: Dropout = Dropout::new(config.hidden_dropout_prob);
        let out_proj: Linear = linear(config.hidden_size,num_labels,vb.pp("out_proj"))?;
        Ok(RobertaClassificationHead { 
            dense: dense, 
            dropout: dropout, 
            out_proj: out_proj 
        })
    }
    fn forward(&self, features: Tensor, train: bool) -> Result<Tensor> {
        let x: Tensor = features.get_on_dim(1,0)?;
        let x: Tensor = self.dropout.forward(&x, train)?;
        let x: Tensor = x.unsqueeze(1)?;
        let x: Tensor = self.dense.forward(&x)?;
        let x: Tensor = x.tanh()?;
        let x: Tensor = self.dropout.forward(&x,train)?;
        let x: Tensor = self.out_proj.forward(&x)?;
        let x: Tensor = x.get_on_dim(1,0)?;
        Ok(x)
    }
}

pub struct RobertaForSequenceClassification{
    roberta: RobertaModel,
    classifier: RobertaClassificationHead
}

impl RobertaForSequenceClassification{
    pub fn load(vb: VarBuilder, config: &RobertaConfig, num_labels: usize) -> Result<RobertaForSequenceClassification> {
        let roberta: RobertaModel = RobertaModel::load(vb.pp("roberta"),config)?;
        let classifier: RobertaClassificationHead = RobertaClassificationHead::load(vb.pp("classifier"),config, num_labels)?;
        Ok(RobertaForSequenceClassification{
            roberta: roberta,
            classifier: classifier
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor,attention_mask :&Tensor, train: bool) -> Result<Tensor> {
        let x: Tensor = self.roberta.forward(input_ids, token_type_ids, attention_mask, train)?;
        let x: Tensor = self.classifier.forward(x, train)?;
        Ok(x)
    }
}