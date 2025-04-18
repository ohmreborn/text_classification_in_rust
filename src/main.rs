pub mod utils;
pub mod layer;

use std::path::PathBuf;
use std::fs;

use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device,DType,Tensor};
use candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use candle_nn::{
    VarBuilder, VarMap,
    ParamsAdamW, AdamW,
    Optimizer
};
use candle_nn::loss::cross_entropy;

use tokenizers::tokenizer::{Tokenizer,
    pad_encodings,
    PaddingParams,
    PaddingStrategy,
    PaddingDirection
};

use layer::model::{RobertaConfig, RobertaForSequenceClassification};
use crate::utils::*;

fn main() -> Result<(),Box<dyn std::error::Error + Send + Sync>>{
    let epoch: usize = 3;
    let lr: f64 = 2e-5;
    let batch_size: usize = 5;
    let train: bool = true;

    // some config
    let model_name: &str = "FacebookAI/roberta-base";
    let tokenizer_path:&str = "tokenizer.json";
    let model_path:&str = "model.safetensors";
    let config_path:&str = "config.json";
    let cache_dir: &str = "myfile";
    let save_file: PathBuf = PathBuf::from("model.safetensors");
    let dataset_id: String = String::from("stanfordnlp/imdb");
    // load datasets file from dataset_id and recive all dataset path in all_datasets_file
    let all_datasets_file: Vec<PathBuf> = get_datasets_path(dataset_id,cache_dir)?;
    // choose device
    let device: Device = if cuda_is_available(){
        Device::new_cuda(0)?
    }else if metal_is_available(){
        Device::new_metal(0)?
    }else{
        Device::Cpu
    };
    
    // load config.json
    let config_file: PathBuf = load(model_name,config_path,cache_dir)?;
    let data:String = fs::read_to_string(&config_file)?;
    let config: RobertaConfig = serde_json::from_str(&data)?;

    // load tokenizer
    let tokenizer_file: PathBuf = load(model_name,tokenizer_path,cache_dir)?;
    let tokenizer: Tokenizer = Tokenizer::from_file(tokenizer_file)?;
    
    // load dataset
    let (mut text,mut target) = get_datasets(&all_datasets_file[0])?;
    assert_eq!(text.len(),target.len());
    let mut datasets: Vec<(String,u32)> = Vec::with_capacity(text.len());
    loop{
        match (text.pop(), target.pop()){
            (Some(s),Some(i)) => datasets.push((s,i)),
            _ => break
        }
    }
    // shuffle datasetes
    shuffle(&mut datasets);

    println!("create data loader");
    // create a data loader
    let n_samples: usize = datasets.len();
    let mut data_loader: Vec<(Vec<String>, Vec<u32>)> = Vec::new();
    for i in (0..n_samples).step_by(batch_size) {
        let end = (i + batch_size).min(n_samples);
        let mut x_batch: Vec<String> = Vec::with_capacity(end-i);
        let mut y_batch: Vec<u32> = Vec::with_capacity(end-i);

        for _ in i..end{
            let (x,y) = match datasets.pop(){
                Some((a,b)) => (a,b),
                _ => panic!("error")
            };
            x_batch.push(x);
            y_batch.push(y);
        }
        data_loader.push((x_batch, y_batch));
    }

    drop(text);
    drop(target);
    drop(datasets);

    println!("load model");
    // load modle file
    let model_file: PathBuf = load(model_name,model_path,cache_dir)?;
    let mut varmap: VarMap = VarMap::new();
    let vb: VarBuilderArgs<Box<dyn SimpleBackend>> = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model: RobertaForSequenceClassification = RobertaForSequenceClassification::load(vb, &config,2)?;
    load_some_layer(&mut varmap, model_file)?;

    // optimizer
    let num_train_step: usize = epoch * data_loader.len();
    let num_warmup: usize = data_loader.len();
    let mut curr_step: usize = 0;
    let param: ParamsAdamW = ParamsAdamW {
        lr: lr,
        ..ParamsAdamW::default()
    };
    let mut optimizer: AdamW = AdamW::new(varmap.all_vars(), param)?;

    // pad config for create a dataloader
    let mut pad_config: PaddingParams = PaddingParams{
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Right,
        pad_to_multiple_of: Some(batch_size),
        pad_id: config.pad_token_id,
        pad_type_id: 0,
        pad_token: String::from("<pad>"),
    };
    println!("start train");
    for _ in 0..epoch{
        for (i, (inp,target)) in data_loader.iter().enumerate(){
            let local_batch: usize = target.len();
            pad_config.pad_to_multiple_of = Some(local_batch);
            
            println!("tokenize text");
            // encode a vector of string to vector ot unsigned int
            let mut encode = tokenizer.encode_batch(inp.clone(),true)?;
            pad_encodings(&mut encode,&pad_config)?;
            let mut batch_encode: Vec<u32> = Vec::new();
            for element in encode.into_iter(){
                let mut val:Vec<u32> = element.get_ids().to_vec(); 
                batch_encode.append(&mut val);
            }
            println!("forward");
            let shape: (usize,usize) = (local_batch,batch_encode.len()/local_batch);
            let target: Tensor = Tensor::from_vec(target.clone(),shape.0,&device)?;
            let input_ids: Tensor = Tensor::from_vec(batch_encode.clone(), shape, &device)?;
            let token_type_ids: Tensor = input_ids.zeros_like()?;
            let attention_mask: Tensor = input_ids.eq(config.pad_token_id)?
                .to_dtype(DType::F32)?
                .broadcast_mul(&Tensor::new(&[f32::MIN],&device)?)?
                .unsqueeze(1)?
                .unsqueeze(1)?;
            let pred: Tensor = model.forward(&input_ids, &token_type_ids, &attention_mask, train)?;
            println!("backward");
            let loss: Tensor = cross_entropy(&pred, &target)?;
            optimizer.backward_step(&loss)?;

            let curr_lr: f64 = new_lr(lr,curr_step,num_train_step,num_warmup);
            optimizer.set_learning_rate(curr_lr);
            curr_step += 1;
            if i% 100 == 0{
                println!("{}",loss);
            }
        }
    }

    match varmap.save(save_file.clone()){
        Ok(()) => println!("save file {} sucess",save_file.display()),
        Err(e) => {
            println!("cannot save file");
            panic!("{}",e);
        }
    };

    Ok(())
}

