use parquet::file::serialized_reader::SerializedFileReader;
use parquet::file::reader::FileReader;
use parquet::record::reader::RowIter;
use parquet::record::Field;

use hf_hub::api::sync::{Api,ApiBuilder,ApiRepo};
use hf_hub::{Cache, Repo, RepoType};
use hf_hub::api::RepoInfo;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::Tensor;
use candle_nn::VarMap;

use std::path::{PathBuf,Path};
use std::fs::File;
use std::error::Error;
use std::env;

use rand::Rng;


pub fn load(model_name:&str,files:&str,cache_dir:&str) -> Result<PathBuf, Box<dyn Error + Send + Sync>>{
    let mut path: PathBuf = env::current_dir()?;
    path.push(cache_dir);
    let cache: Cache = Cache::new(path);
    // let cache = Cache::from_env();
    println!("load {} and save at {:?}",files,cache);
    let api: Api = ApiBuilder::from_cache(cache).build()?;
    let repo: Repo = Repo::with_revision(String::from(model_name), RepoType::Model, String::from("main"));
    let api: ApiRepo = api.repo(repo);
    let result: PathBuf = api.get(files)?;
    Ok(result)
}

pub fn get_datasets_path(dataset_id: String,cache_dir: &str) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>>{
    let mut path: PathBuf = env::current_dir()?;
    path.push(cache_dir);
    let cache: Cache = Cache::new(path);
    // let cache = Cache::from_env();

    let api: Api = ApiBuilder::from_cache(cache).build()?;
    let repo: Repo = Repo::with_revision(
        dataset_id,
        RepoType::Dataset,
        "refs/convert/parquet".to_string(),
    );
    let api: ApiRepo = api.repo(repo);
    let info: RepoInfo = api.info()?;
    let mut all_file: Vec<PathBuf> = Vec::new();
    for element in info.siblings.iter(){
        let filename:&str = &element.rfilename;
        if filename.ends_with(".parquet"){
            all_file.push(
                api.get(filename)?
            );
        }
    }
    Ok(all_file)
}

pub fn get_datasets(path: &PathBuf) -> Result<(Vec<String>,Vec<u32>),Box<dyn Error + Send + Sync>>{
    let file: File = File::open(path)?;
    let reader: SerializedFileReader<File> = SerializedFileReader::new(file)?;
    let row: RowIter = reader.get_row_iter(None)?;

    let mut text: Vec<String> = Vec::new();
    let mut target: Vec<u32> = Vec::new();

    for element in row{
        for (_name, field) in element?.get_column_iter() {
            match field{
                Field::Str(x) => {
                    let value: String = String::from(x);
                    text.push(value);
                },
                Field::Long(x) => {
                    let value: u32 = *x as u32;
                    target.push(value);
                },
                _ => panic!("error tye")
            };
        }
    }
    Ok((text,target))
}

pub fn shuffle<T>(vec:&mut Vec<T>){
    let mut rng = rand::rng();
    // let arr: Vec<u32> = (0..vec.len()).map(|x| x as u32).collect();
    // println!("{:?}",arr);
    for _ in 0..vec.len(){
        let i: usize = rng.random_range(0..vec.len());
        let j: usize = rng.random_range(0..vec.len());
        if i==j{
            continue;
        }
        vec.swap(i,j);
    }
}

// https://docs.rs/candle-nn/latest/src/candle_nn/var_map.rs.html#12-14
pub fn load_some_layer<P: AsRef<Path>>(obj:&mut VarMap, path: P) -> Result<(), Box<dyn Error + Send + Sync>> {
    let path: &Path = path.as_ref();
    let data:MmapedSafetensors = unsafe { MmapedSafetensors::new(path)? };
    let mut tensor_data = obj.data().lock().unwrap();
    for (name, var) in tensor_data.iter_mut() {
        let data: Tensor = match data.load(name, var.device()){
            Ok(t) => t,
            Err(_) => continue
        };
        if let Err(_) = var.set(&data) {
            panic!("file to setting {name}");
            // candle_core::bail!("error setting {name} using data from {path:?}: {err}",)
        }
    }
    Ok(())
}

// pub fn save<P: AsRef<Path>>(obj:&mut VarMap, path: P)  -> Result<(), Box<dyn Error + Send + Sync>> {
//     let tensor_data = obj.data().lock().unwrap();
//     let data = tensor_data.iter().map(|(k, v)| (k, v.as_tensor()));
//     safetensors::tensor::serialize_to_file(data, &None, path.as_ref())?;
//     Ok(())
// }

// https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L100C5-L100C47
pub fn new_lr(lr: f64,curr_step: usize,num_train_step: usize, num_warmup: usize) -> f64{
    if curr_step < num_warmup{
        lr * (curr_step as f64)/(1f64.max(num_warmup as f64))
    }else{
        let a: f64 = (num_train_step - curr_step) as f64;
        let b: f64 = (num_train_step - num_warmup) as f64;
        lr * 0f64.max(a/1f64.max(b))
    }
}