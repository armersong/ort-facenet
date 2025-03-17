use image::GenericImageView;
use ndarray::Array3;
use ndarray::Axis;
use ndarray::CowArray;
use ort::Environment;
use ort::Value;
use std::path::Path;
use std::sync::Arc;

// 加载并预处理图像
fn load_and_preprocess_image(path: &str) -> Vec<f32> {
    let img = image::open(Path::new(path)).expect("Failed to open image");
    let resized_img = img.resize_exact(160, 160, image::imageops::FilterType::Triangle);
    let mut input_data = Vec::new();
    for (_, _, pixel) in resized_img.pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        // let r = pixel[0] as f32 / 127.5 - 1.0;
        // let g = pixel[1] as f32 / 127.5 - 1.0;
        // let b = pixel[2] as f32 / 127.5 - 1.0;
        input_data.push(r);
        input_data.push(g);
        input_data.push(b);
    }
    input_data
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // 检查两个向量的长度是否相同
    assert_eq!(a.len(), b.len(), "向量长度必须相同");

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    // 计算点积和向量的模
    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    // 计算向量的模
    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();

    // 避免除零错误
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    // 计算余弦相似度
    dot_product / (norm_a * norm_b)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化ONNX Runtime环境
    let env = Environment::builder()
        .with_name("facenet_example")
        .build()?;
    let env = Arc::new(env);
    // 加载ONNX模型
    let session = ort::session::SessionBuilder::new(&env)
        .unwrap()
        .with_model_from_file("facenet.onnx")
        .unwrap();

    let mut faces = Vec::new();
    // 加载并预处理图像
    for i in 1..=6 {
        let filename = format!("images/face{}.jpeg", i);
        let input_data = load_and_preprocess_image(&filename);

        // 准备输入张量
        let image = Array3::<f32>::from_shape_vec((3, 160, 160), input_data)?.insert_axis(Axis(0));
        println!("dim: {:?}, sharp: {:?}", image.dim(), image.shape());

        let binding = CowArray::from(image).into_dyn();
        let input_tensor = Value::from_array(session.allocator(), &binding).unwrap();
        // 运行模型
        let outputs = session.run(vec![input_tensor])?;
        for (i, output) in outputs.iter().enumerate() {
            if let Ok(output_array) = output.try_extract::<f32>() {
                // 将数组转换为切片并打印
                if let Some(slice) = output_array.view().as_slice() {
                    faces.push(Vec::from(slice));
                }
            } else {
                println!("Output {} is not an f32 array.", i);
            }
        }
    }
    for i in 0..faces.len() {
        for j in 0..faces.len() {
            if i == j {
                continue;
            }
            let distance = get_educlidean_distance(&faces[i], &faces[j]);
            let similarity = cosine_similarity(&faces[i], &faces[j]);
            println!(
                "{} vs {} distance:{}, similarity: {}",
                i, j, distance, similarity
            );
        }
    }
    Ok(())
}

fn get_educlidean_distance(src: &[f32], dst: &[f32]) -> f32 {
    assert!(
        src.len() == dst.len(),
        "src and dst must have the same length"
    );
    // assert!(src.len() == 128, "src and dst must 128 elements");
    let mut sum = 0.0;
    for i in 0..src.len() {
        sum += (src[i] - dst[i]) * (src[i] - dst[i]);
    }
    let norm = sum.sqrt();
    norm
}
