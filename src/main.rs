use ndarray::ArrayView3;
use std::time::Instant;
use vit_tracker::{BBox, TrackingResult, VitTrack};

#[cfg(feature = "opencv-camera")]
use opencv::{
    core, highgui, imgproc, prelude::*, videoio, Result as CvResult,
};

#[cfg(feature = "opencv-camera")]
fn mat_to_array3(mat: &core::Mat) -> CvResult<ArrayView3<u8>> {
    let bytes = mat.data_bytes().unwrap();
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let channels = mat.channels() as usize;
    
    let array = ArrayView3::from_shape((rows, cols, channels), bytes).unwrap();
    Ok(array)
}

#[cfg(feature = "opencv-camera")]
fn draw_result(frame: &mut core::Mat, result: &TrackingResult, fps: f64) -> CvResult<()> {
    let [x, y, w, h] = result.bbox;

    let color = if result.success {
        core::Scalar::new(0.0, 255.0, 0.0, 0.0) // Green
    } else {
        core::Scalar::new(0.0, 0.0, 255.0, 0.0) // Red
    };

    // Draw bounding box
    imgproc::rectangle(
        frame,
        core::Rect::new(x, y, w, h),
        color,
        2,
        imgproc::LINE_8,
        0,
    )?;

    // Draw center point
    if result.success {
        imgproc::circle(
            frame,
            core::Point::new(x + w / 2, y + h / 2),
            4,
            color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
    }

    // Draw FPS
    imgproc::put_text(
        frame,
        &format!("FPS: {:.1}", fps),
        core::Point::new(10, 30),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        core::Scalar::new(0.0, 0.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    // Draw score
    imgproc::put_text(
        frame,
        &format!("Score: {:.3}", result.score),
        core::Point::new(10, 60),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    // Draw status
    let status = if result.success { "Tracking" } else { "Lost" };
    imgproc::put_text(
        frame,
        status,
        core::Point::new(10, 90),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        imgproc::LINE_8,
        false,
    )?;

    // Draw RKNN label
    imgproc::put_text(
        frame,
        "RKNN NPU (Rust)",
        core::Point::new(10, 120),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    Ok(())
}

#[cfg(feature = "opencv-camera")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/object_tracking_vittrack_2023sep.rknn");

    let camera_id: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(11);

    println!("VitTrack Rust + RKNN");
    println!("====================");
    println!("Model: {}", model_path);
    println!("Camera: {}", camera_id);

    // Create tracker
    println!("\nLoading tracker...");
    let mut tracker = VitTrack::new(model_path)?;
    println!("Tracker loaded!");

    // Open camera
    println!("Opening camera {}...", camera_id);
    let mut cap = videoio::VideoCapture::new(camera_id, videoio::CAP_ANY)?;

    if !cap.is_opened()? {
        return Err(format!("Cannot open camera {}", camera_id).into());
    }

    cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1920.0)?;
    cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 1080.0)?;

    let mut frame = core::Mat::default();
    cap.read(&mut frame)?;

    if frame.empty() {
        return Err("Cannot read frame".into());
    }

    // Select ROI
    println!("\nSelect object to track...");
    let roi = highgui::select_roi(&frame, false, false)?;
    // highgui::destroy_window("Select Object")?;

    if roi.width == 0 || roi.height == 0 {
        return Err("No object selected".into());
    }

    // Initialize tracker
    let image = mat_to_array3(&frame)?;
    let bbox = BBox::new(roi.x, roi.y, roi.width, roi.height);
    tracker.init(&image, bbox);
    let test_result = tracker.update(&image)?;
    println!("TEST: update on same frame: {:?}", test_result);

    println!(
        "Tracker initialized: x={}, y={}, w={}, h={}",
        roi.x, roi.y, roi.width, roi.height
    );
    println!("\nControls: Q - quit, R - reinitialize");

    let mut fps_history: Vec<f64> = Vec::with_capacity(32);
    let mut avg_fps = 0.0;
    loop {
        let start = Instant::now();
        cap.read(&mut frame)?;
        if frame.empty() {
            break;
        }
        let elapsed = start.elapsed().as_micros();
        println!("Read frame: {} usec", elapsed);

        let timer = Instant::now();
        let image = mat_to_array3(&frame)?;
        let elapsed = timer.elapsed().as_micros();
        println!("mat_to_array3: {} usec", elapsed);

        // Track
        let timer = Instant::now();
        let result = tracker.update(&image)?;
        let elapsed = timer.elapsed().as_micros();
        println!("tracker.update: {} usec", elapsed);

        // Draw result
        let timer = Instant::now();
        draw_result(&mut frame, &result, avg_fps)?;
        let elapsed = timer.elapsed().as_micros();
        println!("draw_result: {} usec", elapsed);

        let timer = Instant::now();
        highgui::imshow("VitTrack Rust", &frame)?;
        let elapsed = timer.elapsed().as_micros();
        println!("highgui::imshow: {} usec", elapsed);

        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            break;
        } else if key == 'r' as i32 {
            println!("\nReinitializing...");
            let roi = highgui::select_roi(&frame, false, false)?;
            // highgui::destroy_window("Select Object")?;
            if roi.width > 0 && roi.height > 0 {
                let image = mat_to_array3(&frame)?;
                let bbox = BBox::new(roi.x, roi.y, roi.width, roi.height);
                tracker.init(&image, bbox);
                fps_history.clear();
                println!("Reinitialized!");
            }
        }
        let elapsed = start.elapsed().as_secs_f64();

        // Calculate FPS
        let fps = 1.0 / elapsed;
        fps_history.push(fps);
        if fps_history.len() > 30 {
            fps_history.remove(0);
        }
        avg_fps = fps_history.iter().sum::<f64>() / fps_history.len() as f64;
    }

    if !fps_history.is_empty() {
        let avg_fps: f64 = fps_history.iter().sum::<f64>() / fps_history.len() as f64;
        println!("\nAverage FPS: {:.1}", avg_fps);
    }

    Ok(())
}

#[cfg(not(feature = "opencv-camera"))]
fn main() {
    println!("OpenCV camera support not enabled.");
    println!("Build with: cargo build --features opencv-camera");
}