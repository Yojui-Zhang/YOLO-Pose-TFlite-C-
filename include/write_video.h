

cv::VideoWriter video_writer;
bool stop_signal = false;

// 信號處理函數
void handle_signal(int signal) {
    std::cout << "\nInterrupt signal (" << signal << ") received. Closing video file..." << std::endl;
    stop_signal = true;
    video_writer.release(); // 確保釋放影片寫入器
    exit(0); // 結束程式
}

void write_video(int frame_width, int frame_height, int fps, std::string output_filename){

    // 捕捉 SIGINT 信號（Ctrl+C）
    std::signal(SIGINT, handle_signal);

    // 初始化影片寫入器
    video_writer.open(
        output_filename, 
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
        fps, 
        cv::Size(frame_width, frame_height)
    );

    if (!video_writer.isOpened()) {
        std::cerr << "Error: Could not open video file for writing!" << std::endl;
    }

}