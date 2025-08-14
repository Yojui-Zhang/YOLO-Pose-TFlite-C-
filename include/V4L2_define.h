#define NUM_PLANES				3
#define TEST_BUFFER_NUM			3
#define NUM_SENSORS				1
#define NUM_CARDS				8
#define DEFAULT					4
#define WIDTH					1280
#define HEIGHT					720

#define INFO_LEVEL				4
#define DBG_LEVEL				5
#define ERR_LEVEL				6

/*
 * V4L2 releated data structure definition
 */
struct plane_buffer {
	__u8 *start;
	__u32 plane_size;
	__u32 length;
	size_t offset;
};

struct testbuffer {
	struct plane_buffer planes[NUM_PLANES];
	__u32 nr_plane;
};

struct video_channel {
	int v4l_fd;
	int saved_file_fd;

	__u32 init;
	__u32 on;
	__u32 index;
	__u32 cap_fmt;
	__u32 mem_type;
	__u32 cur_buf_id;
	__u32 frame_num;

	__u32 out_width;
	__u32 out_height;

	char save_file_name[100];
	char v4l_dev_name[100];

	/* fb info */
	__s32 x_offset;
	__s32 y_offset;

	struct testbuffer buffers[TEST_BUFFER_NUM];
	__u32 nr_buffer;

	struct timeval tv1;
	struct timeval tv2;
};

struct v4l2_device {
	struct video_channel video_ch[NUM_SENSORS];
};

struct media_dev {
	struct drm_device *drm_dev;
	struct v4l2_device *v4l2_dev;
	__u32 total_frames_cnt;
};

/*
 * Global variable definition
 */
static sigset_t sigset_v;
static uint32_t log_level;
static uint32_t g_cam;
static uint32_t g_cam_num;
static uint32_t g_num_buffers;
static uint32_t g_num_frames;
static uint32_t g_num_planes;
static uint32_t g_capture_mode;
static uint32_t g_camera_framerate;
static uint32_t g_cap_fmt;
static uint32_t g_mem_type;
static uint32_t g_out_width;
static uint32_t g_out_height;
static uint32_t g_cap_ow;
static uint32_t g_cap_oh;

static bool g_saved_to_file;
static bool g_performance_test;
static bool quitflag;

static char g_v4l_device[NUM_SENSORS][100];
static char g_saved_filename[NUM_SENSORS][100];
static char g_fmt_name[10];

static bool g_cap_hfilp;
static bool g_cap_vfilp;
static int32_t g_cap_alpha;


static struct media_dev media;

int v4l2init(int cam_id);
cv::Mat v4l2Cam();
static int media_device_stop(struct media_dev *media);
static void media_device_cleanup(struct media_dev *media);
static void media_device_close(struct media_dev *media);
static void media_device_free(struct media_dev *media);
