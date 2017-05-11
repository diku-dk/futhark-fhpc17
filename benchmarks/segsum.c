#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include <getopt.h>
static int detail_memory = 0;
static int debugging = 0;
/* Crash and burn. */

#include <stdarg.h>

static const char *fut_progname;

void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* Some simple utilities for wall-clock timing.

   The function get_wall_time() returns the wall time in microseconds
   (with an unspecified offset).
*/

#ifdef _WIN32

#include <windows.h>

int64_t get_wall_time() {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

int64_t get_wall_time() {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

#define FUT_BLOCK_DIM 16
/* The simple OpenCL runtime framework used by Futhark. */

#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define OPENCL_SUCCEED(e) opencl_succeed(e, #e, __FILE__, __LINE__)

static cl_context fut_cl_context;
static cl_command_queue fut_cl_queue;
static const char *cl_preferred_platform = "";
static const char *cl_preferred_device = "";
static int cl_preferred_device_num = 0;
static int cl_debug = 0;

static size_t cl_group_size = 256;
static size_t cl_num_groups = 128;
static size_t cl_tile_size = 32;
static size_t cl_lockstep_width = 1;
static const char* cl_dump_program_to = NULL;
static const char* cl_load_program_from = NULL;

struct opencl_device_option {
  cl_platform_id platform;
  cl_device_id device;
  cl_device_type device_type;
  char *platform_name;
  char *device_name;
};

/* This function must be defined by the user.  It is invoked by
   setup_opencl() after the platform and device has been found, but
   before the program is loaded.  Its intended use is to tune
   constants based on the selected platform and device. */
static void post_opencl_setup(struct opencl_device_option*);

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

static const char* opencl_error_string(unsigned int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

static void opencl_succeed(unsigned int ret,
                    const char *call,
                    const char *file,
                    int line) {
  if (ret != CL_SUCCESS) {
    panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

void set_preferred_platform(const char *s) {
  cl_preferred_platform = s;
}

void set_preferred_device(const char *s) {
  int x = 0;
  if (*s == '#') {
    s++;
    while (isdigit(*s)) {
      x = x * 10 + (*s++)-'0';
    }
    // Skip trailing spaces.
    while (isspace(*s)) {
      s++;
    }
  }
  cl_preferred_device = s;
  cl_preferred_device_num = x;
}

static char* opencl_platform_info(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED(clGetPlatformInfo(platform, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED(clGetPlatformInfo(platform, param, req_bytes, info, NULL));

  return info;
}

static char* opencl_device_info(cl_device_id device,
                                cl_device_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED(clGetDeviceInfo(device, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED(clGetDeviceInfo(device, param, req_bytes, info, NULL));

  return info;
}

static void opencl_all_device_options(struct opencl_device_option **devices_out,
                                      size_t *num_devices_out) {
  size_t num_devices = 0, num_devices_added = 0;

  cl_platform_id *all_platforms;
  cl_uint *platform_num_devices;

  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED(clGetPlatformIDs(0, NULL, &num_platforms));

  // Make room for them.
  all_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  platform_num_devices = calloc(num_platforms, sizeof(cl_uint));

  // Fetch all the platforms.
  OPENCL_SUCCEED(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  // Count the number of devices for each platform, as well as the
  // total number of devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    if (clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                       0, NULL, &platform_num_devices[i]) == CL_SUCCESS) {
      num_devices += platform_num_devices[i];
    } else {
      platform_num_devices[i] = 0;
    }
  }

  // Make room for all the device options.
  struct opencl_device_option *devices =
    calloc(num_devices, sizeof(struct opencl_device_option));

  // Loop through the platforms, getting information about their devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_platform_id platform = all_platforms[i];
    cl_uint num_platform_devices = platform_num_devices[i];

    if (num_platform_devices == 0) {
      continue;
    }

    char *platform_name = opencl_platform_info(platform, CL_PLATFORM_NAME);
    cl_device_id *platform_devices =
      calloc(num_platform_devices, sizeof(cl_device_id));

    // Fetch all the devices.
    OPENCL_SUCCEED(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_platform_devices, platform_devices, NULL));

    // Loop through the devices, adding them to the devices array.
    for (cl_uint i = 0; i < num_platform_devices; i++) {
      char *device_name = opencl_device_info(platform_devices[i], CL_DEVICE_NAME);
      devices[num_devices_added].platform = platform;
      devices[num_devices_added].device = platform_devices[i];
      OPENCL_SUCCEED(clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE,
                                     sizeof(cl_device_type),
                                     &devices[num_devices_added].device_type,
                                     NULL));
      // We don't want the structs to share memory, so copy the platform name.
      // Each device name is already unique.
      devices[num_devices_added].platform_name = strclone(platform_name);
      devices[num_devices_added].device_name = device_name;
      num_devices_added++;
    }
    free(platform_devices);
    free(platform_name);
  }
  free(all_platforms);
  free(platform_num_devices);

  *devices_out = devices;
  *num_devices_out = num_devices;
}

static struct opencl_device_option get_preferred_device() {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  int num_platform_matches = 0;
  int num_device_matches = 0;

  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strstr(device.platform_name, cl_preferred_platform) != NULL &&
        strstr(device.device_name, cl_preferred_device) != NULL &&
        num_device_matches++ == cl_preferred_device_num) {
      // Free all the platform and device names, except the ones we have chosen.
      for (size_t j = 0; j < num_devices; j++) {
        if (j != i) {
          free(devices[j].platform_name);
          free(devices[j].device_name);
        }
      }
      free(devices);
      return device;
    }
  }

  panic(1, "Could not find acceptable OpenCL device.\n");
}

static void describe_device_option(struct opencl_device_option device) {
  fprintf(stderr, "Using platform: %s\n", device.platform_name);
  fprintf(stderr, "Using device: %s\n", device.device_name);
}

static cl_build_status build_opencl_program(cl_program program, cl_device_id device, const char* options) {
  cl_int ret_val = clBuildProgram(program, 1, &device, options, NULL, NULL);

  // Avoid termination due to CL_BUILD_PROGRAM_FAILURE
  if (ret_val != CL_SUCCESS && ret_val != CL_BUILD_PROGRAM_FAILURE) {
    assert(ret_val == 0);
  }

  cl_build_status build_status;
  ret_val = clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_STATUS,
                                  sizeof(cl_build_status),
                                  &build_status,
                                  NULL);
  assert(ret_val == 0);

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    ret_val = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    assert(ret_val == 0);

    build_log = malloc(ret_val_size+1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    assert(ret_val == 0);

    // The spec technically does not say whether the build log is zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);
  }

  return build_status;
}

static cl_program setup_opencl(const char *prelude_src, const char *src) {

  cl_int error;
  cl_platform_id platform;
  cl_device_id device;
  cl_uint platforms, devices;
  size_t max_group_size;

  struct opencl_device_option device_option = get_preferred_device();

  if (cl_debug) {
    describe_device_option(device_option);
  }

  device = device_option.device;
  platform = device_option.platform;

  OPENCL_SUCCEED(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_group_size, NULL));

  size_t max_tile_size = sqrt(max_group_size);

  if (max_group_size < cl_group_size) {
    fprintf(stderr, "Warning: Device limits group size to %zu (setting was %zu)\n",
            max_group_size, cl_group_size);
    cl_group_size = max_group_size;
  }

  if (max_tile_size < cl_tile_size) {
    fprintf(stderr, "Warning: Device limits tile size to %zu (setting was %zu)\n",
            max_tile_size, cl_tile_size);
    cl_tile_size = max_tile_size;
  }

  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)platform,
    0
  };
  // Note that nVidia's OpenCL requires the platform property
  fut_cl_context = clCreateContext(properties, 1, &device, NULL, NULL, &error);
  assert(error == 0);

  fut_cl_queue = clCreateCommandQueue(fut_cl_context, device, 0, &error);
  assert(error == 0);

  // Make sure this function is defined.
  post_opencl_setup(&device_option);

  char *fut_opencl_src = NULL;
  size_t src_size = 0;

  // Maybe we have to read OpenCL source from somewhere else (used for debugging).
  if (cl_load_program_from) {
    FILE *f = fopen(cl_load_program_from, "r");
    assert(f != NULL);
    fseek(f, 0, SEEK_END);
    src_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    fut_opencl_src = malloc(src_size);
    fread(fut_opencl_src, 1, src_size, f);
    fclose(f);
  } else {
    // Build the OpenCL program.  First we have to prepend the prelude to the program source.
    size_t prelude_size = strlen(prelude_src);
    size_t program_size = strlen(src);
    src_size = prelude_size + program_size;
    fut_opencl_src = malloc(src_size + 1);
    strncpy(fut_opencl_src, prelude_src, src_size);
    strncpy(fut_opencl_src+prelude_size, src, src_size-prelude_size);
    fut_opencl_src[src_size] = 0;
  }

  cl_program prog;
  error = 0;
  const char* src_ptr[] = {fut_opencl_src};

  if (cl_dump_program_to) {
    FILE *f = fopen(cl_dump_program_to, "w");
    assert(f != NULL);
    fputs(fut_opencl_src, f);
    fclose(f);
  }

  prog = clCreateProgramWithSource(fut_cl_context, 1, src_ptr, &src_size, &error);
  assert(error == 0);
  char compile_opts[1024];
  snprintf(compile_opts, sizeof(compile_opts), "-DFUT_BLOCK_DIM=%d -DLOCKSTEP_WIDTH=%d -DDEFAULT_GROUP_SIZE=%d -DDEFAULT_NUM_GROUPS=%d  -DDEFAULT_TILE_SIZE=%d", FUT_BLOCK_DIM, cl_lockstep_width, cl_group_size, cl_num_groups, cl_tile_size);
  OPENCL_SUCCEED(build_opencl_program(prog, device, compile_opts));
  free(fut_opencl_src);

  return prog;
}

static const char fut_opencl_prelude[] =
                  "__kernel void dummy_kernel(__global unsigned char *dummy, int n)\n{\n    const int thread_gid = get_global_id(0);\n    \n    if (thread_gid >= n)\n        return;\n}\ntypedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef uchar uint8_t;\ntypedef ushort uint16_t;\ntypedef uint uint32_t;\ntypedef ulong uint64_t;\n#define ALIGNED_LOCAL_MEMORY(m,size) __local unsigned char m[size] __attribute__ ((align))\nstatic inline int8_t add8(int8_t x, int8_t y)\n{\n    return x + y;\n}\nstatic inline int16_t add16(int16_t x, int16_t y)\n{\n    return x + y;\n}\nstatic inline int32_t add32(int32_t x, int32_t y)\n{\n    return x + y;\n}\nstatic inline int64_t add64(int64_t x, int64_t y)\n{\n    return x + y;\n}\nstatic inline int8_t sub8(int8_t x, int8_t y)\n{\n    return x - y;\n}\nstatic inline int16_t sub16(int16_t x, int16_t y)\n{\n    return x - y;\n}\nstatic inline int32_t sub32(int32_t x, int32_t y)\n{\n    return x - y;\n}\nstatic inline int64_t sub64(int64_t x, int64_t y)\n{\n    return x - y;\n}\nstatic inline int8_t mul8(int8_t x, int8_t y)\n{\n    return x * y;\n}\nstatic inline int16_t mul16(int16_t x, int16_t y)\n{\n    return x * y;\n}\nstatic inline int32_t mul32(int32_t x, int32_t y)\n{\n    return x * y;\n}\nstatic inline int64_t mul64(int64_t x, int64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int16_t smin16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int32_t smin32(int32_t x, int32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline char ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline char ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline char ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline char ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline char ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline char ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline char ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline char ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline char slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline char slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline char slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline char slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline char sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline char sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline char sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline char sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int8_t sext_i8_i8(int8_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i8_i16(int8_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i8_i32(int8_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i8_i64(int8_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i16_i8(int16_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i16_i16(int16_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i16_i32(int16_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i16_i64(int16_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i32_i8(int32_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i32_i16(int32_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i32_i32(int32_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i32_i64(int32_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i64_i8(int64_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i64_i16(int64_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i64_i32(int64_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i64_i64(int64_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i8_i8(uint8_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i8_i16(uint8_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i8_i32(uint8_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i8_i64(uint8_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i16_i8(uint16_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i16_i16(uint16_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i16_i32(uint16_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i16_i64(uint16_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i32_i8(uint32_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i32_i16(uint32_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i32_i32(uint32_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i32_i64(uint32_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i64_i8(uint64_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i64_i16(uint64_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i64_i32(uint64_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i64_i64(uint64_t x)\n{\n    return x;\n}\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return x < y ? x : y;\n}\nstatic inline float fmax32(float x, float y)\n{\n    return x < y ? y : x;\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inline char cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline char cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\n#define group_sizze_1032 (DEFAULT_GROUP_SIZE)\n#define group_sizze_1032 (DEFAULT_GROUP_SIZE)\n#define group_sizze_1032 (DEFAULT_GROUP_SIZE)\n#define group_sizze_1032 (DEFAULT_GROUP_SIZE)\n#define group_sizze_1032 (DEFAULT_GROUP_SIZE)\n";
static const char fut_opencl_program[] =
                  "__kernel void fut_kernel_map_transpose_i32(__global int32_t *odata,\n                                           uint odata_offset, __global\n                                           int32_t *idata, uint idata_offset,\n                                           uint width, uint height,\n                                           uint input_size, uint output_size,\n                                           __local int32_t *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_global_id(0);\n    y_index = get_global_id(1);\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowheight_i32(__global int32_t *odata,\n                                                     uint odata_offset, __global\n                                                     int32_t *idata,\n                                                     uint idata_offset,\n                                                     uint width, uint height,\n                                                     uint input_size,\n                                                     uint output_size,\n                                                     uint mulx, __local\n                                                     int32_t *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(0) +\n        get_local_id(1) % mulx * FUT_BLOCK_DIM;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(1) / mulx;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0) / mulx;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(1) +\n        get_local_id(0) % mulx * FUT_BLOCK_DIM;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowwidth_i32(__global int32_t *odata,\n                                                    uint odata_offset, __global\n                                                    int32_t *idata,\n                                                    uint idata_offset,\n                                                    uint width, uint height,\n                                                    uint input_size,\n                                                    uint output_size, uint muly,\n                                                    __local int32_t *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(int32_t);\n    idata += idata_offset / sizeof(int32_t);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(0) / muly;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(1) +\n        get_local_id(0) % muly * FUT_BLOCK_DIM;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(0) +\n        get_local_id(1) % muly * FUT_BLOCK_DIM;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1) / muly;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void map_intra_group_kernel_1008(int32_t sizze_285, int32_t sizze_286,\n                                          __global unsigned char *mem_1061,\n                                          __global unsigned char *mem_1064)\n{\n    int32_t wave_sizze_1123;\n    int32_t group_sizze_1124;\n    char thread_active_1125;\n    int32_t gtid_1002;\n    int32_t global_tid_1008;\n    int32_t local_tid_1009;\n    int32_t group_id_1010;\n    \n    global_tid_1008 = get_global_id(0);\n    local_tid_1009 = get_local_id(0);\n    group_sizze_1124 = get_local_size(0);\n    wave_sizze_1123 = LOCKSTEP_WIDTH;\n    group_id_1010 = get_group_id(0);\n    gtid_1002 = group_id_1010;\n    thread_active_1125 = slt32(gtid_1002, sizze_285);\n    \n    int32_t res_1024;\n    \n    if (thread_active_1125) {\n        int32_t acc_1027 = 0;\n        \n        for (int32_t i_1028 = 0; i_1028 < sizze_286; i_1028++) {\n            int32_t binop_param_y_1030 = *(__global\n                                           int32_t *) &mem_1061[(i_1028 *\n                                                                 sizze_285 +\n                                                                 gtid_1002) *\n                                                                4];\n            int32_t res_1031 = acc_1027 + binop_param_y_1030;\n            int32_t acc_tmp_1126 = res_1031;\n            \n            acc_1027 = acc_tmp_1126;\n        }\n        res_1024 = acc_1027;\n    }\n    if (thread_active_1125) {\n        *(__global int32_t *) &mem_1064[gtid_1002 * 4] = res_1024;\n    }\n}\n__kernel void map_kernel_971(int32_t sizze_285, int32_t sizze_286, __global\n                             unsigned char *mem_1053, __global\n                             unsigned char *mem_1056)\n{\n    int32_t wave_sizze_1119;\n    int32_t group_sizze_1120;\n    char thread_active_1121;\n    int32_t gtid_964;\n    int32_t global_tid_971;\n    int32_t local_tid_972;\n    int32_t group_id_973;\n    \n    global_tid_971 = get_global_id(0);\n    local_tid_972 = get_local_id(0);\n    group_sizze_1120 = get_local_size(0);\n    wave_sizze_1119 = LOCKSTEP_WIDTH;\n    group_id_973 = get_group_id(0);\n    gtid_964 = global_tid_971;\n    thread_active_1121 = slt32(gtid_964, sizze_285);\n    \n    int32_t res_993;\n    \n    if (thread_active_1121) {\n        int32_t acc_996 = 0;\n        \n        for (int32_t i_997 = 0; i_997 < sizze_286; i_997++) {\n            int32_t binop_param_y_999 = *(__global int32_t *) &mem_1053[(i_997 *\n                                                                         sizze_285 +\n                                                                         gtid_964) *\n                                                                        4];\n            int32_t res_1000 = acc_996 + binop_param_y_999;\n            int32_t acc_tmp_1122 = res_1000;\n            \n            acc_996 = acc_tmp_1122;\n        }\n        res_993 = acc_996;\n    }\n    if (thread_active_1121) {\n        *(__global int32_t *) &mem_1056[gtid_964 * 4] = res_993;\n    }\n}\n__kernel void segmented_redomap__large_comm_many_kernel_418(__local volatile\n                                                            int64_t *mem_aligned_0,\n                                                            int32_t sizze_285,\n                                                            int32_t sizze_286,\n                                                            int32_t elements_per_thread_702,\n                                                            int32_t num_groups_per_segment_760,\n                                                            int32_t threads_within_segment_766,\n                                                            __global\n                                                            unsigned char *xss_mem_1048,\n                                                            __global\n                                                            unsigned char *mem_1076)\n{\n    __local volatile char *restrict mem_1073 = mem_aligned_0;\n    int32_t wave_sizze_1132;\n    int32_t group_sizze_1133;\n    char thread_active_1134;\n    int32_t gtid_305;\n    int32_t gtid_417;\n    int32_t global_tid_418;\n    int32_t local_tid_419;\n    int32_t group_id_420;\n    \n    global_tid_418 = get_global_id(0);\n    local_tid_419 = get_local_id(0);\n    group_sizze_1133 = get_local_size(0);\n    wave_sizze_1132 = LOCKSTEP_WIDTH;\n    group_id_420 = get_group_id(0);\n    gtid_305 = squot32(group_id_420, num_groups_per_segment_760);\n    gtid_417 = group_id_420 - squot32(group_id_420,\n                                      num_groups_per_segment_760) *\n        num_groups_per_segment_760;\n    thread_active_1134 = slt32(gtid_305, sizze_285) && slt32(gtid_417,\n                                                             num_groups_per_segment_760);\n    \n    int32_t segment_index_768;\n    int32_t y_769;\n    int32_t y_770;\n    int32_t index_within_segment_771;\n    int32_t y_772;\n    int32_t offset_773;\n    \n    if (thread_active_1134) {\n        segment_index_768 = squot32(group_id_420, num_groups_per_segment_760);\n        y_769 = srem32(group_id_420, num_groups_per_segment_760);\n        y_770 = group_sizze_1032 * y_769;\n        index_within_segment_771 = local_tid_419 + y_770;\n        y_772 = sizze_286 * segment_index_768;\n        offset_773 = index_within_segment_771 + y_772;\n    }\n    \n    int32_t chunk_sizze_774;\n    int32_t remaining_elements_1135 = squot32(sizze_286 -\n                                              index_within_segment_771 +\n                                              threads_within_segment_766 - 1,\n                                              threads_within_segment_766);\n    \n    if (slt32(elements_per_thread_702, remaining_elements_1135)) {\n        chunk_sizze_774 = elements_per_thread_702;\n    } else {\n        chunk_sizze_774 = remaining_elements_1135;\n    }\n    if (thread_active_1134) { }\n    \n    int32_t res_778;\n    int32_t final_result_788;\n    int32_t acc_781 = 0;\n    int32_t groupstream_mapaccum_dummy_chunk_sizze_779 = 1;\n    \n    if (thread_active_1134) {\n        for (int32_t i_780 = 0; i_780 < chunk_sizze_774; i_780++) {\n            int32_t binop_param_y_783;\n            int32_t res_785;\n            \n            binop_param_y_783 = *(__global\n                                  int32_t *) &xss_mem_1048[(offset_773 + i_780 *\n                                                            threads_within_segment_766) *\n                                                           4];\n            res_785 = acc_781 + binop_param_y_783;\n            acc_781 = res_785;\n        }\n    }\n    res_778 = acc_781;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (slt32(local_tid_419, group_sizze_1032) && 1) {\n        *(__local int32_t *) &mem_1073[local_tid_419 * 4] = res_778;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_1136;\n    int32_t my_index_789;\n    int32_t other_offset_790;\n    int32_t binop_param_x_791;\n    int32_t binop_param_y_792;\n    \n    my_index_789 = local_tid_419;\n    other_offset_790 = 0;\n    binop_param_x_791 = *(__local int32_t *) &mem_1073[(local_tid_419 +\n                                                        other_offset_790) * 4];\n    other_offset_790 = 1;\n    while (slt32(other_offset_790, wave_sizze_1132)) {\n        if (slt32(local_tid_419 + other_offset_790, group_sizze_1032) &&\n            ((local_tid_419 - squot32(local_tid_419, wave_sizze_1132) *\n              wave_sizze_1132) & (2 * other_offset_790 - 1)) == 0) {\n            // read array element\n            {\n                binop_param_y_792 = *(volatile __local\n                                      int32_t *) &mem_1073[(local_tid_419 +\n                                                            other_offset_790) *\n                                                           4];\n            }\n            \n            int32_t res_793;\n            \n            if (thread_active_1134) {\n                res_793 = binop_param_x_791 + binop_param_y_792;\n            }\n            binop_param_x_791 = res_793;\n            *(volatile __local int32_t *) &mem_1073[local_tid_419 * 4] =\n                binop_param_x_791;\n        }\n        other_offset_790 *= 2;\n    }\n    skip_waves_1136 = 1;\n    while (slt32(skip_waves_1136, squot32(group_sizze_1133 + wave_sizze_1132 -\n                                          1, wave_sizze_1132))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_790 = skip_waves_1136 * wave_sizze_1132;\n        if ((local_tid_419 - squot32(local_tid_419, wave_sizze_1132) *\n             wave_sizze_1132) == 0 && (squot32(local_tid_419, wave_sizze_1132) &\n                                       (2 * skip_waves_1136 - 1)) == 0) {\n            // read array element\n            {\n                binop_param_y_792 = *(__local\n                                      int32_t *) &mem_1073[(local_tid_419 +\n                                                            other_offset_790) *\n                                                           4];\n            }\n            \n            int32_t res_793;\n            \n            if (thread_active_1134) {\n                res_793 = binop_param_x_791 + binop_param_y_792;\n            }\n            binop_param_x_791 = res_793;\n            *(__local int32_t *) &mem_1073[local_tid_419 * 4] =\n                binop_param_x_791;\n        }\n        skip_waves_1136 *= 2;\n    }\n    final_result_788 = binop_param_x_791;\n    if (local_tid_419 == 0) {\n        *(__global int32_t *) &mem_1076[group_id_420 * 4] = final_result_788;\n    }\n}\n__kernel void segmented_redomap__large_comm_one_kernel_360(__local volatile\n                                                           int64_t *mem_aligned_0,\n                                                           int32_t sizze_285,\n                                                           int32_t sizze_286,\n                                                           int32_t elements_per_thread_719,\n                                                           __global\n                                                           unsigned char *xss_mem_1048,\n                                                           __global\n                                                           unsigned char *mem_1070)\n{\n    __local volatile char *restrict mem_1067 = mem_aligned_0;\n    int32_t wave_sizze_1127;\n    int32_t group_sizze_1128;\n    char thread_active_1129;\n    int32_t gtid_305;\n    int32_t gtid_359;\n    int32_t global_tid_360;\n    int32_t local_tid_361;\n    int32_t group_id_362;\n    \n    global_tid_360 = get_global_id(0);\n    local_tid_361 = get_local_id(0);\n    group_sizze_1128 = get_local_size(0);\n    wave_sizze_1127 = LOCKSTEP_WIDTH;\n    group_id_362 = get_group_id(0);\n    gtid_305 = group_id_362;\n    gtid_359 = group_id_362 - group_id_362;\n    thread_active_1129 = slt32(gtid_305, sizze_285) && slt32(gtid_359, 1);\n    \n    int32_t y_727;\n    int32_t offset_728;\n    \n    if (thread_active_1129) {\n        y_727 = sizze_286 * group_id_362;\n        offset_728 = local_tid_361 + y_727;\n    }\n    \n    int32_t chunk_sizze_729;\n    int32_t remaining_elements_1130 = squot32(sizze_286 - local_tid_361 +\n                                              group_sizze_1032 - 1,\n                                              group_sizze_1032);\n    \n    if (slt32(elements_per_thread_719, remaining_elements_1130)) {\n        chunk_sizze_729 = elements_per_thread_719;\n    } else {\n        chunk_sizze_729 = remaining_elements_1130;\n    }\n    if (thread_active_1129) { }\n    \n    int32_t res_733;\n    int32_t final_result_743;\n    int32_t acc_736 = 0;\n    int32_t groupstream_mapaccum_dummy_chunk_sizze_734 = 1;\n    \n    if (thread_active_1129) {\n        for (int32_t i_735 = 0; i_735 < chunk_sizze_729; i_735++) {\n            int32_t binop_param_y_738;\n            int32_t res_740;\n            \n            binop_param_y_738 = *(__global\n                                  int32_t *) &xss_mem_1048[(offset_728 + i_735 *\n                                                            group_sizze_1032) *\n                                                           4];\n            res_740 = acc_736 + binop_param_y_738;\n            acc_736 = res_740;\n        }\n    }\n    res_733 = acc_736;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (slt32(local_tid_361, group_sizze_1032) && 1) {\n        *(__local int32_t *) &mem_1067[local_tid_361 * 4] = res_733;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_1131;\n    int32_t my_index_744;\n    int32_t other_offset_745;\n    int32_t binop_param_x_746;\n    int32_t binop_param_y_747;\n    \n    my_index_744 = local_tid_361;\n    other_offset_745 = 0;\n    binop_param_x_746 = *(__local int32_t *) &mem_1067[(local_tid_361 +\n                                                        other_offset_745) * 4];\n    other_offset_745 = 1;\n    while (slt32(other_offset_745, wave_sizze_1127)) {\n        if (slt32(local_tid_361 + other_offset_745, group_sizze_1032) &&\n            ((local_tid_361 - squot32(local_tid_361, wave_sizze_1127) *\n              wave_sizze_1127) & (2 * other_offset_745 - 1)) == 0) {\n            // read array element\n            {\n                binop_param_y_747 = *(volatile __local\n                                      int32_t *) &mem_1067[(local_tid_361 +\n                                                            other_offset_745) *\n                                                           4];\n            }\n            \n            int32_t res_748;\n            \n            if (thread_active_1129) {\n                res_748 = binop_param_x_746 + binop_param_y_747;\n            }\n            binop_param_x_746 = res_748;\n            *(volatile __local int32_t *) &mem_1067[local_tid_361 * 4] =\n                binop_param_x_746;\n        }\n        other_offset_745 *= 2;\n    }\n    skip_waves_1131 = 1;\n    while (slt32(skip_waves_1131, squot32(group_sizze_1128 + wave_sizze_1127 -\n                                          1, wave_sizze_1127))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_745 = skip_waves_1131 * wave_sizze_1127;\n        if ((local_tid_361 - squot32(local_tid_361, wave_sizze_1127) *\n             wave_sizze_1127) == 0 && (squot32(local_tid_361, wave_sizze_1127) &\n                                       (2 * skip_waves_1131 - 1)) == 0) {\n            // read array element\n            {\n                binop_param_y_747 = *(__local\n                                      int32_t *) &mem_1067[(local_tid_361 +\n                                                            other_offset_745) *\n                                                           4];\n            }\n            \n            int32_t res_748;\n            \n            if (thread_active_1129) {\n                res_748 = binop_param_x_746 + binop_param_y_747;\n            }\n            binop_param_x_746 = res_748;\n            *(__local int32_t *) &mem_1067[local_tid_361 * 4] =\n                binop_param_x_746;\n        }\n        skip_waves_1131 *= 2;\n    }\n    final_result_743 = binop_param_x_746;\n    if (local_tid_361 == 0) {\n        *(__global int32_t *) &mem_1070[group_id_362 * 4] = final_result_743;\n    }\n}\n__kernel void segmented_redomap__large_comm_one_kernel_468(__local volatile\n                                                           int64_t *mem_aligned_0,\n                                                           int32_t sizze_285,\n                                                           int32_t num_groups_per_segment_760,\n                                                           int32_t elements_per_thread_804,\n                                                           __global\n                                                           unsigned char *mem_1076,\n                                                           __global\n                                                           unsigned char *mem_1082)\n{\n    __local volatile char *restrict mem_1079 = mem_aligned_0;\n    int32_t wave_sizze_1137;\n    int32_t group_sizze_1138;\n    char thread_active_1139;\n    int32_t gtid_305;\n    int32_t gtid_467;\n    int32_t global_tid_468;\n    int32_t local_tid_469;\n    int32_t group_id_470;\n    \n    global_tid_468 = get_global_id(0);\n    local_tid_469 = get_local_id(0);\n    group_sizze_1138 = get_local_size(0);\n    wave_sizze_1137 = LOCKSTEP_WIDTH;\n    group_id_470 = get_group_id(0);\n    gtid_305 = group_id_470;\n    gtid_467 = group_id_470 - group_id_470;\n    thread_active_1139 = slt32(gtid_305, sizze_285) && slt32(gtid_467, 1);\n    \n    int32_t y_812;\n    int32_t offset_813;\n    \n    if (thread_active_1139) {\n        y_812 = num_groups_per_segment_760 * group_id_470;\n        offset_813 = local_tid_469 + y_812;\n    }\n    \n    int32_t chunk_sizze_814;\n    int32_t remaining_elements_1140 = squot32(num_groups_per_segment_760 -\n                                              local_tid_469 + group_sizze_1032 -\n                                              1, group_sizze_1032);\n    \n    if (slt32(elements_per_thread_804, remaining_elements_1140)) {\n        chunk_sizze_814 = elements_per_thread_804;\n    } else {\n        chunk_sizze_814 = remaining_elements_1140;\n    }\n    if (thread_active_1139) { }\n    \n    int32_t res_818;\n    int32_t final_result_828;\n    int32_t acc_821 = 0;\n    int32_t groupstream_mapaccum_dummy_chunk_sizze_819 = 1;\n    \n    if (thread_active_1139) {\n        for (int32_t i_820 = 0; i_820 < chunk_sizze_814; i_820++) {\n            int32_t binop_param_y_823;\n            int32_t res_825;\n            \n            binop_param_y_823 = *(__global int32_t *) &mem_1076[(offset_813 +\n                                                                 i_820 *\n                                                                 group_sizze_1032) *\n                                                                4];\n            res_825 = acc_821 + binop_param_y_823;\n            acc_821 = res_825;\n        }\n    }\n    res_818 = acc_821;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (slt32(local_tid_469, group_sizze_1032) && 1) {\n        *(__local int32_t *) &mem_1079[local_tid_469 * 4] = res_818;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_1141;\n    int32_t my_index_829;\n    int32_t other_offset_830;\n    int32_t binop_param_x_831;\n    int32_t binop_param_y_832;\n    \n    my_index_829 = local_tid_469;\n    other_offset_830 = 0;\n    binop_param_x_831 = *(__local int32_t *) &mem_1079[(local_tid_469 +\n                                                        other_offset_830) * 4];\n    other_offset_830 = 1;\n    while (slt32(other_offset_830, wave_sizze_1137)) {\n        if (slt32(local_tid_469 + other_offset_830, group_sizze_1032) &&\n            ((local_tid_469 - squot32(local_tid_469, wave_sizze_1137) *\n              wave_sizze_1137) & (2 * other_offset_830 - 1)) == 0) {\n            // read array element\n            {\n                binop_param_y_832 = *(volatile __local\n                                      int32_t *) &mem_1079[(local_tid_469 +\n                                                            other_offset_830) *\n                                                           4];\n            }\n            \n            int32_t res_833;\n            \n            if (thread_active_1139) {\n                res_833 = binop_param_x_831 + binop_param_y_832;\n            }\n            binop_param_x_831 = res_833;\n            *(volatile __local int32_t *) &mem_1079[local_tid_469 * 4] =\n                binop_param_x_831;\n        }\n        other_offset_830 *= 2;\n    }\n    skip_waves_1141 = 1;\n    while (slt32(skip_waves_1141, squot32(group_sizze_1138 + wave_sizze_1137 -\n                                          1, wave_sizze_1137))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_830 = skip_waves_1141 * wave_sizze_1137;\n        if ((local_tid_469 - squot32(local_tid_469, wave_sizze_1137) *\n             wave_sizze_1137) == 0 && (squot32(local_tid_469, wave_sizze_1137) &\n                                       (2 * skip_waves_1141 - 1)) == 0) {\n            // read array element\n            {\n                binop_param_y_832 = *(__local\n                                      int32_t *) &mem_1079[(local_tid_469 +\n                                                            other_offset_830) *\n                                                           4];\n            }\n            \n            int32_t res_833;\n            \n            if (thread_active_1139) {\n                res_833 = binop_param_x_831 + binop_param_y_832;\n            }\n            binop_param_x_831 = res_833;\n            *(__local int32_t *) &mem_1079[local_tid_469 * 4] =\n                binop_param_x_831;\n        }\n        skip_waves_1141 *= 2;\n    }\n    final_result_828 = binop_param_x_831;\n    if (local_tid_469 == 0) {\n        *(__global int32_t *) &mem_1082[group_id_470 * 4] = final_result_828;\n    }\n}\n__kernel void segmented_redomap__small_comm_kernel_498(__local volatile\n                                                       int64_t *mem_aligned_0,\n                                                       __local volatile\n                                                       int64_t *mem_aligned_1,\n                                                       int32_t sizze_285,\n                                                       int32_t num_groups_per_segment_760,\n                                                       int32_t num_segments_per_group_837,\n                                                       int32_t active_threads_per_group_842,\n                                                       int32_t active_threads_last_group_847,\n                                                       int32_t y_849, __global\n                                                       unsigned char *mem_1076,\n                                                       __global\n                                                       unsigned char *mem_1085)\n{\n    __local volatile char *restrict mem_1087 = mem_aligned_0;\n    __local volatile char *restrict mem_1090 = mem_aligned_1;\n    int32_t wave_sizze_1142;\n    int32_t group_sizze_1143;\n    char thread_active_1144;\n    int32_t global_tid_498;\n    int32_t local_tid_499;\n    int32_t group_id_500;\n    \n    global_tid_498 = get_global_id(0);\n    local_tid_499 = get_local_id(0);\n    group_sizze_1143 = get_local_size(0);\n    wave_sizze_1142 = LOCKSTEP_WIDTH;\n    group_id_500 = get_group_id(0);\n    thread_active_1144 = 1;\n    \n    char islastgroup_850;\n    int32_t active_thread_this_group_851;\n    char isactive_852;\n    int32_t redtmp_res_854;\n    int32_t x_868;\n    char isfirstinsegment_869;\n    \n    if (thread_active_1144) {\n        islastgroup_850 = group_id_500 == y_849;\n        if (islastgroup_850) {\n            active_thread_this_group_851 = active_threads_last_group_847;\n        } else {\n            active_thread_this_group_851 = active_threads_per_group_842;\n        }\n        isactive_852 = slt32(local_tid_499, active_thread_this_group_851);\n        if (isactive_852) {\n            int32_t x_855;\n            int32_t y_856;\n            int32_t segment_index_857;\n            int32_t index_within_segment_858;\n            int32_t y_859;\n            int32_t offset_860;\n            int32_t binop_param_y_864;\n            \n            x_855 = squot32(local_tid_499, num_groups_per_segment_760);\n            y_856 = group_id_500 * num_segments_per_group_837;\n            segment_index_857 = x_855 + y_856;\n            index_within_segment_858 = srem32(local_tid_499,\n                                              num_groups_per_segment_760);\n            y_859 = num_groups_per_segment_760 * segment_index_857;\n            offset_860 = index_within_segment_858 + y_859;\n            binop_param_y_864 = *(__global int32_t *) &mem_1076[offset_860 * 4];\n            redtmp_res_854 = binop_param_y_864;\n        } else {\n            redtmp_res_854 = 0;\n        }\n        x_868 = srem32(local_tid_499, num_groups_per_segment_760);\n        isfirstinsegment_869 = x_868 == 0;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (slt32(local_tid_499, group_sizze_1032) && 1) {\n        *(__local char *) &mem_1087[local_tid_499] = isfirstinsegment_869;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (slt32(local_tid_499, group_sizze_1032) && 1) {\n        *(__local int32_t *) &mem_1090[local_tid_499 * 4] = redtmp_res_854;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t my_index_874;\n    int32_t other_offset_875;\n    char x_flag_876;\n    int32_t binop_param_x_877;\n    char y_flag_878;\n    int32_t binop_param_y_879;\n    int32_t my_index_1145;\n    int32_t other_offset_1146;\n    char x_flag_1147;\n    int32_t binop_param_x_1148;\n    char y_flag_1149;\n    int32_t binop_param_y_1150;\n    \n    my_index_874 = local_tid_499;\n    if (slt32(local_tid_499, group_sizze_1032)) {\n        y_flag_878 = *(volatile __local char *) &mem_1087[local_tid_499 *\n                                                          sizeof(char)];\n        binop_param_y_879 = *(volatile __local\n                              int32_t *) &mem_1090[local_tid_499 *\n                                                   sizeof(int32_t)];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        int32_t skip_threads_1154 = 1;\n        \n        while (slt32(skip_threads_1154, 32)) {\n            if (slt32(local_tid_499, group_sizze_1032) &&\n                sle32(skip_threads_1154, local_tid_499 - squot32(local_tid_499,\n                                                                 32) * 32)) {\n                // read operands\n                {\n                    x_flag_876 = *(volatile __local\n                                   char *) &mem_1087[(local_tid_499 -\n                                                      skip_threads_1154) *\n                                                     sizeof(char)];\n                    binop_param_x_877 = *(volatile __local\n                                          int32_t *) &mem_1090[(local_tid_499 -\n                                                                skip_threads_1154) *\n                                                               sizeof(int32_t)];\n                }\n                // perform operation\n                {\n                    char new_flag_880;\n                    int32_t seg_lhs_881;\n                    int32_t res_884;\n                    \n                    if (thread_active_1144) {\n                        new_flag_880 = x_flag_876 || y_flag_878;\n                        if (y_flag_878) {\n                            seg_lhs_881 = 0;\n                        } else {\n                            seg_lhs_881 = binop_param_x_877;\n                        }\n                        res_884 = seg_lhs_881 + binop_param_y_879;\n                    }\n                    y_flag_878 = new_flag_880;\n                    binop_param_y_879 = res_884;\n                }\n            }\n            if (sle32(wave_sizze_1142, skip_threads_1154)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (slt32(local_tid_499, group_sizze_1032) &&\n                sle32(skip_threads_1154, local_tid_499 - squot32(local_tid_499,\n                                                                 32) * 32)) {\n                // write result\n                {\n                    *(volatile __local char *) &mem_1087[local_tid_499 *\n                                                         sizeof(char)] =\n                        y_flag_878;\n                    *(volatile __local int32_t *) &mem_1090[local_tid_499 *\n                                                            sizeof(int32_t)] =\n                        binop_param_y_879;\n                }\n            }\n            if (sle32(wave_sizze_1142, skip_threads_1154)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_1154 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_499 - squot32(local_tid_499, 32) * 32) == 31 &&\n            slt32(local_tid_499, group_sizze_1032)) {\n            *(volatile __local char *) &mem_1087[squot32(local_tid_499, 32) *\n                                                 sizeof(char)] = y_flag_878;\n            *(volatile __local int32_t *) &mem_1090[squot32(local_tid_499, 32) *\n                                                    sizeof(int32_t)] =\n                binop_param_y_879;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        if (squot32(local_tid_499, 32) == 0 && slt32(local_tid_499,\n                                                     group_sizze_1032)) {\n            y_flag_1149 = *(volatile __local char *) &mem_1087[local_tid_499 *\n                                                               sizeof(char)];\n            binop_param_y_1150 = *(volatile __local\n                                   int32_t *) &mem_1090[local_tid_499 *\n                                                        sizeof(int32_t)];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            int32_t skip_threads_1155 = 1;\n            \n            while (slt32(skip_threads_1155, 32)) {\n                if ((squot32(local_tid_499, 32) == 0 && slt32(local_tid_499,\n                                                              group_sizze_1032)) &&\n                    sle32(skip_threads_1155, local_tid_499 -\n                          squot32(local_tid_499, 32) * 32)) {\n                    // read operands\n                    {\n                        x_flag_1147 = *(volatile __local\n                                        char *) &mem_1087[(local_tid_499 -\n                                                           skip_threads_1155) *\n                                                          sizeof(char)];\n                        binop_param_x_1148 = *(volatile __local\n                                               int32_t *) &mem_1090[(local_tid_499 -\n                                                                     skip_threads_1155) *\n                                                                    sizeof(int32_t)];\n                    }\n                    // perform operation\n                    {\n                        char new_flag_1151;\n                        int32_t seg_lhs_1152;\n                        int32_t res_1153;\n                        \n                        if (thread_active_1144) {\n                            new_flag_1151 = x_flag_1147 || y_flag_1149;\n                            if (y_flag_1149) {\n                                seg_lhs_1152 = 0;\n                            } else {\n                                seg_lhs_1152 = binop_param_x_1148;\n                            }\n                            res_1153 = seg_lhs_1152 + binop_param_y_1150;\n                        }\n                        y_flag_1149 = new_flag_1151;\n                        binop_param_y_1150 = res_1153;\n                    }\n                }\n                if (sle32(wave_sizze_1142, skip_threads_1155)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if ((squot32(local_tid_499, 32) == 0 && slt32(local_tid_499,\n                                                              group_sizze_1032)) &&\n                    sle32(skip_threads_1155, local_tid_499 -\n                          squot32(local_tid_499, 32) * 32)) {\n                    // write result\n                    {\n                        *(volatile __local char *) &mem_1087[local_tid_499 *\n                                                             sizeof(char)] =\n                            y_flag_1149;\n                        *(volatile __local int32_t *) &mem_1090[local_tid_499 *\n                                                                sizeof(int32_t)] =\n                            binop_param_y_1150;\n                    }\n                }\n                if (sle32(wave_sizze_1142, skip_threads_1155)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_1155 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_499, 32) == 0 || !slt32(local_tid_499,\n                                                        group_sizze_1032))) {\n            // read operands\n            {\n                x_flag_876 = *(volatile __local\n                               char *) &mem_1087[(squot32(local_tid_499, 32) -\n                                                  1) * sizeof(char)];\n                binop_param_x_877 = *(volatile __local\n                                      int32_t *) &mem_1090[(squot32(local_tid_499,\n                                                                    32) - 1) *\n                                                           sizeof(int32_t)];\n            }\n            // perform operation\n            {\n                char new_flag_880;\n                int32_t seg_lhs_881;\n                int32_t res_884;\n                \n                if (thread_active_1144) {\n                    new_flag_880 = x_flag_876 || y_flag_878;\n                    if (y_flag_878) {\n                        seg_lhs_881 = 0;\n                    } else {\n                        seg_lhs_881 = binop_param_x_877;\n                    }\n                    res_884 = seg_lhs_881 + binop_param_y_879;\n                }\n                y_flag_878 = new_flag_880;\n                binop_param_y_879 = res_884;\n            }\n            // write final result\n            {\n                *(volatile __local char *) &mem_1087[local_tid_499 *\n                                                     sizeof(char)] = y_flag_878;\n                *(volatile __local int32_t *) &mem_1090[local_tid_499 *\n                                                        sizeof(int32_t)] =\n                    binop_param_y_879;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_499, 32) == 0) {\n            *(volatile __local char *) &mem_1087[local_tid_499 * sizeof(char)] =\n                y_flag_878;\n            *(volatile __local int32_t *) &mem_1090[local_tid_499 *\n                                                    sizeof(int32_t)] =\n                binop_param_y_879;\n        }\n    }\n    \n    int32_t redoffset_885;\n    int32_t red_res_886;\n    \n    if (thread_active_1144) {\n        if (isactive_852) {\n            int32_t x_887;\n            int32_t y_888;\n            int32_t segment_index_889;\n            int32_t y_891;\n            char islastinseg_892;\n            int32_t redoffset_893;\n            int32_t red_return_elem_894;\n            \n            x_887 = squot32(local_tid_499, num_groups_per_segment_760);\n            y_888 = group_id_500 * num_segments_per_group_837;\n            segment_index_889 = x_887 + y_888;\n            y_891 = num_groups_per_segment_760 - 1;\n            islastinseg_892 = x_868 == y_891;\n            if (islastinseg_892) {\n                redoffset_893 = segment_index_889;\n            } else {\n                redoffset_893 = -1;\n            }\n            if (islastinseg_892) {\n                int32_t x_895 = *(__local int32_t *) &mem_1090[local_tid_499 *\n                                                               4];\n                \n                red_return_elem_894 = x_895;\n            } else {\n                red_return_elem_894 = 0;\n            }\n            redoffset_885 = redoffset_893;\n            red_res_886 = red_return_elem_894;\n        } else {\n            redoffset_885 = -1;\n            red_res_886 = 0;\n        }\n    }\n    if (thread_active_1144 && (sle32(0, redoffset_885) && slt32(redoffset_885,\n                                                                sizze_285))) {\n        *(__global int32_t *) &mem_1085[redoffset_885 * 4] = red_res_886;\n    }\n}\n__kernel void segmented_redomap__small_comm_kernel_651(__local volatile\n                                                       int64_t *mem_aligned_0,\n                                                       __local volatile\n                                                       int64_t *mem_aligned_1,\n                                                       int32_t sizze_285,\n                                                       int32_t sizze_286,\n                                                       int32_t num_segments_per_group_899,\n                                                       int32_t active_threads_per_group_904,\n                                                       int32_t active_threads_last_group_909,\n                                                       int32_t y_911, __global\n                                                       unsigned char *xss_mem_1048,\n                                                       __global\n                                                       unsigned char *mem_1097)\n{\n    __local volatile char *restrict mem_1099 = mem_aligned_0;\n    __local volatile char *restrict mem_1102 = mem_aligned_1;\n    int32_t wave_sizze_1156;\n    int32_t group_sizze_1157;\n    char thread_active_1158;\n    int32_t global_tid_651;\n    int32_t local_tid_652;\n    int32_t group_id_653;\n    \n    global_tid_651 = get_global_id(0);\n    local_tid_652 = get_local_id(0);\n    group_sizze_1157 = get_local_size(0);\n    wave_sizze_1156 = LOCKSTEP_WIDTH;\n    group_id_653 = get_group_id(0);\n    thread_active_1158 = 1;\n    \n    char islastgroup_912;\n    int32_t active_thread_this_group_913;\n    char isactive_914;\n    int32_t redtmp_res_916;\n    int32_t x_930;\n    char isfirstinsegment_931;\n    \n    if (thread_active_1158) {\n        islastgroup_912 = group_id_653 == y_911;\n        if (islastgroup_912) {\n            active_thread_this_group_913 = active_threads_last_group_909;\n        } else {\n            active_thread_this_group_913 = active_threads_per_group_904;\n        }\n        isactive_914 = slt32(local_tid_652, active_thread_this_group_913);\n        if (isactive_914) {\n            int32_t x_917;\n            int32_t y_918;\n            int32_t segment_index_919;\n            int32_t index_within_segment_920;\n            int32_t y_921;\n            int32_t offset_922;\n            int32_t new_index_1037;\n            int32_t binop_y_1039;\n            int32_t new_index_1040;\n            int32_t binop_param_y_926;\n            \n            x_917 = squot32(local_tid_652, sizze_286);\n            y_918 = group_id_653 * num_segments_per_group_899;\n            segment_index_919 = x_917 + y_918;\n            index_within_segment_920 = srem32(local_tid_652, sizze_286);\n            y_921 = sizze_286 * segment_index_919;\n            offset_922 = index_within_segment_920 + y_921;\n            new_index_1037 = squot32(offset_922, sizze_286);\n            binop_y_1039 = new_index_1037 * sizze_286;\n            new_index_1040 = offset_922 - binop_y_1039;\n            binop_param_y_926 = *(__global\n                                  int32_t *) &xss_mem_1048[(new_index_1037 *\n                                                            sizze_286 +\n                                                            new_index_1040) *\n                                                           4];\n            redtmp_res_916 = binop_param_y_926;\n        } else {\n            redtmp_res_916 = 0;\n        }\n        x_930 = srem32(local_tid_652, sizze_286);\n        isfirstinsegment_931 = x_930 == 0;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (slt32(local_tid_652, group_sizze_1032) && 1) {\n        *(__local char *) &mem_1099[local_tid_652] = isfirstinsegment_931;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (slt32(local_tid_652, group_sizze_1032) && 1) {\n        *(__local int32_t *) &mem_1102[local_tid_652 * 4] = redtmp_res_916;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t my_index_936;\n    int32_t other_offset_937;\n    char x_flag_938;\n    int32_t binop_param_x_939;\n    char y_flag_940;\n    int32_t binop_param_y_941;\n    int32_t my_index_1159;\n    int32_t other_offset_1160;\n    char x_flag_1161;\n    int32_t binop_param_x_1162;\n    char y_flag_1163;\n    int32_t binop_param_y_1164;\n    \n    my_index_936 = local_tid_652;\n    if (slt32(local_tid_652, group_sizze_1032)) {\n        y_flag_940 = *(volatile __local char *) &mem_1099[local_tid_652 *\n                                                          sizeof(char)];\n        binop_param_y_941 = *(volatile __local\n                              int32_t *) &mem_1102[local_tid_652 *\n                                                   sizeof(int32_t)];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        int32_t skip_threads_1168 = 1;\n        \n        while (slt32(skip_threads_1168, 32)) {\n            if (slt32(local_tid_652, group_sizze_1032) &&\n                sle32(skip_threads_1168, local_tid_652 - squot32(local_tid_652,\n                                                                 32) * 32)) {\n                // read operands\n                {\n                    x_flag_938 = *(volatile __local\n                                   char *) &mem_1099[(local_tid_652 -\n                                                      skip_threads_1168) *\n                                                     sizeof(char)];\n                    binop_param_x_939 = *(volatile __local\n                                          int32_t *) &mem_1102[(local_tid_652 -\n                                                                skip_threads_1168) *\n                                                               sizeof(int32_t)];\n                }\n                // perform operation\n                {\n                    char new_flag_942;\n                    int32_t seg_lhs_943;\n                    int32_t res_946;\n                    \n                    if (thread_active_1158) {\n                        new_flag_942 = x_flag_938 || y_flag_940;\n                        if (y_flag_940) {\n                            seg_lhs_943 = 0;\n                        } else {\n                            seg_lhs_943 = binop_param_x_939;\n                        }\n                        res_946 = seg_lhs_943 + binop_param_y_941;\n                    }\n                    y_flag_940 = new_flag_942;\n                    binop_param_y_941 = res_946;\n                }\n            }\n            if (sle32(wave_sizze_1156, skip_threads_1168)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (slt32(local_tid_652, group_sizze_1032) &&\n                sle32(skip_threads_1168, local_tid_652 - squot32(local_tid_652,\n                                                                 32) * 32)) {\n                // write result\n                {\n                    *(volatile __local char *) &mem_1099[local_tid_652 *\n                                                         sizeof(char)] =\n                        y_flag_940;\n                    *(volatile __local int32_t *) &mem_1102[local_tid_652 *\n                                                            sizeof(int32_t)] =\n                        binop_param_y_941;\n                }\n            }\n            if (sle32(wave_sizze_1156, skip_threads_1168)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_1168 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_652 - squot32(local_tid_652, 32) * 32) == 31 &&\n            slt32(local_tid_652, group_sizze_1032)) {\n            *(volatile __local char *) &mem_1099[squot32(local_tid_652, 32) *\n                                                 sizeof(char)] = y_flag_940;\n            *(volatile __local int32_t *) &mem_1102[squot32(local_tid_652, 32) *\n                                                    sizeof(int32_t)] =\n                binop_param_y_941;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        if (squot32(local_tid_652, 32) == 0 && slt32(local_tid_652,\n                                                     group_sizze_1032)) {\n            y_flag_1163 = *(volatile __local char *) &mem_1099[local_tid_652 *\n                                                               sizeof(char)];\n            binop_param_y_1164 = *(volatile __local\n                                   int32_t *) &mem_1102[local_tid_652 *\n                                                        sizeof(int32_t)];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            int32_t skip_threads_1169 = 1;\n            \n            while (slt32(skip_threads_1169, 32)) {\n                if ((squot32(local_tid_652, 32) == 0 && slt32(local_tid_652,\n                                                              group_sizze_1032)) &&\n                    sle32(skip_threads_1169, local_tid_652 -\n                          squot32(local_tid_652, 32) * 32)) {\n                    // read operands\n                    {\n                        x_flag_1161 = *(volatile __local\n                                        char *) &mem_1099[(local_tid_652 -\n                                                           skip_threads_1169) *\n                                                          sizeof(char)];\n                        binop_param_x_1162 = *(volatile __local\n                                               int32_t *) &mem_1102[(local_tid_652 -\n                                                                     skip_threads_1169) *\n                                                                    sizeof(int32_t)];\n                    }\n                    // perform operation\n                    {\n                        char new_flag_1165;\n                        int32_t seg_lhs_1166;\n                        int32_t res_1167;\n                        \n                        if (thread_active_1158) {\n                            new_flag_1165 = x_flag_1161 || y_flag_1163;\n                            if (y_flag_1163) {\n                                seg_lhs_1166 = 0;\n                            } else {\n                                seg_lhs_1166 = binop_param_x_1162;\n                            }\n                            res_1167 = seg_lhs_1166 + binop_param_y_1164;\n                        }\n                        y_flag_1163 = new_flag_1165;\n                        binop_param_y_1164 = res_1167;\n                    }\n                }\n                if (sle32(wave_sizze_1156, skip_threads_1169)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if ((squot32(local_tid_652, 32) == 0 && slt32(local_tid_652,\n                                                              group_sizze_1032)) &&\n                    sle32(skip_threads_1169, local_tid_652 -\n                          squot32(local_tid_652, 32) * 32)) {\n                    // write result\n                    {\n                        *(volatile __local char *) &mem_1099[local_tid_652 *\n                                                             sizeof(char)] =\n                            y_flag_1163;\n                        *(volatile __local int32_t *) &mem_1102[local_tid_652 *\n                                                                sizeof(int32_t)] =\n                            binop_param_y_1164;\n                    }\n                }\n                if (sle32(wave_sizze_1156, skip_threads_1169)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_1169 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_652, 32) == 0 || !slt32(local_tid_652,\n                                                        group_sizze_1032))) {\n            // read operands\n            {\n                x_flag_938 = *(volatile __local\n                               char *) &mem_1099[(squot32(local_tid_652, 32) -\n                                                  1) * sizeof(char)];\n                binop_param_x_939 = *(volatile __local\n                                      int32_t *) &mem_1102[(squot32(local_tid_652,\n                                                                    32) - 1) *\n                                                           sizeof(int32_t)];\n            }\n            // perform operation\n            {\n                char new_flag_942;\n                int32_t seg_lhs_943;\n                int32_t res_946;\n                \n                if (thread_active_1158) {\n                    new_flag_942 = x_flag_938 || y_flag_940;\n                    if (y_flag_940) {\n                        seg_lhs_943 = 0;\n                    } else {\n                        seg_lhs_943 = binop_param_x_939;\n                    }\n                    res_946 = seg_lhs_943 + binop_param_y_941;\n                }\n                y_flag_940 = new_flag_942;\n                binop_param_y_941 = res_946;\n            }\n            // write final result\n            {\n                *(volatile __local char *) &mem_1099[local_tid_652 *\n                                                     sizeof(char)] = y_flag_940;\n                *(volatile __local int32_t *) &mem_1102[local_tid_652 *\n                                                        sizeof(int32_t)] =\n                    binop_param_y_941;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_652, 32) == 0) {\n            *(volatile __local char *) &mem_1099[local_tid_652 * sizeof(char)] =\n                y_flag_940;\n            *(volatile __local int32_t *) &mem_1102[local_tid_652 *\n                                                    sizeof(int32_t)] =\n                binop_param_y_941;\n        }\n    }\n    \n    int32_t redoffset_947;\n    int32_t red_res_948;\n    \n    if (thread_active_1158) {\n        if (isactive_914) {\n            int32_t x_949;\n            int32_t y_950;\n            int32_t segment_index_951;\n            int32_t y_953;\n            char islastinseg_954;\n            int32_t redoffset_955;\n            int32_t red_return_elem_956;\n            \n            x_949 = squot32(local_tid_652, sizze_286);\n            y_950 = group_id_653 * num_segments_per_group_899;\n            segment_index_951 = x_949 + y_950;\n            y_953 = sizze_286 - 1;\n            islastinseg_954 = x_930 == y_953;\n            if (islastinseg_954) {\n                redoffset_955 = segment_index_951;\n            } else {\n                redoffset_955 = -1;\n            }\n            if (islastinseg_954) {\n                int32_t x_957 = *(__local int32_t *) &mem_1102[local_tid_652 *\n                                                               4];\n                \n                red_return_elem_956 = x_957;\n            } else {\n                red_return_elem_956 = 0;\n            }\n            redoffset_947 = redoffset_955;\n            red_res_948 = red_return_elem_956;\n        } else {\n            redoffset_947 = -1;\n            red_res_948 = 0;\n        }\n    }\n    if (thread_active_1158 && (sle32(0, redoffset_947) && slt32(redoffset_947,\n                                                                sizze_285))) {\n        *(__global int32_t *) &mem_1097[redoffset_947 * 4] = red_res_948;\n    }\n}\n";
static cl_kernel fut_kernel_map_transpose_i32;
static int fut_kernel_map_transpose_i32total_runtime = 0;
static int fut_kernel_map_transpose_i32runs = 0;
static cl_kernel fut_kernel_map_transpose_lowheight_i32;
static int fut_kernel_map_transpose_lowheight_i32total_runtime = 0;
static int fut_kernel_map_transpose_lowheight_i32runs = 0;
static cl_kernel fut_kernel_map_transpose_lowwidth_i32;
static int fut_kernel_map_transpose_lowwidth_i32total_runtime = 0;
static int fut_kernel_map_transpose_lowwidth_i32runs = 0;
static cl_kernel map_intra_group_kernel_1008;
static int map_intra_group_kernel_1008total_runtime = 0;
static int map_intra_group_kernel_1008runs = 0;
static cl_kernel map_kernel_971;
static int map_kernel_971total_runtime = 0;
static int map_kernel_971runs = 0;
static cl_kernel segmented_redomap__large_comm_many_kernel_418;
static int segmented_redomap__large_comm_many_kernel_418total_runtime = 0;
static int segmented_redomap__large_comm_many_kernel_418runs = 0;
static cl_kernel segmented_redomap__large_comm_one_kernel_360;
static int segmented_redomap__large_comm_one_kernel_360total_runtime = 0;
static int segmented_redomap__large_comm_one_kernel_360runs = 0;
static cl_kernel segmented_redomap__large_comm_one_kernel_468;
static int segmented_redomap__large_comm_one_kernel_468total_runtime = 0;
static int segmented_redomap__large_comm_one_kernel_468runs = 0;
static cl_kernel segmented_redomap__small_comm_kernel_498;
static int segmented_redomap__small_comm_kernel_498total_runtime = 0;
static int segmented_redomap__small_comm_kernel_498runs = 0;
static cl_kernel segmented_redomap__small_comm_kernel_651;
static int segmented_redomap__small_comm_kernel_651total_runtime = 0;
static int segmented_redomap__small_comm_kernel_651runs = 0;
void setup_opencl_and_load_kernels()

{
    cl_int error;
    cl_program prog = setup_opencl(fut_opencl_prelude, fut_opencl_program);
    
    {
        fut_kernel_map_transpose_i32 = clCreateKernel(prog,
                                                      "fut_kernel_map_transpose_i32",
                                                      &error);
        assert(error == 0);
        if (debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_i32");
    }
    {
        fut_kernel_map_transpose_lowheight_i32 = clCreateKernel(prog,
                                                                "fut_kernel_map_transpose_lowheight_i32",
                                                                &error);
        assert(error == 0);
        if (debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_lowheight_i32");
    }
    {
        fut_kernel_map_transpose_lowwidth_i32 = clCreateKernel(prog,
                                                               "fut_kernel_map_transpose_lowwidth_i32",
                                                               &error);
        assert(error == 0);
        if (debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_lowwidth_i32");
    }
    {
        map_intra_group_kernel_1008 = clCreateKernel(prog,
                                                     "map_intra_group_kernel_1008",
                                                     &error);
        assert(error == 0);
        if (debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "map_intra_group_kernel_1008");
    }
    {
        map_kernel_971 = clCreateKernel(prog, "map_kernel_971", &error);
        assert(error == 0);
        if (debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_971");
    }
    {
        segmented_redomap__large_comm_many_kernel_418 = clCreateKernel(prog,
                                                                       "segmented_redomap__large_comm_many_kernel_418",
                                                                       &error);
        assert(error == 0);
        if (debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "segmented_redomap__large_comm_many_kernel_418");
    }
    {
        segmented_redomap__large_comm_one_kernel_360 = clCreateKernel(prog,
                                                                      "segmented_redomap__large_comm_one_kernel_360",
                                                                      &error);
        assert(error == 0);
        if (debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "segmented_redomap__large_comm_one_kernel_360");
    }
    {
        segmented_redomap__large_comm_one_kernel_468 = clCreateKernel(prog,
                                                                      "segmented_redomap__large_comm_one_kernel_468",
                                                                      &error);
        assert(error == 0);
        if (debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "segmented_redomap__large_comm_one_kernel_468");
    }
    {
        segmented_redomap__small_comm_kernel_498 = clCreateKernel(prog,
                                                                  "segmented_redomap__small_comm_kernel_498",
                                                                  &error);
        assert(error == 0);
        if (debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "segmented_redomap__small_comm_kernel_498");
    }
    {
        segmented_redomap__small_comm_kernel_651 = clCreateKernel(prog,
                                                                  "segmented_redomap__small_comm_kernel_651",
                                                                  &error);
        assert(error == 0);
        if (debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "segmented_redomap__small_comm_kernel_651");
    }
}
void post_opencl_setup(struct opencl_device_option *option)
{
    if (strcmp(option->platform_name, "NVIDIA CUDA") == 0 &&
        option->device_type == CL_DEVICE_TYPE_GPU) {
        cl_lockstep_width = 32;
        if (debugging)
            fprintf(stderr, "Setting lockstep width to: %d\n",
                    cl_lockstep_width);
    }
    if (strcmp(option->platform_name, "AMD Accelerated Parallel Processing") ==
        0 && option->device_type == CL_DEVICE_TYPE_GPU) {
        cl_lockstep_width = 64;
        if (debugging)
            fprintf(stderr, "Setting lockstep width to: %d\n",
                    cl_lockstep_width);
    }
}
int64_t peak_mem_usage_device = 0;
int64_t cur_mem_usage_device = 0;
struct memblock_device {
    int *references;
    cl_mem mem;
    int64_t size;
} ;
static void memblock_unref_device(struct memblock_device *block)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (detail_memory)
            fprintf(stderr,
                    "Unreferencing block in space 'device': %d references remaining.\n",
                    *block->references);
        if (*block->references == 0) {
            cur_mem_usage_device -= block->size;
            OPENCL_SUCCEED(clReleaseMemObject(block->mem));
            free(block->references);
            block->references = NULL;
            if (detail_memory)
                fprintf(stderr, "%ld bytes freed (now allocated: %ld bytes)\n",
                        block->size, cur_mem_usage_device);
        }
    }
}
static void memblock_alloc_device(struct memblock_device *block, int32_t size)
{
    memblock_unref_device(block);
    
    cl_int clCreateBuffer_succeeded_1228;
    
    block->mem = clCreateBuffer(fut_cl_context, CL_MEM_READ_WRITE, size >
                                0 ? size : 1, NULL,
                                &clCreateBuffer_succeeded_1228);
    OPENCL_SUCCEED(clCreateBuffer_succeeded_1228);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    cur_mem_usage_device += size;
    if (detail_memory)
        fprintf(stderr,
                "Allocated %d bytes in space 'device' (now allocated: %ld bytes)",
                size, cur_mem_usage_device);
    if (cur_mem_usage_device > peak_mem_usage_device) {
        peak_mem_usage_device = cur_mem_usage_device;
        if (detail_memory)
            fprintf(stderr, " (new peak).\n", peak_mem_usage_device);
    } else if (detail_memory)
        fprintf(stderr, ".\n");
}
static void memblock_set_device(struct memblock_device *lhs,
                                struct memblock_device *rhs)
{
    memblock_unref_device(lhs);
    (*rhs->references)++;
    *lhs = *rhs;
}
int64_t peak_mem_usage_local = 0;
int64_t cur_mem_usage_local = 0;
struct memblock_local {
    int *references;
    unsigned char mem;
    int64_t size;
} ;
static void memblock_unref_local(struct memblock_local *block)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (detail_memory)
            fprintf(stderr,
                    "Unreferencing block in space 'local': %d references remaining.\n",
                    *block->references);
        if (*block->references == 0) {
            cur_mem_usage_local -= block->size;
            free(block->references);
            block->references = NULL;
            if (detail_memory)
                fprintf(stderr, "%ld bytes freed (now allocated: %ld bytes)\n",
                        block->size, cur_mem_usage_local);
        }
    }
}
static void memblock_alloc_local(struct memblock_local *block, int32_t size)
{
    memblock_unref_local(block);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    cur_mem_usage_local += size;
    if (detail_memory)
        fprintf(stderr,
                "Allocated %d bytes in space 'local' (now allocated: %ld bytes)",
                size, cur_mem_usage_local);
    if (cur_mem_usage_local > peak_mem_usage_local) {
        peak_mem_usage_local = cur_mem_usage_local;
        if (detail_memory)
            fprintf(stderr, " (new peak).\n", peak_mem_usage_local);
    } else if (detail_memory)
        fprintf(stderr, ".\n");
}
static void memblock_set_local(struct memblock_local *lhs,
                               struct memblock_local *rhs)
{
    memblock_unref_local(lhs);
    (*rhs->references)++;
    *lhs = *rhs;
}
int64_t peak_mem_usage_default = 0;
int64_t cur_mem_usage_default = 0;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
} ;
static void memblock_unref(struct memblock *block)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (detail_memory)
            fprintf(stderr,
                    "Unreferencing block in default space: %d references remaining.\n",
                    *block->references);
        if (*block->references == 0) {
            cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            block->references = NULL;
            if (detail_memory)
                fprintf(stderr, "%ld bytes freed (now allocated: %ld bytes)\n",
                        block->size, cur_mem_usage_default);
        }
    }
}
static void memblock_alloc(struct memblock *block, int32_t size)
{
    memblock_unref(block);
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    cur_mem_usage_default += size;
    if (detail_memory)
        fprintf(stderr,
                "Allocated %d bytes in default space (now allocated: %ld bytes)",
                size, cur_mem_usage_default);
    if (cur_mem_usage_default > peak_mem_usage_default) {
        peak_mem_usage_default = cur_mem_usage_default;
        if (detail_memory)
            fprintf(stderr, " (new peak).\n", peak_mem_usage_default);
    } else if (detail_memory)
        fprintf(stderr, ".\n");
}
static void memblock_set(struct memblock *lhs, struct memblock *rhs)
{
    memblock_unref(lhs);
    (*rhs->references)++;
    *lhs = *rhs;
}
struct tuple_ { } ;
struct tuple_int32_t_device_mem_int32_t {
    int32_t elem_0;
    struct memblock_device elem_1;
    int32_t elem_2;
} ;
static struct tuple_
futhark_map_transpose_opencl_i32(struct memblock_device destmem_0,
                                 int32_t destoffset_1,
                                 struct memblock_device srcmem_2,
                                 int32_t srcoffset_3, int32_t num_arrays_4,
                                 int32_t x_elems_5, int32_t y_elems_6,
                                 int32_t in_elems_7, int32_t out_elems_8);
static struct tuple_int32_t_device_mem_int32_t
futhark_main(int64_t xss_mem_sizze_1047, struct memblock_device xss_mem_1048,
             int32_t sizze_285, int32_t sizze_286);
static inline float futhark_log32(float x)
{
    return log(x);
}
static inline float futhark_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futhark_exp32(float x)
{
    return exp(x);
}
static inline float futhark_cos32(float x)
{
    return cos(x);
}
static inline float futhark_sin32(float x)
{
    return sin(x);
}
static inline float futhark_acos32(float x)
{
    return acos(x);
}
static inline float futhark_asin32(float x)
{
    return asin(x);
}
static inline double futhark_atan32(float x)
{
    return atan(x);
}
static inline float futhark_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline char futhark_isnan32(float x)
{
    return isnan(x);
}
static inline char futhark_isinf32(float x)
{
    return isinf(x);
}
static inline double futhark_log64(double x)
{
    return log(x);
}
static inline double futhark_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futhark_exp64(double x)
{
    return exp(x);
}
static inline double futhark_cos64(double x)
{
    return cos(x);
}
static inline double futhark_sin64(double x)
{
    return sin(x);
}
static inline double futhark_acos64(double x)
{
    return acos(x);
}
static inline double futhark_asin64(double x)
{
    return asin(x);
}
static inline double futhark_atan64(double x)
{
    return atan(x);
}
static inline double futhark_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline char futhark_isnan64(double x)
{
    return isnan(x);
}
static inline char futhark_isinf64(double x)
{
    return isinf(x);
}
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int8_t sext_i8_i8(int8_t x)
{
    return x;
}
static inline int16_t sext_i8_i16(int8_t x)
{
    return x;
}
static inline int32_t sext_i8_i32(int8_t x)
{
    return x;
}
static inline int64_t sext_i8_i64(int8_t x)
{
    return x;
}
static inline int8_t sext_i16_i8(int16_t x)
{
    return x;
}
static inline int16_t sext_i16_i16(int16_t x)
{
    return x;
}
static inline int32_t sext_i16_i32(int16_t x)
{
    return x;
}
static inline int64_t sext_i16_i64(int16_t x)
{
    return x;
}
static inline int8_t sext_i32_i8(int32_t x)
{
    return x;
}
static inline int16_t sext_i32_i16(int32_t x)
{
    return x;
}
static inline int32_t sext_i32_i32(int32_t x)
{
    return x;
}
static inline int64_t sext_i32_i64(int32_t x)
{
    return x;
}
static inline int8_t sext_i64_i8(int64_t x)
{
    return x;
}
static inline int16_t sext_i64_i16(int64_t x)
{
    return x;
}
static inline int32_t sext_i64_i32(int64_t x)
{
    return x;
}
static inline int64_t sext_i64_i64(int64_t x)
{
    return x;
}
static inline uint8_t zext_i8_i8(uint8_t x)
{
    return x;
}
static inline uint16_t zext_i8_i16(uint8_t x)
{
    return x;
}
static inline uint32_t zext_i8_i32(uint8_t x)
{
    return x;
}
static inline uint64_t zext_i8_i64(uint8_t x)
{
    return x;
}
static inline uint8_t zext_i16_i8(uint16_t x)
{
    return x;
}
static inline uint16_t zext_i16_i16(uint16_t x)
{
    return x;
}
static inline uint32_t zext_i16_i32(uint16_t x)
{
    return x;
}
static inline uint64_t zext_i16_i64(uint16_t x)
{
    return x;
}
static inline uint8_t zext_i32_i8(uint32_t x)
{
    return x;
}
static inline uint16_t zext_i32_i16(uint32_t x)
{
    return x;
}
static inline uint32_t zext_i32_i32(uint32_t x)
{
    return x;
}
static inline uint64_t zext_i32_i64(uint32_t x)
{
    return x;
}
static inline uint8_t zext_i64_i8(uint64_t x)
{
    return x;
}
static inline uint16_t zext_i64_i16(uint64_t x)
{
    return x;
}
static inline uint32_t zext_i64_i32(uint64_t x)
{
    return x;
}
static inline uint64_t zext_i64_i64(uint64_t x)
{
    return x;
}
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return x < y ? x : y;
}
static inline float fmax32(float x, float y)
{
    return x < y ? y : x;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return x < y ? x : y;
}
static inline double fmax64(double x, double y)
{
    return x < y ? y : x;
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static int detail_timing = 0;
static
struct tuple_ futhark_map_transpose_opencl_i32(struct memblock_device destmem_0,
                                               int32_t destoffset_1,
                                               struct memblock_device srcmem_2,
                                               int32_t srcoffset_3,
                                               int32_t num_arrays_4,
                                               int32_t x_elems_5,
                                               int32_t y_elems_6,
                                               int32_t in_elems_7,
                                               int32_t out_elems_8)
{
    if (in_elems_7 == out_elems_8 && ((num_arrays_4 == 1 || x_elems_5 *
                                       y_elems_6 == in_elems_7) && (x_elems_5 ==
                                                                    1 ||
                                                                    y_elems_6 ==
                                                                    1))) {
        if (in_elems_7 * sizeof(int32_t) > 0) {
            OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, srcmem_2.mem,
                                               destmem_0.mem, srcoffset_3,
                                               destoffset_1, in_elems_7 *
                                               sizeof(int32_t), 0, NULL, NULL));
            if (debugging)
                OPENCL_SUCCEED(clFinish(fut_cl_queue));
        }
    } else {
        if (sle32(x_elems_5, squot32(16, 2)) && slt32(16, y_elems_6)) {
            int32_t muly_9 = squot32(16, x_elems_5);
            int32_t new_height_10;
            
            new_height_10 = squot32(y_elems_6 + muly_9 - 1, muly_9);
            OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowwidth_i32,
                                          0, sizeof(destmem_0.mem),
                                          &destmem_0.mem));
            OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowwidth_i32,
                                          1, sizeof(destoffset_1),
                                          &destoffset_1));
            OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowwidth_i32,
                                          2, sizeof(srcmem_2.mem),
                                          &srcmem_2.mem));
            OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowwidth_i32,
                                          3, sizeof(srcoffset_3),
                                          &srcoffset_3));
            OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowwidth_i32,
                                          4, sizeof(x_elems_5), &x_elems_5));
            OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowwidth_i32,
                                          5, sizeof(y_elems_6), &y_elems_6));
            OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowwidth_i32,
                                          6, sizeof(in_elems_7), &in_elems_7));
            OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowwidth_i32,
                                          7, sizeof(out_elems_8),
                                          &out_elems_8));
            OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowwidth_i32,
                                          8, sizeof(muly_9), &muly_9));
            OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowwidth_i32,
                                          9, 272 * sizeof(int32_t), NULL));
            if (1 * (x_elems_5 + srem32(16 - srem32(x_elems_5, 16), 16)) *
                (new_height_10 + srem32(16 - srem32(new_height_10, 16), 16)) *
                num_arrays_4 != 0) {
                const size_t global_work_sizze_1171[3] = {x_elems_5 +
                                                          srem32(16 -
                                                                 srem32(x_elems_5,
                                                                        16),
                                                                 16),
                                                          new_height_10 +
                                                          srem32(16 -
                                                                 srem32(new_height_10,
                                                                        16),
                                                                 16),
                                                          num_arrays_4};
                const size_t local_work_sizze_1175[3] = {16, 16, 1};
                int64_t time_start_1172, time_end_1173;
                
                if (debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "fut_kernel_map_transpose_lowwidth_i32");
                    fprintf(stderr, "%zu", global_work_sizze_1171[0]);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%zu", global_work_sizze_1171[1]);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%zu", global_work_sizze_1171[2]);
                    fprintf(stderr, "].\n");
                    time_start_1172 = get_wall_time();
                }
                OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                                      fut_kernel_map_transpose_lowwidth_i32,
                                                      3, NULL,
                                                      global_work_sizze_1171,
                                                      local_work_sizze_1175, 0,
                                                      NULL, NULL));
                if (debugging) {
                    OPENCL_SUCCEED(clFinish(fut_cl_queue));
                    time_end_1173 = get_wall_time();
                    
                    long time_diff_1174 = time_end_1173 - time_start_1172;
                    
                    if (detail_timing) {
                        fut_kernel_map_transpose_lowwidth_i32total_runtime +=
                            time_diff_1174;
                        fut_kernel_map_transpose_lowwidth_i32runs++;
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "fut_kernel_map_transpose_lowwidth_i32",
                                (int) time_diff_1174);
                    }
                }
            }
        } else {
            if (sle32(y_elems_6, squot32(16, 2)) && slt32(16, x_elems_5)) {
                int32_t mulx_11 = squot32(16, y_elems_6);
                int32_t new_width_12;
                
                new_width_12 = squot32(x_elems_5 + mulx_11 - 1, mulx_11);
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowheight_i32,
                                              0, sizeof(destmem_0.mem),
                                              &destmem_0.mem));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowheight_i32,
                                              1, sizeof(destoffset_1),
                                              &destoffset_1));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowheight_i32,
                                              2, sizeof(srcmem_2.mem),
                                              &srcmem_2.mem));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowheight_i32,
                                              3, sizeof(srcoffset_3),
                                              &srcoffset_3));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowheight_i32,
                                              4, sizeof(x_elems_5),
                                              &x_elems_5));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowheight_i32,
                                              5, sizeof(y_elems_6),
                                              &y_elems_6));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowheight_i32,
                                              6, sizeof(in_elems_7),
                                              &in_elems_7));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowheight_i32,
                                              7, sizeof(out_elems_8),
                                              &out_elems_8));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowheight_i32,
                                              8, sizeof(mulx_11), &mulx_11));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_lowheight_i32,
                                              9, 272 * sizeof(int32_t), NULL));
                if (1 * (new_width_12 + srem32(16 - srem32(new_width_12, 16),
                                               16)) * (y_elems_6 + srem32(16 -
                                                                          srem32(y_elems_6,
                                                                                 16),
                                                                          16)) *
                    num_arrays_4 != 0) {
                    const size_t global_work_sizze_1176[3] = {new_width_12 +
                                                              srem32(16 -
                                                                     srem32(new_width_12,
                                                                            16),
                                                                     16),
                                                              y_elems_6 +
                                                              srem32(16 -
                                                                     srem32(y_elems_6,
                                                                            16),
                                                                     16),
                                                              num_arrays_4};
                    const size_t local_work_sizze_1180[3] = {16, 16, 1};
                    int64_t time_start_1177, time_end_1178;
                    
                    if (debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "fut_kernel_map_transpose_lowheight_i32");
                        fprintf(stderr, "%zu", global_work_sizze_1176[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_1176[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_1176[2]);
                        fprintf(stderr, "].\n");
                        time_start_1177 = get_wall_time();
                    }
                    OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                                          fut_kernel_map_transpose_lowheight_i32,
                                                          3, NULL,
                                                          global_work_sizze_1176,
                                                          local_work_sizze_1180,
                                                          0, NULL, NULL));
                    if (debugging) {
                        OPENCL_SUCCEED(clFinish(fut_cl_queue));
                        time_end_1178 = get_wall_time();
                        
                        long time_diff_1179 = time_end_1178 - time_start_1177;
                        
                        if (detail_timing) {
                            fut_kernel_map_transpose_lowheight_i32total_runtime +=
                                time_diff_1179;
                            fut_kernel_map_transpose_lowheight_i32runs++;
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "fut_kernel_map_transpose_lowheight_i32",
                                    (int) time_diff_1179);
                        }
                    }
                }
            } else {
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                              sizeof(destmem_0.mem),
                                              &destmem_0.mem));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                              sizeof(destoffset_1),
                                              &destoffset_1));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                              sizeof(srcmem_2.mem),
                                              &srcmem_2.mem));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                              sizeof(srcoffset_3),
                                              &srcoffset_3));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                              sizeof(x_elems_5), &x_elems_5));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                              sizeof(y_elems_6), &y_elems_6));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                              sizeof(in_elems_7), &in_elems_7));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7,
                                              sizeof(out_elems_8),
                                              &out_elems_8));
                OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 8,
                                              272 * sizeof(int32_t), NULL));
                if (1 * (x_elems_5 + srem32(16 - srem32(x_elems_5, 16), 16)) *
                    (y_elems_6 + srem32(16 - srem32(y_elems_6, 16), 16)) *
                    num_arrays_4 != 0) {
                    const size_t global_work_sizze_1181[3] = {x_elems_5 +
                                                              srem32(16 -
                                                                     srem32(x_elems_5,
                                                                            16),
                                                                     16),
                                                              y_elems_6 +
                                                              srem32(16 -
                                                                     srem32(y_elems_6,
                                                                            16),
                                                                     16),
                                                              num_arrays_4};
                    const size_t local_work_sizze_1185[3] = {16, 16, 1};
                    int64_t time_start_1182, time_end_1183;
                    
                    if (debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "fut_kernel_map_transpose_i32");
                        fprintf(stderr, "%zu", global_work_sizze_1181[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_1181[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_1181[2]);
                        fprintf(stderr, "].\n");
                        time_start_1182 = get_wall_time();
                    }
                    OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                                          fut_kernel_map_transpose_i32,
                                                          3, NULL,
                                                          global_work_sizze_1181,
                                                          local_work_sizze_1185,
                                                          0, NULL, NULL));
                    if (debugging) {
                        OPENCL_SUCCEED(clFinish(fut_cl_queue));
                        time_end_1183 = get_wall_time();
                        
                        long time_diff_1184 = time_end_1183 - time_start_1182;
                        
                        if (detail_timing) {
                            fut_kernel_map_transpose_i32total_runtime +=
                                time_diff_1184;
                            fut_kernel_map_transpose_i32runs++;
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "fut_kernel_map_transpose_i32",
                                    (int) time_diff_1184);
                        }
                    }
                }
            }
        }
    }
    
    struct tuple_ retval_1170;
    
    return retval_1170;
}
static
struct tuple_int32_t_device_mem_int32_t futhark_main(int64_t xss_mem_sizze_1047,
                                                     struct memblock_device xss_mem_1048,
                                                     int32_t sizze_285,
                                                     int32_t sizze_286)
{
    int32_t out_memsizze_1117;
    struct memblock_device out_mem_1116;
    
    out_mem_1116.references = NULL;
    
    int32_t out_arrsizze_1118;
    char outer_suff_par_1001 = slt32(65536, sizze_285);
    int32_t group_sizze_1032;
    
    group_sizze_1032 = cl_group_size;
    
    int32_t group_available_par_1033 = sizze_285 * group_sizze_1032;
    char group_suff_par_1034 = slt32(65536, group_available_par_1033);
    int64_t binop_y_1050 = sext_i32_i64(sizze_286);
    int64_t binop_x_1051 = 4 * binop_y_1050;
    int64_t binop_y_1052 = sext_i32_i64(sizze_285);
    int64_t bytes_1049 = binop_x_1051 * binop_y_1052;
    struct memblock_device mem_1053;
    
    mem_1053.references = NULL;
    memblock_alloc_device(&mem_1053, bytes_1049);
    
    int64_t bytes_1054 = 4 * binop_y_1052;
    struct memblock_device mem_1056;
    
    mem_1056.references = NULL;
    memblock_alloc_device(&mem_1056, bytes_1054);
    
    struct memblock_device mem_1061;
    
    mem_1061.references = NULL;
    memblock_alloc_device(&mem_1061, bytes_1049);
    
    struct memblock_device mem_1064;
    
    mem_1064.references = NULL;
    memblock_alloc_device(&mem_1064, bytes_1054);
    
    int32_t total_num_elements_691 = sizze_286 * sizze_285;
    int32_t num_groups_hint_694;
    
    num_groups_hint_694 = cl_num_groups;
    
    int32_t y_695 = sizze_285 - 1;
    int32_t x_696 = num_groups_hint_694 + y_695;
    int32_t num_groups_per_segment_hint_697 = squot32(x_696, sizze_285);
    int32_t x_698 = group_sizze_1032 * num_groups_per_segment_hint_697;
    int32_t y_699 = x_698 - 1;
    int32_t x_700 = sizze_286 + y_699;
    int32_t elements_per_thread_702 = squot32(x_700, x_698);
    char cond_703 = elements_per_thread_702 == 1;
    int32_t num_groups_per_segment_704;
    
    if (cond_703) {
        int32_t y_705 = group_sizze_1032 - 1;
        int32_t x_706 = sizze_286 + y_705;
        int32_t x_707 = squot32(x_706, group_sizze_1032);
        
        num_groups_per_segment_704 = x_707;
    } else {
        num_groups_per_segment_704 = num_groups_per_segment_hint_697;
    }
    
    int32_t x_708 = squot32(group_sizze_1032, 2);
    char cond_709 = slt32(x_708, sizze_286);
    char cond_711 = num_groups_per_segment_704 == 1;
    struct memblock_device mem_1070;
    
    mem_1070.references = NULL;
    memblock_alloc_device(&mem_1070, bytes_1054);
    
    int32_t group_sizze_749 = group_sizze_1032;
    int32_t elements_per_thread_758 = elements_per_thread_702;
    int32_t num_groups_per_segment_760;
    
    if (cond_703) {
        int32_t y_761 = group_sizze_1032 - 1;
        int32_t x_762 = sizze_286 + y_761;
        int32_t x_763 = squot32(x_762, group_sizze_1032);
        
        num_groups_per_segment_760 = x_763;
    } else {
        num_groups_per_segment_760 = num_groups_per_segment_hint_697;
    }
    
    int32_t num_groups_764 = sizze_285 * num_groups_per_segment_760;
    int32_t num_threads_765 = num_groups_764 * group_sizze_1032;
    int32_t threads_within_segment_766 = group_sizze_1032 *
            num_groups_per_segment_760;
    int64_t binop_y_1075 = sext_i32_i64(num_groups_764);
    int64_t bytes_1074 = 4 * binop_y_1075;
    struct memblock_device mem_1076;
    
    mem_1076.references = NULL;
    memblock_alloc_device(&mem_1076, bytes_1074);
    
    int64_t binop_y_1072 = sext_i32_i64(group_sizze_1032);
    int64_t bytes_1071 = 4 * binop_y_1072;
    char cond_796 = slt32(x_708, num_groups_per_segment_760);
    struct memblock_device mem_1082;
    
    mem_1082.references = NULL;
    memblock_alloc_device(&mem_1082, bytes_1054);
    
    struct memblock_device mem_1085;
    
    mem_1085.references = NULL;
    memblock_alloc_device(&mem_1085, bytes_1054);
    
    struct memblock_device mem_1097;
    
    mem_1097.references = NULL;
    memblock_alloc_device(&mem_1097, bytes_1054);
    
    struct memblock_local mem_1067;
    
    mem_1067.references = NULL;
    
    struct memblock_local mem_1073;
    
    mem_1073.references = NULL;
    
    struct memblock_local mem_1079;
    
    mem_1079.references = NULL;
    
    struct memblock_local mem_1087;
    
    mem_1087.references = NULL;
    
    struct memblock_local mem_1090;
    
    mem_1090.references = NULL;
    
    struct memblock_local mem_1099;
    
    mem_1099.references = NULL;
    
    struct memblock_local mem_1102;
    
    mem_1102.references = NULL;
    
    int64_t res_mem_sizze_1107;
    struct memblock_device res_mem_1108;
    
    res_mem_1108.references = NULL;
    if (outer_suff_par_1001) {
        int32_t y_986 = group_sizze_1032 - 1;
        int32_t x_987 = sizze_285 + y_986;
        int32_t num_groups_988 = squot32(x_987, group_sizze_1032);
        int32_t num_threads_989 = num_groups_988 * group_sizze_1032;
        struct tuple_ call_ret_1187;
        
        call_ret_1187 = futhark_map_transpose_opencl_i32(mem_1053, 0,
                                                         xss_mem_1048, 0, 1,
                                                         sizze_286, sizze_285,
                                                         sizze_285 * sizze_286,
                                                         sizze_285 * sizze_286);
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_971, 0, sizeof(sizze_285),
                                      &sizze_285));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_971, 1, sizeof(sizze_286),
                                      &sizze_286));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_971, 2, sizeof(mem_1053.mem),
                                      &mem_1053.mem));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_971, 3, sizeof(mem_1056.mem),
                                      &mem_1056.mem));
        if (1 * (num_groups_988 * group_sizze_1032) != 0) {
            const size_t global_work_sizze_1188[1] = {num_groups_988 *
                         group_sizze_1032};
            const size_t local_work_sizze_1192[1] = {group_sizze_1032};
            int64_t time_start_1189, time_end_1190;
            
            if (debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_kernel_971");
                fprintf(stderr, "%zu", global_work_sizze_1188[0]);
                fprintf(stderr, "].\n");
                time_start_1189 = get_wall_time();
            }
            OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_971,
                                                  1, NULL,
                                                  global_work_sizze_1188,
                                                  local_work_sizze_1192, 0,
                                                  NULL, NULL));
            if (debugging) {
                OPENCL_SUCCEED(clFinish(fut_cl_queue));
                time_end_1190 = get_wall_time();
                
                long time_diff_1191 = time_end_1190 - time_start_1189;
                
                if (detail_timing) {
                    map_kernel_971total_runtime += time_diff_1191;
                    map_kernel_971runs++;
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "map_kernel_971", (int) time_diff_1191);
                }
            }
        }
        memblock_set_device(&res_mem_1108, &mem_1056);
        res_mem_sizze_1107 = bytes_1054;
    } else {
        int64_t res_mem_sizze_1105;
        struct memblock_device res_mem_1106;
        
        res_mem_1106.references = NULL;
        if (group_suff_par_1034) {
            int32_t y_1017 = group_sizze_1032 - 1;
            int32_t x_1018 = sizze_285 + y_1017;
            int32_t num_groups_1019 = squot32(x_1018, group_sizze_1032);
            int32_t num_threads_1020 = num_groups_1019 * group_sizze_1032;
            struct tuple_ call_ret_1193;
            
            call_ret_1193 = futhark_map_transpose_opencl_i32(mem_1061, 0,
                                                             xss_mem_1048, 0, 1,
                                                             sizze_286,
                                                             sizze_285,
                                                             sizze_285 *
                                                             sizze_286,
                                                             sizze_285 *
                                                             sizze_286);
            OPENCL_SUCCEED(clSetKernelArg(map_intra_group_kernel_1008, 0,
                                          sizeof(sizze_285), &sizze_285));
            OPENCL_SUCCEED(clSetKernelArg(map_intra_group_kernel_1008, 1,
                                          sizeof(sizze_286), &sizze_286));
            OPENCL_SUCCEED(clSetKernelArg(map_intra_group_kernel_1008, 2,
                                          sizeof(mem_1061.mem), &mem_1061.mem));
            OPENCL_SUCCEED(clSetKernelArg(map_intra_group_kernel_1008, 3,
                                          sizeof(mem_1064.mem), &mem_1064.mem));
            if (1 * (num_groups_1019 * group_sizze_1032) != 0) {
                const size_t global_work_sizze_1194[1] = {num_groups_1019 *
                             group_sizze_1032};
                const size_t local_work_sizze_1198[1] = {group_sizze_1032};
                int64_t time_start_1195, time_end_1196;
                
                if (debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "map_intra_group_kernel_1008");
                    fprintf(stderr, "%zu", global_work_sizze_1194[0]);
                    fprintf(stderr, "].\n");
                    time_start_1195 = get_wall_time();
                }
                OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                                      map_intra_group_kernel_1008,
                                                      1, NULL,
                                                      global_work_sizze_1194,
                                                      local_work_sizze_1198, 0,
                                                      NULL, NULL));
                if (debugging) {
                    OPENCL_SUCCEED(clFinish(fut_cl_queue));
                    time_end_1196 = get_wall_time();
                    
                    long time_diff_1197 = time_end_1196 - time_start_1195;
                    
                    if (detail_timing) {
                        map_intra_group_kernel_1008total_runtime +=
                            time_diff_1197;
                        map_intra_group_kernel_1008runs++;
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "map_intra_group_kernel_1008",
                                (int) time_diff_1197);
                    }
                }
            }
            memblock_set_device(&res_mem_1106, &mem_1064);
            res_mem_sizze_1105 = bytes_1054;
        } else {
            int64_t res_mem_sizze_1103;
            struct memblock_device res_mem_1104;
            
            res_mem_1104.references = NULL;
            if (cond_709) {
                int64_t x_mem_sizze_1093;
                struct memblock_device x_mem_1094;
                
                x_mem_1094.references = NULL;
                if (cond_711) {
                    int32_t group_sizze_713 = group_sizze_1032;
                    int32_t y_716 = group_sizze_1032 - 1;
                    int32_t x_717 = sizze_286 + y_716;
                    int32_t elements_per_thread_719 = squot32(x_717,
                                                              group_sizze_1032);
                    
                    if (debugging) {
                        fprintf(stderr, "%s: ", "num_segment");
                        fprintf(stderr, "%di32", sizze_285);
                        fprintf(stderr, "\n");
                    }
                    if (debugging) {
                        fprintf(stderr, "%s: ", "segment_size");
                        fprintf(stderr, "%di32", sizze_286);
                        fprintf(stderr, "\n");
                    }
                    if (debugging) {
                        fprintf(stderr, "%s: ", "num_groups");
                        fprintf(stderr, "%di32", sizze_285);
                        fprintf(stderr, "\n");
                    }
                    if (debugging) {
                        fprintf(stderr, "%s: ", "group_size");
                        fprintf(stderr, "%di32", group_sizze_713);
                        fprintf(stderr, "\n");
                    }
                    if (debugging) {
                        fprintf(stderr, "%s: ", "elements_per_thread");
                        fprintf(stderr, "%di32", elements_per_thread_719);
                        fprintf(stderr, "\n");
                    }
                    if (debugging) {
                        fprintf(stderr, "%s: ", "num_groups_per_segment");
                        fprintf(stderr, "%di32", 1);
                        fprintf(stderr, "\n");
                    }
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_360,
                                                  0, bytes_1071, NULL));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_360,
                                                  1, sizeof(sizze_285),
                                                  &sizze_285));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_360,
                                                  2, sizeof(sizze_286),
                                                  &sizze_286));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_360,
                                                  3,
                                                  sizeof(elements_per_thread_719),
                                                  &elements_per_thread_719));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_360,
                                                  4, sizeof(xss_mem_1048.mem),
                                                  &xss_mem_1048.mem));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_360,
                                                  5, sizeof(mem_1070.mem),
                                                  &mem_1070.mem));
                    if (1 * (sizze_285 * group_sizze_1032) != 0) {
                        const size_t global_work_sizze_1199[1] = {sizze_285 *
                                     group_sizze_1032};
                        const size_t local_work_sizze_1203[1] =
                                     {group_sizze_1032};
                        int64_t time_start_1200, time_end_1201;
                        
                        if (debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "segmented_redomap__large_comm_one_kernel_360");
                            fprintf(stderr, "%zu", global_work_sizze_1199[0]);
                            fprintf(stderr, "].\n");
                            time_start_1200 = get_wall_time();
                        }
                        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                                              segmented_redomap__large_comm_one_kernel_360,
                                                              1, NULL,
                                                              global_work_sizze_1199,
                                                              local_work_sizze_1203,
                                                              0, NULL, NULL));
                        if (debugging) {
                            OPENCL_SUCCEED(clFinish(fut_cl_queue));
                            time_end_1201 = get_wall_time();
                            
                            long time_diff_1202 = time_end_1201 -
                                 time_start_1200;
                            
                            if (detail_timing) {
                                segmented_redomap__large_comm_one_kernel_360total_runtime +=
                                    time_diff_1202;
                                segmented_redomap__large_comm_one_kernel_360runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "segmented_redomap__large_comm_one_kernel_360",
                                        (int) time_diff_1202);
                            }
                        }
                    }
                    memblock_set_device(&x_mem_1094, &mem_1070);
                    x_mem_sizze_1093 = bytes_1054;
                } else {
                    if (debugging) {
                        fprintf(stderr, "%s: ", "num_segment");
                        fprintf(stderr, "%di32", sizze_285);
                        fprintf(stderr, "\n");
                    }
                    if (debugging) {
                        fprintf(stderr, "%s: ", "segment_size");
                        fprintf(stderr, "%di32", sizze_286);
                        fprintf(stderr, "\n");
                    }
                    if (debugging) {
                        fprintf(stderr, "%s: ", "num_groups");
                        fprintf(stderr, "%di32", num_groups_764);
                        fprintf(stderr, "\n");
                    }
                    if (debugging) {
                        fprintf(stderr, "%s: ", "group_size");
                        fprintf(stderr, "%di32", group_sizze_749);
                        fprintf(stderr, "\n");
                    }
                    if (debugging) {
                        fprintf(stderr, "%s: ", "elements_per_thread");
                        fprintf(stderr, "%di32", elements_per_thread_758);
                        fprintf(stderr, "\n");
                    }
                    if (debugging) {
                        fprintf(stderr, "%s: ", "num_groups_per_segment");
                        fprintf(stderr, "%di32", num_groups_per_segment_760);
                        fprintf(stderr, "\n");
                    }
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_many_kernel_418,
                                                  0, bytes_1071, NULL));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_many_kernel_418,
                                                  1, sizeof(sizze_285),
                                                  &sizze_285));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_many_kernel_418,
                                                  2, sizeof(sizze_286),
                                                  &sizze_286));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_many_kernel_418,
                                                  3,
                                                  sizeof(elements_per_thread_702),
                                                  &elements_per_thread_702));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_many_kernel_418,
                                                  4,
                                                  sizeof(num_groups_per_segment_760),
                                                  &num_groups_per_segment_760));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_many_kernel_418,
                                                  5,
                                                  sizeof(threads_within_segment_766),
                                                  &threads_within_segment_766));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_many_kernel_418,
                                                  6, sizeof(xss_mem_1048.mem),
                                                  &xss_mem_1048.mem));
                    OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_many_kernel_418,
                                                  7, sizeof(mem_1076.mem),
                                                  &mem_1076.mem));
                    if (1 * (num_groups_764 * group_sizze_1032) != 0) {
                        const size_t global_work_sizze_1204[1] =
                                     {num_groups_764 * group_sizze_1032};
                        const size_t local_work_sizze_1208[1] =
                                     {group_sizze_1032};
                        int64_t time_start_1205, time_end_1206;
                        
                        if (debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "segmented_redomap__large_comm_many_kernel_418");
                            fprintf(stderr, "%zu", global_work_sizze_1204[0]);
                            fprintf(stderr, "].\n");
                            time_start_1205 = get_wall_time();
                        }
                        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                                              segmented_redomap__large_comm_many_kernel_418,
                                                              1, NULL,
                                                              global_work_sizze_1204,
                                                              local_work_sizze_1208,
                                                              0, NULL, NULL));
                        if (debugging) {
                            OPENCL_SUCCEED(clFinish(fut_cl_queue));
                            time_end_1206 = get_wall_time();
                            
                            long time_diff_1207 = time_end_1206 -
                                 time_start_1205;
                            
                            if (detail_timing) {
                                segmented_redomap__large_comm_many_kernel_418total_runtime +=
                                    time_diff_1207;
                                segmented_redomap__large_comm_many_kernel_418runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "segmented_redomap__large_comm_many_kernel_418",
                                        (int) time_diff_1207);
                            }
                        }
                    }
                    
                    int64_t step_two_kernel_result_mem_sizze_1091;
                    struct memblock_device step_two_kernel_result_mem_1092;
                    
                    step_two_kernel_result_mem_1092.references = NULL;
                    if (cond_796) {
                        int32_t group_sizze_798 = group_sizze_1032;
                        int32_t y_801 = group_sizze_1032 - 1;
                        int32_t x_802 = num_groups_per_segment_760 + y_801;
                        int32_t elements_per_thread_804 = squot32(x_802,
                                                                  group_sizze_1032);
                        
                        if (debugging) {
                            fprintf(stderr, "%s: ", "num_segment");
                            fprintf(stderr, "%di32", sizze_285);
                            fprintf(stderr, "\n");
                        }
                        if (debugging) {
                            fprintf(stderr, "%s: ", "segment_size");
                            fprintf(stderr, "%di32",
                                    num_groups_per_segment_760);
                            fprintf(stderr, "\n");
                        }
                        if (debugging) {
                            fprintf(stderr, "%s: ", "num_groups");
                            fprintf(stderr, "%di32", sizze_285);
                            fprintf(stderr, "\n");
                        }
                        if (debugging) {
                            fprintf(stderr, "%s: ", "group_size");
                            fprintf(stderr, "%di32", group_sizze_798);
                            fprintf(stderr, "\n");
                        }
                        if (debugging) {
                            fprintf(stderr, "%s: ", "elements_per_thread");
                            fprintf(stderr, "%di32", elements_per_thread_804);
                            fprintf(stderr, "\n");
                        }
                        if (debugging) {
                            fprintf(stderr, "%s: ", "num_groups_per_segment");
                            fprintf(stderr, "%di32", 1);
                            fprintf(stderr, "\n");
                        }
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_468,
                                                      0, bytes_1071, NULL));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_468,
                                                      1, sizeof(sizze_285),
                                                      &sizze_285));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_468,
                                                      2,
                                                      sizeof(num_groups_per_segment_760),
                                                      &num_groups_per_segment_760));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_468,
                                                      3,
                                                      sizeof(elements_per_thread_804),
                                                      &elements_per_thread_804));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_468,
                                                      4, sizeof(mem_1076.mem),
                                                      &mem_1076.mem));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__large_comm_one_kernel_468,
                                                      5, sizeof(mem_1082.mem),
                                                      &mem_1082.mem));
                        if (1 * (sizze_285 * group_sizze_1032) != 0) {
                            const size_t global_work_sizze_1209[1] =
                                         {sizze_285 * group_sizze_1032};
                            const size_t local_work_sizze_1213[1] =
                                         {group_sizze_1032};
                            int64_t time_start_1210, time_end_1211;
                            
                            if (debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "segmented_redomap__large_comm_one_kernel_468");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_1209[0]);
                                fprintf(stderr, "].\n");
                                time_start_1210 = get_wall_time();
                            }
                            OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                                                  segmented_redomap__large_comm_one_kernel_468,
                                                                  1, NULL,
                                                                  global_work_sizze_1209,
                                                                  local_work_sizze_1213,
                                                                  0, NULL,
                                                                  NULL));
                            if (debugging) {
                                OPENCL_SUCCEED(clFinish(fut_cl_queue));
                                time_end_1211 = get_wall_time();
                                
                                long time_diff_1212 = time_end_1211 -
                                     time_start_1210;
                                
                                if (detail_timing) {
                                    segmented_redomap__large_comm_one_kernel_468total_runtime +=
                                        time_diff_1212;
                                    segmented_redomap__large_comm_one_kernel_468runs++;
                                    fprintf(stderr,
                                            "kernel %s runtime: %ldus\n",
                                            "segmented_redomap__large_comm_one_kernel_468",
                                            (int) time_diff_1212);
                                }
                            }
                        }
                        memblock_set_device(&step_two_kernel_result_mem_1092,
                                            &mem_1082);
                        step_two_kernel_result_mem_sizze_1091 = bytes_1054;
                    } else {
                        int32_t group_sizze_836 = group_sizze_1032;
                        int32_t num_segments_per_group_837 =
                                squot32(group_sizze_1032,
                                        num_groups_per_segment_760);
                        int32_t y_838 = num_segments_per_group_837 - 1;
                        int32_t x_839 = sizze_285 + y_838;
                        int32_t num_groups_840 = squot32(x_839,
                                                         num_segments_per_group_837);
                        int32_t num_threads_841 = num_groups_840 *
                                group_sizze_1032;
                        int32_t active_threads_per_group_842 =
                                num_groups_per_segment_760 *
                                num_segments_per_group_837;
                        int32_t x_843 = srem32(sizze_285,
                                               num_segments_per_group_837);
                        char cond_844 = x_843 == 0;
                        int32_t seg_in_last_group_845;
                        
                        if (cond_844) {
                            seg_in_last_group_845 = num_segments_per_group_837;
                        } else {
                            seg_in_last_group_845 = x_843;
                        }
                        
                        int32_t active_threads_last_group_847 =
                                num_groups_per_segment_760 *
                                seg_in_last_group_845;
                        int32_t y_849 = num_groups_840 - 1;
                        
                        if (debugging) {
                            fprintf(stderr, "%s: ", "num_segment");
                            fprintf(stderr, "%di32", sizze_285);
                            fprintf(stderr, "\n");
                        }
                        if (debugging) {
                            fprintf(stderr, "%s: ", "segment_size");
                            fprintf(stderr, "%di32",
                                    num_groups_per_segment_760);
                            fprintf(stderr, "\n");
                        }
                        if (debugging) {
                            fprintf(stderr, "%s: ", "num_groups");
                            fprintf(stderr, "%di32", num_groups_840);
                            fprintf(stderr, "\n");
                        }
                        if (debugging) {
                            fprintf(stderr, "%s: ", "group_size");
                            fprintf(stderr, "%di32", group_sizze_836);
                            fprintf(stderr, "\n");
                        }
                        if (debugging) {
                            fprintf(stderr, "%s: ", "num_segments_per_group");
                            fprintf(stderr, "%di32",
                                    num_segments_per_group_837);
                            fprintf(stderr, "\n");
                        }
                        if (debugging) {
                            fprintf(stderr, "%s: ", "active_threads_per_group");
                            fprintf(stderr, "%di32",
                                    active_threads_per_group_842);
                            fprintf(stderr, "\n");
                        }
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_498,
                                                      0, binop_y_1072, NULL));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_498,
                                                      1, bytes_1071, NULL));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_498,
                                                      2, sizeof(sizze_285),
                                                      &sizze_285));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_498,
                                                      3,
                                                      sizeof(num_groups_per_segment_760),
                                                      &num_groups_per_segment_760));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_498,
                                                      4,
                                                      sizeof(num_segments_per_group_837),
                                                      &num_segments_per_group_837));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_498,
                                                      5,
                                                      sizeof(active_threads_per_group_842),
                                                      &active_threads_per_group_842));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_498,
                                                      6,
                                                      sizeof(active_threads_last_group_847),
                                                      &active_threads_last_group_847));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_498,
                                                      7, sizeof(y_849),
                                                      &y_849));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_498,
                                                      8, sizeof(mem_1076.mem),
                                                      &mem_1076.mem));
                        OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_498,
                                                      9, sizeof(mem_1085.mem),
                                                      &mem_1085.mem));
                        if (1 * (num_groups_840 * group_sizze_1032) != 0) {
                            const size_t global_work_sizze_1214[1] =
                                         {num_groups_840 * group_sizze_1032};
                            const size_t local_work_sizze_1218[1] =
                                         {group_sizze_1032};
                            int64_t time_start_1215, time_end_1216;
                            
                            if (debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "segmented_redomap__small_comm_kernel_498");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_1214[0]);
                                fprintf(stderr, "].\n");
                                time_start_1215 = get_wall_time();
                            }
                            OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                                                  segmented_redomap__small_comm_kernel_498,
                                                                  1, NULL,
                                                                  global_work_sizze_1214,
                                                                  local_work_sizze_1218,
                                                                  0, NULL,
                                                                  NULL));
                            if (debugging) {
                                OPENCL_SUCCEED(clFinish(fut_cl_queue));
                                time_end_1216 = get_wall_time();
                                
                                long time_diff_1217 = time_end_1216 -
                                     time_start_1215;
                                
                                if (detail_timing) {
                                    segmented_redomap__small_comm_kernel_498total_runtime +=
                                        time_diff_1217;
                                    segmented_redomap__small_comm_kernel_498runs++;
                                    fprintf(stderr,
                                            "kernel %s runtime: %ldus\n",
                                            "segmented_redomap__small_comm_kernel_498",
                                            (int) time_diff_1217);
                                }
                            }
                        }
                        memblock_set_device(&step_two_kernel_result_mem_1092,
                                            &mem_1085);
                        step_two_kernel_result_mem_sizze_1091 = bytes_1054;
                    }
                    memblock_set_device(&x_mem_1094,
                                        &step_two_kernel_result_mem_1092);
                    x_mem_sizze_1093 = step_two_kernel_result_mem_sizze_1091;
                    memblock_unref_device(&step_two_kernel_result_mem_1092);
                }
                memblock_set_device(&res_mem_1104, &x_mem_1094);
                res_mem_sizze_1103 = x_mem_sizze_1093;
                memblock_unref_device(&x_mem_1094);
            } else {
                int32_t group_sizze_898 = group_sizze_1032;
                int32_t num_segments_per_group_899 = squot32(group_sizze_1032,
                                                             sizze_286);
                int32_t y_900 = num_segments_per_group_899 - 1;
                int32_t x_901 = sizze_285 + y_900;
                int32_t num_groups_902 = squot32(x_901,
                                                 num_segments_per_group_899);
                int32_t num_threads_903 = num_groups_902 * group_sizze_1032;
                int32_t active_threads_per_group_904 = sizze_286 *
                        num_segments_per_group_899;
                int32_t x_905 = srem32(sizze_285, num_segments_per_group_899);
                char cond_906 = x_905 == 0;
                int32_t seg_in_last_group_907;
                
                if (cond_906) {
                    seg_in_last_group_907 = num_segments_per_group_899;
                } else {
                    seg_in_last_group_907 = x_905;
                }
                
                int32_t active_threads_last_group_909 = sizze_286 *
                        seg_in_last_group_907;
                int32_t y_911 = num_groups_902 - 1;
                
                if (debugging) {
                    fprintf(stderr, "%s: ", "num_segment");
                    fprintf(stderr, "%di32", sizze_285);
                    fprintf(stderr, "\n");
                }
                if (debugging) {
                    fprintf(stderr, "%s: ", "segment_size");
                    fprintf(stderr, "%di32", sizze_286);
                    fprintf(stderr, "\n");
                }
                if (debugging) {
                    fprintf(stderr, "%s: ", "num_groups");
                    fprintf(stderr, "%di32", num_groups_902);
                    fprintf(stderr, "\n");
                }
                if (debugging) {
                    fprintf(stderr, "%s: ", "group_size");
                    fprintf(stderr, "%di32", group_sizze_898);
                    fprintf(stderr, "\n");
                }
                if (debugging) {
                    fprintf(stderr, "%s: ", "num_segments_per_group");
                    fprintf(stderr, "%di32", num_segments_per_group_899);
                    fprintf(stderr, "\n");
                }
                if (debugging) {
                    fprintf(stderr, "%s: ", "active_threads_per_group");
                    fprintf(stderr, "%di32", active_threads_per_group_904);
                    fprintf(stderr, "\n");
                }
                OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_651,
                                              0, binop_y_1072, NULL));
                OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_651,
                                              1, bytes_1071, NULL));
                OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_651,
                                              2, sizeof(sizze_285),
                                              &sizze_285));
                OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_651,
                                              3, sizeof(sizze_286),
                                              &sizze_286));
                OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_651,
                                              4,
                                              sizeof(num_segments_per_group_899),
                                              &num_segments_per_group_899));
                OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_651,
                                              5,
                                              sizeof(active_threads_per_group_904),
                                              &active_threads_per_group_904));
                OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_651,
                                              6,
                                              sizeof(active_threads_last_group_909),
                                              &active_threads_last_group_909));
                OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_651,
                                              7, sizeof(y_911), &y_911));
                OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_651,
                                              8, sizeof(xss_mem_1048.mem),
                                              &xss_mem_1048.mem));
                OPENCL_SUCCEED(clSetKernelArg(segmented_redomap__small_comm_kernel_651,
                                              9, sizeof(mem_1097.mem),
                                              &mem_1097.mem));
                if (1 * (num_groups_902 * group_sizze_1032) != 0) {
                    const size_t global_work_sizze_1219[1] = {num_groups_902 *
                                 group_sizze_1032};
                    const size_t local_work_sizze_1223[1] = {group_sizze_1032};
                    int64_t time_start_1220, time_end_1221;
                    
                    if (debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "segmented_redomap__small_comm_kernel_651");
                        fprintf(stderr, "%zu", global_work_sizze_1219[0]);
                        fprintf(stderr, "].\n");
                        time_start_1220 = get_wall_time();
                    }
                    OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                                          segmented_redomap__small_comm_kernel_651,
                                                          1, NULL,
                                                          global_work_sizze_1219,
                                                          local_work_sizze_1223,
                                                          0, NULL, NULL));
                    if (debugging) {
                        OPENCL_SUCCEED(clFinish(fut_cl_queue));
                        time_end_1221 = get_wall_time();
                        
                        long time_diff_1222 = time_end_1221 - time_start_1220;
                        
                        if (detail_timing) {
                            segmented_redomap__small_comm_kernel_651total_runtime +=
                                time_diff_1222;
                            segmented_redomap__small_comm_kernel_651runs++;
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "segmented_redomap__small_comm_kernel_651",
                                    (int) time_diff_1222);
                        }
                    }
                }
                memblock_set_device(&res_mem_1104, &mem_1097);
                res_mem_sizze_1103 = bytes_1054;
            }
            memblock_set_device(&res_mem_1106, &res_mem_1104);
            res_mem_sizze_1105 = res_mem_sizze_1103;
            memblock_unref_device(&res_mem_1104);
        }
        memblock_set_device(&res_mem_1108, &res_mem_1106);
        res_mem_sizze_1107 = res_mem_sizze_1105;
        memblock_unref_device(&res_mem_1106);
    }
    memblock_set_device(&out_mem_1116, &res_mem_1108);
    out_arrsizze_1118 = sizze_285;
    out_memsizze_1117 = res_mem_sizze_1107;
    
    struct tuple_int32_t_device_mem_int32_t retval_1186;
    
    retval_1186.elem_0 = out_memsizze_1117;
    retval_1186.elem_1.references = NULL;
    memblock_set_device(&retval_1186.elem_1, &out_mem_1116);
    retval_1186.elem_2 = out_arrsizze_1118;
    memblock_unref_device(&out_mem_1116);
    memblock_unref_device(&mem_1053);
    memblock_unref_device(&mem_1056);
    memblock_unref_device(&mem_1061);
    memblock_unref_device(&mem_1064);
    memblock_unref_device(&mem_1070);
    memblock_unref_device(&mem_1076);
    memblock_unref_device(&mem_1082);
    memblock_unref_device(&mem_1085);
    memblock_unref_device(&mem_1097);
    memblock_unref_local(&mem_1067);
    memblock_unref_local(&mem_1073);
    memblock_unref_local(&mem_1079);
    memblock_unref_local(&mem_1087);
    memblock_unref_local(&mem_1090);
    memblock_unref_local(&mem_1099);
    memblock_unref_local(&mem_1102);
    memblock_unref_device(&res_mem_1108);
    return retval_1186;
}
#include <inttypes.h>

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  int (*elem_reader)(void*);
};

static int peekc() {
  int c = getchar();
  ungetc(c,stdin);
  return c;
}

static int next_is_not_constituent() {
  int c = peekc();
  return c == EOF || !isalnum(c);
}

static void skipspaces() {
  int c = getchar();
  if (isspace(c)) {
    skipspaces();
  } else if (c == '-' && peekc() == '-') {
    // Skip to end of line.
    for (; c != '\n' && c != EOF; c = getchar());
    // Next line may have more spaces.
    skipspaces();
  } else if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int read_str_elem(struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(struct array_reader *reader, int dims) {
  int c;
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));
  while (1) {
    skipspaces();

    c = getchar();
    if (c == ']') {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (c == ',') {
      skipspaces();
      c = getchar();
      if (c == '[') {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ungetc(c, stdin);
        ret = read_str_elem(reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (c == EOF) {
      ret = 1;
      break;
    } else if (first) {
      if (c == '[') {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ungetc(c, stdin);
        ret = read_str_elem(reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(const char *type_name, int64_t *shape, int64_t dims) {
  char c;
  if (scanf("empty") == EOF) {
    return 1;
  }

  c = getchar();
  if (c != '(') {
    return 1;
  }

  for (int i = 0; i < dims-1; i++) {
    c = getchar();
    if (c != '[') {
      return 1;
    }
    c = getchar();
    if (c != ']') {
      return 1;
    }
  }

  int n = strlen(type_name);
  for (int i = 0; i < n; i++) {
    c = getchar();
    if (c != type_name[i]) {
      return 1;
    }
  }

  if (getchar() != ')') {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    shape[i] = 0;
  }

  return 0;
}

static int read_str_array(int64_t elem_size, int (*elem_reader)(void*),
                      const char *type_name,
                      void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  int64_t read_dims = 0;
  while (1) {
    int c;
    skipspaces();
    c = getchar();
    if (c=='[') {
      read_dims++;
    } else {
      if (c != EOF) {
        ungetc(c, stdin);
      }
      break;
    }
  }

  if (read_dims == 0) {
    return read_str_empty_array(type_name, shape, dims);
  }

  if (read_dims != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(&reader, dims);

  *data = reader.elems;

  return ret;
}

static int read_str_int8(void* dest) {
  skipspaces();
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  int x;
  if (scanf("%i", &x) == 1) {
    *(int8_t*)dest = x;
    scanf("i8");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_str_int16(void* dest) {
  skipspaces();
  if (scanf("%"SCNi16, (int16_t*)dest) == 1) {
    scanf("i16");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_str_int32(void* dest) {
  skipspaces();
  if (scanf("%"SCNi32, (int32_t*)dest) == 1) {
    scanf("i32");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_str_int64(void* dest) {
  skipspaces();
  if (scanf("%"SCNi64, (int64_t*)dest) == 1) {
    scanf("i64");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_str_float(void* dest) {
  skipspaces();
  if (scanf("%f", (float*)dest) == 1) {
    scanf("f32");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_str_double(void* dest) {
  skipspaces();
  if (scanf("%lf", (double*)dest) == 1) {
    scanf("f64");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_str_bool(void* dest) {
  /* This is a monstrous hack.  Maybe we should get a proper lexer in here. */
  char b[4];
  skipspaces();
  if (scanf("%4c", b) == 1) {
    if (strncmp(b, "true", 4) == 0) {
      *(char*)dest = 1;
      return 0;
    } else if (strncmp(b, "fals", 4) == 0 && getchar() == 'e') {
      *(char*)dest = 0;
      return 0;
    } else {
      return 1;
    }
  } else {
    return 1;
  }
}

// taken from http://esr.ibiblio.org/?p=5095
#define IS_BIG_ENDIAN (*(uint16_t *)"\0\xff" < 0x100)

#define READ_BINARY_VERSION 1

typedef struct {
  const char binname[4]; // used for parsing binary date
  const char* type_name; // used for printing
  const int size;
} primtype_info_t;

const primtype_info_t FUTHARK_PRIMTYPES[] = {
  {.binname = "  i8", .type_name = "i8",   .size = 1},
  {.binname = " i16", .type_name = "i16",  .size = 2},
  {.binname = " i32", .type_name = "i32",  .size = 4},
  {.binname = " i64", .type_name = "i64",  .size = 8},
  {.binname = " f32", .type_name = "f32",  .size = 4},
  {.binname = " f64", .type_name = "f64",  .size = 8},
  {.binname = "bool", .type_name = "bool", .size = 1},
};

// These indices should match up with the information above
typedef enum {
  FUTHARK_INT8 = 0,
  FUTHARK_INT16 = 1,
  FUTHARK_INT32 = 2,
  FUTHARK_INT64 = 3,
  FUTHARK_FLOAT32 = 4,
  FUTHARK_FLOAT64 = 5,
  FUTHARK_BOOL = 6,

  // Please add new types above this line -- we exploit that enums are just
  // ints, and use this value to loop through all types we know.
  FUTHARK_NUM_PRIMTYPES
} primtype_t;


////////////////////////////////////////////////////////////////////////////////
// Little endian
////////////////////////////////////////////////////////////////////////////////

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_2byte(void* dest) {
  int num_elems_read = fread(dest, 2, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_4byte(void* dest) {
  int num_elems_read = fread(dest, 4, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_8byte(void* dest) {
  int num_elems_read = fread(dest, 8, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

////////////////////////////////////////////////////////////////////////////////
// Big endian
////////////////////////////////////////////////////////////////////////////////

static int read_be_2byte(void* dest) {
  char* destc = (char*) dest;
  int num_matched = scanf("%c%c", destc+1, destc+0);
  return num_matched == 2 ? 0 : 1;
}

static int read_be_4byte(void* dest) {
  char* destc = (char*) dest;
  int num_matched = scanf("%c%c%c%c", destc+3, destc+2, destc+1, destc+0);
  return num_matched == 4 ? 0 : 1;
}

static int read_be_8byte(void* dest) {
  char* destc = (char*) dest;
  int num_matched = scanf("%c%c%c%c%c%c%c%c", destc+7, destc+6, destc+5,
                          destc+4, destc+3, destc+2, destc+1, destc+0);
  return num_matched == 8 ? 0 : 1;
}

////////////////////////////////////////////////////////////////////////////////
// General value interface
////////////////////////////////////////////////////////////////////////////////

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != READ_BINARY_VERSION) {
      panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, READ_BINARY_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static primtype_t read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { panic(1, "binary-input: Couldn't read element type.\n"); }

  for (int i=0; i<FUTHARK_NUM_PRIMTYPES; i++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if ( (read_binname[0] == FUTHARK_PRIMTYPES[i].binname[0]) &&
         (read_binname[1] == FUTHARK_PRIMTYPES[i].binname[1]) &&
         (read_binname[2] == FUTHARK_PRIMTYPES[i].binname[2]) &&
         (read_binname[3] == FUTHARK_PRIMTYPES[i].binname[3]) ) {
      return i;
    }
  }
  panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
}

static void read_bin_ensure_scalar(primtype_t expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  primtype_t bin_type_enum = read_bin_read_type_enum();
  if (bin_type_enum != expected_type) {
    panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          FUTHARK_PRIMTYPES[expected_type].type_name,
          FUTHARK_PRIMTYPES[bin_type_enum].type_name);
  }
}

static int read_int8(void* dest) {
  if (read_is_binary()) {
    read_bin_ensure_scalar(FUTHARK_INT8);
    return read_byte(dest);
  }
  return read_str_int8(dest);
}

static int read_int16(void* dest) {
  if (read_is_binary()) {
    read_bin_ensure_scalar(FUTHARK_INT16);
    if (IS_BIG_ENDIAN) {
      return read_be_2byte(dest);
    } else {
      return read_le_2byte(dest);
    }
  }
  return read_str_int16(dest);
}

static int read_int32(void* dest) {
  if (read_is_binary()) {
    read_bin_ensure_scalar(FUTHARK_INT32);
    if (IS_BIG_ENDIAN) {
      return read_be_4byte(dest);
    } else {
      return read_le_4byte(dest);
    }
  }
  return read_str_int32(dest);
}

static int read_int64(void* dest) {
  if (read_is_binary()) {
    read_bin_ensure_scalar(FUTHARK_INT64);
    if (IS_BIG_ENDIAN) {
      return read_be_8byte(dest);
    } else {
      return read_le_8byte(dest);
    }
  }
  return read_str_int64(dest);
}

static int read_float(void* dest) {
  if (read_is_binary()) {
    read_bin_ensure_scalar(FUTHARK_FLOAT32);
    if (IS_BIG_ENDIAN) {
      return read_be_4byte(dest);
    } else {
      return read_le_4byte(dest);
    }
  }
  return read_str_float(dest);
}

static int read_double(void* dest) {
  if (read_is_binary()) {
    read_bin_ensure_scalar(FUTHARK_FLOAT64);
    if (IS_BIG_ENDIAN) {
      return read_be_8byte(dest);
    } else {
      return read_le_8byte(dest);
    }
  }
  return read_str_double(dest);
}

static int read_bool(void* dest) {
  if (read_is_binary()) {
    read_bin_ensure_scalar(FUTHARK_BOOL);
    return read_byte(dest);
  }
  return read_str_bool(dest);
}

////////////////////////////////////////////////////////////////////////////////
// General array interface
////////////////////////////////////////////////////////////////////////////////

static int read_array(primtype_t expected_type, int64_t elem_size, int (*elem_reader)(void*),
                      const char *type_name, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(elem_size, elem_reader, type_name, data, shape, dims);
  }

  // now we know it is binary :)
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  primtype_t bin_type_enum = read_bin_read_type_enum();
  const primtype_info_t bin_primtype = FUTHARK_PRIMTYPES[bin_type_enum];
  if (expected_type != bin_type_enum) {
    panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, FUTHARK_PRIMTYPES[expected_type].type_name, dims, bin_primtype.type_name);
  }

  if (elem_size != bin_primtype.size) {
    panic(1, "binary-input: The RTS expected type %s to use %i bytes per element, but the call to `read_array` tells me to use %i bytes per element.\n",
          bin_primtype.type_name, bin_primtype.size, elem_size);
  }

  uint64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    uint64_t bin_shape;
    ret = IS_BIG_ENDIAN ? read_be_8byte(&bin_shape) : read_le_8byte(&bin_shape);
    if (ret != 0) { panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i); }
    elem_count *= bin_shape;
    shape[i] = (int64_t) bin_shape;
  }

  void* tmp = realloc(*data, elem_count * elem_size);
  if (tmp == NULL) {
    panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  size_t num_elems_read = fread(*data, elem_size, elem_count, stdin);
  if (num_elems_read != elem_count) {
    panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    char* elems = (char*) *data;
    for (uint64_t i=0; i<elem_count; i++) {
      char* elem = elems+(i*elem_size);
      for (int j=0; j<elem_size/2; j++) {
        char head = elem[j];
        int tail_index = elem_size-1-j;
        elem[j] = elem[tail_index];
        elem[tail_index] = head;
      }
    }
  }

  return 0;
}

static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
int parse_options(int argc, char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"memory-reporting", no_argument,
                                            NULL, 3}, {"entry-point",
                                                       required_argument, NULL,
                                                       4}, {"platform",
                                                            required_argument,
                                                            NULL, 5}, {"device",
                                                                       required_argument,
                                                                       NULL, 6},
                                           {"synchronous", no_argument, NULL,
                                            7}, {"group-size",
                                                 required_argument, NULL, 8},
                                           {"num-groups", required_argument,
                                            NULL, 9}, {"dump-opencl",
                                                       required_argument, NULL,
                                                       10}, {"load-opencl",
                                                             required_argument,
                                                             NULL, 11}, {0, 0,
                                                                         0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:me:p:d:s", long_options,
                             NULL)) != -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s\n", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s\n", optarg);
        }
        if (ch == 3 || ch == 'm')
            detail_memory = 1;
        if (ch == 4 || ch == 'e')
            entry_point = optarg;
        if (ch == 5 || ch == 'p')
            set_preferred_platform(optarg);
        if (ch == 6 || ch == 'd')
            set_preferred_device(optarg);
        if (ch == 7 || ch == 's')
            debugging = 1;
        if (ch == 8)
            cl_group_size = atoi(optarg);
        if (ch == 9)
            cl_num_groups = atoi(optarg);
        if (ch == 10)
            cl_dump_program_to = optarg;
        if (ch == 11)
            cl_load_program_from = optarg;
        if (ch == ':')
            panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?')
            panic(-1, "Unknown option %s\n", argv[optind - 1]);
    }
    return optind;
}
void entry_main()

{
    int64_t t_start, t_end;
    int time_runs;
    int64_t xss_mem_sizze_1047;
    struct memblock xss_mem_1048;
    
    xss_mem_1048.references = NULL;
    memblock_alloc(&xss_mem_1048, 0);
    
    int32_t sizze_285;
    int32_t sizze_286;
    struct tuple_int32_t_device_mem_int32_t main_ret_1224;
    
    {
        int64_t shape[2];
        
        if (read_array(FUTHARK_INT32, sizeof(int32_t), read_str_int32, "i32",
                       (void **) &xss_mem_1048.mem, shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[][]i32");
        sizze_285 = shape[0];
        sizze_286 = shape[1];
        xss_mem_sizze_1047 = sizeof(int32_t) * shape[0] * shape[1];
        xss_mem_1048.size = xss_mem_sizze_1047;
    }
    
    struct memblock_device xss_mem_device_1225;
    
    xss_mem_device_1225.references = NULL;
    memblock_alloc_device(&xss_mem_device_1225, xss_mem_1048.size);
    if (xss_mem_1048.size > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            xss_mem_device_1225.mem, CL_TRUE, 0,
                                            xss_mem_1048.size,
                                            xss_mem_1048.mem + 0, 0, NULL,
                                            NULL));
    
    int32_t out_memsizze_1117;
    struct memblock out_mem_1116;
    
    out_mem_1116.references = NULL;
    
    int32_t out_arrsizze_1118;
    
    if (perform_warmup) {
        time_runs = 0;
        t_start = get_wall_time();
        main_ret_1224 = futhark_main(xss_mem_sizze_1047, xss_mem_device_1225,
                                     sizze_285, sizze_286);
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        memblock_unref_device(&main_ret_1224.elem_1);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        if (run == num_runs - 1)
            detail_timing = 1;
        t_start = get_wall_time();
        main_ret_1224 = futhark_main(xss_mem_sizze_1047, xss_mem_device_1225,
                                     sizze_285, sizze_286);
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        if (run < num_runs - 1) {
            memblock_unref_device(&main_ret_1224.elem_1);
        }
    }
    memblock_unref(&xss_mem_1048);
    out_memsizze_1117 = main_ret_1224.elem_0;
    memblock_alloc(&out_mem_1116, main_ret_1224.elem_1.size);
    if (main_ret_1224.elem_1.size > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                           main_ret_1224.elem_1.mem, CL_TRUE, 0,
                                           main_ret_1224.elem_1.size,
                                           out_mem_1116.mem + 0, 0, NULL,
                                           NULL));
    out_arrsizze_1118 = main_ret_1224.elem_2;
    if (out_arrsizze_1118 * 1 == 0)
        printf("empty(%s)", "i32");
    else {
        int print_i_1226;
        
        putchar('[');
        for (print_i_1226 = 0; print_i_1226 < out_arrsizze_1118;
             print_i_1226++) {
            int32_t *print_elem_1227 = (int32_t *) out_mem_1116.mem +
                    print_i_1226 * 1;
            
            fprintf(stdout, "%di32", *print_elem_1227);
            if (print_i_1226 != out_arrsizze_1118 - 1)
                printf(", ");
        }
        putchar(']');
    }
    printf("\n");
    memblock_unref_device(&main_ret_1224.elem_1);
}
typedef void entry_point_fun();
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                entry_main}};
    int parsed_options = parse_options(argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    setup_opencl_and_load_kernels();
    
    int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
    entry_point_fun *entry_point_fun = NULL;
    
    for (int i = 0; i < num_entry_points; i++) {
        if (strcmp(entry_points[i].name, entry_point) == 0) {
            entry_point_fun = entry_points[i].fun;
            break;
        }
    }
    if (entry_point_fun == NULL) {
        fprintf(stderr,
                "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                entry_point);
        for (int i = 0; i < num_entry_points; i++)
            fprintf(stderr, "%s\n", entry_points[i].name);
        return 1;
    }
    entry_point_fun();
    if (runtime_file != NULL)
        fclose(runtime_file);
    
    int total_runtime = 0;
    int total_runs = 0;
    
    if (debugging) {
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_i32                  executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                fut_kernel_map_transpose_i32runs,
                (long) fut_kernel_map_transpose_i32total_runtime /
                (fut_kernel_map_transpose_i32runs !=
                 0 ? fut_kernel_map_transpose_i32runs : 1),
                (long) fut_kernel_map_transpose_i32total_runtime);
        total_runtime += fut_kernel_map_transpose_i32total_runtime;
        total_runs += fut_kernel_map_transpose_i32runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_lowheight_i32        executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                fut_kernel_map_transpose_lowheight_i32runs,
                (long) fut_kernel_map_transpose_lowheight_i32total_runtime /
                (fut_kernel_map_transpose_lowheight_i32runs !=
                 0 ? fut_kernel_map_transpose_lowheight_i32runs : 1),
                (long) fut_kernel_map_transpose_lowheight_i32total_runtime);
        total_runtime += fut_kernel_map_transpose_lowheight_i32total_runtime;
        total_runs += fut_kernel_map_transpose_lowheight_i32runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_lowwidth_i32         executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                fut_kernel_map_transpose_lowwidth_i32runs,
                (long) fut_kernel_map_transpose_lowwidth_i32total_runtime /
                (fut_kernel_map_transpose_lowwidth_i32runs !=
                 0 ? fut_kernel_map_transpose_lowwidth_i32runs : 1),
                (long) fut_kernel_map_transpose_lowwidth_i32total_runtime);
        total_runtime += fut_kernel_map_transpose_lowwidth_i32total_runtime;
        total_runs += fut_kernel_map_transpose_lowwidth_i32runs;
        fprintf(stderr,
                "Kernel map_intra_group_kernel_1008                   executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_intra_group_kernel_1008runs,
                (long) map_intra_group_kernel_1008total_runtime /
                (map_intra_group_kernel_1008runs !=
                 0 ? map_intra_group_kernel_1008runs : 1),
                (long) map_intra_group_kernel_1008total_runtime);
        total_runtime += map_intra_group_kernel_1008total_runtime;
        total_runs += map_intra_group_kernel_1008runs;
        fprintf(stderr,
                "Kernel map_kernel_971                                executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_971runs, (long) map_kernel_971total_runtime /
                (map_kernel_971runs != 0 ? map_kernel_971runs : 1),
                (long) map_kernel_971total_runtime);
        total_runtime += map_kernel_971total_runtime;
        total_runs += map_kernel_971runs;
        fprintf(stderr,
                "Kernel segmented_redomap__large_comm_many_kernel_418 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                segmented_redomap__large_comm_many_kernel_418runs,
                (long) segmented_redomap__large_comm_many_kernel_418total_runtime /
                (segmented_redomap__large_comm_many_kernel_418runs !=
                 0 ? segmented_redomap__large_comm_many_kernel_418runs : 1),
                (long) segmented_redomap__large_comm_many_kernel_418total_runtime);
        total_runtime +=
            segmented_redomap__large_comm_many_kernel_418total_runtime;
        total_runs += segmented_redomap__large_comm_many_kernel_418runs;
        fprintf(stderr,
                "Kernel segmented_redomap__large_comm_one_kernel_360  executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                segmented_redomap__large_comm_one_kernel_360runs,
                (long) segmented_redomap__large_comm_one_kernel_360total_runtime /
                (segmented_redomap__large_comm_one_kernel_360runs !=
                 0 ? segmented_redomap__large_comm_one_kernel_360runs : 1),
                (long) segmented_redomap__large_comm_one_kernel_360total_runtime);
        total_runtime +=
            segmented_redomap__large_comm_one_kernel_360total_runtime;
        total_runs += segmented_redomap__large_comm_one_kernel_360runs;
        fprintf(stderr,
                "Kernel segmented_redomap__large_comm_one_kernel_468  executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                segmented_redomap__large_comm_one_kernel_468runs,
                (long) segmented_redomap__large_comm_one_kernel_468total_runtime /
                (segmented_redomap__large_comm_one_kernel_468runs !=
                 0 ? segmented_redomap__large_comm_one_kernel_468runs : 1),
                (long) segmented_redomap__large_comm_one_kernel_468total_runtime);
        total_runtime +=
            segmented_redomap__large_comm_one_kernel_468total_runtime;
        total_runs += segmented_redomap__large_comm_one_kernel_468runs;
        fprintf(stderr,
                "Kernel segmented_redomap__small_comm_kernel_498      executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                segmented_redomap__small_comm_kernel_498runs,
                (long) segmented_redomap__small_comm_kernel_498total_runtime /
                (segmented_redomap__small_comm_kernel_498runs !=
                 0 ? segmented_redomap__small_comm_kernel_498runs : 1),
                (long) segmented_redomap__small_comm_kernel_498total_runtime);
        total_runtime += segmented_redomap__small_comm_kernel_498total_runtime;
        total_runs += segmented_redomap__small_comm_kernel_498runs;
        fprintf(stderr,
                "Kernel segmented_redomap__small_comm_kernel_651      executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                segmented_redomap__small_comm_kernel_651runs,
                (long) segmented_redomap__small_comm_kernel_651total_runtime /
                (segmented_redomap__small_comm_kernel_651runs !=
                 0 ? segmented_redomap__small_comm_kernel_651runs : 1),
                (long) segmented_redomap__small_comm_kernel_651total_runtime);
        total_runtime += segmented_redomap__small_comm_kernel_651total_runtime;
        total_runs += segmented_redomap__small_comm_kernel_651runs;
    }
    if (debugging)
        fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                total_runs, total_runtime);
    if (detail_memory) {
        fprintf(stderr, "Peak memory usage for space 'device': %ld bytes.\n",
                peak_mem_usage_device);
        fprintf(stderr, "Peak memory usage for space 'local': %ld bytes.\n",
                peak_mem_usage_local);
        fprintf(stderr, "Peak memory usage for default space: %ld bytes.\n",
                peak_mem_usage_default);
    }
    return 0;
}
