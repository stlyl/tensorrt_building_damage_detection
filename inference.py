import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from torchvision import transforms
CLASSES=('ignore','crack', 'spall', 'rebar')
PALETTE=[
        [0, 0, 0],[0, 0, 255], [255, 0, 0], [0, 255, 0]]#bgr

def out_to_rgb_np(out):
    CLASSES=('ignore','crack', 'spall', 'rebar')
    PALETTE=[
        [0, 0, 0],[0, 0, 255], [255, 0, 0], [0, 255, 0]]#bgr
    softmax_output = np.exp(out) / np.sum(np.exp(out), axis=0, keepdims=True)  # 使用np.exp和np.sum计算softmax概率分布
    out = np.argmax(softmax_output, axis=0).astype(float)  # 使用np.argmax获取预测结果的索引，并转换为浮点数类型
    # out = out.transpose((1,2,0))
    palette = np.array(PALETTE)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[out == label, :] = color
    return color_seg
# 加载TensorRT引擎
def load_engine(engine_path):
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

# 创建TensorRT上下文
def create_context(engine):
    context = engine.create_execution_context()
    return context

# 加载图像并进行预处理
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))  # 根据模型需求调整大小
    image = np.float32(image) / 255.0      # 归一化到0-1范围
    image = np.transpose(image, (2, 0, 1))  # 转换为CHW格式
    return image

# 执行推理
def perform_inference(context, input_data):
    input_bindings = []
    output_bindings = []
    stream = cuda.Stream()

    # 分配输入和输出内存
    for binding in range(context.engine.num_bindings):
        size = trt.volume(context.engine.get_binding_shape(binding)) * \
               context.engine.max_batch_size
        dtype = trt.nptype(context.engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        if context.engine.binding_is_input(binding):
            input_bindings.append({'host': host_mem, 'device': device_mem})
        else:
            output_bindings.append({'host': host_mem, 'device': device_mem})
        # input_bindings.append({'host': host_mem, 'device': device_mem})
        # output_bindings.append({'host': host_mem, 'device': device_mem})

    # 将输入数据传输到设备内存
    np.copyto(input_bindings[0]['host'], input_data.ravel())
    cuda.memcpy_htod_async(input_bindings[0]['device'], input_bindings[0]['host'], stream)

    # 执行推理
    context.execute_async_v2(bindings=[inp['device'] for inp in input_bindings] +
                                      [out['device'] for out in output_bindings],
                             stream_handle=stream.handle)

    # 等待推理完成
    stream.synchronize()

    # 将输出数据从设备内存复制到主机内存
    output_data = []
    for out in output_bindings:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        output_data.append(out['host'].reshape(context.engine.get_binding_shape(1)))

    # 返回输出数据
    return output_data

# 保存灰度图像
def save_grayscale_image(image_data, save_path):
    image_data = np.squeeze(image_data)   # 去除维度为1的维度
    
    image_data = out_to_rgb_np(image_data)
    
    # image_data = np.clip(image_data, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, image_data)

# 主函数
def main():
    engine_path = '/home/lyl/project/tensorrt_demo/model.engine'  # TensorRT引擎文件路径
    image_path = '/home/lyl/project/tensorrt_demo/rebar0004902.jpg'    # 输入RGB图像文件路径
    output_path = '/home/lyl/project/tensorrt_demo/output.jpg'  # 输出灰度图像保存路径

    # 加载TensorRT引擎
    engine = load_engine(engine_path)

    # 创建TensorRT上下文
    context = create_context(engine)

    # 加载图像并进行预处理
    input_data = preprocess_image(image_path)

    # 执行推理
    output_data = perform_inference(context, input_data)

    # 保存灰度图像
    save_grayscale_image(output_data, output_path)

if __name__ == '__main__':
    main()