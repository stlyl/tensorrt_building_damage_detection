import torch
import numpy as np
import tensorrt as trt
import pycuda.autoinit  #负责数据初始化，内存管理，销毁等
import pycuda.driver as cuda  #GPU CPU之间的数据传输
from model.unet import Unet
import cv2


# 1、模型转为onnx
def toonnx(weight_path="/home/lyl/project/tensorrt_demo/model/best_epoch_weights.pth",output_path="/home/lyl/project/tensorrt_demo/model.onnx"):
    model = Unet(num_classes = 4, backbone="vgg")
    # 反序列化权重参数
    # model.load_state_dict(torch.load(weight_path),strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weight_path))#, map_location=device))
    model.eval()
    # 定义输入名称，list结构，可能有多个输入
    input_names = ['input']
    # 定义输出名称，list结构，可能有多个输出
    output_names = ['output']
    # 构造输入用以验证onnx模型的正确性
    input = torch.rand(1, 3, 256, 256)
    # 导出
    torch.onnx.export(model, input, output_path,
                            export_params=True,
                            opset_version=11,
                            do_constant_folding=True,
                            input_names=input_names,
                            output_names=output_names)

# 2、onnx转engine文件
def onnx2engine(onnx_path = "/home/lyl/project/tensorrt_demo/model.onnx",engine_path = "/home/lyl/project/tensorrt_demo/model.engine"):
    logger = trt.Logger(trt.Logger.WARNING)
    # 创建构建器builder
    builder = trt.Builder(logger)
    # 预创建网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # 加载onnx解析器
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        pass  # Error handling code here
    # builder配置
    config = builder.create_builder_config()
    # 分配显存作为工作区间，一般建议为显存一半的大小
    config.max_workspace_size = 2 << 30  # 1 Mi
    serialized_engine = builder.build_serialized_network(network, config)
    # 序列化生成engine文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("generate file success!")

# 3、使用engine文件推理
def deploy():
    # 加载和预处理图像数据
    image = cv2.imread('image.jpg')
    resized_image = cv2.resize(image, (256, 256))
    normalized_image = (resized_image / 255.0).astype(np.float32)
    # 创建输入Tensor
    input_shape = (1, 3, 224, 224)  # 假设输入Tensor的形状为(1, 3, 224, 224)
    input_tensor = np.zeros(input_shape, dtype=np.float32)
    # 将图像数据复制到输入Tensor中
    np.copyto(input_tensor, normalized_image.ravel())


    # 创建logger：日志记录器
    logger = trt.Logger(trt.Logger.WARNING)
    # 创建runtime并反序列化生成engine
    with open("/home/lyl/project/tensorrt_demo/model.engine", "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
    # 创建cuda流
    stream = cuda.Stream()
    # 创建context并进行推理
    with engine.create_execution_context() as context:
        # 分配CPU锁页内存和GPU显存
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output. 该数据等同于原始模型的输出数据
        return h_output
def deploy2():
    # 定义输入数据的大小
    INPUT_H = 256
    INPUT_W = 256
    INPUT_C = 3
    # 分配内存
    input_data = np.empty((1, INPUT_C, INPUT_H, INPUT_W), dtype=np.float32)
    output_data = np.empty((1, 4), dtype=np.float32)
    # 加载和预处理图像
    image = cv2.imread("/home/lyl/project/tensorrt_demo/crack000664.jpg")
    image = cv2.resize(image, (INPUT_W, INPUT_H))
    image = image.astype(np.float32)
    image = (image / 255.0)  # 归一化
    # 创建logger：日志记录器
    logger = trt.Logger(trt.Logger.WARNING)
    # 创建runtime并反序列化生成engine
    with open("/home/lyl/project/tensorrt_demo/model.engine", "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    stream = cuda.Stream()
    with engine.create_execution_context() as context:
        # 创建host buffer
        input_host_buffer = cuda.pagelocked_empty(trt.volume((256,256,3)), dtype=np.float32)
        # 将数据复制到host buffer中
        input_host_buffer[:] = image#.transpose((2, 0, 1))  # 转换为CHW格式
        # 创建device buffer
        input_device_buffer = cuda.mem_alloc(input_host_buffer.nbytes)
        output_device_buffer = cuda.mem_alloc(output_data.nbytes)
        # 将数据从host buffer复制到device buffer中
        cuda.memcpy_htod(input_device_buffer, input_host_buffer)
        # 执行推理
        context.execute(batch_size=1, bindings=[int(input_device_buffer), int(output_device_buffer)])
        # 获取输出数据
        cuda.memcpy_dtoh(output_data, output_device_buffer)
        # 后处理并保存输出图像
        # output_image = postprocess(output_data)  # 根据需要进行后处理
        # output_image = (output_image * 255.0).clip(0, 255).astype(np.uint8)  # 还原为0-255范围的像素值
        cv2.imwrite('/home/lyl/project/tensorrt_demo/output_image.jpg', output_data)

# # toonnx()
# # onnx2engine()
# deploy2()



