# 创建文件: check_dna.py
import importlib

try:
    import dynamic_network_architectures

    print("dynamic_network_architectures可用模块:", dir(dynamic_network_architectures))

    # 查看子模块
    try:
        import dynamic_network_architectures.architectures

        print("\ndynamic_network_architectures.architectures模块:",
              dir(dynamic_network_architectures.architectures))

        # 检查UNet模块
        try:
            import dynamic_network_architectures.architectures.unet

            print("\nUNet模块内容:",
                  dir(dynamic_network_architectures.architectures.unet))

            # 检查是否有PlainConvUNet
            if hasattr(dynamic_network_architectures.architectures.unet, 'PlainConvUNet'):
                print("\n找到PlainConvUNet! 正确的导入路径是:")
                print("from dynamic_network_architectures.architectures.unet import PlainConvUNet")
        except ImportError:
            print("无法导入unet模块")
    except ImportError:
        print("无法导入architectures子模块")

except ImportError:
    print("dynamic_network_architectures包未安装。请运行:")
    print("pip install dynamic-network-architectures")