#Presented by KeJi
#Date: 2025-12-30

import platform
import os
import subprocess
import psutil

# CPU型号数据库
CPU_DATABASE = {
    # Intel 12代 Alder Lake
    197: {
        "family": "Intel 12th Gen Alder Lake",
        "models": ["i9-12900K", "i9-12900", "i7-12700K", "i7-12700", "i5-12600K", "i5-12600"],
        "description": "Intel 12代酷睿处理器，采用混合架构（P-core + E-core）"
    },
    # Intel 11代 Rocket Lake
    167: {
        "family": "Intel 11th Gen Rocket Lake",
        "models": ["i9-11900K", "i7-11700K", "i5-11600K"],
        "description": "Intel 11代酷睿处理器"
    },
    # Intel 10代 Comet Lake
    165: {
        "family": "Intel 10th Gen Comet Lake",
        "models": ["i9-10900K", "i7-10700K", "i5-10600K"],
        "description": "Intel 10代酷睿处理器"
    },
    # AMD Ryzen 5000
    80: {
        "family": "AMD Ryzen 5000 Series",
        "models": ["Ryzen 9 5950X", "Ryzen 7 5800X", "Ryzen 5 5600X"],
        "description": "AMD Zen 3架构处理器"
    }
}

def get_cpu_model():
    """尝试获取具体CPU型号"""
    try:
        # 方法1: 使用platform模块
        if platform.system() == "Windows":
            # 尝试使用wmic（如果可用）
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name", "/value"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Name=' in line:
                            return line.split('Name=')[1].strip()
            except:
                pass
            
            # 方法2: 使用PowerShell
            try:
                result = subprocess.run(
                    ["powershell", "-Command", "Get-WmiObject -Class Win32_Processor | Select-Object -ExpandProperty Name"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except:
                pass
            
            # 方法3: 读取环境变量
            processor = os.environ.get("PROCESSOR_IDENTIFIER", "")
            if processor:
                return processor
        
        # Linux/macOS
        elif platform.system() in ["Linux", "Darwin"]:
            try:
                # 读取/proc/cpuinfo
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            return line.split(':')[1].strip()
            except:
                pass
        
        # 方法4: 使用platform模块
        return platform.processor()
        
    except Exception as e:
        return f"未知 ({str(e)})"

def decode_cpu_model(family, model, stepping):
    """解码CPU型号信息"""
    info = {}
    
    # 查找数据库
    if model in CPU_DATABASE:
        db_info = CPU_DATABASE[model]
        info['family'] = db_info['family']
        info['possible_models'] = db_info['models']
        info['description'] = db_info['description']
    else:
        # 通用识别
        if family == 6:
            if model >= 160:
                info['family'] = "Intel 10代或更新"
            else:
                info['family'] = "Intel Core 系列"
        elif family == 0x17:  # AMD
            info['family'] = "AMD Ryzen 系列"
        else:
            info['family'] = f"未知 (Family {family})"
        
        info['possible_models'] = ["需要进一步检测"]
        info['description'] = "无法识别具体型号"
    
    # 步进信息
    info['stepping'] = stepping
    
    return info

def get_cpu_info():
    """获取CPU信息"""
    print("=" * 60)
    print("CPU 信息")
    print("=" * 60)
    
    # 基础信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    
    # 尝试获取具体型号
    cpu_model = get_cpu_model()
    print(f"\n处理器型号: {cpu_model}")
    
    # 从环境变量获取详细信息
    if platform.system() == "Windows":
        processor = os.environ.get("PROCESSOR_IDENTIFIER", "未知")
        arch = os.environ.get("PROCESSOR_ARCHITECTURE", "未知")
        num_cpus = os.environ.get("NUMBER_OF_PROCESSORS", "未知")
        
        print(f"处理器标识: {processor}")
        print(f"处理器架构: {arch}")
        print(f"处理器数量: {num_cpus}")
        
        # 解析Family/Model/Stepping
        if "Family" in processor:
            try:
                parts = processor.split()
                family = None
                model = None
                stepping = None
                
                for i, part in enumerate(parts):
                    if part == "Family" and i+1 < len(parts):
                        family = int(parts[i+1])
                    elif part == "Model" and i+1 < len(parts):
                        model = int(parts[i+1])
                    elif part == "Stepping" and i+1 < len(parts):
                        stepping = int(parts[i+1])
                
                if family and model:
                    decoded = decode_cpu_model(family, model, stepping)
                    print(f"\n架构家族: {decoded['family']}")
                    print(f"可能型号: {', '.join(decoded['possible_models'])}")
                    print(f"描述: {decoded['description']}")
            except:
                pass
    
    # 使用psutil获取详细信息
    try:
        print(f"\n物理核心数: {psutil.cpu_count(logical=False)}")
        print(f"逻辑核心数: {psutil.cpu_count(logical=True)}")
        
        # CPU频率
        freq = psutil.cpu_freq()
        if freq:
            print(f"当前频率: {freq.current:.2f} MHz")
            print(f"最大频率: {freq.max:.2f} MHz")
            print(f"最小频率: {freq.min:.2f} MHz")
        
        # CPU使用率
        print(f"\n当前CPU使用率: {psutil.cpu_percent(interval=1)}%")
        
    except Exception as e:
        print(f"\n无法获取详细CPU信息: {e}")

def get_memory_info():
    """获取内存信息"""
    print("\n" + "=" * 60)
    print("内存 信息")
    print("=" * 60)
    
    try:
        # 虚拟内存
        vm = psutil.virtual_memory()
        print(f"总内存: {vm.total / (1024**3):.2f} GB")
        print(f"可用内存: {vm.available / (1024**3):.2f} GB")
        print(f"已用内存: {vm.used / (1024**3):.2f} GB")
        print(f"内存使用率: {vm.percent}%")
        
        # 交换内存
        swap = psutil.swap_memory()
        if swap.total > 0:
            print(f"\n交换内存总大小: {swap.total / (1024**3):.2f} GB")
            print(f"交换内存使用率: {swap.percent}%")
        else:
            print("\n交换内存: 未启用")
            
    except Exception as e:
        print(f"无法获取内存信息: {e}")

def main():
    """主函数"""
    print("系统硬件信息检测")
    print("=" * 60)
    
    get_cpu_info()
    get_memory_info()
    
    print("\n" + "=" * 60)
    print("检测完成")
    print("=" * 60)
    
    # 提示安装cpuinfo库
    print("\n提示：安装 py-cpuinfo 库可获取更详细信息:")
    print("pip install py-cpuinfo")

if __name__ == "__main__":
    main()
