#!/usr/bin/env python3
"""
系统状态检查脚本

检查qm_final3系统的完整状态，包括：
1. 代码完整性检查
2. 依赖项检查
3. 配置文件验证
4. 系统性能测试
5. 部署准备状态
"""

import os
import sys
import importlib
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemChecker:
    """系统状态检查器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            "overall_status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "summary": {},
            "recommendations": []
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        logger.info("开始系统状态检查...")
        
        # 1. 代码结构检查
        self._check_code_structure()
        
        # 2. 依赖项检查
        self._check_dependencies()
        
        # 3. 配置文件检查
        self._check_configuration()
        
        # 4. 导入测试
        self._check_imports()
        
        # 5. 系统功能测试
        self._check_system_functionality()
        
        # 6. 性能基准测试
        self._check_performance()
        
        # 7. 部署准备检查
        self._check_deployment_readiness()
        
        # 生成总结
        self._generate_summary()
        
        logger.info("系统状态检查完成")
        return self.results
    
    def _check_code_structure(self):
        """检查代码结构"""
        logger.info("检查代码结构...")
        
        required_files = [
            "main.py",
            "requirements.txt",
            "DEPLOYMENT.md",
            ".gitignore",
            "layers/__init__.py",
            "layers/base_layer.py",
            "layers/input_layer.py",
            "layers/fusion_layer.py",
            "layers/mapping_layer.py",
            "layers/generation_layer.py",
            "layers/rendering_layer.py",
            "layers/therapy_layer.py",
            "core/utils.py",
            "core/__init__.py",
            "configs/six_layer_architecture.yaml",
            "performance_optimizer.py"
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        # 统计代码行数
        total_lines = 0
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except:
                pass
        
        self.results["checks"]["code_structure"] = {
            "status": "pass" if not missing_files else "warning",
            "existing_files": len(existing_files),
            "missing_files": missing_files,
            "total_python_files": len(python_files),
            "total_lines_of_code": total_lines,
            "completion_percentage": (len(existing_files) / len(required_files)) * 100
        }
        
        if missing_files:
            self.results["recommendations"].append({
                "type": "code_structure",
                "message": f"缺少以下文件: {', '.join(missing_files)}"
            })
    
    def _check_dependencies(self):
        """检查依赖项"""
        logger.info("检查依赖项...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.results["checks"]["dependencies"] = {
                "status": "fail",
                "message": "requirements.txt 文件不存在"
            }
            return
        
        # 读取requirements
        with open(requirements_file, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f.readlines() 
                          if line.strip() and not line.startswith('#')]
        
        # 检查核心依赖
        core_deps = ['numpy', 'torch', 'matplotlib', 'scipy', 'yaml']
        optional_deps = ['pygame', 'opencv-python', 'librosa', 'pyaudio']
        
        installed_deps = []
        missing_deps = []
        optional_missing = []
        
        for dep in core_deps:
            if self._is_package_installed(dep):
                installed_deps.append(dep)
            else:
                missing_deps.append(dep)
        
        for dep in optional_deps:
            if not self._is_package_installed(dep):
                optional_missing.append(dep)
        
        self.results["checks"]["dependencies"] = {
            "status": "pass" if not missing_deps else "fail",
            "total_requirements": len(requirements),
            "core_dependencies": {
                "installed": installed_deps,
                "missing": missing_deps
            },
            "optional_dependencies": {
                "missing": optional_missing
            }
        }
        
        if missing_deps:
            self.results["recommendations"].append({
                "type": "dependencies",
                "message": f"缺少核心依赖: {', '.join(missing_deps)}"
            })
    
    def _is_package_installed(self, package_name: str) -> bool:
        """检查包是否已安装"""
        try:
            importlib.import_module(package_name.replace('-', '_'))
            return True
        except ImportError:
            return False
    
    def _check_configuration(self):
        """检查配置文件"""
        logger.info("检查配置文件...")
        
        config_file = self.project_root / "configs/six_layer_architecture.yaml"
        
        if not config_file.exists():
            self.results["checks"]["configuration"] = {
                "status": "fail",
                "message": "配置文件不存在"
            }
            return
        
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查配置完整性
            required_sections = ['system', 'layers', 'logging']
            missing_sections = [s for s in required_sections if s not in config]
            
            layer_sections = ['input_layer', 'fusion_layer', 'mapping_layer', 
                            'generation_layer', 'rendering_layer', 'therapy_layer']
            
            if 'layers' in config:
                missing_layers = [l for l in layer_sections if l not in config['layers']]
            else:
                missing_layers = layer_sections
            
            self.results["checks"]["configuration"] = {
                "status": "pass" if not missing_sections and not missing_layers else "warning",
                "config_sections": list(config.keys()) if config else [],
                "missing_sections": missing_sections,
                "missing_layers": missing_layers,
                "has_system_config": 'system' in config,
                "has_layer_config": 'layers' in config
            }
            
        except Exception as e:
            self.results["checks"]["configuration"] = {
                "status": "fail",
                "message": f"配置文件解析错误: {str(e)}"
            }
    
    def _check_imports(self):
        """检查导入"""
        logger.info("检查导入...")
        
        import_tests = [
            ("layers", "__all__"),
            ("layers.base_layer", "LayerPipeline"),
            ("layers.input_layer", "InputLayer"),
            ("layers.fusion_layer", "FusionLayer"),
            ("layers.mapping_layer", "MappingLayer"),
            ("layers.generation_layer", "GenerationLayer"),
            ("layers.rendering_layer", "RenderingLayer"),
            ("layers.therapy_layer", "TherapyLayer"),
            ("core.utils", "setup_logging"),
            ("performance_optimizer", "PerformanceOptimizer")
        ]
        
        successful_imports = []
        failed_imports = []
        
        for module_name, class_name in import_tests:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    successful_imports.append(f"{module_name}.{class_name}")
                else:
                    failed_imports.append(f"{module_name}.{class_name} (attribute not found)")
            except Exception as e:
                failed_imports.append(f"{module_name}.{class_name} ({str(e)})")
        
        self.results["checks"]["imports"] = {
            "status": "pass" if not failed_imports else "fail",
            "successful_imports": successful_imports,
            "failed_imports": failed_imports,
            "success_rate": len(successful_imports) / len(import_tests) * 100
        }
        
        if failed_imports:
            self.results["recommendations"].append({
                "type": "imports",
                "message": f"导入失败: {', '.join(failed_imports)}"
            })
    
    def _check_system_functionality(self):
        """检查系统功能"""
        logger.info("检查系统功能...")
        
        try:
            # 尝试导入主系统类
            sys.path.insert(0, str(self.project_root))
            from main import QMFinal3System
            
            # 创建系统实例
            system = QMFinal3System()
            
            # 检查层数
            layer_count = len(system.layers)
            expected_layers = 6
            
            # 检查管道
            has_pipeline = system.pipeline is not None
            
            # 获取系统状态
            status = system.get_system_status()
            
            self.results["checks"]["system_functionality"] = {
                "status": "pass" if layer_count == expected_layers and has_pipeline else "warning",
                "layer_count": layer_count,
                "expected_layers": expected_layers,
                "has_pipeline": has_pipeline,
                "system_status": status["system_info"],
                "all_layers_enabled": all(layer.get("enabled", False) for layer in status["layer_statuses"])
            }
            
            # 清理
            system.shutdown() if hasattr(system, 'shutdown') else None
            
        except Exception as e:
            self.results["checks"]["system_functionality"] = {
                "status": "fail",
                "error": str(e)
            }
    
    def _check_performance(self):
        """检查性能"""
        logger.info("检查性能...")
        
        try:
            # 简单性能测试
            import time
            import psutil
            
            # CPU和内存使用
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # 磁盘空间
            disk_usage = psutil.disk_usage(str(self.project_root))
            disk_free_gb = disk_usage.free / (1024**3)
            
            self.results["checks"]["performance"] = {
                "status": "pass",
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory_percent,
                "disk_free_gb": disk_free_gb,
                "performance_acceptable": cpu_percent < 80 and memory_percent < 80 and disk_free_gb > 1.0
            }
            
        except Exception as e:
            self.results["checks"]["performance"] = {
                "status": "warning",
                "error": str(e)
            }
    
    def _check_deployment_readiness(self):
        """检查部署准备状态"""
        logger.info("检查部署准备状态...")
        
        deployment_files = [
            "DEPLOYMENT.md",
            "requirements.txt",
            ".gitignore",
            "main.py"
        ]
        
        git_ready = all((self.project_root / f).exists() for f in deployment_files)
        
        # 检查是否在git仓库中
        git_dir = self.project_root / ".git"
        in_git_repo = git_dir.exists()
        
        # 检查关键目录
        required_dirs = ["layers", "core", "configs"]
        dirs_exist = all((self.project_root / d).exists() for d in required_dirs)
        
        self.results["checks"]["deployment_readiness"] = {
            "status": "pass" if git_ready and dirs_exist else "warning",
            "deployment_files_present": git_ready,
            "required_directories_exist": dirs_exist,
            "in_git_repository": in_git_repo,
            "ready_for_github": git_ready and dirs_exist
        }
        
        if not git_ready:
            self.results["recommendations"].append({
                "type": "deployment",
                "message": "部署文件不完整，请检查DEPLOYMENT.md、requirements.txt等文件"
            })
    
    def _generate_summary(self):
        """生成总结"""
        checks = self.results["checks"]
        
        # 统计检查结果
        pass_count = sum(1 for check in checks.values() if check.get("status") == "pass")
        warning_count = sum(1 for check in checks.values() if check.get("status") == "warning")
        fail_count = sum(1 for check in checks.values() if check.get("status") == "fail")
        
        total_checks = len(checks)
        
        # 确定整体状态
        if fail_count > 0:
            overall_status = "fail"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "pass"
        
        self.results["overall_status"] = overall_status
        self.results["summary"] = {
            "total_checks": total_checks,
            "pass_count": pass_count,
            "warning_count": warning_count,
            "fail_count": fail_count,
            "success_rate": (pass_count / total_checks) * 100 if total_checks > 0 else 0,
            "ready_for_deployment": overall_status in ["pass", "warning"]
        }
    
    def print_report(self):
        """打印报告"""
        print("\n" + "="*50)
        print("qm_final3 系统状态检查报告")
        print("="*50)
        
        print(f"检查时间: {self.results['timestamp']}")
        print(f"整体状态: {self.results['overall_status'].upper()}")
        
        # 总结
        summary = self.results["summary"]
        print(f"\n总检查项目: {summary['total_checks']}")
        print(f"通过: {summary['pass_count']}")
        print(f"警告: {summary['warning_count']}")
        print(f"失败: {summary['fail_count']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        
        # 详细检查结果
        print("\n详细检查结果:")
        for check_name, check_result in self.results["checks"].items():
            status = check_result.get("status", "unknown")
            print(f"  {check_name}: {status.upper()}")
            
            if status == "fail" and "error" in check_result:
                print(f"    错误: {check_result['error']}")
            elif status == "warning" and "message" in check_result:
                print(f"    警告: {check_result['message']}")
        
        # 建议
        if self.results["recommendations"]:
            print("\n建议:")
            for i, rec in enumerate(self.results["recommendations"], 1):
                print(f"  {i}. [{rec['type']}] {rec['message']}")
        
        # 部署状态
        deployment_check = self.results["checks"].get("deployment_readiness", {})
        if deployment_check.get("ready_for_github"):
            print("\n✅ 系统已准备好推送到GitHub")
        else:
            print("\n⚠️  系统尚未完全准备好部署")
        
        print("\n" + "="*50)

def main():
    """主函数"""
    checker = SystemChecker()
    results = checker.run_all_checks()
    checker.print_report()
    
    # 保存结果到文件
    with open("system_check_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细报告已保存到: system_check_report.json")

if __name__ == "__main__":
    main()