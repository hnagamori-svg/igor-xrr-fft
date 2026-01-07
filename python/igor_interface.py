"""
Python-Igor Pro 連携モジュール

PythonとIgor Pro間のデータ交換機能を提供。
ITXファイル形式でのWaveエクスポート対応。
"""

import numpy as np
import os
import tempfile
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class IgorWave:
    """Igor Pro Wave"""
    name: str
    data: np.ndarray
    x_start: float = 0.0
    x_delta: float = 1.0
    units: str = ""


class IgorProInterface:
    """Igor Pro連携インターフェース"""

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.data_file = os.path.join(self.temp_dir, "igor_python_data.json")
        self.wave_dir = os.path.join(self.temp_dir, "igor_waves")
        os.makedirs(self.wave_dir, exist_ok=True)

    def save_wave(self, wave: IgorWave) -> str:
        """WaveをITXファイルとして保存"""
        filepath = os.path.join(self.wave_dir, f"{wave.name}.itx")
        self._write_itx(filepath, wave)
        return filepath

    def _write_itx(self, filepath: str, wave: IgorWave):
        """ITX形式で書き込み"""
        with open(filepath, "w") as f:
            f.write("IGOR
")
            f.write(f"WAVES/{wave.name}
")
            f.write("BEGIN
")
            for val in wave.data:
                f.write(f"  {val:.15g}
")
            f.write("END
")
            if wave.x_delta != 1.0 or wave.x_start != 0.0:
                f.write(f"X SetScale/x= {wave.x_start},{wave.x_delta}, "z_{wave.units}", {wave.name}
")

    def load_wave(self, filepath: str) -> IgorWave:
        """ITXファイルからWaveを読み込み"""
        return self._read_itx(filepath)

    def _read_itx(self, filepath: str) -> IgorWave:
        """ITX形式を読み込み"""
        with open(filepath, "r") as f:
            lines = f.readlines()

        name = "wave"
        data = []
        in_data = False

        for line in lines:
            line = line.strip()
            if line.startswith("WAVES"):
                parts = line.split("/")
                if len(parts) > 1:
                    name = parts[1].strip()
            elif line == "BEGIN":
                in_data = True
            elif line == "END":
                in_data = False
            elif in_data:
                try:
                    data.append(float(line))
                except ValueError:
                    pass

        return IgorWave(name=name, data=np.array(data))

    def save_analysis_results(self, results: Dict[str, Any], filename: str = "xrr_results.json"):
        """解析結果をJSONで保存"""
        filepath = os.path.join(self.temp_dir, filename)

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            return obj

        with open(filepath, "w") as f:
            json.dump(results, f, default=convert, indent=2)

        return filepath

    def generate_igor_commands(self, wave_files: List[str]) -> str:
        """Wave読み込みコマンドを生成"""
        commands = []
        for fp in wave_files:
            fp_escaped = fp.replace("\", ":")
            commands.append(f"LoadWave/O/Q "{fp_escaped}"")
        return "
".join(commands)


class XRRIgorBridge:
    """XRR解析とIgor Pro間のブリッジ"""

    def __init__(self, interface: Optional[IgorProInterface] = None):
        self.interface = interface or IgorProInterface()

    def export_xrr_results(self, result, prefix: str = "XRR") -> List[str]:
        """XRR解析結果をIgor Waveとしてエクスポート"""
        wave_files = []

        wave_files.append(self.interface.save_wave(
            IgorWave(name=f"{prefix}_2theta", data=result.two_theta_raw, units="deg")))
        wave_files.append(self.interface.save_wave(
            IgorWave(name=f"{prefix}_intensity", data=result.intensity_raw, units="counts")))

        if result.fft_result is not None:
            wave_files.append(self.interface.save_wave(
                IgorWave(name=f"{prefix}_fft_thickness", data=result.fft_result.freq, units="nm")))
            wave_files.append(self.interface.save_wave(
                IgorWave(name=f"{prefix}_fft_amp", data=result.fft_result.amplitude, units="")))

        if result.fit_result is not None:
            fit = result.fit_result
            wave_files.append(self.interface.save_wave(
                IgorWave(name=f"{prefix}_fit_x", data=fit.x_fit, units="nm")))
            wave_files.append(self.interface.save_wave(
                IgorWave(name=f"{prefix}_fit_total", data=fit.y_fit, units="")))
            wave_files.append(self.interface.save_wave(
                IgorWave(name=f"{prefix}_gauss1", data=fit.y_gauss1, units="")))
            wave_files.append(self.interface.save_wave(
                IgorWave(name=f"{prefix}_gauss2", data=fit.y_gauss2, units="")))
            wave_files.append(self.interface.save_wave(
                IgorWave(name=f"{prefix}_gauss3", data=fit.y_gauss3, units="")))
            wave_files.append(self.interface.save_wave(
                IgorWave(name=f"{prefix}_noise", data=fit.y_noise, units="")))

        return wave_files

    def get_thickness_summary(self, result) -> Dict[str, float]:
        """膜厚サマリーを取得"""
        if result.fit_result is None:
            return {}

        fit = result.fit_result
        return {
            "layer1_nm": fit.thickness_layer1,
            "layer2_nm": fit.thickness_layer2,
            "total_nm": fit.thickness_total,
            "r_squared": fit.r_squared
        }

    def save_thickness_report(self, result, filepath: str):
        """膜厚レポートを保存"""
        if result.fit_result is None:
            return

        fit = result.fit_result
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("XRR FFT 膜厚解析結果
")
            f.write("=" * 40 + "

")
            f.write(f"上層 (層1): {fit.thickness_layer1:.2f} nm
")
            f.write(f"下層 (層2): {fit.thickness_layer2:.2f} nm
")
            f.write(f"合計膜厚: {fit.thickness_total:.2f} nm

")
            f.write(f"フィット品質 (R^2): {fit.r_squared:.4f}
")
            f.write(f"残差標準偏差: {fit.residual_std:.4e}
")
