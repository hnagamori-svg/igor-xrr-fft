"""
Python-Igor Pro Interface Module

This module provides communication between Python and Igor Pro
using the Igor Pro Socket Connection or file-based data exchange.
"""

import numpy as np
import os
import tempfile
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class IgorWave:
    """Represents an Igor Pro wave"""
    name: str
    data: np.ndarray
    x_start: float = 0.0
    x_delta: float = 1.0
    units: str = ""


class IgorProInterface:
    """Interface for communicating with Igor Pro"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.data_file = os.path.join(self.temp_dir, "igor_python_data.json")
        self.wave_dir = os.path.join(self.temp_dir, "igor_waves")
        os.makedirs(self.wave_dir, exist_ok=True)
    
    def save_wave(self, wave: IgorWave) -> str:
        """Save a wave to a file for Igor Pro to read"""
        filepath = os.path.join(self.wave_dir, f"{wave.name}.itx")
        self._write_itx(filepath, wave)
        return filepath
    
    def _write_itx(self, filepath: str, wave: IgorWave):
        """Write wave data in Igor Text (ITX) format"""
        with open(filepath, 'w') as f:
            f.write("IGOR\n")
            f.write(f"WAVES/{wave.name}\n")
            f.write("BEGIN\n")
            for val in wave.data:
                f.write(f"  {val:.15g}\n")
            f.write("END\n")
            if wave.x_delta != 1.0 or wave.x_start != 0.0:
                f.write(f"X SetScale/x= {wave.x_start},{wave.x_delta}, \"z_{wave.units}\", {wave.name}\n")
    
    def load_wave(self, filepath: str) -> IgorWave:
        """Load a wave from an ITX file"""
        return self._read_itx(filepath)
    
    def _read_itx(self, filepath: str) -> IgorWave:
        """Read wave data from Igor Text (ITX) format"""
        with open(filepath, 'r') as f:
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
        """Save analysis results to JSON file"""
        filepath = os.path.join(self.temp_dir, filename)
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(results, f, default=convert, indent=2)
        
        return filepath
    
    def generate_igor_commands(self, wave_files: List[str]) -> str:
        """Generate Igor commands to load waves"""
        commands = []
        for fp in wave_files:
            fp_escaped = fp.replace("\\", ":")
            commands.append(f'LoadWave/O/Q "{fp_escaped}"')
        return "\n".join(commands)


class XRRIgorBridge:
    """Bridge between XRR analysis and Igor Pro"""
    
    def __init__(self, interface: Optional[IgorProInterface] = None):
        self.interface = interface or IgorProInterface()
    
    def export_xrr_results(self, result, prefix: str = "XRR") -> List[str]:
        """Export XRR analysis results as Igor waves"""
        wave_files = []
        
        # Original data
        wave_files.append(self.interface.save_wave(
            IgorWave(name=f"{prefix}_2theta", data=result.two_theta_raw, units="deg")))
        wave_files.append(self.interface.save_wave(
            IgorWave(name=f"{prefix}_intensity", data=result.intensity_raw, units="counts")))
        
        # FFT results
        if result.fft_result is not None:
            wave_files.append(self.interface.save_wave(
                IgorWave(name=f"{prefix}_fft_thickness", data=result.fft_result.freq, units="nm")))
            wave_files.append(self.interface.save_wave(
                IgorWave(name=f"{prefix}_fft_amp", data=result.fft_result.amplitude, units="")))
        
        # Fit results
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
        """Get thickness summary from analysis results"""
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
        """Save thickness report to text file"""
        if result.fit_result is None:
            return
        
        fit = result.fit_result
        with open(filepath, 'w') as f:
            f.write("XRR FFT Thickness Analysis Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Top layer (Layer 1): {fit.thickness_layer1:.2f} nm\n")
            f.write(f"Base layer (Layer 2): {fit.thickness_layer2:.2f} nm\n")
            f.write(f"Total thickness: {fit.thickness_total:.2f} nm\n\n")
            f.write(f"Fit quality (R^2): {fit.r_squared:.4f}\n")
            f.write(f"Residual std: {fit.residual_std:.4e}\n")
