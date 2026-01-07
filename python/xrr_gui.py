"""
XRR FFT Thickness Analysis - Main GUI Application

This script provides a GUI interface for XRR FFT thickness analysis
and can trigger Igor Pro for advanced plotting.
"""

import sys
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xrr_fft_core import (
    analyze_xrr, XRRParameters, WindowType, detect_critical_angle
)
from igor_interface import IgorProInterface, XRRIgorBridge


class XRRFFTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XRR FFT Thickness Analysis")
        self.root.geometry("1200x800")
        
        # Data storage
        self.two_theta = None
        self.intensity = None
        self.result = None
        
        # Igor interface
        self.igor_bridge = XRRIgorBridge()
        
        self._create_widgets()
    
    def _create_widgets(self):
        # Main frames
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ==== Data Selection ====
        data_frame = ttk.Labelframe(left_frame, text="Data Selection", padding="10")
        data_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(data_frame, text="Load Data File", command=self.load_data).pack(fill=tk.X)
        self.file_label = ttk.Label(data_frame, text="No file loaded")
        self.file_label.pack(fill=tk.X)
        
        # ==== Parameters ====
        param_frame = ttk.Labelframe(left_frame, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=5)
        
        # Wavelength
        ttk.Label(param_frame, text="X-ray Wavelength (nm):").pack(anchor=tk.W)
        self.wavelength_var = tk.DoubleVar(value=0.154)
        ttk.Entry(param_frame, textvariable=self.wavelength_var).pack(fill=tk.X)
        
        # Critical Angle
        ttk.Label(param_frame, text="Critical Angle (deg):").pack(anchor=tk.W)
        self.crit_angle_var = tk.DoubleVar(value=0.6)
        ttk.Entry(param_frame, textvariable=self.crit_angle_var).pack(fill=tk.X)
        ttk.Button(param_frame, text="Auto Detect", command=self.auto_detect_critical_angle).pack(fill=tk.X)
        
        # Interpolation Points
        ttk.Label(param_frame, text="Interpolation Points:").pack(anchor=tk.W)
        self.interp_pts_var = tk.IntVar(value=1000)
        ttk.Entry(param_frame, textvariable=self.interp_pts_var).pack(fill=tk.X)
        
        # ==== FFT Options ====
        fft_frame = ttk.Labelframe(left_frame, text="FFT Options", padding="10")
        fft_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(fft_frame, text="Zero Padding Factor:").pack(anchor=tk.W)
        self.zero_pad_var = tk.IntVar(value=10)
        ttk.Entry(fft_frame, textvariable=self.zero_pad_var).pack(fill=tk.X)
        
        ttk.Label(fft_frame, text="Window Function:").pack(anchor=tk.W)
        self.window_var = tk.StringVar(value="Hamming")
        window_combo = ttk.Combobox(fft_frame, textvariable=self.window_var,
                                    values=["None", "Hanning", "Hamming", "Flat-top"])
        window_combo.pack(fill=tk.X)
        
        # ==== Buttons ====
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Analyze", command=self.run_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Export to Igor", command=self.export_to_igor).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Save Report", command=self.save_report).pack(fill=tk.X, pady=2)
        
        # ==== Results ====
        result_frame = ttk.Labelframe(left_frame, text="Results", padding="10")
        result_frame.pack(fill=tk.X, pady=5)
        
        self.result_text = tk.Text(result_frame, height=8, width=35)
        self.result_text.pack(fill=tk.X)
        self.result_text.insert(tk.END, "No analysis run yet")
        
        # ==== Plot Area ====
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        filepath = filedialog.askopenfilename(
            title="Select XRR Data File",
            filetypes=[("Text files", "*.txt *.dat *.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        try:
            data = np.loadtxt(filepath)
            if data.shape[1] >= 2:
                self.two_theta = data[:, 0]
                self.intensity = data[:, 1]
                self.file_label.config(text=os.path.basename(filepath))
                self._plot_raw_data()
            else:
                messagebox.showerror("Error", "Data file must have at least 2 columns")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def _plot_raw_data(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.semilogy(self.two_theta, self.intensity, 'b-', label='Raw Data')
        ax.axvline(self.crit_angle_var.get(), color='r', linestyle='--', label='Critical Angle')
        ax.set_xlabel('2\u03b8 (deg)')
        ax.set_ylabel('Intensity (counts)')
        ax.set_title('Raw XRR Data')
        ax.legend()
        self.canvas.draw()
    
    def auto_detect_critical_angle(self):
        if self.two_theta is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        crit_angle = detect_critical_angle(self.two_theta, self.intensity)
        self.crit_angle_var.set(round(crit_angle, 3))
        self._plot_raw_data()
    
    def _get_window_type(self):
        window_map = {
            "None": WindowType.NONE,
            "Hanning": WindowType.HANNING,
            "Hamming": WindowType.HAMMING,
            "Flat-top": WindowType.FLATTOP
        }
        return window_map.get(self.window_var.get(), WindowType.HAMMING)
    
    def run_analysis(self):
        if self.two_theta is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        try:
            params = XRRParameters(
                wavelength=self.wavelength_var.get(),
                critical_angle=self.crit_angle_var.get(),
                interpolation_points=self.interp_pts_var.get(),
                zero_padding_factor=self.zero_pad_var.get(),
                window_type=self._get_window_type()
            )
            
            self.result = analyze_xrr(self.two_theta, self.intensity, params)
            self._plot_results()
            self._update_results_text()
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")

    def _plot_results(self):
        self.fig.clear()
        
        # Subplot 1: Raw data
        ax1 = self.fig.add_subplot(221)
        ax1.semilogy(self.two_theta, self.intensity, 'b-', label='Raw Data')
        ax1.axvline(self.crit_angle_var.get(), color='r', linestyle='--', label='Critical Angle')
        ax1.set_xlabel('2\u03b8 (deg)')
        ax1.set_ylabel('Intensity (counts)')
        ax1.set_title('Raw XRR Data')
        ax1.legend(fontsize=8)
        
        # Subplot 2: Fresnel corrected
        ax2 = self.fig.add_subplot(222)
        mask = ~np.isnan(self.result.q_space)
        ax2.plot(self.result.q_space[mask], self.result.intensity_corrected[mask], 'g-')
        ax2.set_xlabel('Q (nm^{-1})')
        ax2.set_ylabel('Fresnel Corrected Intensity')
        ax2.set_title('Fresnel Corrected Data')
        
        # Subplot 3: FFT Spectrum
        ax3 = self.fig.add_subplot(223)
        fft = self.result.fft_result
        
        # Plot data points
        ax3.plot(fft.freq, fft.amplitude, 'o-', color='gray', markersize=2, label='Data')
        
        if self.result.fit_result is not None:
            fit = self.result.fit_result
            ax3.plot(fit.x_fit, fit.y_fit, '-', color='orange', lw=2, label='Fit')
            ax3.plot(fit.x_fit, fit.y_gauss1, '--', color='blue', label='Layer 1')
            ax3.plot(fit.x_fit, fit.y_gauss2, '--', color='purple', label='Layer 2')
            ax3.plot(fit.x_fit, fit.y_gauss3, '--', color='teal', label='Total')
            ax3.plot(fit.x_fit, fit.y_noise, '--', color='brown', label='1/f Noise')
        
        ax3.set_xlim(0, 35)
        ax3.set_ylim(0, 0.45)
        ax3.set_xlabel('Thickness (nm)')
        ax3.set_ylabel('Normalized FFT Amplitude')
        ax3.set_title('FFT Spectrum')
        ax3.legend(fontsize=7)
        
        # Subplot 4: Residuals
        ax4 = self.fig.add_subplot(224)
        if self.result.fit_result is not None:
            fit = self.result.fit_result
            # Interpolate fit to data points
            from scipy import interpolate
            fit_interp = interpolate.interp1d(fit.x_fit, fit.y_fit, 
                                              bounds_error=False, fill_value=0)
            mask = (fft.freq > 1.1) & (fft.freq < 35)
            residuals = fft.amplitude[mask] - fit_interp(fft.freq[mask])
            ax4.plot(fft.freq[mask], residuals, 'k-', lw=0.5)
            ax4.axhline(0, color='r', linestyle='--')
            ax4.set_xlabel('Thickness (nm)')
            ax4.set_ylabel('Residual')
            ax4.set_title(f'Residuals (R^2={fit.r_squared:.4f})')
        else:
            ax4.text(0.5, 0.5, 'No fit available', transform=ax4.transAxes,
                    ha='center', va='center')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _update_results_text(self):
        self.result_text.delete(1.0, tk.END)
        
        if self.result.fit_result is None:
            self.result_text.insert(tk.END, "Fitting failed\n")
            return
        
        fit = self.result.fit_result
        text = f"""Thickness Analysis Results
=========================

Top layer (Layer 1): {fit.thickness_layer1:.2f} nm
Base layer (Layer 2): {fit.thickness_layer2:.2f} nm
Total thickness: {fit.thickness_total:.2f} nm

Fit quality (R^2): {fit.r_squared:.4f}
"""
        self.result_text.insert(tk.END, text)

    def export_to_igor(self):
        if self.result is None:
            messagebox.showwarning("Warning", "Please run analysis first")
            return
        
        try:
            wave_files = self.igor_bridge.export_xrr_results(self.result)
            commands = self.igor_bridge.interface.generate_igor_commands(wave_files)
            
            # Show commands in a dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Igor Pro Commands")
            dialog.geometry("500x300")
            
            ttk.Label(dialog, text="Copy these commands to Igor Pro command line:").pack(pady=10)
            
            text = tk.Text(dialog, height=15, width=60)
            text.pack(fill=tk.BOTH, expand=True, padx=10)
            text.insert(tk.END, commands)
            
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
            
            messagebox.showinfo("Success", f"Exported {len(wave_files)} waves to ITX files")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def save_report(self):
        if self.result is None or self.result.fit_result is None:
            messagebox.showwarning("Warning", "Please run analysis first")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if not filepath:
            return
        
        try:
            self.igor_bridge.save_thickness_report(self.result, filepath)
            
            # Also save the figure
            fig_path = filepath.replace('.txt', '.png')
            self.fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            
            messagebox.showinfo("Success", f"Report saved to {filepath}\nFigure saved to {fig_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {e}")


def main():
    root = tk.Tk()
    app = XRRFFTApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
