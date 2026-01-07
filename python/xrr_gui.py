"""
XRR FFT 膜厚解析 - GUIアプリケーション

X線反射率データのFFT解析を行うGUIインターフェース。
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
        self.root.title("XRR FFT 膜厚解析")
        self.root.geometry("1200x800")
        
        # データ storage
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
        
        # ==== データ選択 ====
        data_frame = ttk.Labelframe(left_frame, text="データ選択", padding="10")
        data_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(data_frame, text="データ読み込み", command=self.load_data).pack(fill=tk.X)
        self.file_label = ttk.Label(data_frame, text="ファイル未選択")
        self.file_label.pack(fill=tk.X)
        
        # ==== Parameters ====
        param_frame = ttk.Labelframe(left_frame, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=5)
        
        # Wavelength
        ttk.Label(param_frame, text="X線波長 (nm):").pack(anchor=tk.W)
        self.wavelength_var = tk.DoubleVar(value=0.154)
        ttk.Entry(param_frame, textvariable=self.wavelength_var).pack(fill=tk.X)
        
        # Critical Angle
        ttk.Label(param_frame, text="臨界角 (度):").pack(anchor=tk.W)
        self.crit_angle_var = tk.DoubleVar(value=0.6)
        ttk.Entry(param_frame, textvariable=self.crit_angle_var).pack(fill=tk.X)
        ttk.Button(param_frame, text="自動検出", command=self.auto_detect_critical_angle).pack(fill=tk.X)
        
        # Interpolation Points
        ttk.Label(param_frame, text="補間点数:").pack(anchor=tk.W)
        self.interp_pts_var = tk.IntVar(value=1000)
        ttk.Entry(param_frame, textvariable=self.interp_pts_var).pack(fill=tk.X)
        
        # ==== FFTオプション ====
        fft_frame = ttk.Labelframe(left_frame, text="FFTオプション", padding="10")
        fft_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(fft_frame, text="ゼロパディング倍率:").pack(anchor=tk.W)
        self.zero_pad_var = tk.IntVar(value=10)
        ttk.Entry(fft_frame, textvariable=self.zero_pad_var).pack(fill=tk.X)
        
        ttk.Label(fft_frame, text="窓関数:").pack(anchor=tk.W)
        self.window_var = tk.StringVar(value="Hamming")
        window_combo = ttk.Combobox(fft_frame, textvariable=self.window_var,
                                    values=["None", "Hanning", "Hamming", "Flat-top"])
        window_combo.pack(fill=tk.X)
        
        # ==== Buttons ====
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="解析実行", command=self.run_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Igorへ出力", command=self.export_to_igor).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="レポート保存", command=self.save_report).pack(fill=tk.X, pady=2)
        
        # ==== Results ====
        result_frame = ttk.Labelframe(left_frame, text="解析結果", padding="10")
        result_frame.pack(fill=tk.X, pady=5)
        
        self.result_text = tk.Text(result_frame, height=8, width=35)
        self.result_text.pack(fill=tk.X)
        self.result_text.insert(tk.END, "解析未実行")
        
        # ==== Plot Area ====
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        filepath = filedialog.askopenfilename(
            title="XRRデータファイル選択",
            filetypes=[("テキストファイル", "*.txt *.dat *.csv"), ("全てのファイル", "*.*")]
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
                messagebox.showerror("エラー", "2列以上のデータが必要です")
        except Exception as e:
            messagebox.showerror("エラー", f"読み込み失敗: {e}")
    
    def _plot_raw_data(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.semilogy(self.two_theta, self.intensity, 'b-', label='生データ')
        ax.axvline(self.crit_angle_var.get(), color='r', linestyle='--', label='Critical Angle')
        ax.set_xlabel('2\u03b8 (deg)')
        ax.set_ylabel('強度 (カウント)')
        ax.set_title('XRR生データ')
        ax.legend()
        self.canvas.draw()
    
    def auto_detect_critical_angle(self):
        if self.two_theta is None:
            messagebox.showwarning("警告", "先にデータを読み込んでください")
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
            messagebox.showwarning("警告", "先にデータを読み込んでください")
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
            messagebox.showerror("エラー", f"解析失敗: {e}")

    def _plot_results(self):
        self.fig.clear()
        
        # Subplot 1: Raw data
        ax1 = self.fig.add_subplot(221)
        ax1.semilogy(self.two_theta, self.intensity, 'b-', label='生データ')
        ax1.axvline(self.crit_angle_var.get(), color='r', linestyle='--', label='Critical Angle')
        ax1.set_xlabel('2\u03b8 (deg)')
        ax1.set_ylabel('強度 (カウント)')
        ax1.set_title('XRR生データ')
        ax1.legend(fontsize=8)
        
        # Subplot 2: Fresnel corrected
        ax2 = self.fig.add_subplot(222)
        mask = ~np.isnan(self.result.q_space)
        ax2.plot(self.result.q_space[mask], self.result.intensity_corrected[mask], 'g-')
        ax2.set_xlabel('Q (nm^{-1})')
        ax2.set_ylabel('フレネル補正強度')
        ax2.set_title('フレネル補正データ')
        
        # Subplot 3: FFTスペクトル
        ax3 = self.fig.add_subplot(223)
        fft = self.result.fft_result
        
        # Plot data points
        ax3.plot(fft.freq, fft.amplitude, 'o-', color='gray', markersize=2, label='データ')
        
        if self.result.fit_result is not None:
            fit = self.result.fit_result
            ax3.plot(fit.x_fit, fit.y_fit, '-', color='orange', lw=2, label='フィット')
            ax3.plot(fit.x_fit, fit.y_gauss1, '--', color='blue', label='層1')
            ax3.plot(fit.x_fit, fit.y_gauss2, '--', color='purple', label='層2')
            ax3.plot(fit.x_fit, fit.y_gauss3, '--', color='teal', label='合計')
            ax3.plot(fit.x_fit, fit.y_noise, '--', color='brown', label='1/fノイズ')
        
        ax3.set_xlim(0, 35)
        ax3.set_ylim(0, 0.45)
        ax3.set_xlabel('膜厚 (nm)')
        ax3.set_ylabel('正規化FFT振幅')
        ax3.set_title('FFTスペクトル')
        ax3.legend(fontsize=7)
        
        # Subplot 4: 残差
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
            ax4.set_xlabel('膜厚 (nm)')
            ax4.set_ylabel('残差')
            ax4.set_title(f'残差 (R^2={fit.r_squared:.4f})')
        else:
            ax4.text(0.5, 0.5, 'フィット無し', transform=ax4.transAxes,
                    ha='center', va='center')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _update_results_text(self):
        self.result_text.delete(1.0, tk.END)
        
        if self.result.fit_result is None:
            self.result_text.insert(tk.END, "フィッティング失敗\n")
            return
        
        fit = self.result.fit_result
        text = f"""膜厚解析結果
=========================

上層 (層1): {fit.thickness_layer1:.2f} nm
下層 (層2): {fit.thickness_layer2:.2f} nm
合計膜厚: {fit.thickness_total:.2f} nm

フィット品質 (R^2): {fit.r_squared:.4f}
"""
        self.result_text.insert(tk.END, text)

    def export_to_igor(self):
        if self.result is None:
            messagebox.showwarning("警告", "先に解析を実行してください")
            return
        
        try:
            wave_files = self.igor_bridge.export_xrr_results(self.result)
            commands = self.igor_bridge.interface.generate_igor_commands(wave_files)
            
            # Show commands in a dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Igor Proコマンド")
            dialog.geometry("500x300")
            
            ttk.Label(dialog, text="以下のコマンドをIgor Proで実行:").pack(pady=10)
            
            text = tk.Text(dialog, height=15, width=60)
            text.pack(fill=tk.BOTH, expand=True, padx=10)
            text.insert(tk.END, commands)
            
            ttk.Button(dialog, text="閉じる", command=dialog.destroy).pack(pady=10)
            
            messagebox.showinfo("成功", f"出力完了 {len(wave_files)} 個のWaveをITXファイルに出力")
            
        except Exception as e:
            messagebox.showerror("エラー", f"出力失敗: {e}")
    
    def save_report(self):
        if self.result is None or self.result.fit_result is None:
            messagebox.showwarning("警告", "先に解析を実行してください")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="レポート保存",
            defaultextension=".txt",
            filetypes=[("テキストファイル", "*.txt")]
        )
        if not filepath:
            return
        
        try:
            self.igor_bridge.save_thickness_report(self.result, filepath)
            
            # Also save the figure
            fig_path = filepath.replace('.txt', '.png')
            self.fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            
            messagebox.showinfo("成功", f"レポート保存: {filepath}\n図保存: {fig_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"保存失敗: {e}")


def main():
    root = tk.Tk()
    app = XRRFFTApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
