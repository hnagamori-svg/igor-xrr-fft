# XRR FFT 膜厚解析ツール

X線反射率（XRR）データをFFT解析し、薄膜の膜厚を決定するためのツールです。

## 参考文献

Lammel et al., Appl. Phys. Lett. 117, 213106 (2020)
"Fast Fourier transform and multi-Gaussian fitting of XRR data to determine the thickness of ALD grown thin films within the initial growth regime"
DOI: 10.1063/5.0024991

## インストール

1. Python依存パッケージのインストール:
```bash
pip install numpy scipy matplotlib
```

2. Igor Pro連携（オプション）:
   - `igor_procedures/XRR_FFT_Analysis.ipf` をIgor Proceduresフォルダにコピー

## 使用方法

### Python GUI

```bash
cd python
python xrr_gui.py
```

### Python API

```python
from xrr_fft_core import analyze_xrr, XRRParameters
import numpy as np

# データ読み込み
data = np.loadtxt("your_xrr_data.txt")
two_theta = data[:, 0]
intensity = data[:, 1]

# パラメータ設定
params = XRRParameters(
    wavelength=0.154,         # Cu Ka (nm)
    critical_angle=0.6,       # 臨界角 (度)
    interpolation_points=1000,
    zero_padding_factor=10
)

# 解析実行
result = analyze_xrr(two_theta, intensity, params)

# 膜厚結果
if result.fit_result:
    print(f"上層膜厚: {result.fit_result.thickness_layer1:.2f} nm")
    print(f"下層膜厚: {result.fit_result.thickness_layer2:.2f} nm")
    print(f"合計膜厚: {result.fit_result.thickness_total:.2f} nm")
```

### Igor Pro

1. プロシージャファイルを読み込み
2. メニューから Analysis -> XRR FFT Thickness Analysis を選択
3. 2theta波形と強度波形を選択
4. 「Analyze」をクリック

## アルゴリズム概要

1. **Q空間変換**: 2thetaからQへの変換（臨界角補正付き）
2. **フレネル補正**: Q^4を乗じてフレネル減衰を補償
3. **補間**: 等間隔Qへのリサンプリング
4. **窓関数適用**: Hamming窓等を適用
5. **FFT**: ゼロパディング付き高速フーリエ変換
6. **マルチガウシアンフィッティング**: 3つのガウシアン + 1/f^alpha ノイズ

## 出力

- 上層膜厚 (nm)
- 下層膜厚 (nm)
- 合計膜厚 (nm)
- フィッティング品質 (R^2)

## ファイル構成

```
/python
    xrr_fft_core.py      # 解析コア関数
    igor_interface.py    # Igor Pro連携
    xrr_gui.py           # GUIアプリケーション
    generate_sample_data.py  # サンプルデータ生成

/igor_procedures
    XRR_FFT_Analysis.ipf # Igor Proプロシージャ

/examples
    sample_xrr_data.txt  # サンプルデータ
```
