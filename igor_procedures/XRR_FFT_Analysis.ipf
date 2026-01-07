#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3
#pragma version=1.0
#pragma IgorVersion=8.0

// XRR FFT 膜厚解析
// 参考文献: Lammel et al., Appl. Phys. Lett. 117, 213106 (2020)
// アルゴリズム: arXiv:2008.04626v2

Menu "Analysis"
    "-"
    "XRR FFT 膜厚解析...", /Q, XRRFFTPanel()
    "XRRパネル表示", /Q, XRRShowPanel()
    "XRRパネル非表示", /Q, XRRHidePanel()
End

Function XRRShowPanel()
    DoWindow XRRFFTPanel
    if (V_flag == 0)
        XRRFFTPanel()
    else
        DoWindow/F XRRFFTPanel
    endif
End

Function XRRHidePanel()
    DoWindow XRRFFTPanel
    if (V_flag != 0)
        DoWindow/HIDE=1 XRRFFTPanel
    endif
End

// 起動時に自動で開く場合はコメント解除:
// Function AfterCompiledHook()
//     XRRShowPanel()
// End

Function XRRInitGlobals()
    String dfSave = GetDataFolder(1)
    NewDataFolder/O/S root:Packages
    NewDataFolder/O/S root:Packages:XRR_FFT
    Variable/G gWavelength = 0.154
    Variable/G gCriticalAngle = 0.3
    Variable/G gAutoDetect = 1
    Variable/G gWindowType = 1
    Variable/G gZeroPadding = 4
    Variable/G gQmin = 0
    Variable/G gQmax = 0
    Variable/G gDoFit = 1
    Variable/G gThickness1 = 0
    Variable/G gThickness2 = 0
    Variable/G gThicknessTotal = 0
    Variable/G gFitError = 0
    String/G gTwoThetaWave = ""
    String/G gIntensityWave = ""
    String/G gStatusText = "準備完了"
    SetDataFolder dfSave
End

Function/S XRRGetWaveList()
    String list = WaveList("*", ";", "DIMS:1")
    if (strlen(list) == 0)
        return "_none_"
    endif
    return "_none_;" + list
End

Function XRRUpdateStatus(msg)
    String msg
    SVAR status = root:Packages:XRR_FFT:gStatusText
    status = msg
    DoUpdate
End


Window XRRFFTPanel() : Panel
    if (!DataFolderExists("root:Packages:XRR_FFT"))
        XRRInitGlobals()
    endif
    PauseUpdate; Silent 1
    DoWindow XRRFFTPanel
    if (V_flag != 0)
        DoWindow/F XRRFFTPanel
        return
    endif
    NewPanel/K=1/W=(100,100,520,620) as "XRR FFT Thickness Analysis"
    DoWindow/C XRRFFTPanel
    ModifyPanel cbRGB=(61166,61166,61166)
    
    GroupBox grpData, pos={10,10}, size={400,100}, title="データ選択"
    TitleBox title2Theta, pos={20,35}, size={80,20}, title="2theta波形:", frame=0
    PopupMenu popup2Theta, pos={100,32}, size={200,20}
    PopupMenu popup2Theta, mode=1, value=#"XRRGetWaveList()"
    PopupMenu popup2Theta, proc=XRRWavePopupProc
    TitleBox titleInt, pos={20,60}, size={80,20}, title="強度波形:", frame=0
    PopupMenu popupIntensity, pos={100,57}, size={200,20}
    PopupMenu popupIntensity, mode=1, value=#"XRRGetWaveList()"
    PopupMenu popupIntensity, proc=XRRWavePopupProc
    Button btnRefresh, pos={320,45}, size={80,25}, title="更新"
    Button btnRefresh, proc=XRRRefreshButtonProc
    
    GroupBox grpParams, pos={10,120}, size={400,130}, title="解析パラメータ"
    TitleBox titleWL, pos={20,145}, size={100,20}, title="波長 (nm):", frame=0
    SetVariable setWavelength, pos={130,142}, size={100,20}
    SetVariable setWavelength, value=root:Packages:XRR_FFT:gWavelength
    SetVariable setWavelength, limits={0.01,1,0.001}, format="%.4f"
    TitleBox titleCA, pos={20,170}, size={100,20}, title="臨界角 (度):", frame=0
    SetVariable setCritAngle, pos={140,167}, size={90,20}
    SetVariable setCritAngle, value=root:Packages:XRR_FFT:gCriticalAngle
    SetVariable setCritAngle, limits={0,5,0.01}, format="%.3f"
    CheckBox chkAutoDetect, pos={250,170}, size={120,20}, title="自動検出"
    CheckBox chkAutoDetect, variable=root:Packages:XRR_FFT:gAutoDetect
    Button btnDetect, pos={320,195}, size={80,25}, title="検出実行"
    Button btnDetect, proc=XRRDetectButtonProc
    TitleBox titleQmin, pos={20,200}, size={100,20}, title="Q最小 (1/nm):", frame=0
    SetVariable setQmin, pos={130,197}, size={100,20}
    SetVariable setQmin, value=root:Packages:XRR_FFT:gQmin
    SetVariable setQmin, limits={0,100,0.1}, format="%.2f"
    TitleBox titleQmax, pos={20,225}, size={100,20}, title="Q最大 (1/nm):", frame=0
    SetVariable setQmax, pos={130,222}, size={100,20}
    SetVariable setQmax, value=root:Packages:XRR_FFT:gQmax
    SetVariable setQmax, limits={0,100,0.1}, format="%.2f"
    TitleBox titleQmaxNote, pos={240,225}, size={100,20}, title="(0=自動)", frame=0
    
    GroupBox grpFFT, pos={10,260}, size={400,80}, title="FFTオプション"
    TitleBox titleWin, pos={20,285}, size={100,20}, title="窓関数:", frame=0
    PopupMenu popupWindow, pos={130,282}, size={120,20}
    PopupMenu popupWindow, mode=2, value="None;Hamming;Hanning;Flat-top"
    PopupMenu popupWindow, proc=XRRWindowPopupProc
    TitleBox titlePad, pos={20,310}, size={100,20}, title="ゼロパディング:", frame=0
    PopupMenu popupPadding, pos={130,307}, size={80,20}
    PopupMenu popupPadding, mode=3, value="1x;2x;4x;8x;16x"
    PopupMenu popupPadding, proc=XRRPaddingPopupProc
    CheckBox chkDoFit, pos={250,285}, size={140,20}, title="マルチガウシアンフィット"
    CheckBox chkDoFit, variable=root:Packages:XRR_FFT:gDoFit
    
    Button btnAnalyze, pos={50,355}, size={120,35}, title="解析"
    Button btnAnalyze, proc=XRRAnalyzeButtonProc, fColor=(0,52224,0)
    Button btnClear, pos={190,355}, size={80,35}, title="クリア"
    Button btnClear, proc=XRRClearButtonProc
    Button btnExport, pos={290,355}, size={80,35}, title="出力"
    Button btnExport, proc=XRRExportButtonProc
    
    GroupBox grpResults, pos={10,400}, size={400,95}, title="解析結果 - 膜厚"
    TitleBox titleT1, pos={20,425}, size={120,20}, title="層1:", frame=0
    ValDisplay valThick1, pos={130,422}, size={150,20}
    ValDisplay valThick1, value=#"root:Packages:XRR_FFT:gThickness1"
    ValDisplay valThick1, format="%.2f nm"
    TitleBox titleT2, pos={20,450}, size={120,20}, title="層2:", frame=0
    ValDisplay valThick2, pos={130,447}, size={150,20}
    ValDisplay valThick2, value=#"root:Packages:XRR_FFT:gThickness2"
    ValDisplay valThick2, format="%.2f nm"
    TitleBox titleTT, pos={20,475}, size={120,20}, title="合計:", frame=0, fStyle=1
    ValDisplay valThickTotal, pos={130,472}, size={150,20}
    ValDisplay valThickTotal, value=#"root:Packages:XRR_FFT:gThicknessTotal"
    ValDisplay valThickTotal, format="%.2f nm"
    
    TitleBox titleStatus, pos={10,505}, size={400,20}
    TitleBox titleStatus, variable=root:Packages:XRR_FFT:gStatusText, frame=1
EndMacro


Function XRRWavePopupProc(pa) : PopupMenuControl
    STRUCT WMPopupAction &pa
    if (pa.eventCode != 2)
        return 0
    endif
    SVAR twoThetaWave = root:Packages:XRR_FFT:gTwoThetaWave
    SVAR intensityWave = root:Packages:XRR_FFT:gIntensityWave
    strswitch(pa.ctrlName)
        case "popup2Theta":
            twoThetaWave = pa.popStr
            break
        case "popupIntensity":
            intensityWave = pa.popStr
            break
    endswitch
    return 0
End

Function XRRWindowPopupProc(pa) : PopupMenuControl
    STRUCT WMPopupAction &pa
    if (pa.eventCode != 2)
        return 0
    endif
    NVAR windowType = root:Packages:XRR_FFT:gWindowType
    windowType = pa.popNum - 1
    return 0
End

Function XRRPaddingPopupProc(pa) : PopupMenuControl
    STRUCT WMPopupAction &pa
    if (pa.eventCode != 2)
        return 0
    endif
    NVAR zeroPadding = root:Packages:XRR_FFT:gZeroPadding
    switch(pa.popNum)
        case 1:
            zeroPadding = 1
            break
        case 2:
            zeroPadding = 2
            break
        case 3:
            zeroPadding = 4
            break
        case 4:
            zeroPadding = 8
            break
        case 5:
            zeroPadding = 16
            break
    endswitch
    return 0
End

Function XRRRefreshButtonProc(ba) : ButtonControl
    STRUCT WMButtonAction &ba
    if (ba.eventCode != 2)
        return 0
    endif
    PopupMenu popup2Theta, win=XRRFFTPanel, mode=1
    PopupMenu popupIntensity, win=XRRFFTPanel, mode=1
    XRRUpdateStatus("波形リスト更新完了")
    return 0
End

Function XRRDetectButtonProc(ba) : ButtonControl
    STRUCT WMButtonAction &ba
    if (ba.eventCode != 2)
        return 0
    endif
    XRRAutoDetectCriticalAngle()
    return 0
End

Function XRRAnalyzeButtonProc(ba) : ButtonControl
    STRUCT WMButtonAction &ba
    if (ba.eventCode != 2)
        return 0
    endif
    XRRDoAnalysis()
    return 0
End

Function XRRClearButtonProc(ba) : ButtonControl
    STRUCT WMButtonAction &ba
    if (ba.eventCode != 2)
        return 0
    endif
    NVAR t1 = root:Packages:XRR_FFT:gThickness1
    NVAR t2 = root:Packages:XRR_FFT:gThickness2
    NVAR tt = root:Packages:XRR_FFT:gThicknessTotal
    t1 = 0
    t2 = 0
    tt = 0
    KillWaves/Z XRR_Q, XRR_Intensity, XRR_Fresnel, XRR_Interp
    KillWaves/Z XRR_FFT_Freq, XRR_FFT_Amp, XRR_FFT_Fit
    DoWindow XRRFFTGraph
    if (V_flag != 0)
        DoWindow/K XRRFFTGraph
    endif
    XRRUpdateStatus("結果クリア")
    return 0
End

Function XRRExportButtonProc(ba) : ButtonControl
    STRUCT WMButtonAction &ba
    if (ba.eventCode != 2)
        return 0
    endif
    XRRExportResults()
    return 0
End

Function XRRAutoDetectCriticalAngle()
    SVAR twoThetaWave = root:Packages:XRR_FFT:gTwoThetaWave
    SVAR intensityWave = root:Packages:XRR_FFT:gIntensityWave
    NVAR criticalAngle = root:Packages:XRR_FFT:gCriticalAngle
    if (cmpstr(twoThetaWave, "_none_") == 0 || strlen(twoThetaWave) == 0)
        XRRUpdateStatus("エラー: 2theta波形を選択してください")
        return -1
    endif
    if (cmpstr(intensityWave, "_none_") == 0 || strlen(intensityWave) == 0)
        XRRUpdateStatus("エラー: 強度波形を選択してください")
        return -1
    endif
    Wave/Z wTwoTheta = $twoThetaWave
    Wave/Z wInt = $intensityWave
    if (!WaveExists(wTwoTheta) || !WaveExists(wInt))
        XRRUpdateStatus("エラー: 選択した波形が見つかりません")
        return -1
    endif
    XRRUpdateStatus("臨界角検出中...")
    WaveStats/Q wInt
    Variable maxIdx = V_maxloc
    Variable twoThetaAtMax = wTwoTheta[maxIdx]
    criticalAngle = twoThetaAtMax / 2
    Variable halfMax = V_max * 0.5
    Variable i, n = numpnts(wInt)
    for (i = maxIdx; i < n; i += 1)
        if (wInt[i] < halfMax)
            criticalAngle = wTwoTheta[i] / 2
            break
        endif
    endfor
    XRRUpdateStatus("臨界角: " + num2str(criticalAngle) + " deg")
    return 0
End


Function XRRDoAnalysis()
    SVAR twoThetaWave = root:Packages:XRR_FFT:gTwoThetaWave
    SVAR intensityWave = root:Packages:XRR_FFT:gIntensityWave
    NVAR wavelength = root:Packages:XRR_FFT:gWavelength
    NVAR criticalAngle = root:Packages:XRR_FFT:gCriticalAngle
    NVAR autoDetect = root:Packages:XRR_FFT:gAutoDetect
    NVAR windowType = root:Packages:XRR_FFT:gWindowType
    NVAR zeroPadding = root:Packages:XRR_FFT:gZeroPadding
    NVAR Qmin = root:Packages:XRR_FFT:gQmin
    NVAR Qmax = root:Packages:XRR_FFT:gQmax
    NVAR doFit = root:Packages:XRR_FFT:gDoFit
    
    if (cmpstr(twoThetaWave, "_none_") == 0 || strlen(twoThetaWave) == 0)
        XRRUpdateStatus("エラー: 2theta波形を選択")
        return -1
    endif
    if (cmpstr(intensityWave, "_none_") == 0 || strlen(intensityWave) == 0)
        XRRUpdateStatus("エラー: 強度波形を選択")
        return -1
    endif
    
    Wave/Z wTwoTheta = $twoThetaWave
    Wave/Z wInt = $intensityWave
    
    if (!WaveExists(wTwoTheta) || !WaveExists(wInt))
        XRRUpdateStatus("エラー: 波形が見つかりません")
        return -1
    endif
    if (numpnts(wTwoTheta) != numpnts(wInt))
        XRRUpdateStatus("エラー: 波形の長さが一致しません")
        return -1
    endif
    
    XRRUpdateStatus("解析開始...")
    
    if (autoDetect)
        XRRAutoDetectCriticalAngle()
    endif
    
    Variable n = numpnts(wTwoTheta)
    Variable i
    
    // ステップ1: 2thetaをQ空間へ変換（臨界角補正付き）
    XRRUpdateStatus("Q空間へ変換中...")
    Make/O/D/N=(n) XRR_Q, XRR_Intensity
    
    Variable thetaCrit = criticalAngle * pi / 180
    Variable lambda = wavelength
    
    for (i = 0; i < n; i += 1)
        Variable twoTheta = wTwoTheta[i]
        Variable theta = twoTheta / 2 * pi / 180
        Variable sinTheta = sin(theta)
        Variable sinThetaC = sin(thetaCrit)
        Variable arg = sinTheta^2 - sinThetaC^2
        if (arg > 0)
            XRR_Q[i] = (4 * pi / lambda) * sqrt(arg)
            XRR_Intensity[i] = wInt[i]
        else
            XRR_Q[i] = 0
            XRR_Intensity[i] = NaN
        endif
    endfor
    
    // 無効なデータ点を除去
    Variable validCount = 0
    for (i = 0; i < n; i += 1)
        if (XRR_Q[i] > 0 && numtype(XRR_Intensity[i]) == 0)
            validCount += 1
        endif
    endfor
    
    Make/O/D/N=(validCount) XRR_Q_valid, XRR_Int_valid
    Variable j = 0
    for (i = 0; i < n; i += 1)
        if (XRR_Q[i] > 0 && numtype(XRR_Intensity[i]) == 0)
            XRR_Q_valid[j] = XRR_Q[i]
            XRR_Int_valid[j] = XRR_Intensity[i]
            j += 1
        endif
    endfor
    
    // ステップ2: フレネル補正（theta^4を乗算）
    XRRUpdateStatus("フレネル補正適用中...")
    Make/O/D/N=(validCount) XRR_Fresnel
    
    Variable sinThetaCSq = sin(thetaCrit)^2
    for (i = 0; i < validCount; i += 1)
        Variable sinThSq = (XRR_Q_valid[i] * lambda / (4 * pi))^2 + sinThetaCSq
        Variable thetaRad = asin(sqrt(sinThSq))
        Variable thetaDeg = thetaRad * 180 / pi
        XRR_Fresnel[i] = XRR_Int_valid[i] * (thetaDeg^4)
    endfor
    
    // ステップ3: Q範囲設定
    Variable Qmin_use = Qmin
    Variable Qmax_use = Qmax
    if (Qmin_use <= 0)
        Qmin_use = XRR_Q_valid[0]
    endif
    if (Qmax_use <= 0)
        Qmax_use = XRR_Q_valid[validCount-1]
    endif
    
    // ステップ4: 等間隔Qへ補間
    XRRUpdateStatus("データ補間中...")
    Variable Qrange = Qmax_use - Qmin_use
    Variable dQ = Qrange / (validCount - 1)
    Variable nInterp = floor(Qrange / dQ) + 1
    
    Make/O/D/N=(nInterp) XRR_Q_interp, XRR_Fresnel_interp
    for (i = 0; i < nInterp; i += 1)
        XRR_Q_interp[i] = Qmin_use + i * dQ
    endfor
    Interpolate2/T=3/I=3/Y=XRR_Fresnel_interp XRR_Q_valid, XRR_Fresnel /X=XRR_Q_interp

    
    // ステップ5: 窓関数適用
    XRRUpdateStatus("窓関数適用中...")
    Make/O/D/N=(nInterp) XRR_Windowed
    
    for (i = 0; i < nInterp; i += 1)
        Variable w = 1
        Variable x = i / (nInterp - 1)
        switch(windowType)
            case 1:  // ハミング窓
                w = 0.54 - 0.46 * cos(2 * pi * x)
                break
            case 2:  // ハニング窓
                w = 0.5 * (1 - cos(2 * pi * x))
                break
            case 3:  // フラットトップ窓
                w = 0.21557895 - 0.41663158*cos(2*pi*x) + 0.277263158*cos(4*pi*x)
                w -= 0.083578947*cos(6*pi*x) + 0.006947368*cos(8*pi*x)
                break
        endswitch
        XRR_Windowed[i] = XRR_Fresnel_interp[i] * w
    endfor
    
    // ステップ6: ゼロパディングとFFT
    XRRUpdateStatus("FFT実行中...")
    Variable nFFT = nInterp * zeroPadding
    nFFT = 2^ceil(log(nFFT)/log(2))
    
    Make/O/D/N=(nFFT) XRR_Padded
    XRR_Padded = 0
    XRR_Padded[0, nInterp-1] = XRR_Windowed[p]
    
    FFT/OUT=3/DEST=XRR_FFT_Complex XRR_Padded
    
    Make/O/D/N=(nFFT/2) XRR_FFT_Amp, XRR_FFT_Freq
    Variable df = 1 / (nFFT * dQ)
    
    for (i = 0; i < nFFT/2; i += 1)
        XRR_FFT_Freq[i] = i * df
        Variable re = real(XRR_FFT_Complex[i])
        Variable im = imag(XRR_FFT_Complex[i])
        XRR_FFT_Amp[i] = sqrt(re^2 + im^2)
    endfor
    
    WaveStats/Q XRR_FFT_Amp
    XRR_FFT_Amp /= V_max
    
    // ステップ7: マルチガウシアンフィッティング
    if (doFit)
        XRRUpdateStatus("マルチガウシアンフィッティング中...")
        XRRFitMultiGaussian()
    endif
    
    XRRDisplayResults()
    XRRUpdateStatus("解析完了")
    
    KillWaves/Z XRR_Q, XRR_Intensity, XRR_Q_valid, XRR_Int_valid
    KillWaves/Z XRR_Padded, XRR_FFT_Complex, XRR_Windowed
    KillWaves/Z XRR_Q_interp, XRR_Fresnel_interp, XRR_Fresnel
    
    return 0
End


Function XRRFitMultiGaussian()
    Wave/Z amp = XRR_FFT_Amp
    Wave/Z freq = XRR_FFT_Freq
    
    if (!WaveExists(amp) || !WaveExists(freq))
        XRRUpdateStatus("Error: FFT data not found")
        return -1
    endif
    
    Variable n = numpnts(amp)
    Variable startIdx = 5
    
    Duplicate/O amp, XRR_PeakFind
    XRR_PeakFind[0, startIdx] = 0
    
    // ピーク1
    WaveStats/Q XRR_PeakFind
    Variable peak1Idx = V_maxloc
    Variable peak1Amp = V_max
    Variable peak1Pos = freq[peak1Idx]
    Variable peakWidth = 5
    XRR_PeakFind[max(0, peak1Idx-peakWidth), min(n-1, peak1Idx+peakWidth)] = 0
    
    // ピーク2
    WaveStats/Q XRR_PeakFind
    Variable peak2Idx = V_maxloc
    Variable peak2Amp = V_max
    Variable peak2Pos = freq[peak2Idx]
    XRR_PeakFind[max(0, peak2Idx-peakWidth), min(n-1, peak2Idx+peakWidth)] = 0
    
    // ピーク3
    WaveStats/Q XRR_PeakFind
    Variable peak3Idx = V_maxloc
    Variable peak3Amp = V_max
    Variable peak3Pos = freq[peak3Idx]
    
    KillWaves/Z XRR_PeakFind
    
    // ピークを位置順にソート
    Make/O/D/N=3 tempPos = {peak1Pos, peak2Pos, peak3Pos}
    Make/O/D/N=3 tempAmp = {peak1Amp, peak2Amp, peak3Amp}
    Sort tempPos, tempPos, tempAmp
    
    peak1Pos = tempPos[0]
    peak1Amp = tempAmp[0]
    peak2Pos = tempPos[1]
    peak2Amp = tempAmp[1]
    peak3Pos = tempPos[2]
    peak3Amp = tempAmp[2]
    
    KillWaves/Z tempPos, tempAmp
    
    Variable sigma = (freq[1] - freq[0]) * 3
    
    // 係数: A1,pos1,sig1, A2,pos2,sig2, A3,pos3,sig3, alpha, C0
    Make/O/D/N=11 XRR_FitCoefs
    XRR_FitCoefs[0] = peak1Amp
    XRR_FitCoefs[1] = peak1Pos
    XRR_FitCoefs[2] = sigma
    XRR_FitCoefs[3] = peak2Amp
    XRR_FitCoefs[4] = peak2Pos
    XRR_FitCoefs[5] = sigma
    XRR_FitCoefs[6] = peak3Amp
    XRR_FitCoefs[7] = peak3Pos
    XRR_FitCoefs[8] = sigma
    XRR_FitCoefs[9] = 1
    XRR_FitCoefs[10] = 0.01
    
    // 拘束条件: pos1 = pos3 - pos2
    String constraints = "K1=K7-K4;"
    
    Variable V_FitError = 0
    try
        FuncFit/Q/N/H="00000000000" XRRGauss3Func, XRR_FitCoefs, amp /X=freq /C={constraints}
    catch
        FuncFit/Q/N XRRGauss3Func, XRR_FitCoefs, amp /X=freq
    endtry
    
    NVAR t1 = root:Packages:XRR_FFT:gThickness1
    NVAR t2 = root:Packages:XRR_FFT:gThickness2
    NVAR tt = root:Packages:XRR_FFT:gThicknessTotal
    
    t1 = abs(XRR_FitCoefs[1])
    t2 = abs(XRR_FitCoefs[4])
    tt = abs(XRR_FitCoefs[7])
    
    Make/O/D/N=(numpnts(freq)) XRR_FFT_Fit
    XRR_FFT_Fit = XRRGauss3Func(XRR_FitCoefs, freq[p])
    
    return 0
End

Function XRRGauss3Func(w, x) : FitFunc
    Wave w
    Variable x
    
    Variable result = 0
    result += w[0] * exp(-((x - w[1])^2) / (2 * w[2]^2))
    result += w[3] * exp(-((x - w[4])^2) / (2 * w[5]^2))
    result += w[6] * exp(-((x - w[7])^2) / (2 * w[8]^2))
    if (x > 0)
        result += w[10] / (x^w[9])
    endif
    return result
End


Function XRRDisplayResults()
    NVAR t1 = root:Packages:XRR_FFT:gThickness1
    NVAR t2 = root:Packages:XRR_FFT:gThickness2
    NVAR tt = root:Packages:XRR_FFT:gThicknessTotal
    
    DoWindow XRRFFTGraph
    if (V_flag != 0)
        DoWindow/K XRRFFTGraph
    endif
    
    Display/K=1/W=(450,100,1050,600) as "XRR FFT解析結果"
    DoWindow/C XRRFFTGraph
    
    Wave/Z amp = XRR_FFT_Amp
    Wave/Z freq = XRR_FFT_Freq
    Wave/Z fit = XRR_FFT_Fit
    
    if (WaveExists(amp) && WaveExists(freq))
        AppendToGraph amp vs freq
        ModifyGraph mode(XRR_FFT_Amp)=0, lsize(XRR_FFT_Amp)=1.5
        ModifyGraph rgb(XRR_FFT_Amp)=(0,0,65535)
        
        if (WaveExists(fit))
            AppendToGraph fit vs freq
            ModifyGraph mode(XRR_FFT_Fit)=0, lsize(XRR_FFT_Fit)=2
            ModifyGraph rgb(XRR_FFT_Fit)=(65535,0,0)
        endif
    endif
    
    Label left "FFT振幅 (正規化)"
    Label bottom "膜厚 (nm)"
    SetAxis/A left
    SetAxis bottom 0, 200
    
    ModifyGraph grid=1, gridRGB=(48000,48000,48000)
    ModifyGraph tick=2, mirror=1
    
    String legendStr = "\s(XRR_FFT_Amp) FFT Data
"
    if (WaveExists(fit))
        legendStr += "\s(XRR_FFT_Fit) Multi-Gaussian Fit
"
    endif
    legendStr += "
"
    legendStr += "Layer 1: " + num2str(t1) + " nm
"
    legendStr += "Layer 2: " + num2str(t2) + " nm
"
    legendStr += "Total: " + num2str(tt) + " nm"
    
    Legend/C/N=legend1/J/A=RT legendStr
    
    if (t1 > 0)
        SetDrawEnv xcoord=bottom, ycoord=prel, linefgc=(0,52224,0), dash=2
        DrawLine t1, 0, t1, 1
    endif
    if (t2 > 0)
        SetDrawEnv xcoord=bottom, ycoord=prel, linefgc=(52224,0,52224), dash=2
        DrawLine t2, 0, t2, 1
    endif
    if (tt > 0)
        SetDrawEnv xcoord=bottom, ycoord=prel, linefgc=(65535,32768,0), dash=2
        DrawLine tt, 0, tt, 1
    endif
End

Function XRRExportResults()
    NVAR t1 = root:Packages:XRR_FFT:gThickness1
    NVAR t2 = root:Packages:XRR_FFT:gThickness2
    NVAR tt = root:Packages:XRR_FFT:gThicknessTotal
    NVAR wavelength = root:Packages:XRR_FFT:gWavelength
    NVAR criticalAngle = root:Packages:XRR_FFT:gCriticalAngle
    NVAR windowType = root:Packages:XRR_FFT:gWindowType
    NVAR zeroPadding = root:Packages:XRR_FFT:gZeroPadding
    SVAR twoThetaWave = root:Packages:XRR_FFT:gTwoThetaWave
    SVAR intensityWave = root:Packages:XRR_FFT:gIntensityWave
    
    Variable refNum
    String fileName
    
    Open/D/T=".txt" refNum
    fileName = S_fileName
    
    if (strlen(fileName) == 0)
        XRRUpdateStatus("出力キャンセル")
        return -1
    endif
    
    Open refNum as fileName
    
    fprintf refNum, "XRR FFT Thickness Analysis Results

"
    fprintf refNum, "===================================

"
    fprintf refNum, "Date: %s

", Secs2Date(DateTime, -2)
    fprintf refNum, "Time: %s



", Secs2Time(DateTime, 3)
    
    fprintf refNum, "Input Data:

"
    fprintf refNum, "  2theta Wave: %s

", twoThetaWave
    fprintf refNum, "  Intensity Wave: %s



", intensityWave
    
    fprintf refNum, "Analysis Parameters:

"
    fprintf refNum, "  Wavelength: %.4f nm

", wavelength
    fprintf refNum, "  Critical Angle: %.3f deg

", criticalAngle
    
    String winName
    switch(windowType)
        case 0:
            winName = "None"
            break
        case 1:
            winName = "Hamming"
            break
        case 2:
            winName = "Hanning"
            break
        case 3:
            winName = "Flat-top"
            break
    endswitch
    fprintf refNum, "  Window Function: %s

", winName
    fprintf refNum, "  Zero Padding: %dx



", zeroPadding
    
    fprintf refNum, "Results - Film Thickness:

"
    fprintf refNum, "  Layer 1: %.2f nm

", t1
    fprintf refNum, "  Layer 2: %.2f nm

", t2
    fprintf refNum, "  Total:   %.2f nm



", tt
    
    fprintf refNum, "Algorithm Reference:

"
    fprintf refNum, "  Lammel et al., Appl. Phys. Lett. 117, 213106 (2020)

"
    fprintf refNum, "  DOI: 10.1063/5.0024991

"
    fprintf refNum, "  arXiv: 2008.04626v2

"
    
    Close refNum
    
    XRRUpdateStatus("出力完了: " + fileName)
    return 0
End

// ファイル終端
