#### Readme
##### 最小功能工具
1. Frequency_Response_Plot
    - 讀取耳機頻響，將其繪製成頻域圖
        - 讀取單一型號
        - 讀取兩個型號
            - 包含Raw/Target 頻響圖
            - Target-Raw 誤差
    - 虛擬耳機技術
        - 將Raw耳機轉換為Target頻響參數，經正規化與增益限制
        - 可匯出對應之Equalization.csv
    - 工具命名
        - iPy_Frequency_Response_Plot
        - Py_Frequency_Response_Plot
    - 匯出命名
        - raw_FR.png(Frequency Response 頻響)
        - raw_Target_FR.png(Frequency Response 頻響)
        - raw_Target_Equalization.csv
        - raw_Target_Equalization_FR.png(Frequency Response 頻響)
2. Equalization_Audacity
    - 將Equalization頻響，轉換成Audacity之曲線等化器參數
    - 工具命名
        - iPy_Equalization_Audacity
        - Py_Equalization_Audacity
    - 匯出命名
        - raw_Target_Audacity_config.txt(新版本適用>2.3.2)
        - raw_Target_Audacity_config.xml(舊版本適用<=2.3.2)
3. Wav_Plot
    - 將Wav讀取後，將其繪製成對應之頻域圖與時域圖
        - 讀取單一Wav
        - 讀取多Wav
    - 比較兩者Wav之圖型，疊加並且匯出
    - 工具命名
        - iPy_Wav_Plot
        - Py_Wav_Plot
    - 匯出命名
        - raw_Wav_FD.png(Frequency Domain頻域)
        - raw_Wav_TD.png(Time Domain時域)
        - raw_Target_Wav_FD.png(Frequency Domain頻域)
        - raw_Target_Wav_TD.png(Time Domain時域)
