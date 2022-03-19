---
title: Equalizer_Workflow
tags: jurytest, python, EQ validation
description: Equalizer_Workflow
---
# Equalizer_Workflow

## 工具整合-Py_Equalizer_Workflow
> * 整合底下四種工具，合而為單一工具
> * 虛擬耳機技術，將Raw耳機轉換為Target頻響參數，經正規化與增益限制，同時輸出對應之產出檔案
### Usage

參數說明：
| console     | code variable    | 說明                         |
| ----------- | ---------------- | ---------------------------- |
| -raw        | Raw_csv          | 原始耳機頻響csv(required)    |
| -output     | output_name      | 輸出地址與檔名               |
| -target     | Target_csv       | 目標耳機頻響csv              |
| -v          | Virtualize       | 是否進行虛擬化耳機           |
| -l          | Limit            | 目標耳機增益限制，預設無限制 |
| -x          | xfig             | Plot匯出X軸大小，預設6       |
| -y          | yfig             | Plot匯出Y軸大小，預設4       |
| -raw_wav    | Raw_wav          | 原始合成音頻                 |
| -target_wav | Target_wav       | 目標合成音頻                 |
| -TD         | Time_Domain      | 時域圖顯示，預設True         |
| -FD         | Frequency_Domain | 頻域圖顯示，預設True         |
| -xml        | xml              | 是否輸出xml檔案              |

### 1.Multiple Output with Virtualize虛擬化耳機整合輸出選項
> **限制：**
> 需要同時有Raw_csv and Target_csv，並且Mode需要設定為"multi"
```python=
python Equalizer_WorkFlow.py -m="multi" -raw="./Sony MH750.csv" -target="./Shure SE215.csv" -output="MH750_to_SE215" -xml=True
``` 
默認輸出之檔案有以下：
> 未設定情況下輸出outputs檔案夾：
> Audacity_config.txt
> Virtualize_filter.csv
> Equalization FR Plot.png
可依照需求進行增減

### 2. **Py_Octave_Transfer**：
* 轉換AutoEQ(695 nodes)原有的frequency為標準之1/12 Octave(121 nodes)
> **限制：**
> 僅支持單一檔案轉換：Raw_csv，並且Mode需要設定為"octave"

```python=
python Equalizer_WorkFlow.py -m="octave" -raw="./Sony MH750.csv" -output="MH750_octave"
```
### 3. **Py_Frequency_Response**
* 讀取耳機頻響，將其繪製成頻域圖
* 虛擬耳機技術，將Raw耳機轉換為Target頻響參數，經正規化與增益限制
> **限制：**
> Mode需要設定為"fr"
> 
> 支持三種模式：
> - 單一模式(匯出TD/FD)
> - Compare模式(匯出比對TD/FD圖)
> - Virtualize模式(匯出虛擬耳機之Filter與比對TD/FD圖)

```python=
python Equalizer_WorkFlow.py -m="fr" -raw="./Sony MH750.csv" -target="./Shure SE215.csv" -output="MH750_octave" -v=True -l=5 -x=10 -y=6
```

### 4. **Py_Equalization_Audacity**
* 讀取頻響值，匯出Audacity的EQ設定檔
> **限制：**
> 僅支持單一檔案轉換：Raw_csv，並且Mode需要設定為"audacity"

```python=
python Equalizer_WorkFlow.py -m="audacity" -raw="./Sony MH750.csv" -target="./Shure SE215.csv" -output="MH750_octave" -xml=True
```

### 5. **Py_Wav_Plot**
* 讀取音頻/具有濾波器的音頻
* 匯出頻域圖/時域圖
> **限制：**
> Mode需要設定為"wav"
> 
> 支持兩種模式：
> - 單一模式(匯出TD/FD)
> - Compare模式(匯出比對TD/FD圖)

```python=
python Equalizer_WorkFlow.py -m="wav" -raw="./Sony MH750.csv" -target="./Shure SE215.csv" -output="MH750_octave" -v=True 
```

---

## 介紹：
>  在這個Equalizer部分中總共有四個Module被使用
1. **Py_Octave_Transfer**：
    * 轉換AutoEQ(695 nodes)原有的frequency為標準之1/12 Octave(121 nodes)
2. **Py_Frequency_Response**
    * 讀取耳機頻響，將其繪製成頻域圖
    * 虛擬耳機技術，將Raw耳機轉換為Target頻響參數，經正規化與增益限制
3. **Py_Equalization_Audacity**
    * 讀取頻響值，匯出Audacity的EQ設定檔
4. **Py_Wav_Plot**
    * 讀取音頻/具有濾波器的音頻
    * 匯出頻域圖/時域圖

---

以下是整體活動圖：
![](https://i.imgur.com/iPf6Ezk.png)

---
**實際操作**
> 涵蓋實際操作範例

> **Prerequisites**
> 
> - 所有csv皆需要按照以下格式匯入:
>     - 含有Frequency欄位名稱，用以標註頻率(Frequency)
>     - 含有raw欄位名稱，用以標註震幅(Gain)
> 
> |Frequency|raw|
> |----|----|
> |20|3.0|
> |30|2.5|
> |40|3.0|

## 1. **Py_Octave_Transfer**
> * 轉換AutoEQ(695 nodes)原有的frequency為標準之1/12 Octave(121 nodes)

### Usage
參數說明：
```md
-input = --Raw_csv = 原始耳機頻響csv
-output = --output_name = 輸出地址與檔名(預設為./octave_result.csv)
```
### Export Transfered csv
```python=
python Py_Octave_Transfer.py -input="Xiaomi Piston 2.csv" -output="./octave_MI.csv"
```
經輸出後之CSV即為標準之1/12 Octave(121 nodes)檔案

---
## 2. **Py_Frequency_Response**
> * 讀取耳機頻響，將其繪製成頻域圖
> * 虛擬耳機技術，將Raw耳機轉換為Target頻響參數，經正規化與增益限制
### Usage

參數說明：
| -console | code variable | 說明                         |
| -------- | ------------- | ---------------------------- |
| -raw     | Raw_csv       | 原始耳機頻響csv(required)    |
| -output  | output_name   | 輸出地址與檔名               |
| -target  | Target_csv    | 目標耳機頻響csv              |
| -v       | Virtualize    | 是否進行虛擬化耳機           |
| -l       | Limit         | 目標耳機增益限制，預設無限制 |
| -x       | xfig          | Plot匯出X軸大小，預設6       |
| -y       | yfig          | Plot匯出Y軸大小，預設4       |
### 1.Draw the plot by Raw.csv
繪出單一頻響圖
```python=
python Py_Frequency_Response_Plot.py -raw="./Raw.csv" -outputdir="./Result/Raw" -x=20 -y=10
```
輸出：
> ------Raw Plot-----
Have save fig into ./Result/Raw_FR.png
------Raw Plot-----
![](https://i.imgur.com/COOYBVp.png)
### 2.Draw the plot with Raw.csv and Target.csv
同時繪出兩個頻響圖並且匯出兩者誤差
```python=
python Py_Frequency_Response_Plot.py -raw="./Raw.csv" -outputdir="./Result/Raw_Target" -target="./Target.csv" -x=20 -y=10
```
輸出：
> ------Compare Plot-----
Have save fig into ./Result/Raw_Target_FR.png
------Compare Plot-----
#### 圖型標記說明：
| 圖內簡標          | 詳細說明              |
| ----------------- | --------------------- |
| Raw               | 原有頻響值            |
| Error             | Raw - Target          |
| Target            | 目標頻響值            |
|實際輸出圖|![](https://i.imgur.com/touIySN.png)

### 3.Equalization the plot with Raw.csv and Target.csv
- 虛擬耳機技術
- 產出Raw_Target.csv
- 產出Raw_Target_Equalization.png
```python=
python Py_Frequency_Response_Plot.py -raw="./Raw.csv" -output="./Result/Raw_Target" -target="./Target.csv" -v=True -l=5 -x=20 -y=10
```
輸出：
> 
> ------Virtualize Script Start-----
Have save fig into ./Result/Raw_Target_Equalization.png
Have save csv into ./Result/Raw_Target.csv
------Virtualize Script End-----
#### 圖型標記說明：
| 圖內簡標          | 詳細說明              |
| ----------------- | --------------------- |
| Raw               | 原有頻響值            |
| Error             | Raw - Target          |
| LimitEqualization | 耳機增益限制之頻響    |
| Equalization      | Target - Raw          |
| Equalized         | Raw+LimitEqualization |
| Target            | 目標頻響值            |
|實際輸出圖|![](https://i.imgur.com/WDzPs3r.png)|


---
## 3. **Py_Equalization_Audacity**
> * 讀取頻響值，匯出Audacity的EQ設定檔
### Usage
參數說明：
```md
-input = --Raw_csv = 原始耳機頻響csv
-output = --output_name = 輸出地址與檔名
-xml = --xml = 是否輸出xml檔案
```
### Export only txt
```python=
python Py_Equalization_Audacity.py -input="../Result/MI_MH750.csv" -output="./MI_MH750"
```
輸出：
> ------Export-----
Have save txt into ./MI_MH750_Audacity_config.txt
------Export-----
![](https://i.imgur.com/aymsk0Q.png)
### Export txt and xml
```python=
python Py_Equalization_Audacity.py -input="../Result/MI_MH750.csv" -output="./MI_MH750" -xml=True
```
輸出：
> ------Export-----
Have save txt into ./MI_MH750_Audacity_config.txt
Have save xml into ./MI_MH750_Audacity_config.xml
------Export-----
![](https://i.imgur.com/yBRv9JJ.png)
---
## 4. **Py_Wav_Plot**
> * 讀取音頻/具有濾波器的音頻
> * 匯出頻域圖/時域圖
### Usage
參數說明：
| console | code variable    | 說明                      |
| ------- | ---------------- | ------------------------- |
| -raw    | Raw_wav          | 原始耳機頻響wav(required) |
| -output | output_name      | 輸出地址與檔名            |
| -target | Target_wav       | 目標耳機頻響wav           |
| -TD     | Time_Domain      | 時域圖顯示，預設True      |
| -FD     | Frequency_Domain | 頻域圖顯示，預設True      |
| -x      | xfig             | Plot匯出X軸大小，預設6    |
| -y      | yfig             | Plot匯出Y軸大小，預設4    |
### 1.Draw the plot by Raw.wav
繪出單一音檔之時域圖和頻域圖
```python=
python Py_Wav_Plot.py -raw="../Test/oBRD.wav" -output="BRD_AutoEQ"
```
輸出：
> ------Raw Plot-----
Have save fig into ./Result/Raw_FR.png
------Raw Plot-----
#### 圖型標記說明：
| 圖內簡標 | 詳細說明   |
| -------- | ---------- |
| Raw      | 原有頻響值 |
| Target   | 目標頻響值 |
|頻域圖(FD)|![](https://i.imgur.com/5gyHjVI.png)|
|時域圖(TD)|![](https://i.imgur.com/ZdQ4Udm.png)|
### 2.Draw the plot with Raw.wav and Target.wav
同時繪出兩個檔案之對比時域圖和頻域圖
```python=
python Py_Wav_Plot.py -raw="../Test/oBRD.wav" -target="../Test/re_audio.wav" -output="BRD_AutoEQ" -x=25 -y=10
```
輸出：
> ------Compare mode-----
number of samplerate = 44100
number of channels = 1
length = 18.62047619047619s
number of samplerate = 44100
number of channels = 1
length = 18.62047619047619s
Have save fig into BRD_AutoEQ_Wav_TD_twice.png
Have save fig into BRD_AutoEQ_Wav_FD_twice.png
------Compare mode-----
#### 圖型標記說明：
| 圖內簡標 | 詳細說明   |
| -------- | ---------- |
| Raw      | 原有頻響值 |
| Target   | 目標頻響值 |
|頻域圖(FD)|![](https://i.imgur.com/8bDI0SJ.png)|
|時域圖(TD)|![](https://i.imgur.com/28p2ggM.png)|

---


