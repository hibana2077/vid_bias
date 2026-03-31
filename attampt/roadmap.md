**Spatial BG ratio** 要先知道哪裡是 foreground / background，沒有 segmentation、bbox、mask 時會很勉強，不適合放在你的最小實驗設計裡。

你這題最穩的做法，還是照 VDU 的思路，把 dataset 當成不同 **structural pressure**，再用 attention map 萃出幾個**不需要額外標註**、但能支撐 claim 的統計量。VDU 強調的核心就是：dataset 的 motion complexity、temporal span、hierarchical structure 會驅動模型學到不同 inductive bias；而 comparative review 那篇的好處是，它示範了可以用 **average rank**、**confusion matrix** 這種很乾淨的 summary 來整理比較。   

下面我幫你整理成可以直接拿去 proposal / paper 初稿用的版本。

---

## 一、主實驗表（精簡且可直接執行）

| Dataset 組別           | Datasets                                      | 這組資料在考什麼                  | 你要驗證的 bias / claim                                                           | 主要比較方式                                          | 要看的 attention 現象                                  | 主要統計量                                                                                            |
| -------------------- | --------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **短時序、粗動作**          | HMDB51, UCF101, MSR Daily Act                 | 動作明顯、片段短、通常不用太長時間推理       | 模型是否偏向「只看少數明顯 frame 就下判斷」；K400/K600 類 pretrain 是否更容易學到這種短時序偏好                | backbone 間比；同 backbone 不同 weights 比             | attention 是否集中在少數 frame；是否多數 head 只看鄰近 frame      | **Temporal Concentration**、**Local Attention Ratio**、**Head Diversity**                          |
| **細粒度動作**            | Diving48, FineGym                             | 類別差異小，靠細微姿態/時間差異分辨        | K400 類 pretrain 是否容易抓大動作但忽略細節；SSv2 類 pretrain 是否較能捕捉細粒度時序變化                  | K400 vs SSv2 最重要；再看 backbone 差異                 | attention 是否比粗動作資料更分散；不同 head 是否會看不同時刻，而不是全部擠在同一段 | **Temporal Concentration**、**Temporal Coverage**、**Head Diversity**、**Class Confusion**          |
| **程序式 / 階層式長時序**     | 50 Salads, Breakfast, MPII Cooking2, YouCook2 | 要看多步驟流程，不是只看一個瞬間          | 模型是否有 **short-range bias**：只看最近片段，不會整合前後步驟；instructional pretrain 是否較能處理長程依賴 | HowTo100M / VideoPrism vs K400 類最關鍵；再比 backbone | attention 是否能跨較遠時間連接；是否有覆蓋多個步驟，而不是只盯一小段           | **Long-range Attention Ratio**、**Local Attention Ratio**、**Temporal Coverage**、**Step Coverage** |
| **Egocentric 手-物互動** | GTEA Gaze                                     | 第一人稱視角、手部操作多、ego-motion 強 | 模型是否被相機晃動或局部動作帶偏；SSv2 / instructional pretrain 是否較能穩定看完整互動過程                 | weights 比較比 backbone 更重要；再看 backbone            | attention 是否只黏在局部瞬間；是否在不同 sample 間非常不穩定           | **Attention Stability**、**Temporal Coverage**、**Long-range Attention Ratio**、**Class Confusion** |

### 這張表的用法

你的核心比較只做三條線就夠：

1. **同 backbone，不同 pretrain weights**
   看 pretraining bias。

2. **不同 backbone，盡量對齊 pretrain source**
   看 architecture bias。

3. **每個 dataset group 再做 group-level summary**
   用一個平均排名收尾，避免表太炸。這種 summary 方式和 review 那篇用 average rank 的整理方式一致。 

---

## 二、統計量定義表（都不需要 segmentation / mask）

下面這張表只放**真的能從 attention map 直接算**、而且能支撐你老師那個 dataset-driven claim 的量。

| 統計量                                   | 直觀意思                        | 怎麼算（簡單版）                                                   | 數值大代表什麼           | 最適合支持哪種 claim                                      |
| ------------------------------------- | --------------------------- | ---------------------------------------------------------- | ----------------- | -------------------------------------------------- |
| **Temporal Concentration (TC)**       | attention 有沒有擠在很少數幾個 frame  | 把每個 frame 收到的 attention 加總後正規化；看前 20% frame 吃掉多少 attention | 越大 = 越只看少數 frame  | 模型有沒有「短時序偷懶」；粗動作資料是否容易讓模型只看關鍵瞬間                    |
| **Temporal Coverage (TCo)**           | attention 有沒有分布到整段影片        | 統計 attention 超過小門檻的 frame 比例                               | 越大 = 看過更多時間位置     | 細粒度或程序式資料是否要求更廣的時間覆蓋                               |
| **Local Attention Ratio (LAR)**       | 模型是不是只看鄰近 frame             | 對每個 query frame，看它分給時間距離 ≤ k 的 frame 的 attention 比例，再平均    | 越大 = 越偏近距離        | short-range bias；程序式資料上若 LAR 太高，代表模型可能只看附近步驟       |
| **Long-range Attention Ratio (LRAR)** | 模型有沒有跨很遠時間做連結               | 對每個 query frame，看它分給時間距離 > k 的 frame 的 attention 比例，再平均    | 越大 = 越能連到遠距 frame | 長時序、程序式資料是否真的逼出長程推理                                |
| **Head Diversity (HDiv)**             | 不同 attention head 是不是在看不同東西 | 把每個 head 的時間 attention 分布拿去兩兩比較，取平均差異                      | 越大 = 各 head 分工較多  | 細粒度資料是否需要多種 temporal cue；模型是不是所有 head 都在看同一段       |
| **Attention Stability (AS)**          | 同一類樣本的 attention 模式穩不穩      | 同一類內，把 sample 的 attention 分布兩兩比較，算平均相似度                    | 越大 = 同類樣本更穩定      | egocentric 資料是否讓模型不穩；某種 pretrain 是否讓 attention 更一致 |
| **Step Coverage (SC)**                | 對多步驟影片，attention 有沒有碰到多個步驟  | 只用有 step annotation 的資料，把每一步內 attention 加總；看有幾個 step 超過門檻  | 越大 = 有看到更多步驟      | 程序式 / 階層式資料最關鍵；直接支撐 hierarchical reasoning claim   |
| **Class Confusion (CC)**              | 哪些類別容易被混淆                   | 一般 confusion matrix；再特別看細粒度相似類別對                           | 越高 = 混淆更嚴重        | Diving48 / FineGym 這種相近動作是否暴露 bias                 |
| **Average Rank (AVRank)**             | 不同資料集上整體表現穩不穩               | 每個 dataset 內把模型排序，再對 rank 取平均                              | 越小 = 整體更穩、更平均     | 收斂成一個乾淨總結，適合主表最後一欄                                 |

---

## 三、我建議你最後只留這 6 個統計量

如果你想最精簡、最像一篇乾淨 extension，我建議主文只留：

1. **Temporal Concentration**
2. **Local Attention Ratio**
3. **Long-range Attention Ratio**
4. **Head Diversity**
5. **Step Coverage**（只有 procedural datasets 用）
6. **Average Rank**

這 6 個最夠用。

原因很簡單：

* **TC + LAR**：抓「只看少數瞬間／只看近距離」這種短視偏差。
* **LRAR + SC**：抓「有沒有真的整合長程、多步驟資訊」。
* **HDiv**：抓「不同 head 有沒有分工，還是全部在重複看同一段」。
* **AVRank**：把一堆 dataset 的結果收斂成一個 reviewer 看得懂的 summary。這種做法和 comparative review 的整理邏輯很一致。 

---

## 四、你可以直接寫進 proposal 的簡短定義

下面這段你幾乎可以原封不動貼進方法章節。

### 1. Temporal Concentration

先把所有 attention map 沿空間 token 加總，只留下時間維度。若大部分 attention 都落在很少數 frame，代表模型傾向用少數瞬間做判斷，而不是整體理解影片。

### 2. Local Attention Ratio

對每個 frame，統計它把多少 attention 分給附近幾個 frame。這個值高，表示模型偏向使用短距離資訊；在程序式資料上若太高，通常代表它沒有真正做長程整合。

### 3. Long-range Attention Ratio

和上面相反，這個量看模型有多少 attention 會連到較遠的 frame。對長時序與多步驟影片，這個量應該要比較高，否則模型很可能只是在做局部判斷。

### 4. Head Diversity

比較不同 head 的 attention 分布是否相似。若所有 head 都看一樣的時間位置，代表 head 間分工有限；若不同 head 會關注不同時間段，通常表示模型有學到較豐富的時序線索。

### 5. Step Coverage

對有 step annotation 的資料，統計有多少個步驟得到足夠 attention。若模型只盯著其中一兩步，代表它對流程理解不完整；若能覆蓋較多步驟，較能支持 hierarchical / procedural reasoning 的 claim。VDU 也特別強調這類資料需要多尺度、跨步驟的 temporal reasoning。 

### 6. Average Rank

在每個 dataset 上對模型排名，再把排名取平均。它不是要取代 accuracy，而是用來總結哪個模型或哪組預訓練在不同結構壓力下表現更穩定。這跟 comparative review 用平均排名做整體比較的邏輯是一致的。 

---

## 五、最後給你一個最終精簡版

如果你現在只想交一版最乾淨的表，我建議你就交這個：

### 表 A：實驗設定總表

| 組別             | Datasets                                      | 主要問題                   | 比較重點                           | attention 證據    | 量化指標                  |
| -------------- | --------------------------------------------- | ---------------------- | ------------------------------ | --------------- | --------------------- |
| 短時序粗動作         | HMDB51, UCF101, MSR Daily Act                 | 模型是否只靠少數明顯 frame 判斷    | backbone / K400-K600 weights   | 是否集中在少數時間點      | TC, LAR, HDiv         |
| 細粒度動作          | Diving48, FineGym                             | 模型是否忽略細微時序差異           | K400 vs SSv2；再比 backbone       | 是否能分散到更多關鍵片段    | TC, TCo, HDiv, CC     |
| 程序式長時序         | 50 Salads, Breakfast, MPII Cooking2, YouCook2 | 模型是否有 short-range bias | HowTo100M / VideoPrism vs K400 | 是否跨步驟連結         | LRAR, LAR, SC, AVRank |
| Egocentric HOI | GTEA Gaze                                     | 模型是否被 ego-motion 帶偏    | weights 比較優先                   | attention 是否不穩定 | AS, TCo, LRAR, CC     |

### 表 B：統計量定義表

| 指標     | 定義                           | 解讀                |
| ------ | ---------------------------- | ----------------- |
| TC     | 前 20% frame 吃掉的 attention 比例 | 高 = 太依賴少數 frame   |
| TCo    | attention 超過門檻的 frame 比例     | 高 = 時間覆蓋較完整       |
| LAR    | attention 落在鄰近 frame 的比例     | 高 = 偏短距離資訊        |
| LRAR   | attention 落在遠距 frame 的比例     | 高 = 較能做長程連結       |
| HDiv   | 不同 head attention 分布的平均差異    | 高 = head 分工較多     |
| AS     | 同類樣本間 attention 分布相似度        | 高 = attention 較穩定 |
| SC     | 被足夠 attention 覆蓋的步驟數比例       | 高 = 程序理解較完整       |
| CC     | confusion matrix 中相似類別的混淆程度  | 高 = 細粒度辨識較差       |
| AVRank | 各 dataset 排名的平均              | 小 = 整體更穩定         |

如果你要，我下一則可以直接幫你把這兩張表改成 **論文風格的 LaTeX 版本**。
