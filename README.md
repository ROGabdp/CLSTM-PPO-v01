# CLSTM-PPO 股票交易系統

基於論文 **《A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks》**（Zou et al., 2023）實作的自動化股票交易系統，用於道瓊工業指數（DJI）回測。

## 論文參考

- **標題**：A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks
- **作者**：Jie Zou, Jiashu Lou, Baohua Wang, Sixue Liu
- **來源**：[arXiv:2212.02721](https://arxiv.org/abs/2212.02721)

## 模型架構

CLSTM-PPO 模型使用串聯式 LSTM 架構：

1. **LSTM 特徵提取器**：提取時序特徵（T=30） → 下採樣至 15 步 → Flatten → Linear 層（符合論文 15×128 輸入維度）
2. **RecurrentPPO 策略網路**：基於 LSTM 的策略網路（隱藏層：512）進行決策

![Architecture](reference/figure%2002.jpg)

## 專案結構

```
CLSTM-PPO-v01/
├── main.py                      # 統一入口
├── train.py                     # 滾動視窗訓練腳本
├── backtest.py                  # 回測與評估
├── config.py                    # 配置常數
├── data_fetcher.py              # 數據下載與技術指標
├── requirements.txt             # 依賴套件
├── envs/
│   └── multi_stock_trading_env.py  # Gym 環境
├── models/
│   └── lstm_feature_extractor.py   # LSTM 特徵提取器
├── models_saved/                # 已訓練模型
├── results/                     # 回測結果
├── data/                        # 股票數據快取
└── reference/                   # 論文與圖表
```

## 安裝

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境 (Windows)
.\venv\Scripts\activate

# 安裝依賴套件
pip install -r requirements.txt
```

## 快速開始

### 1. 驗證環境
```bash
python main.py test
```

### 2. 訓練模型
```bash
# 快速測試（1000 步）
python main.py train --test_run

# 完整訓練（預設 100k 步）
python main.py train

# 論文方式：滾動視窗訓練（每視窗訓練 500k 步）
python main.py train --rolling --timesteps 500000
```

### 3. 回測
```bash
# 回測已訓練模型
python main.py backtest --model_path models_saved/clstm_ppo_final.zip
```

## 論文規格

### 狀態空間（181 維）
| 成分 | 維度 | 說明 |
|------|------|------|
| 餘額 (b_t) | 1 | 可用現金 |
| 價格 (p_t) | 30 | 調整後收盤價 |
| 持股 (h_t) | 30 | 持有股數 |
| MACD (M_t) | 30 | 移動平均收斂發散 |
| RSI (R_t) | 30 | 相對強弱指標 |
| CCI (C_t) | 30 | 商品通道指標 |
| ADX (X_t) | 30 | 平均趨向指標 |

### 環境參數
| 參數 | 值 | 論文參考 |
|------|-----|---------|
| 初始資金 | $1,000,000 | Section 3.1.5 |
| 最大交易股數 | 100 | Section 3.1.5 |
| 交易成本 | 0.1% | Section 3.1.3 |
| 獎勵縮放 | 1e-4 | Section 3.1.5 |

### 模型參數
| 參數 | 值 | 論文參考 |
|------|-----|---------|
| 時間窗口 (T) | 30 天 | Table 2 |
| 特徵輸出維度 | 128 | Line 877 |
| PPO LSTM 隱藏層 | 512 | Table 3 |
| 折扣因子 (γ) | 0.99 | Table 1 |
| 學習率 | 3e-4 | Table 1 |
| Clip Range | 0.2 | Table 1 |

### 訓練/測試策略 (Rolling Window)
本系統嚴格遵循論文的滾動訓練機制：
- **滾動重訓練 (Rolling Retrain)**：每 3 個月滑動一次視窗，使用累積數據進行訓練。
- **持續學習 (Continuous Learning)**：模型在進入下一個視窗時保留參數繼續訓練，而非重置（`reset_num_timesteps=False`）。
- **參數凍結測試 (Deterministic Testing)**：在每個滾動視窗的 Out-Sample 測試階段，強制凍結模型參數（`deterministic=True`），確保測試結果真實反映當前模型能力。

- **風險控制 (Turbulence Index)**：當市場動盪指數（Turbulence）超過閾值時，強制執行**平倉 (Sell All)** 策略以規避風險（Reference: Paper Line 578）。
- **訓練期**：2009-01-01 至 2015-12-31（初始）
- **測試期**：2016-01-01 至 2020-05-08
- **滾動頻率**：每 3 個月

## 回測結果

### 論文目標 vs 實際結果

| 指標 | 論文目標 | 我們的結果 | 說明 |
|------|---------|-----------|------|
| CR (累積報酬) | 90.81% | 68.17% | 75% |
| MER (最大盈利率) | 113.50% | 97.90% | 86% |
| MPB (最大回撤) | 46.51% | 34.56% | ✅ 更優 |
| SR (夏普比率) | 1.1540 | 0.7172 | 62% |
| 最終淨值 | ~$1.9M | $1.68M | 盈利 |

> **注意**：若要更接近論文結果，建議增加訓練步數至 500k+

## DJI 30 成分股（2009-2020 期間有效）

```
AXP, AAPL, BA, CAT, CSCO, CVX, DD, DIS, GE, GS,
HD, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT,
NKE, PFE, PG, TRV, UNH, HON, V, VZ, WMT, XOM
```

## 授權條款

本專案僅供教育與研究用途。

## 引用

```bibtex
@article{zou2023novel,
  title={A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks},
  author={Zou, Jie and Lou, Jiashu and Wang, Baohua and Liu, Sixue},
  journal={Expert Systems with Applications},
  year={2023}
}
```
